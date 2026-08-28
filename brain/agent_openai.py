"""OpenAI agent loop on the Responses API, mirroring brain/agent.py.

Uses `/v1/responses` rather than chat completions because that is the only
surface exposing the built-in `web_search` tool alongside custom functions —
without it this path could only reason over the stored corpus, which is a real
handicap on a breaking-news day.

Shape differences from the Anthropic loop, all handled here rather than behind a
leaky abstraction: a flat `input` item list instead of `messages`, function tools
declared flat (not nested under a `function` key), `function_call` /
`function_call_output` items instead of tool_use/tool_result blocks, and
`reasoning: {effort}` instead of `output_config.effort`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx

import db
from brain import tools

log = logging.getLogger("mia.agent_openai")

_URL = "https://api.openai.com/v1/responses"

# USD per million tokens: (input, cached input, output). Verified against
# OpenAI's published pricing 2026-08-26.
PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.6": (5.00, 0.50, 30.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.2": (1.75, 0.175, 14.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4o": (2.50, 1.25, 10.00),
}

# Web search is billed per call on top of tokens.
WEB_SEARCH_USD = 10.0 / 1000

_REASONING = ("gpt-5", "o3", "o4")


def price(model: str, usage: dict, searches: int = 0) -> float:
    key = next((k for k in sorted(PRICING, key=len, reverse=True) if model.startswith(k)), None)
    inp, cached, out = PRICING.get(key, (2.5, 0.5, 10.0))
    total_in = usage.get("input_tokens", 0)
    cached_in = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    fresh_in = max(0, total_in - cached_in)
    return (
        fresh_in * inp / 1e6
        + cached_in * cached / 1e6
        + usage.get("output_tokens", 0) * out / 1e6
        + searches * WEB_SEARCH_USD
    )


def to_responses_tools(defs: list[dict], web_search: bool) -> list[dict]:
    """Anthropic tool definitions -> Responses API tool schema.

    Note the flat shape: `name`/`parameters` sit at the top level of the tool
    object, unlike chat completions where they nest under `function`.
    """
    out = [
        {
            "type": "function",
            "name": d["name"],
            "description": d["description"],
            "parameters": d.get("input_schema") or {"type": "object", "properties": {}},
        }
        for d in defs
    ]
    if web_search:
        out.append({"type": "web_search"})
    return out


def _retry_after(resp: httpx.Response) -> float:
    """Seconds to wait, from the header or the message body.

    OpenAI's 429 states exactly how long to wait ("try again in 5.63s"). A fixed
    short backoff ignores it and burns the retry budget before the window opens.
    """
    header = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-tokens")
    if header:
        try:
            return min(float(str(header).rstrip("smh")), 90.0)
        except ValueError:
            pass
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", resp.text, re.I)
    if m:
        secs = float(m.group(1)) / (1000 if m.group(2).lower() == "ms" else 1)
        return min(secs, 90.0)
    return 0.0


def _stream_once(payload: dict) -> dict:
    """One streamed Responses call, returning the final response object.

    Streaming is not an optimisation here, it is a requirement: OpenAI refuses
    any non-streamed request that could exceed ten minutes, and a digest at high
    reasoning effort over a 13k-token brief does exactly that. The scheduled
    16:05 cycle failed outright on `Streaming is required for operations that
    may take longer than 10 minutes` while the same call at low effort passed,
    which is the worst kind of bug — it only appears when the work is hardest.

    Nothing is rendered incrementally; the terminal `response.completed` event
    carries the same object the non-streamed endpoint would have returned.
    """
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    final: dict | None = None
    # Citations also arrive as their own events. They are usually present on the
    # terminal object too, but collecting both costs nothing and the merge
    # de-duplicates by URL.
    streamed_annotations: list[dict] = []
    with httpx.stream("POST", _URL, headers=headers,
                      json={**payload, "stream": True}, timeout=1800) as r:
        if r.status_code != 200:
            r.read()
            raise _HTTPError(r.status_code, r)
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                evt = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            kind = evt.get("type") or ""
            # `incomplete` is a real terminal state, not an error: it is how the
            # output ceiling reports itself, and the loop continues from it.
            if kind in ("response.completed", "response.incomplete"):
                final = evt.get("response")
            elif kind.endswith("annotation.added"):
                ann = evt.get("annotation")
                if isinstance(ann, dict):
                    streamed_annotations.append(ann)
            elif kind == "response.failed":
                err = (evt.get("response") or {}).get("error") or {}
                raise RuntimeError(f"openai response.failed: {err.get('message', evt)}")
            elif kind == "error":
                raise RuntimeError(f"openai stream error: {evt.get('message', evt)}")
    if final is None:
        raise RuntimeError("openai stream ended without a terminal response event")
    if streamed_annotations:
        final.setdefault("_streamed_annotations", streamed_annotations)
    return final


class _HTTPError(Exception):
    def __init__(self, status: int, resp: httpx.Response):
        super().__init__(f"HTTP {status}: {resp.text[:200]}")
        self.status = status
        self.resp = resp


def _post(payload: dict) -> dict:
    last: Exception | None = None
    for attempt in range(5):
        try:
            return _stream_once(payload)
        except _HTTPError as exc:
            last = exc
            if exc.status not in (429, 500, 502, 503):
                raise RuntimeError(f"openai {exc}") from exc
            wait = _retry_after(exc.resp) or (2 ** attempt)
            # Token-per-minute limits need a real pause, not a token gesture.
            wait = max(wait + 1.0, 5.0) if exc.status == 429 else wait
            log.warning("HTTP %s — waiting %.1fs before retry %d/5",
                        exc.status, wait, attempt + 1)
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("openai attempt %d/5 failed: %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"openai failed after retries: {last}")


# OpenAI wraps inline citations in Private Use Area delimiters — U+E200 opens,
# U+E201 closes, U+E202 separates — expecting the client to swap the span for a
# real link using the response's url_citation annotations. Taking the raw text
# leaves sequences like "\ue200cite\ue202turn0search1\ue201" in the prose,
# which render as garbage boxes. The span's contents ("turn0search1") is an
# internal search-result handle, not a URL, so there is nothing to recover from
# it locally: strip the markers and let the annotation list carry attribution.
_CITE_SPAN = re.compile("\ue200[^\ue201]*\ue201")
_CITE_STRAY = re.compile("[\ue200-\ue206]")


def strip_citation_markers(text: str) -> str:
    if not text:
        return text
    text = _CITE_SPAN.sub("", text)
    text = _CITE_STRAY.sub("", text)
    # Stripping mid-sentence can leave a doubled space or a space before punctuation.
    text = re.sub(r" {2,}", " ", text)
    return re.sub(r" +([.,;:)])", r"\1", text)


def _text_of(output: list[dict]) -> str:
    parts = []
    for item in output:
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    parts.append(c.get("text", ""))
    return strip_citation_markers("\n".join(p for p in parts if p).strip())


def _citations(output: list[dict], streamed: list[dict] | None = None) -> list[dict]:
    """Surface url_citation annotations so sources stay traceable.

    Reads both the message content and any annotations seen as stream events.
    A single-turn probe carries them on the terminal object, but the last digest
    finished with an empty source list even though a search ran two turns
    earlier, so both channels are merged and de-duplicated rather than trusting
    either alone.
    """
    cites = []
    for item in output:
        if item.get("type") != "message":
            continue
        for c in item.get("content", []):
            for a in c.get("annotations") or []:
                if a.get("type") == "url_citation":
                    cites.append({"url": a.get("url"), "title": a.get("title")})
    for a in streamed or []:
        if a.get("type") == "url_citation":
            cites.append({"url": a.get("url"), "title": a.get("title")})
    return cites


def run_agent(
    *,
    system: str,
    user_message: str,
    model: str,
    purpose: str,
    max_turns: int = 6,
    max_tokens: int = 12000,
    effort: str = "medium",
    use_web_search: bool = True,
) -> dict:
    """Drive an OpenAI Responses tool loop to completion."""
    tool_defs = to_responses_tools(tools.DEFINITIONS, use_web_search)
    conversation: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    calls: list[dict] = []
    all_citations: list[dict] = []
    spent = 0.0
    parts: list[str] = []
    stopped = None
    turn = 0

    for turn in range(max_turns):
        payload: dict = {
            "model": model,
            "input": conversation,
            "tools": tool_defs,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if model.startswith(_REASONING):
            payload["reasoning"] = {"effort": effort}

        try:
            data = _post(payload)
        except Exception as exc:  # noqa: BLE001
            stopped = f"error: {exc}"
            log.error("openai agent failed: %s", exc)
            break

        output = data.get("output", [])
        usage = data.get("usage", {})
        searches = sum(1 for o in output if o.get("type") == "web_search_call")
        usd = price(model, usage, searches)
        spent += usd
        db.execute(
            """INSERT INTO api_calls
                 (provider,model,purpose,input_tokens,output_tokens,cache_read,web_searches,usd)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                "openai", model, f"{purpose}:turn{turn}",
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                searches, usd,
            ),
        )
        log.info(
            "%s:turn%s via %s: $%.5f (in=%s cached=%s out=%s web=%s)",
            purpose, turn, model, usd, usage.get("input_tokens"),
            (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
            usage.get("output_tokens"), searches,
        )
        from brain import observe

        observe.record_llm(
            spec=f"openai:{model}", purpose=f"{purpose}:turn{turn}",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0), usd=usd,
        )

        all_citations.extend(
            _citations(output, data.get("_streamed_annotations"))
        )
        for o in output:
            if o.get("type") == "web_search_call":
                calls.append({"tool": "web_search", "args": o.get("action", {}), "chars": 0})

        # Every output item is echoed back so reasoning and tool state carry
        # forward — the Responses API is stateless when `store` is false.
        conversation.extend(output)

        fn_calls = [o for o in output if o.get("type") == "function_call"]
        if not fn_calls:
            text = _text_of(output)
            if text:
                parts.append(text)
            if data.get("status") == "incomplete" and turn + 1 < max_turns:
                reason = (data.get("incomplete_details") or {}).get("reason")
                if reason == "max_output_tokens":
                    log.warning("hit output ceiling — requesting continuation")
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Your previous response was cut off by the output limit. "
                            "Continue from exactly where it stopped. Do not repeat "
                            "text you already wrote."
                        ),
                    })
                    continue
            break

        for fc in fn_calls:
            try:
                args = json.loads(fc.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            log.info("tool: %s(%s)", fc["name"], json.dumps(args, default=str)[:160])
            result = tools.dispatch(fc["name"], args)
            calls.append({"tool": fc["name"], "args": args, "chars": len(result)})
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": fc["call_id"],
                    "output": result,
                }
            )

    # De-duplicate citations while preserving order.
    seen, cites = set(), []
    for c in all_citations:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            cites.append(c)

    return {
        "text": "\n\n".join(p for p in parts if p).strip(),
        "tool_calls": calls,
        "citations": cites,
        "turns": turn + 1,
        "usd": round(spent, 5),
        "stopped": stopped,
    }
