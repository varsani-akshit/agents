"""Gemini agent loop, mirroring brain/agent.py and brain/agent_openai.py.

Runs the whole system on one provider: Gemini Flash handles classification and
extraction already, and this adds the tool loop the brief and the ask path need.

Two Gemini-specific details, both discovered by probing rather than assumed:

  Custom functions and Google Search can be used together, but only when
  `toolConfig.includeServerSideToolInvocations` is set — without it the API
  rejects the request outright with a 400 naming the flag.

  Gemini's function schema is an OpenAPI subset, not full JSON Schema. Keys it
  does not recognise (additionalProperties, $schema, enum on non-strings) cause
  a 400, so tool definitions are translated rather than passed through.

Streamed for the same reason as the OpenAI path: a high-effort brief runs for
minutes, and a single long-blocking request is the fragile way to wait.
"""
from __future__ import annotations

import json
import logging
import os
import time

import httpx

import db
from brain import tools

log = logging.getLogger("alfred.agent_gemini")

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# USD per million tokens: (input, output). Verified against Google's published
# pricing 2026-08-27; Flash is roughly a tenth of gpt-5.1 on output.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-pro-latest": (1.25, 10.00),
    "gemini-3.7-flash": (0.30, 2.50),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}

# Schema keys Gemini's OpenAPI subset accepts. Anything else is dropped.
_SCHEMA_KEYS = {"type", "description", "properties", "required", "items", "enum",
                "nullable", "format"}


def price(model: str, usage: dict) -> float:
    key = next((k for k in sorted(PRICING, key=len, reverse=True) if model.startswith(k)), None)
    inp, out = PRICING.get(key, (0.30, 2.50))
    prompt = usage.get("promptTokenCount", 0)
    # Thinking tokens are billed as output but reported separately.
    completion = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return prompt * inp / 1e6 + completion * out / 1e6


def _clean_schema(node):
    """Strip JSON Schema keys Gemini rejects, recursively."""
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k not in _SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: _clean_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _clean_schema(v)
        elif k == "type" and isinstance(v, str):
            out[k] = v.upper()
        else:
            out[k] = v
    return out


def to_gemini_tools(defs: list[dict], web_search: bool) -> list[dict]:
    declarations = [
        {
            "name": d["name"],
            "description": d["description"],
            "parameters": _clean_schema(
                d.get("input_schema") or {"type": "object", "properties": {}}),
        }
        for d in defs
    ]
    out: list[dict] = [{"functionDeclarations": declarations}]
    if web_search:
        out.append({"googleSearch": {}})
    return out


def _stream(model: str, payload: dict) -> dict:
    """One streamed generateContent call, merged into a single response object."""
    url = f"{_BASE}/{model}:streamGenerateContent"
    params = {"key": os.environ["GEMINI_API_KEY"], "alt": "sse"}
    parts: list[dict] = []
    usage: dict = {}
    grounding: dict = {"webSearchQueries": [], "groundingChunks": []}
    finish = None

    with httpx.stream("POST", url, params=params, json=payload, timeout=1800) as r:
        if r.status_code != 200:
            r.read()
            raise _GeminiError(r.status_code, r.text)
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if not chunk:
                continue
            try:
                evt = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if evt.get("usageMetadata"):
                usage = evt["usageMetadata"]
            for cand in evt.get("candidates", []) or []:
                finish = cand.get("finishReason") or finish
                for p in (cand.get("content") or {}).get("parts", []) or []:
                    parts.append(p)
                gm = cand.get("groundingMetadata") or {}
                grounding["webSearchQueries"].extend(gm.get("webSearchQueries") or [])
                grounding["groundingChunks"].extend(gm.get("groundingChunks") or [])
    return {"parts": parts, "usage": usage, "grounding": grounding, "finish": finish}


class _GeminiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


def _post(model: str, payload: dict) -> dict:
    last: Exception | None = None
    for attempt in range(5):
        try:
            return _stream(model, payload)
        except _GeminiError as exc:
            last = exc
            if exc.status not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"gemini {exc}") from exc
            wait = min(4 * (attempt + 1), 45)
            log.warning("HTTP %s — waiting %ss before retry %d/5", exc.status, wait, attempt + 1)
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("gemini attempt %d/5 failed: %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini failed after retries: {last}")


def _citations(grounding: dict) -> list[dict]:
    seen, out = set(), []
    for chunk in grounding.get("groundingChunks") or []:
        web = chunk.get("web") or {}
        uri = web.get("uri")
        if uri and uri not in seen:
            seen.add(uri)
            out.append({"url": uri, "title": web.get("title")})
    return out


def run_agent(
    *,
    system: str,
    user_message: str,
    model: str,
    purpose: str,
    max_turns: int = 8,
    max_tokens: int = 16000,
    effort: str = "medium",
    use_web_search: bool = True,
) -> dict:
    """Drive a Gemini tool loop to completion. Same contract as the other loops."""
    tool_defs = to_gemini_tools(tools.DEFINITIONS, use_web_search)
    contents: list[dict] = [{"role": "user", "parts": [{"text": user_message}]}]
    calls: list[dict] = []
    all_citations: list[dict] = []
    text_parts: list[str] = []
    spent = 0.0
    stopped = None
    turn = 0

    # Gemini exposes reasoning as a token budget rather than a named effort.
    budget = {"low": 2048, "medium": 8192, "high": 24576}.get(effort, 8192)

    for turn in range(max_turns):
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "tools": tool_defs,
            # Without this, combining our functions with googleSearch is a 400.
            "toolConfig": {"includeServerSideToolInvocations": True},
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "thinkingConfig": {"thinkingBudget": budget},
            },
        }
        try:
            data = _post(model, payload)
        except Exception as exc:  # noqa: BLE001
            stopped = f"error: {exc}"
            log.error("gemini agent failed: %s", exc)
            break

        usd = price(model, data["usage"])
        spent += usd
        searches = len(data["grounding"].get("webSearchQueries") or [])
        db.execute(
            """INSERT INTO api_calls
                 (provider,model,purpose,input_tokens,output_tokens,cache_read,web_searches,usd)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            ("gemini", model, f"{purpose}:turn{turn}",
             data["usage"].get("promptTokenCount", 0),
             data["usage"].get("candidatesTokenCount", 0)
             + data["usage"].get("thoughtsTokenCount", 0),
             (data["usage"].get("cachedContentTokenCount") or 0),
             searches, usd),
        )
        log.info("%s:turn%s via %s: $%.5f (in=%s out=%s search=%s)",
                 purpose, turn, model, usd,
                 data["usage"].get("promptTokenCount"),
                 data["usage"].get("candidatesTokenCount"), searches)

        all_citations.extend(_citations(data["grounding"]))

        fn_calls = [p["functionCall"] for p in data["parts"] if "functionCall" in p]
        turn_text = "".join(p.get("text", "") for p in data["parts"] if "text" in p)

        # Echo the model's own turn back before answering it, or the loop loses
        # the thread of what it asked for.
        contents.append({"role": "model", "parts": data["parts"] or [{"text": turn_text}]})

        if not fn_calls:
            if turn_text:
                text_parts.append(turn_text)
            if data["finish"] == "MAX_TOKENS" and turn + 1 < max_turns:
                log.warning("hit output ceiling — requesting continuation")
                contents.append({"role": "user", "parts": [{"text":
                    "Your previous response was cut off by the output limit. "
                    "Continue from exactly where it stopped. Do not repeat text "
                    "you already wrote."}]})
                continue
            break

        responses = []
        for fc in fn_calls:
            args = fc.get("args") or {}
            log.info("tool: %s(%s)", fc["name"], json.dumps(args, default=str)[:160])
            result = tools.dispatch(fc["name"], args)
            calls.append({"tool": fc["name"], "args": args, "chars": len(result)})
            responses.append({"functionResponse": {
                "name": fc["name"], "response": {"result": result}}})
        contents.append({"role": "user", "parts": responses})

    seen, cites = set(), []
    for c in all_citations:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            cites.append(c)

    return {
        "text": "\n\n".join(t for t in text_parts if t).strip(),
        "tool_calls": calls,
        "citations": cites,
        "turns": turn + 1,
        "usd": round(spent, 5),
        "stopped": stopped,
    }
