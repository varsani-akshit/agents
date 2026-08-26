"""OpenAI-flavoured agent loop, mirroring brain/agent.py.

The two providers differ enough that one loop cannot serve both: Anthropic uses
`tool_use`/`tool_result` content blocks with explicit `cache_control` breakpoints
and an `effort` knob, while OpenAI uses `tool_calls` on the assistant message,
`role: "tool"` replies, automatic prefix caching, and `reasoning_effort`.

Keeping them as separate loops behind one shared toolbox is cheaper than an
abstraction that hides the differences badly.
"""
from __future__ import annotations

import json
import logging
import os
import time

import httpx

import db
from brain import tools

log = logging.getLogger("mia.agent_openai")

_URL = "https://api.openai.com/v1/chat/completions"

# USD per million tokens: (input, cached input, output). Verified against
# OpenAI's published pricing 2026-08-26.
PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.2": (1.75, 0.175, 14.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4o": (2.50, 1.25, 10.00),
}

# Models that expose `reasoning_effort`. Sending it to a non-reasoning model is
# a 400, so it is gated rather than always-on.
_REASONING = ("gpt-5", "o3", "o4")


def price(model: str, usage: dict) -> float:
    key = next((k for k in sorted(PRICING, key=len, reverse=True) if model.startswith(k)), None)
    inp, cached, out = PRICING.get(key, (2.5, 0.5, 10.0))
    total_in = usage.get("prompt_tokens", 0)
    cached_in = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    fresh_in = max(0, total_in - cached_in)
    return (
        fresh_in * inp / 1e6
        + cached_in * cached / 1e6
        + usage.get("completion_tokens", 0) * out / 1e6
    )


def to_openai_tools(defs: list[dict]) -> list[dict]:
    """Anthropic tool definitions -> OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d["description"],
                "parameters": d.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for d in defs
    ]


def _post(payload: dict) -> dict:
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = httpx.post(
                _URL,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json=payload,
                timeout=600,
            )
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"openai HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"openai failed after retries: {last}")


def run_agent(
    *,
    system: str,
    user_message: str,
    model: str,
    purpose: str,
    max_turns: int = 6,
    max_tokens: int = 10000,
    effort: str = "medium",
) -> dict:
    """Drive an OpenAI tool-use loop to completion.

    No web-search tool: OpenAI's hosted search is a different surface than
    Anthropic's server-side tool, so the OpenAI path runs on stored memory and
    computed statistics only. That is a real capability difference, not a
    detail — say so when comparing outputs.
    """
    oa_tools = to_openai_tools(tools.DEFINITIONS)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    calls: list[dict] = []
    spent = 0.0
    parts: list[str] = []
    stopped = None

    for turn in range(max_turns):
        payload: dict = {
            "model": model,
            "messages": messages,
            "tools": oa_tools,
            "max_completion_tokens": max_tokens,
        }
        if model.startswith(_REASONING):
            payload["reasoning_effort"] = effort
        else:
            payload["temperature"] = 0.3

        try:
            data = _post(payload)
        except Exception as exc:  # noqa: BLE001
            stopped = f"error: {exc}"
            log.error("openai agent failed: %s", exc)
            break

        usage = data.get("usage", {})
        usd = price(model, usage)
        spent += usd
        db.execute(
            """INSERT INTO api_calls
                 (provider,model,purpose,input_tokens,output_tokens,cache_read,usd)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (
                "openai", model, f"{purpose}:turn{turn}",
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                usd,
            ),
        )
        log.info(
            "%s:turn%s via %s: $%.5f (in=%s cached=%s out=%s)",
            purpose, turn, model, usd, usage.get("prompt_tokens"),
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            usage.get("completion_tokens"),
        )

        choice = data["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            if msg.get("content"):
                parts.append(msg["content"])
            if choice.get("finish_reason") == "length" and turn + 1 < max_turns:
                log.warning("hit token ceiling — requesting continuation")
                messages.append({"role": "assistant", "content": msg.get("content") or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was cut off by the output limit. "
                        "Continue from exactly where it stopped. Do not repeat "
                        "text you already wrote."
                    ),
                })
                continue
            break

        messages.append(msg)
        for tc in tool_calls:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            log.info("tool: %s(%s)", fn["name"], json.dumps(args, default=str)[:160])
            result = tools.dispatch(fn["name"], args)
            calls.append({"tool": fn["name"], "args": args, "chars": len(result)})
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )

    return {
        "text": "\n\n".join(p for p in parts if p).strip(),
        "tool_calls": calls,
        "turns": min(turn + 1, max_turns),
        "usd": round(spent, 5),
        "stopped": stopped,
    }
