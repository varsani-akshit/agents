"""The agentic loop: model + tools, with a bounded number of turns.

Shared by the digest cycle and interactive `ask`. Uses a manual loop rather than
the SDK tool runner because every iteration needs a budget check — an unattended
agent that can call tools indefinitely is the realistic way to burn a prepaid
balance, so the loop must be able to stop itself mid-flight.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import config
from brain import client, tools

log = logging.getLogger("mia.agent")


def load_principles() -> str:
    """Concatenate the principles corpus — the standing intellectual frame."""
    parts = []
    for path in sorted(config.PRINCIPLES_DIR.glob("*.md")):
        parts.append(path.read_text())
    return "\n\n---\n\n".join(parts)


def _clear_cache_marks(messages: list[dict]) -> None:
    """Only the newest turn should carry a breakpoint (max 4 per request)."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)


def run_agent(
    *,
    system: list[dict] | str,
    user_message: str,
    model: str,
    purpose: str,
    max_turns: int = 8,
    max_tokens: int = 8000,
    use_web_search: bool = True,
    effort: str = "high",
) -> dict:
    """Drive the tool-use loop to completion. Returns text, transcript, and cost."""
    tool_defs = list(tools.DEFINITIONS)
    if use_web_search:
        tool_defs.append(tools.WEB_SEARCH_TOOL)

    # The whole message history is resent on every turn of a tool loop, so an
    # uncached prompt is re-billed at full price each turn — the dominant cost in
    # a multi-turn agent. A cache breakpoint on the newest turn means each turn
    # reads the accumulated prefix at ~10% of input price instead.
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    calls: list[dict] = []
    spent_start = client.spend_total()
    stopped_reason = None

    for turn in range(max_turns):
        try:
            resp = client.complete(
                model=model,
                purpose=f"{purpose}:turn{turn}",
                system=system,
                messages=messages,
                tools=tool_defs,
                max_tokens=max_tokens,
                estimated_usd=0.08,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
            )
        except client.BudgetExceeded as exc:
            stopped_reason = f"budget: {exc}"
            log.warning("agent halted — %s", exc)
            break
        except Exception as exc:  # noqa: BLE001
            stopped_reason = f"error: {type(exc).__name__}: {exc}"
            log.error("agent call failed: %s", exc)
            break

        # A paused turn means a server-side tool hit its iteration cap; re-send
        # to resume rather than treating the partial answer as final.
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue

        if resp.stop_reason == "refusal":
            stopped_reason = "refusal"
            break

        tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not tool_uses:
            messages.append({"role": "assistant", "content": resp.content})
            return {
                "text": client.text_of(resp),
                "messages": messages,
                "tool_calls": calls,
                "turns": turn + 1,
                "usd": round(client.spend_total() - spent_start, 5),
                "stopped": stopped_reason,
            }

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            args = tu.input if isinstance(tu.input, dict) else {}
            log.info("tool: %s(%s)", tu.name, json.dumps(args, default=str)[:160])
            payload = tools.dispatch(tu.name, args)
            calls.append({"tool": tu.name, "args": args, "chars": len(payload)})
            results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": payload}
            )
        # Move the cache breakpoint to the newest turn: the prior turns become a
        # cached prefix, so only the delta is billed at full rate.
        _clear_cache_marks(messages)
        if results:
            results[-1]["cache_control"] = {"type": "ephemeral"}
        messages.append({"role": "user", "content": results})

    # Fell out of the loop: either turns exhausted or halted mid-flight.
    last_text = ""
    for m in reversed(messages):
        if m["role"] == "assistant":
            blocks = m["content"] if isinstance(m["content"], list) else []
            text = "\n".join(
                b.text for b in blocks if getattr(b, "type", "") == "text"
            ).strip()
            if text:
                last_text = text
                break
    return {
        "text": last_text,
        "messages": messages,
        "tool_calls": calls,
        "turns": max_turns,
        "usd": round(client.spend_total() - spent_start, 5),
        "stopped": stopped_reason or "max_turns",
    }
