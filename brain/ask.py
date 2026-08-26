"""Interactive research mode — the conversational interface over full memory.

Same agent and same toolbox as the digest, different prompt and entry point.
That is the whole point: asking "why did silver lag gold last week?" pulls the
stored analyses from that week, the measured correlation history, the relationship
graph, and a fresh web search, because all of those are already tools.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import config
from brain import agent, client
from memory import store, world_model

log = logging.getLogger("mia.ask")

ROLE = """You are MIA, a senior macro strategist answering a direct question from
the investor you work for. Your domain is precious metals, fiat currencies,
sovereign debt, central bank policy, and crypto.

Answer the question that was asked. Lead with the answer, then the evidence.

You have the full history of what this system has ingested and concluded. Use it:
search stored memory to find what was reported and what you previously concluded,
check measured statistics for any number, traverse the relationship graph for
indirect connections, and search the web when the stored corpus is genuinely
insufficient or the question concerns something breaking right now.

Investigate before answering. A question about a specific past period should
send you to memory for that period, not to your general knowledge. If the stored
data cannot answer it, say so explicitly rather than filling the gap from
training data — and say what you would need.

The analysis discipline in your principles is binding here too: every number from
a tool result, correlations quoted as measured, confidence stated honestly, and
the distinction between observation, mechanism, interpretation, and speculation
kept visible.

Write in prose, not headers, unless the answer genuinely needs structure. Be
substantive but not padded — this reader is sophisticated and does not need
macro concepts explained from scratch."""


def ask(
    question: str,
    *,
    max_turns: int = 8,
    save: bool = True,
    model: str | None = None,
    use_web_search: bool = True,
) -> dict:
    """Answer one question with full tool access. Returns text plus provenance."""
    system = [
        {"type": "text", "text": ROLE},
        {
            "type": "text",
            "text": agent.load_principles(),
            "cache_control": {"type": "ephemeral"},
        },
    ]

    wm = world_model.current_body()
    user = f"""Current time: {datetime.now(timezone.utc).isoformat()}

Your standing world model (written by the most recent analysis cycle):
{wm[:4000]}

---
Question: {question}"""

    result = agent.run_agent(
        system=system,
        user_message=user,
        model=model or config.ASK_MODEL,
        purpose="ask",
        max_turns=max_turns,
        max_tokens=12000,
        use_web_search=use_web_search,
        effort="high",
    )

    answer = result["text"]
    analysis_id = None
    if save and answer:
        analysis_id = store.save_analysis(
            "answer",
            f"Q: {question[:160]}",
            answer,
            meta={
                "question": question,
                "tool_calls": result["tool_calls"],
                "turns": result["turns"],
                "usd": result["usd"],
            },
        )

    return {
        "question": question,
        "answer": answer,
        "analysis_id": analysis_id,
        "tools_used": [c["tool"] for c in result["tool_calls"]],
        "turns": result["turns"],
        "usd": result["usd"],
        "stopped": result.get("stopped"),
    }
