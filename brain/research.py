"""The Deep Researcher: an on-demand investigation of one question.

The proven deep-research shape, applied to a single query: a supervisor breaks
the question into facets, context-isolated sub-researchers work each facet in
parallel with live search and the full toolbox, and exactly one synthesis call
writes the note. Triggered from the dashboard or chat; the result is stored,
so an investigation is an asset rather than a chat message that scrolls away.
"""
from __future__ import annotations

import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import db
from brain import agent_gemini, observe, router

log = logging.getLogger("mia.research")

MAX_FACETS = 4

_FACET_SCHEMA = {
    "type": "object",
    "properties": {
        "facets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "facet": {"type": "string", "description": "the sub-question"},
                    "angle": {"type": "string",
                              "description": "how to attack it: which sources, which measurements"},
                },
                "required": ["facet", "angle"],
            },
        },
        "measurements": {"type": "array", "items": {"type": "string"},
                         "description": "series/statistics worth computing for this question"},
    },
    "required": ["facets"],
}

_RESEARCHER_SYSTEM = """You are a sub-researcher on one facet of a larger
question. Work ONLY your facet; the supervisor holds the whole.

Method: search the live web (at least two searches), fetch the one or two most
load-bearing articles in full, check search_memory for what the desk already
knows, and use run_python when your facet needs a number computed from the
price/FRED series. Then COMPRESS: return a tight markdown report —
### Findings (dated facts with numbers), ### Evidence (quotes with source
URLs), ### Judgement (what it means for the question, clearly marked as
judgement). Under 500 words. Only cite URLs you actually saw."""


def run(question: str, *, trigger: str = "ask") -> dict:
    """Investigate one question end to end. Returns the stored note."""
    started = datetime.now(timezone.utc)
    with observe.run("deep-research", trigger=trigger) as rec:
        rec.set_input({"question": question})

        # ── Supervisor: decompose ────────────────────────────────────────────
        with observe.stage("plan", kind="generic") as sp:
            plan, _ = router.complete_json(
                "deep",
                system="You direct a research team. Split the question into "
                "2-4 genuinely independent facets a researcher can pursue "
                "alone. Facets must not overlap.",
                user=f"Question: {question}\nToday: {started.date().isoformat()}",
                schema=_FACET_SCHEMA, purpose="research.plan", max_tokens=4000,
            )
            facets = plan.get("facets", [])[:MAX_FACETS]
            sp.set_output(plan)

        # ── Sub-researchers, parallel and context-isolated ───────────────────
        def _facet(idx_facet):
            idx, facet = idx_facet
            with observe.stage(f"facet:{idx+1}", kind="generic", input=facet) as sp:
                result = agent_gemini.run_agent(
                    system=_RESEARCHER_SYSTEM,
                    user_message=(
                        f"The larger question: {question}\n\n"
                        f"Your facet: {facet['facet']}\n"
                        f"Angle: {facet['angle']}\n"
                        f"Now: {datetime.now(timezone.utc).isoformat()}"
                    ),
                    model="gemini-flash-latest",
                    purpose=f"research.facet{idx+1}",
                    max_turns=6, max_tokens=8000, effort="medium",
                    use_web_search=True,
                )
                sp.set_output({"report": result.get("text", ""),
                               "citations": result.get("citations", [])})
                return {"facet": facet["facet"], "report": result.get("text", ""),
                        "citations": result.get("citations", []),
                        "usd": result.get("usd", 0)}

        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(contextvars.copy_context().run, _facet, (i, f))
                       for i, f in enumerate(facets)]
            reports = [f.result() for f in futures]

        # ── Synthesis: the single writer ─────────────────────────────────────
        citations, seen = [], set()
        for r in reports:
            for c in r["citations"]:
                if c.get("url") and c["url"] not in seen:
                    seen.add(c["url"])
                    citations.append(c)

        with observe.stage("synthesis", kind="generic") as sp:
            body, spec, usd = router.complete_text(
                "premium", premium_site="research_synthesis",
                system=(
                    "You are Alfred, writing a research note for your reader. "
                    "British English, formal register, no American colloquialism. "
                    "Structure: # headline stating the answer · *standfirst* · "
                    "## The Answer (direct, first) · ## The Evidence (by theme, "
                    "not by researcher, quotes and numbers inline with source "
                    "links) · ## What Would Change This View. 600-1200 words. "
                    "Numbers only from the reports; cite only URLs they contain. "
                    "Where researchers disagree, say so rather than averaging."
                ),
                user=(
                    f"Question: {question}\n\n"
                    + "\n\n---\n\n".join(
                        f"## Facet: {r['facet']}\n{r['report']}" for r in reports)
                    + "\n\n# Source URLs seen\n"
                    + json.dumps(citations, default=str)[:4000]
                ),
                purpose="research.synthesis", max_tokens=20000,
            )
            sp.set_output({"chars": len(body), "spec": spec})

        total = _spend(started)
        row = db.one(
            """INSERT INTO research_notes (question, body, facets, usd)
               VALUES (%s,%s,%s,%s) RETURNING id""",
            (question, body, json.dumps(
                [{k: r[k] for k in ("facet", "report", "citations")} for r in reports],
                default=str), round(total, 5)),
        )
        note_id = row["id"]
        rec.set_output({"note_id": note_id, "usd": total,
                        "facets": len(reports), "sources": len(citations)})
        return {"ok": True, "note_id": note_id, "body": body,
                "usd": total, "facets": len(reports), "citations": citations}


def _spend(since: datetime) -> float:
    row = db.one(
        "SELECT coalesce(sum(usd),0) s FROM api_calls WHERE purpose LIKE 'research.%%' AND created_at >= %s",
        (since,))
    return float(row["s"]) if row else 0.0
