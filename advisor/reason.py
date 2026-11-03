"""The case for a call: why this, why at this price, and what would break it.

Everything else Alfred produces is a measurement or a summary. The strategy
fired. The record says +0.84R. The news read says neutral. All true, all
assembled — and none of it an argument. Reading three facts side by side and
deciding is work the system was leaving to the reader.

This writes the argument. It takes the same evidence the rest of the pipeline
already produced and asks one question of it: on this evidence, is there a
reason to buy this at this price, and what is it?

Two rules it is built around:

- **Numbers from code, reasoning from the model.** Every figure in the output
  is passed in, already computed. The model may not produce a price, a
  percentage or a date; it reasons about the ones it is given. A model that
  invents a stop loses the only property that makes this worth reading.

- **A case that cannot be wrong is not a case.** The invalidation is required,
  and it must be observable — a level, a report, a condition that flips. "The
  thesis may not play out" is a refusal to commit, and is rejected as such.

The output is deliberately allowed to say *don't*. A system that produces a
reason to buy for every setup it is handed is not reasoning, it is decorating.
"""
from __future__ import annotations

import json
import logging

import db

log = logging.getLogger("alfred.reason")

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["take", "take_smaller", "wait", "skip"]},
        "headline": {"type": "string"},
        "why_now": {"type": "string"},
        "why_this_price": {"type": "string"},
        "what_must_hold": {"type": "array", "items": {"type": "string"}},
        "what_would_break_it": {"type": "string"},
        "against": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "moderate", "low"]},
    },
    "required": ["verdict", "headline", "why_now", "why_this_price", "what_must_hold",
                 "what_would_break_it", "against", "confidence"],
    "additionalProperties": False,
}

SYSTEM = """You make the case for one trade setup, or argue against it.

You are given a setup that a tested strategy produced, the strategy's own record, the market's current conditions, and a read of the news around the name. All the numbers are already computed and correct.

Write the reasoning that connects them. Not a summary of them — the reader can already see the figures. What they cannot see is whether, taken together, these facts amount to a reason to commit money at this price.

Hard rules:

- Never state a number that was not given to you. No prices, percentages, dates, or counts of your own. Refer to the ones provided.
- The case must be falsifiable. `what_would_break_it` is one observable development or level that would tell the reader the argument is wrong — not "it might not work".
- `against` is the strongest honest argument for NOT taking this. It is required even when you recommend taking it. If you cannot find one, say what would have to be true for a sceptic to be right.
- Do not re-judge the chart pattern; the strategy already did. Judge whether the surrounding evidence supports acting on it.
- Plain, direct English. No hedging stacks ("may potentially possibly"), no broker-speak, no exhortation. You are advising one careful reader, not selling.

verdict:
- take — the evidence supports acting at the stated price.
- take_smaller — the setup is sound but something (conditions, an event inside the hold, a thin record on this name) argues for less than the suggested size.
- wait — the setup is sound but the price or the timing is not; say what to wait for.
- skip — the evidence does not support it. Use this when it is true. Most setups in most weeks do not deserve money.

why_now: two or three sentences on what has changed that makes this worth attention now, drawing on the conditions and the news read.
why_this_price: two or three sentences on what the entry and stop imply — what the reader is risking per unit, what has to happen for it to work, and whether the distance to the stop is reasonable given how the name moves.
what_must_hold: two to four short conditions that have to stay true for the case to survive.
headline: one sentence, under twenty words, that states the case."""


def case(call: dict, *, store: bool = True) -> dict | None:
    """Write the case for one call. Returns the reasoning, or None if it could
    not be written — which is reported rather than filled with a guess."""
    from advisor import story
    from brain import router

    payload = _evidence(call)
    if payload is None:
        return None
    try:
        out, spec = router.complete_json(
            "reason", system=SYSTEM, user=json.dumps(payload, default=str, ensure_ascii=False),
            schema=SCHEMA, purpose="reason.case", max_tokens=1200)
    except Exception as exc:  # noqa: BLE001 - a missing case is better than a wrong one
        log.warning("case for %s/%s failed: %s", call.get("symbol"), call.get("strategy"), exc)
        return None

    out["model"] = spec
    out["evidence"] = {"news_articles": len(payload.get("news") or []),
                       "has_record": payload["record"]["expectancy_r"] is not None,
                       "news_direction": (payload.get("news_read") or {}).get("direction")}
    if store and call.get("id"):
        db.execute("UPDATE calls SET reasoning=%s, reasoned_at=now() WHERE id=%s",
                   (json.dumps(out, default=str), call["id"]))
    return out


def _evidence(call: dict) -> dict | None:
    """Everything the case may reason from, assembled in code.

    Deliberately flat and literal: the model's job is to argue, so anything it
    would otherwise have to infer — the risk per unit, how wide the stop is
    against the name's usual range — is computed here where it can be checked.
    """
    from advisor import bare, story

    d = call.get("detail") or {}
    limit = _f(call.get("limit_price") or d.get("limit"))
    stop = _f(call.get("stop_price") or d.get("stop"))
    if limit is None or stop is None:
        return None
    risk = limit - stop

    try:
        dos = story.dossier(call)
    except Exception as exc:  # noqa: BLE001 - reason from what we have
        log.info("dossier for %s failed: %s", call.get("symbol"), exc)
        dos = {}

    read = call.get("story") or {}
    return {
        "instrument": {
            "name": dos.get("name") or bare(call["symbol"]),
            "ticker": bare(call["symbol"]),
            "exchange": dos.get("exchange") or call.get("region"),
            "currency": dos.get("currency"),
            "recent_moves": dos.get("moves") or {},
        },
        "setup": {
            "strategy": d.get("label") or call.get("strategy"),
            "what_the_strategy_looks_for": d.get("summary"),
            "why_it_fired": d.get("reasons") or [],
            "buy_limit": round(limit, 4),
            "stop": round(stop, 4),
            "target": _f(call.get("target_price") or d.get("target")),
            "risk_per_unit": round(risk, 4),
            "stop_distance_pct": _round(d.get("stop_pct"), 2),
            "suggested_quantity": d.get("quantity"),
            "typical_hold_sessions": d.get("median_days"),
            "signal_date": str(call.get("signal_date") or ""),
        },
        "record": {
            "expectancy_r": _round((d.get("evidence") or {}).get("expectancy_r")
                                   or d.get("expectancy_r"), 3),
            "this_name_expectancy_r": _round(d.get("name_expectancy_r"), 3),
            "this_name_trades": d.get("name_trades"),
            "passes_the_gate": bool(call.get("passed_gate")),
            "why_not": d.get("fails") or [],
            "note": "expectancy_r is the average result per trade in R, where "
                    "one R is the distance from entry to stop.",
        },
        "conditions": {
            "states": dos.get("conditions") or [],
            "strategy_on_today": d.get("regime_on"),
            "why": d.get("regime_why"),
            "world": dos.get("world"),
        },
        "news_read": {
            "direction": read.get("direction"),
            "strength": read.get("strength"),
            "thesis": read.get("thesis"),
            "risk": read.get("risk"),
            "event_risk": read.get("event_risk"),
            "flags": read.get("flags") or [],
        } if read else None,
        "events_inside_the_hold": dos.get("events_in_window") or [],
        "news": [{"when": n.get("when"), "source": n.get("source"), "title": n.get("title")}
                 for n in (dos.get("news") or [])[:12]],
    }


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _round(v, n):
    f = _f(v)
    return None if f is None else round(f, n)


def write_new(limit: int = 6) -> dict:
    """Cases for the calls that have a news read but no case yet.

    Live tickets first. The queue runs to a hundred and fifty, and a cap of
    six a run ordered by recency meant the setups actually sitting on the
    dashboard — which are a few days old by the time they are standing there
    — never reached the front, while cases were written for calls nobody
    could act on. Whether a ticket exists for the call is the honest measure
    of whether the reader needs the argument.

    Only calls that passed the gate: a case for a setup nobody may act on
    costs the same as one for a setup they might.
    """
    rows = db.query(
        """SELECT c.* FROM calls c
           LEFT JOIN ideas i
             ON i.symbol = c.symbol AND i.strategy = c.strategy
            AND i.status IN ('open', 'filled')
           WHERE c.story IS NOT NULL AND c.reasoning IS NULL AND c.passed_gate
             AND c.status = 'pending'
           ORDER BY (i.id IS NOT NULL) DESC, c.created_at DESC,
                    c.score DESC NULLS LAST
           LIMIT %s""", (limit,))
    done, skipped = 0, 0
    for row in rows:
        out = case(row)
        if out is None:
            skipped += 1
            continue
        done += 1
        log.info("case %s %s: %s — %s", row["symbol"], row["strategy"],
                 out.get("verdict"), out.get("headline"))
    return {"written": done, "skipped": skipped, "considered": len(rows)}
