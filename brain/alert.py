"""Critical-alert writer.

Code has already decided that something happened. The model's only job is to say
concisely what it was and why it might matter — with a hard rule against
inventing a cause. Runs on the cheap model, no tools, low token ceiling, so the
alert path stays fast and nearly free.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import config
import db
from brain import client, llm
from signals import stats

log = logging.getLogger("mia.alert")

SYSTEM = """You write terse macro alerts for one sophisticated investor.

A code-based rule has already determined this is material — do not re-argue
whether it matters, and do not hedge about whether it is significant.

Write:
1. One headline line: what happened, with the number. Max 15 words.
2. Two to four sentences: the immediate context and the most plausible reading.
   If several readings are possible, say so rather than picking one.
3. One line starting "Watch:" — the specific next observation that would confirm
   or refute that reading.

Hard rules:
- Use only numbers present in the data given to you. Never introduce a figure.
- If no cause is evident in the supplied context, say the move is unexplained by
  available information. Do not invent a driver. "Gold rose on debasement fears"
  when no such news exists is exactly the failure this system must avoid.
- No preamble, no sign-off, no markdown headers. Plain text."""


def _context_for(event: dict) -> dict:
    """Assemble only what the model is allowed to reason from."""
    pack = stats.latest_pack() or {}
    perf = {p["symbol"]: p for p in pack.get("performance", [])}
    ctx: dict = {
        "event": {
            "rule": event["rule"],
            "severity": event["severity"],
            "symbol": event.get("symbol"),
            "detail": event.get("detail"),
        },
        "yield_curve_bp": pack.get("yield_curve"),
        "_units": (
            "Yield instruments (US2Y/US5Y/US10Y/US30Y) are quoted as percent "
            "yields. Their daily change is given ONLY in basis points, as "
            "`chg_1d_basis_points`. Do not describe a yield move in percent, and "
            "never read a percent figure as basis points."
        ),
    }
    sym = event.get("symbol")

    def _describe(symbol: str, p: dict) -> dict:
        """Emit a unit-unambiguous view. Percent change on a yield is meaningless
        to a reader and invites misreading it as basis points, so rates carry bp
        only and prices carry percent only."""
        is_rate = symbol.startswith("US") and symbol.endswith("Y")
        if is_rate:
            return {
                "current_yield_pct": p.get("last"),
                "chg_1d_basis_points": p.get("chg_1d_bp"),
                "chg_1w_basis_points": p.get("chg_1w_bp"),
                "z_1d_move": p.get("z_1d_move"),
            }
        return {
            "last_usd": p.get("last"),
            "chg_1d_pct": p.get("chg_1d_pct"),
            "chg_1w_pct": p.get("chg_1w_pct"),
            "chg_1m_pct": p.get("chg_1m_pct"),
            "z_1d_move": p.get("z_1d_move"),
        }

    if sym and sym in perf:
        ctx["instrument"] = _describe(sym, perf[sym])
    # Cross-asset context: whether this move is idiosyncratic or part of a
    # broader one.
    ctx["other_moves"] = {s: _describe(s, p) for s, p in perf.items() if s != sym}
    if event.get("doc_id"):
        doc = db.one(
            "SELECT title, summary, source, source_tier, url FROM documents WHERE id=%s",
            (event["doc_id"],),
        )
        if doc:
            ctx["document"] = doc

    recent = db.query(
        """SELECT title, summary, source, source_tier FROM documents
           WHERE fetched_at > now() - interval '8 hours'
             AND urgency IN ('Critical','High')
           ORDER BY urgency_score DESC, source_tier LIMIT 8"""
    )
    ctx["recent_high_urgency_news"] = recent
    return ctx


def write(event: dict) -> dict:
    ctx = _context_for(event)
    try:
        resp = client.complete(
            model=llm.parse_spec(config.ALERT_SPEC)[1],
            purpose="alert",
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(ctx, default=str)[:12000]}],
            max_tokens=500,
            estimated_usd=0.005,
        )
        text = client.text_of(resp)
    except client.BudgetExceeded as exc:
        log.warning("alert written without model (%s)", exc)
        text = _fallback_text(event)
    except Exception as exc:  # noqa: BLE001
        log.error("alert generation failed: %s", exc)
        text = _fallback_text(event)

    title = f"{event['severity']}: {event.get('symbol') or event['rule']}"
    return {
        "title": title,
        "text": text,
        "event_id": event.get("id"),
        "rule": event["rule"],
        "severity": event["severity"],
        "symbol": event.get("symbol"),
        "detail": event.get("detail"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _fallback_text(event: dict) -> str:
    """Deterministic alert when the model is unavailable — never lose the signal."""
    d = event.get("detail") or {}
    bits = ", ".join(f"{k}={v}" for k, v in d.items() if isinstance(v, (int, float, str)))
    return (
        f"{event['rule']} fired for {event.get('symbol') or 'system'} "
        f"({event['severity']}). {bits}\n"
        "Watch: model commentary unavailable — inspect the stats pack directly."
    )
