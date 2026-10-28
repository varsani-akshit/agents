"""The scan, written as an email of tickets.

Plain text first — it is what reads on a lock screen — with an HTML twin for
the inbox. Every ticket carries its reasons and its evidence, and every email
says the same thing at the top: advice, not orders; you place them.

Sent only when there is something to act on: a new ticket, or a holding that
has just closed below its trailing stop. A quiet day sends nothing.
"""
from __future__ import annotations

import html

from notify import email

REGION_NAME = {"IN": "India", "ASX": "Australia", "US": "US"}
BROKER_NAME = {"kite": "Kite", "webull": "Webull"}
HEADER = ("Advice only — Alfred never places orders. If you act on a ticket, "
          "place it yourself in the broker named.")


def _money(ccy: str, v) -> str:
    sym = {"INR": "₹", "AUD": "A$", "USD": "US$"}.get(ccy, ccy + " ")
    return f"{sym}{v:,.2f}" if v is not None else "—"


def _evidence_line(e: dict) -> str:
    def pct(v):
        return f"{v * 100:.0f}%" if v is not None else "—"

    return (f"Simulated on {e['symbols']} names over 12 years. Recent period "
            f"({e['region_trades']} trades since {e['recent_from']}) {e['expectancy_r']:+.2f}R "
            f"per trade, win rate {pct(e['win_rate'])}, profit factor "
            f"{(e['profit_factor'] or 0):.2f}; older period {e['older_expectancy_r']:+.2f}R; "
            f"random entries {e['random_expectancy_r']:+.2f}R. A 100-trade run lost money in "
            f"{pct(e['chance_100_trade_loss'])} of reshuffles; worst-case drawdown "
            f"~{(e['p95_drawdown_r'] or 0):.0f}R. On this name: "
            f"{e['name_trades'] or 0} trades, {(e['name_expectancy_r'] or 0):+.2f}R.")


def ticket_text(t: dict) -> str:
    ccy = t["currency"]
    qty = "size it yourself" if t["quantity"] is None else f"{t['quantity']:g}"
    lines = [
        f"{t['action']} {t['display']} ({t['market']}) · in {BROKER_NAME[t['broker']]}"
        + ("   [PREVIEW — did not pass the gate]" if t.get("preview") else ""),
        f"  Limit {_money(ccy, t['limit'])} · stop-loss {_money(ccy, t['stop'])} "
        f"({t['stop_pct']:+.1f}%)"
        + (f" · target {_money(ccy, t['target'])}" if t["target"] else " · no fixed target")
        + f" · quantity {qty}"
        + (f" · risks {_money(ccy, t['risk_amount'])}" if t.get("risk_amount") else ""),
        f"  Good until {t['valid_until']:%a %d %b}. Strategy: {t['label']}.",
        "  Why:",
        *[f"   - {r}" for r in t["reasons"]],
        *[f"   ! {n}" for n in t["notes"]],
        f"  Evidence: {_evidence_line(t['evidence'])}",
    ]
    return "\n".join(lines)


def holdings_text(risks: list[dict]) -> str:
    out = ["Your holdings — trailing stop levels (risk levels, not gated trades):"]
    for h in risks:
        if h.get("note"):
            out.append(f"  {h['display']:<12} {h['note']}")
            continue
        flag = ("JUST CLOSED BELOW" if h["newly_below"] else
                "below" if h["below_stop"] else "above")
        out.append(f"  {h['display']:<12} close {_money(h['currency'], h['close'])} · "
                   f"stop {_money(h['currency'], h['stop'])} ({h['stop_gap_pct']:+.1f}%) · "
                   f"{flag}" + (" · under its 200-day average" if h["below_200d"] else ""))
    return "\n".join(out)


def compose(result: dict, regime: str | None = None) -> tuple[str, str, str] | None:
    """Subject, text and HTML — or None when there is nothing to act on."""
    tickets = result["tickets"]
    fresh = [h for h in result["holdings"] if h["newly_below"]]
    if not tickets and not fresh:
        return None
    region = REGION_NAME.get(result["region"], result["region"])
    bits = []
    if tickets:
        bits.append(f"{len(tickets)} idea{'s' if len(tickets) != 1 else ''}")
    if fresh:
        bits.append(f"{len(fresh)} holding{'s' if len(fresh) != 1 else ''} broke a stop")
    subject = f"Alfred · {region} close {result['as_of']:%d %b} · " + " · ".join(bits)

    parts = [HEADER, ""]
    if regime:
        parts += [f"Macro backdrop (Alfred's world model): {regime}", ""]
    for t in tickets:
        parts += [ticket_text(t), ""]
    if result.get("left_out"):
        parts += [f"{result['left_out']} more signals fired today; these are the "
                  f"{len(tickets)} strongest by 12-month momentum. The rest are on the "
                  "Ideas page only if you ask for a preview scan.", ""]
    if fresh:
        parts += ["Holdings that just closed below their trailing stop:"]
        parts += [f"  {h['display']}: close {_money(h['currency'], h['close'])}, stop "
                  f"{_money(h['currency'], h['stop'])}. Consider whether the reason you hold it "
                  "still stands." for h in fresh]
        parts.append("")
    parts.append(holdings_text(result["holdings"]))
    text = "\n".join(parts)
    page = (f'<div style="font:14px/1.55 -apple-system,Segoe UI,sans-serif;color:#111;'
            f'max-width:680px"><pre style="white-space:pre-wrap;font:13px/1.55 '
            f'ui-monospace,Menlo,monospace">{html.escape(text)}</pre></div>')
    return subject, text, page


def send(result: dict, regime: str | None = None) -> str | None:
    composed = compose(result, regime)
    if not composed:
        return None
    subject, text, page = composed
    # Keyed on content (the default): a rerun of the same scan is deduplicated,
    # while a later scan that finds a new ticket still gets through.
    return email.send(subject, text, page)
