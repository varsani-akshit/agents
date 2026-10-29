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

REGION_NAME = {"IN": "India", "ASX": "Australia", "US": "US", "CRYPTO": "Crypto"}
BROKER_NAME = {"kite": "Kite", "webull": "Webull", "coinbase": "Coinbase"}
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
            f"{pct(e['chance_100_trade_loss'])} of reshuffles. On this name: "
            f"{e['name_trades'] or 0} trades, {(e['name_expectancy_r'] or 0):+.2f}R."
            + _account_line(e.get("account")))


def _account_line(a: dict | None) -> str:
    """The account replay, which is the honest drawdown: trades that lose
    together are counted together, unlike the per-trade reshuffle."""
    if not a:
        return ""

    def p(v, fmt="{:+.1f}%"):
        return fmt.format(v * 100) if v is not None else "—"

    return (f" Followed as one account since {a['since']}: {p(a['cagr'])} a year vs "
            f"{p(a['index_cagr'])} for {a['index']}, worst fall {p(a['max_drawdown'], '{:.0f}%')} "
            f"vs {p(a['index_max_drawdown'], '{:.0f}%')}, Sharpe {a['sharpe'] or 0:.2f} vs "
            f"{a['index_sharpe'] or 0:.2f}. " + _verdict(a))


def _verdict(a: dict) -> str:
    gap = (a["cagr"] or 0) - (a["index_cagr"] or 0)
    if abs(gap) < 0.01:
        ret = "About the index's return"
    elif gap > 0:
        ret = f"{gap * 100:.1f} points a year more than the index"
    else:
        ret = f"{-gap * 100:.1f} points a year less than the index"
    falls = ("with smaller falls" if (a["max_drawdown"] or -1) > (a["index_max_drawdown"] or -1)
             else "with deeper falls")
    return f"{ret}, {falls}."


def ticket_text(t: dict) -> str:
    ccy = t["currency"]
    qty = "size it yourself" if t["quantity"] is None else f"{t['quantity']:g}"
    lines = [
        f"{t['action']} {t['display']} on {t.get('venue') or t['market']} · in {BROKER_NAME[t['broker']]}"
        + ("   [PREVIEW — did not pass the gate]" if t.get("preview") else ""),
        f"  Limit {_money(ccy, t['limit'])} · stop-loss {_money(ccy, t['stop'])} "
        f"({t['stop_pct']:+.1f}%)"
        + (f" · target {_money(ccy, t['target'])}" if t["target"] else " · no fixed target")
        + f" · quantity {qty}"
        + (f" · risks {_money(ccy, t['risk_amount'])}" if t.get("risk_amount") else ""),
        f"  Good until {t['valid_until']:%a %d %b}. Strategy: {t['label']}."
        + (f" Typical hold: about {t['evidence']['median_days']} sessions."
           if t.get("evidence", {}).get("median_days") else ""),
        "  Why:",
        *[f"   - {r}" for r in t["reasons"]],
        *[f"   ! {n}" for n in t["notes"]],
        *([f"  Funding: {t['funding']}"] if t.get("funding") else []),
        f"  Evidence: {_evidence_line(t['evidence'])}",
        *_story_lines(t.get("story")),
    ]
    return "\n".join(lines)


def _story_lines(s: dict | None) -> list[str]:
    """What the news says about the setup — a read, not a gate (yet)."""
    if not s or not s.get("direction"):
        return []
    out = [f"  News: {s['direction'].upper()} (strength {s.get('strength', 0):.1f}) — {s.get('thesis', '')}"]
    if s.get("risk"):
        out.append(f"  Watch for: {s['risk']}")
    if s.get("event_risk") and s.get("events"):
        out.append(f"  Inside the hold: {'; '.join(s['events'][:3])}")
    return out


def holdings_text(risks: list[dict]) -> str:
    out = ["Your holdings — trailing stop levels (risk levels, not gated trades):"]
    for h in risks:
        if h.get("note"):
            out.append(f"  {h['display']:<12} {h['note']}")
            continue
        if h.get("core"):
            tr = h.get("trim")
            out.append(f"  {h['display']:<12} close {_money(h['currency'], h['close'])} · core, "
                       "held long-term (no stop)"
                       + (f" · stretched: optionally trim up to {tr['units']} units, buy back near "
                          f"{_money(h['currency'], tr['rebuy'])}" if tr else ""))
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
    held_back = result.get("held_back") or []
    fresh = [h for h in result["holdings"] if h["newly_below"]]
    exits = result.get("exits") or []
    closed = result.get("closed") or []
    if not tickets and not fresh and not exits and not closed and not held_back:
        return None
    region = REGION_NAME.get(result["region"], result["region"])
    bits = []
    if exits:
        bits.append(f"SELL {', '.join(e['symbol'].split('.')[0] for e in exits)}")
    if tickets:
        bits.append(f"{len(tickets)} idea{'s' if len(tickets) != 1 else ''}")
    if held_back and not tickets:
        bits.append(f"{len(held_back)} held back by the news")
    if fresh:
        bits.append(f"{len(fresh)} holding{'s' if len(fresh) != 1 else ''} broke a stop")
    subject = f"Alfred · {region} close {result['as_of']:%d %b} · " + " · ".join(bits)

    parts = [HEADER, ""]
    if regime:
        parts += [f"Macro backdrop (Alfred's world model): {regime}", ""]
    for e in exits:
        gain = (e["last"] / e["fill_price"] - 1) * 100 if e.get("fill_price") else None
        parts += [f"SELL {e['symbol'].split('.')[0]} at the next open · in "
                  f"{BROKER_NAME[e['broker']]}",
                  f"  Why: {e['reason']} ({e['strategy'].replace('_', ' ')})."
                  + (f" Last close {_money(e['currency'], e['last'])}, "
                     f"{gain:+.1f}% from the ticket's fill." if gain is not None else ""),
                  ""]
    for e in closed:
        parts += [f"Closed: {e['symbol'].split('.')[0]} — {e['reason']} at "
                  f"{_money(e['currency'], e['exit_price'])} ({e['r']:+.2f}R). If your stop order "
                  "was in place, this already happened in your account.", ""]
    for t in tickets:
        parts += [ticket_text(t), ""]
    if held_back:
        parts += ["Held back by the news — these passed the gate, but the news read contradicts them:"]
        for t in held_back:
            s = t.get("story") or {}
            parts += [f"  {t['display']} on {t.get('venue') or t['market']} ({t['label']}): "
                      f"{s.get('thesis', '')}",
                      f"    Would change the read: {s.get('risk', '—')}"]
        parts += ["  Still recorded and scored; if you take one anyway, the levels are on the Ideas page.", ""]
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
