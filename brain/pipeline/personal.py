"""The reader's own book, as context for the desk.

A brief is about the reader's money, so the Marshal ranks stories by how much
they touch it and the Editor writes a "Your money" section from it. Everything
here is read from what the brokers and the advisor have already stored — no
model, no network — and every figure is measured.
"""
from __future__ import annotations

import logging
from datetime import timezone

import db

log = logging.getLogger("mia.personal")


def book() -> dict:
    """Holdings, open tickets, names near a trigger, and money set aside."""
    out: dict = {"holdings": [], "tickets": [], "near": [], "funds": {}, "markets_sending": []}
    try:
        from advisor import bare, bars, settings
        from advisor.scan import holding_risk
        from advisor.strategies import indicators
        from brokers import portfolio as pf
    except Exception as exc:  # noqa: BLE001 - a brief must not fail on the advisor
        log.warning("personal context unavailable: %s", exc)
        return out

    held = bars.holdings()
    for broker in ("kite", "webull"):
        data = pf.latest(broker)
        for ln in (data or {}).get("lines") or []:
            avg, last = ln["avg_price"], ln["last_price"]
            row = {"broker": broker, "symbol": ln["symbol"], "exchange": ln["exchange"],
                   "currency": ln["currency"], "quantity": float(ln["quantity"]),
                   "value": round(ln["value"] or 0), "last": last and float(last),
                   "pnl_pct": round((float(last) / float(avg) - 1) * 100, 1)
                   if avg and last else None}
            out["holdings"].append(row)
    # Trend state per holding, from the same bars the advisor reads.
    by_name = {bare(t): t for t in held}
    for row in out["holdings"]:
        t = by_name.get(row["symbol"])
        if not t:
            continue
        try:
            x = bars.load(t, years=2)
            if len(x) >= 260:
                r = holding_risk(t, indicators(x), held[t])
                row["below_trailing_stop"] = r["below_stop"]
                row["below_200d"] = r["below_200d"]
        except Exception:  # noqa: BLE001
            pass
    out["holdings"].sort(key=lambda r: -(r["value"] or 0))

    out["tickets"] = [
        {"symbol": bare(r["symbol"]), "market": r["market"], "strategy": r["strategy"],
         "limit": r["limit_price"], "stop": r["stop_price"], "status": r["status"]}
        for r in db.query("""SELECT symbol, market, strategy, limit_price, stop_price, status
                             FROM ideas WHERE status IN ('open','filled')
                             ORDER BY created_at DESC LIMIT 15""")]
    out["near"] = [
        {"symbol": (r["detail"] or {}).get("display") or bare(r["symbol"]), "region": r["region"],
         "waiting": (r["detail"] or {}).get("waiting")}
        for r in db.query("""SELECT region, symbol, detail FROM scan_candidates
                             WHERE kind = 'near'
                             ORDER BY (detail->>'level') IS NULL,
                                      (detail->>'distance_pct')::float LIMIT 30""")][:15]
    try:
        from brain import news_match

        out["news"] = [
            {"symbol": bare(r["symbol"]), "name": r["name"], "title": r["title"],
             "source": r["source"], "tier": r["source_tier"], "url": r["url"],
             "at": r["at"].astimezone(timezone.utc).strftime("%d %b %H:%M"), "read": r["impact"],
             "session_move_pct": r["move_pct"]}
            for r in news_match.recent(hours=36, per_name=3)]
    except Exception as exc:  # noqa: BLE001
        log.warning("news matches unavailable: %s", exc)
        out["news"] = []
    out["funds"] = settings().get("new_funds") or {}
    out["markets_sending"] = [r["region"] for r in db.query(
        "SELECT region FROM portfolio_backtests WHERE (stats->>'passed')::boolean")]
    return out


def as_text(b: dict, max_holdings: int = 25) -> str:
    """The book as compact lines for a prompt."""
    if not b.get("holdings") and not b.get("tickets"):
        return "(no portfolio on file)"
    lines = ["Holdings (largest first; value in the account currency):"]
    for h in b["holdings"][:max_holdings]:
        flags = []
        if h.get("below_trailing_stop"):
            flags.append("below its trailing stop")
        if h.get("below_200d"):
            flags.append("under its 200-day average")
        lines.append(f"  {h['symbol']} ({h['exchange']}, {h['broker']}) value {h['currency']} "
                     f"{h['value']:,} · {h['pnl_pct']:+}% vs cost"
                     + (f" · {', '.join(flags)}" if flags else ""))
    if b.get("news"):
        lines.append("News about the reader's own names, last 36 hours (matched by a judging "
                     "model — verify before relying; 'read' is that model's good/bad call and is "
                     "sometimes wrong, the session move is measured):")
        for n in b["news"]:
            mv = "" if n["session_move_pct"] is None else f", {n['symbol']} {n['session_move_pct']:+.1f}% that session"
            lines.append(f"  {n['symbol']}: \"{n['title']}\" — {n['source']} (tier {n['tier']}), "
                         f"{n['at']} UTC, read: {n['read']}{mv}. {n['url'] or ''}")
    if b.get("tickets"):
        lines.append("Open tickets from Alfred's advisor:")
        lines += [f"  {t['symbol']} ({t['market']}) {t['strategy']} limit {t['limit']:.2f} "
                  f"stop {t['stop']:.2f} [{t['status']}]" for t in b["tickets"]]
    if b.get("near"):
        lines.append("Names close to an advisor trigger:")
        lines += [f"  {n['symbol']} ({n['region']}): waiting for {n['waiting']}" for n in b["near"]]
    if b.get("funds"):
        lines.append("Money set aside to add for good opportunities: "
                     + ", ".join(f"{k}: {v:,}" for k, v in b["funds"].items())
                     + " (Kite in INR, Webull in AUD).")
    if b.get("markets_sending"):
        lines.append("Advisor markets currently allowed to send tickets: "
                     + ", ".join(b["markets_sending"]))
    return "\n".join(lines)
