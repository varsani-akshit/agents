"""The advisor's scheduled work: one cycle per market close, one weekly re-test."""
from __future__ import annotations

import db
from advisor import REGIONS, bars, ledger, market_of, report, scan, simulate


def cycle(region: str, *, send: bool = True) -> dict:
    """After a close: top up prices, settle open ideas, scan, email if needed."""
    fetched = bars.refresh(list(REGIONS[region]), period="1mo")
    settled = ledger.update()
    result = scan.scan(region, store=send)
    # The Story read for each ticket, so the email says what the news says.
    from advisor import story

    for t in result["tickets"]:
        try:
            t["story"] = story.for_ticket(t, region)
        except Exception:  # noqa: BLE001 - a ticket goes out without its read
            t["story"] = None
    # Exits matter only for what you actually hold: an idea you never placed
    # needs no sell email.
    held = set(bars.holdings())
    mine = lambda e: market_of(e["symbol"]) in REGIONS[region] and e["symbol"] in held  # noqa: E731
    result["exits"] = [e for e in settled["exit_due"] if mine(e)]
    result["closed"] = [e for e in settled["closed"] if mine(e)]
    scan.save_candidates(result, result["exits"])
    sent = report.send(result, scan.regime()) if send else None
    ids = [t["id"] for t in result["tickets"] if t.get("id")]
    if sent and ids:
        db.execute("UPDATE ideas SET emailed_at = now() WHERE id = ANY(%s)", (ids,))
    return {"region": region, "bars": fetched, "updated": settled["updated"],
            "exits": len(result["exits"]), "closed": len(result["closed"]),
            "tickets": len(result["tickets"]), "stored": result["stored"],
            "holdings_broke_stop": sum(h["newly_below"] for h in result["holdings"]),
            "emailed": bool(sent), "gate": result["gate"]}


def retest() -> dict:
    """Every two hours: a full simulation run, stored (see advisor/runs.py).

    The gate is not a one-time exam. A strategy that stops working stops
    sending ideas the run its evidence fades, without anyone deciding so.
    """
    from advisor import runs

    return runs.run()
