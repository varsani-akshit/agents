"""The advisor's scheduled work: one cycle per market close, one weekly re-test."""
from __future__ import annotations

from advisor import REGIONS, bars, ledger, report, scan, simulate


def cycle(region: str, *, send: bool = True) -> dict:
    """After a close: top up prices, settle open ideas, scan, email if needed."""
    fetched = bars.refresh(list(REGIONS[region]), period="1mo")
    settled = ledger.update()
    result = scan.scan(region)
    sent = report.send(result, scan.regime()) if send else None
    return {"region": region, "bars": fetched, **settled,
            "tickets": len(result["tickets"]), "stored": result["stored"],
            "holdings_broke_stop": sum(h["newly_below"] for h in result["holdings"]),
            "emailed": bool(sent), "gate": result["gate"]}


def retest() -> dict:
    """Weekly: re-run every simulation on the latest history.

    The gate is not a one-time exam. A strategy that stops working stops
    sending ideas the week its evidence fades, without anyone deciding so.
    """
    from advisor import portfolio_sim

    report_ = simulate.run()
    out = {f"{name}/{region}": ok for (name, region), (ok, _s) in report_.items()}
    for region in REGIONS:
        acct = portfolio_sim.run(region)
        out[f"account/{region}"] = acct and {
            k: (acct.get(k) or {}).get("cagr") for k in ("strategy", "index", "random_picks")}
    return out
