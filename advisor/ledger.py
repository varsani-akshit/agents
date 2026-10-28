"""What happened to every idea Alfred sent.

Each idea is replayed bar by bar from its signal date with `simulate.play` —
the same fill, stop, target, exit and cost rules that produced the evidence
for sending it. Whether you placed the ticket or not, this is what the ticket
would have done, which makes the scoreboard the honest forward test of the
backtest: if live results drift below the evidence, it shows here first.
"""
from __future__ import annotations

import logging

import pandas as pd

import db
from advisor import bars, market_of, settings
from advisor.simulate import play, summarize
from advisor.strategies import STRATEGIES, indicators

log = logging.getLogger("alfred.ledger")

_STATUS = {"pending": "open", "expired": "expired", "open": "filled", "closed": "closed"}


def update() -> dict:
    cfg = settings()
    changed = 0
    for idea in db.query("SELECT * FROM ideas WHERE status IN ('open', 'filled')"):
        strat = STRATEGIES.get(idea["strategy"])
        x = bars.load(idea["symbol"], years=2)
        if strat is None or x.empty:
            continue
        x = indicators(x)
        try:
            i = x.index.get_loc(pd.Timestamp(idea["signal_date"]))
        except KeyError:
            continue
        sig = strat.rules(x)
        cost = cfg["round_trip_cost_pct"][market_of(idea["symbol"])] / 100
        out = play(x["open"].to_numpy(), x["high"].to_numpy(), x["low"].to_numpy(),
                   x["close"].to_numpy(), sig["exit"].to_numpy(), i,
                   idea["limit_price"], idea["stop_price"], idea["target_price"],
                   cfg["validity_sessions"], strat.max_days, cost)
        status = _STATUS[out.status]
        d = lambda k: x.index[k].date() if k is not None else None  # noqa: E731
        db.execute(
            """UPDATE ideas SET status=%s, fill_date=%s, fill_price=%s, exit_date=%s,
                 exit_price=%s, exit_reason=%s, r_multiple=%s WHERE id=%s""",
            (status, d(out.fill_i), out.fill, d(out.exit_i), out.exit, out.reason,
             out.r, idea["id"]))
        changed += status != idea["status"]
    return {"updated": changed}


def scoreboard() -> dict:
    rows = db.query("SELECT strategy, market, r_multiple FROM ideas WHERE status='closed'")
    counts = {r["status"]: r["n"] for r in db.query(
        "SELECT status, count(*) n FROM ideas GROUP BY status")}
    by = {}
    for r in rows:
        by.setdefault(r["strategy"], []).append(r["r_multiple"])
    return {"overall": summarize([r["r_multiple"] for r in rows]),
            "by_strategy": {k: summarize(v) for k, v in by.items()},
            "counts": counts}
