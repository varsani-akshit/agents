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
    """Replay every live idea. Returns what changed, and which exits are due.

    `exit_due`: the strategy's own exit fired on the latest close, so the
    position should be sold at the next open — the one moment an exit email
    is still actionable. `closed`: ideas that finished this run (stop, target,
    time, or yesterday's exit signal), for the record.
    """
    changed = 0
    exit_due, closed = [], []
    for idea in db.query("SELECT * FROM ideas WHERE status IN ('open', 'filled')"):
        strat = STRATEGIES.get(idea["strategy"])
        x = bars.load(idea["symbol"], years=2)
        if strat is None or x.empty:
            continue
        x = indicators(x)
        if strat.name == "momentum_rotation":
            # Its exit depends on the name's rank within the region that day.
            x = _with_rank(idea["symbol"], x)
        out = replay_one(idea["symbol"], x, strat, idea["signal_date"], idea["limit_price"],
                         idea["stop_price"], idea["target_price"])
        if out is None:
            continue
        status = _STATUS[out.status]
        d = lambda k: x.index[k].date() if k is not None else None  # noqa: E731
        db.execute(
            """UPDATE ideas SET status=%s, fill_date=%s, fill_price=%s, exit_date=%s,
                 exit_price=%s, exit_reason=%s, r_multiple=%s WHERE id=%s""",
            (status, d(out.fill_i), out.fill, d(out.exit_i), out.exit, out.reason,
             out.r, idea["id"]))
        changed += status != idea["status"]
        brief = {"id": idea["id"], "symbol": idea["symbol"], "broker": idea["broker"],
                 "strategy": idea["strategy"], "currency": idea["currency"],
                 "fill_price": out.fill, "last": float(x["close"].iloc[-1])}
        if out.status == "open" and out.reason == "exit at next open":
            exit_due.append({**brief, "reason": "the strategy's exit fired at today's close"})
        elif status == "closed" and idea["status"] != "closed":
            closed.append({**brief, "reason": out.reason, "exit_price": out.exit, "r": out.r})
    return {"updated": changed, "exit_due": exit_due, "closed": closed}


def replay_one(symbol: str, x: pd.DataFrame, strat, signal_date, limit: float, stop: float,
               target: float | None):
    """One ticket through play(), from its signal day: the single accounting
    the backtest, the ledger and the call record share. None if the signal
    day is not in the bars."""
    cfg = settings()
    try:
        i = x.index.get_loc(pd.Timestamp(signal_date))
    except KeyError:
        return None
    sig = strat.rules(x)
    cost = cfg["round_trip_cost_pct"][market_of(symbol)] / 100
    return play(x["open"].to_numpy(), x["high"].to_numpy(), x["low"].to_numpy(),
                x["close"].to_numpy(), sig["exit"].to_numpy(), i, limit, stop, target,
                cfg["validity_sessions"], strat.max_days, cost)


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


def _with_rank(symbol: str, x: pd.DataFrame) -> pd.DataFrame:
    from advisor import REGIONS, region_of
    from advisor.strategies import add_momentum_rank

    region = REGIONS[region_of(market_of(symbol))]
    frames = {s: indicators(bars.load(s, years=2)) for s in bars.universe()
              if market_of(s) in region and s != symbol}
    frames = {s: f for s, f in frames.items() if len(f) > 260}
    frames[symbol] = x
    add_momentum_rank(frames)
    return frames[symbol]
