"""Same-day strategies on hourly bars, put through the same gate.

Every trade opens and closes inside one session: entry at the open of the
hour after the signal, a stop at the first hour's low, an optional target,
and out at the last hour's close whatever happens. Long-only.

Three textbook rules, fixed in advance:
  opening-hour breakout  an hourly close above the first hour's high, in a
                         name already above its 20-session average
  gap and go             opens 1%+ above the prior close and the first hour
                         holds above its open
  gap-down reversal      opens 1.5%+ below the prior close and the first hour
                         closes above its open; target the prior close

Judged exactly like the daily strategies — costs, an older and a recent
period, random entries with the same stop and exit, 2,000 reshuffled runs —
then replayed per market as one account (at most a few trades a day, 0.5%
risk each) against simply holding the index. Hourly history is about three
years, so the recent period is one year: a thinner test than twelve years of
daily data, and the page says so.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import db
from advisor import REGIONS, bars, hourly, market_of, settings
from advisor.simulate import monte_carlo, summarize

log = logging.getLogger("alfred.intraday")

MIN_RISK = 0.003          # a stop closer than 0.3% of price is noise, not risk


@dataclass(frozen=True)
class IntradayStrategy:
    name: str
    label: str
    summary: str


STRATEGIES = {s.name: s for s in (
    IntradayStrategy("id_orb", "Opening-hour breakout",
                     "Intraday: an hourly close above the first hour's high, in a name above its "
                     "20-session average; stop at the first hour's low; out by the close"),
    IntradayStrategy("id_gap_go", "Gap and go",
                     "Intraday: opens 1%+ above yesterday's close and the first hour holds; "
                     "stop at the first hour's low; out by the close"),
    IntradayStrategy("id_gap_reversal", "Gap-down reversal",
                     "Intraday: opens 1.5%+ below yesterday's close and the first hour closes "
                     "green; target yesterday's close; out by the close"),
)}


@dataclass
class Trade:
    symbol: str
    session: object
    strategy: str
    entry_hour: int
    r: float
    reason: str


def _sessions(df: pd.DataFrame):
    """(session date, o, h, l, c arrays, prior close, prior 20-session average)."""
    closes = df.groupby("session")["close"].last()
    sma20 = closes.rolling(20).mean().shift()
    prior = closes.shift()
    for day, g in df.groupby("session", sort=True):
        if len(g) < 4 or pd.isna(prior.get(day)):
            continue
        yield (day, g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(),
               g["close"].to_numpy(), float(prior[day]), sma20.get(day))


def _play(o, h, l, c, k, stop, target, cost):
    """Enter at the open of bar k+1; stop, target, or out at the last close."""
    if k + 1 >= len(c):
        return None
    entry = o[k + 1]
    risk = entry - stop
    if risk <= entry * MIN_RISK:
        return None
    for j in range(k + 1, len(c)):
        if l[j] <= stop:
            px = min(o[j], stop) if j > k + 1 else min(entry, stop)
            return (px - entry) / risk - cost * entry / risk, "stop"
        if target and h[j] >= target and j > k + 1:
            return (max(o[j], target) - entry) / risk - cost * entry / risk, "target"
    return (c[-1] - entry) / risk - cost * entry / risk, "close"


def _signal(name: str, o, h, l, c, prior, sma20):
    """The bar index k whose close fires the strategy, and its stop and target."""
    h0, l0 = h[0], l[0]
    if name == "id_orb":
        if sma20 is None or pd.isna(sma20) or prior <= sma20:
            return None
        for k in range(1, len(c) - 1):
            if c[k] > h0:
                return k, l0, None
        return None
    if name == "id_gap_go":
        if o[0] >= prior * 1.01 and c[0] > o[0]:
            return 0, l0, None
        return None
    if name == "id_gap_reversal":
        if o[0] <= prior * 0.985 and c[0] > o[0]:
            return 0, l0, prior
        return None
    return None


def backtest(symbol: str, df: pd.DataFrame, name: str, cost: float,
             rng: np.random.Generator | None = None) -> list[Trade]:
    """Every session's trade under one strategy — or, with `rng`, the same
    stop and exit from a random hour on the strategy's own signal days."""
    out = []
    for day, o, h, l, c, prior, sma20 in _sessions(df):
        sig = _signal(name, o, h, l, c, prior, sma20)
        if not sig:
            continue
        k, stop, target = sig
        if rng is not None:
            # A random hour from the signal onward, never before it: the day
            # is only known to be a signal day once the signal has fired, so a
            # random entry earlier that day would be using the day's outcome.
            k = int(rng.integers(k, len(c) - 1))
        res = _play(o, h, l, c, k, stop, target, cost)
        if res:
            out.append(Trade(symbol, day, name, k + 1, res[0], res[1]))
    return out


# ───────────────────────────────── the run ──────────────────────────────────
def run() -> dict:
    """Simulate every intraday strategy in every market; replay; store evidence.

    Returns {(strategy, region): (passed, stats)} like simulate.run, plus
    {("account", region): stats} for the replays.
    """
    from advisor.portfolio_sim import judge_account, metrics
    from advisor.simulate import judge

    cfg = settings()
    icfg = cfg["intraday"]
    report: dict = {}
    for region, markets in REGIONS.items():
        syms = [s for s in bars.universe() if market_of(s) in markets]
        frames = {s: df for s in syms if len(df := hourly.load(s)) > 400}
        if not frames:
            continue
        last = max(df.index[-1] for df in frames.values())
        cutoff = (last - pd.DateOffset(years=icfg["recent_period_years"])).date()
        rng = np.random.default_rng(17)
        all_trades: dict[str, list[Trade]] = {}
        for name, strat in STRATEGIES.items():
            trades, base = [], []
            for sym, df in frames.items():
                cost = icfg["round_trip_cost_pct"][market_of(sym)] / 100
                trades += backtest(sym, df, name, cost)
                base += backtest(sym, df, name, cost, rng=rng)
            all_trades[name] = trades
            older = summarize([t.r for t in trades if t.session < cutoff])
            recent = summarize([t.r for t in trades if t.session >= cutoff])
            baseline = summarize([t.r for t in base if t.session >= cutoff])
            mc = monte_carlo([t.r for t in trades if t.session >= cutoff])
            passed, fails = judge(older, recent, baseline, mc)
            exits = {}
            for t in trades:
                exits[t.reason] = exits.get(t.reason, 0) + 1
            stats = {"all": summarize([t.r for t in trades]), "older": older, "recent": recent,
                     "random_recent": baseline, "monte_carlo": mc, "fails": fails,
                     "symbols": len(frames), "recent_from": str(cutoff), "through": str(last.date()),
                     "label": strat.label, "exits": exits, "median_days": 0, "intraday": True,
                     "signals_per_day": round(len(trades) / max(1, len({t.session for t in trades})), 2)}
            _store(name, region, passed, stats)
            report[(name, region)] = (passed, stats)
            log.info("%s %s: %s", name, region, "PASS" if passed else fails)

        # One account: the passing strategies' trades, at most N a day, 0.5% risk each.
        passing = [n for n in STRATEGIES if report[(n, region)][0]]
        acct = _account(region, [t for n in passing for t in all_trades[n]], icfg, cfg)
        if acct:
            acct["passed"], acct["fails"] = judge_account(acct)
            acct["strategies"] = passing
            report[("account", region)] = acct
    return report


def _account(region: str, trades: list[Trade], icfg: dict, cfg: dict) -> dict | None:
    from advisor.portfolio_sim import metrics

    if not trades:
        return None
    risk = cfg["risk_per_trade_pct"] / 100
    by_day: dict = {}
    for t in sorted(trades, key=lambda t: (t.session, t.entry_hour)):
        by_day.setdefault(t.session, []).append(t)
    days = sorted(by_day)
    daily = pd.Series([sum(t.r for t in by_day[d][:icfg["max_trades_per_day"]]) * risk for d in days],
                      index=pd.to_datetime(days))
    index_sym = cfg["benchmark"][region]
    idx = bars.load(index_sym, years=4)["close"]
    cal = idx.index[(idx.index >= daily.index[0]) & (idx.index <= daily.index[-1])]
    eq = (1 + daily.reindex(cal).fillna(0)).cumprod()
    bench = idx.reindex(cal).ffill()
    bench = bench / bench.iloc[0]
    weekly = pd.DataFrame({"strategy": eq, "index": bench}).resample("W").last()
    curve = {"dates": [str(d.date()) for d in weekly.index],
             **{k: [None if pd.isna(v) else round(float(v), 4) for v in weekly[k]] for k in weekly}}
    return {"curve": curve, "strategy": metrics(eq), "index": metrics(bench), "index_symbol": index_sym,
            "trades": summarize([t.r for d in days for t in by_day[d][:icfg["max_trades_per_day"]]]),
            "active_days": len(days), "from": str(cal[0].date()), "through": str(cal[-1].date())}


def _store(strategy: str, region: str, passed: bool, stats: dict) -> None:
    from advisor.simulate import _finite

    db.execute("DELETE FROM strategy_evidence WHERE strategy=%s AND market=%s", (strategy, region))
    db.execute("INSERT INTO strategy_evidence (strategy, market, symbol, passed, stats) "
               "VALUES (%s,%s,'',%s,%s)", (strategy, region, passed, json.dumps(_finite(stats), default=str)))
