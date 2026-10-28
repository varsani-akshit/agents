"""Replay the strategies over history and decide which may send ideas.

One function, `play`, decides what happens to a ticket: whether the limit
fills, and when and where it exits. The backtest calls it for every historical
signal; the ledger calls it for every idea actually sent. So the scoreboard of
real recommendations is measured by exactly the rules that earned them a place
in your inbox — there is no second, kinder accounting.

Every assumption leans against the strategy:
  - a limit fills only if the day trades at or below it; a gap below fills at
    the open (that one is in your favour, and real);
  - if the stop and the target are both inside one day's range, the stop
    happened first;
  - a gap through the stop exits at the open, not at the stop;
  - the strategy's own exit is acted on at the next open, not at the close
    that triggered it;
  - round-trip costs from conf/advisor.yaml come off every trade.

Results are in R: profit divided by the risk planned at entry (limit − stop).
+1R means the trade made what it risked; −1R is a clean stop-out.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import db
from advisor import REGIONS, market_of, settings
from advisor.strategies import STRATEGIES, Strategy, indicators

log = logging.getLogger("alfred.simulate")

WARMUP = 220   # bars before the first signal: the 200-day average must exist


@dataclass
class Outcome:
    status: str            # pending | expired | open | closed
    fill_i: int | None = None
    fill: float | None = None
    exit_i: int | None = None
    exit: float | None = None
    reason: str | None = None
    r: float | None = None


def play(o, h, l, c, exit_rule, i: int, limit: float, stop: float,
         target: float | None, validity: int, max_days: int,
         cost_frac: float) -> Outcome:
    """What happens to a ticket raised at the close of bar i."""
    n = len(c)
    risk = limit - stop
    if not (risk > 0 and limit > 0):
        return Outcome("expired")
    fill_i = None
    for j in range(i + 1, min(i + 1 + validity, n)):
        if l[j] <= limit:
            fill_i, fill = j, min(o[j], limit)
            break
    if fill_i is None:
        return Outcome("pending" if i + validity >= n else "expired")

    has_target = target is not None and not math.isnan(target)
    for k in range(fill_i, n):
        held = k - fill_i + 1
        if l[k] <= stop:
            px = min(fill, stop) if k == fill_i else min(o[k], stop)
            return _closed(fill_i, fill, k, px, "stop", risk, cost_frac)
        if has_target and k > fill_i and h[k] >= target:
            return _closed(fill_i, fill, k, max(o[k], target), "target", risk, cost_frac)
        if exit_rule[k]:
            if k + 1 < n:
                return _closed(fill_i, fill, k + 1, o[k + 1], "signal", risk, cost_frac)
            return Outcome("open", fill_i, fill, reason="exit at next open")
        if held >= max_days:
            return _closed(fill_i, fill, k, c[k], "time", risk, cost_frac)
    return Outcome("open", fill_i, fill)


def _closed(fill_i, fill, k, px, reason, risk, cost_frac) -> Outcome:
    r = (px - fill) / risk - cost_frac * fill / risk
    return Outcome("closed", fill_i, fill, k, px, reason, r)


# ───────────────────────────────── backtest ─────────────────────────────────
@dataclass
class Trade:
    symbol: str
    signal: pd.Timestamp
    filled: pd.Timestamp
    exited: pd.Timestamp
    r: float
    reason: str


def backtest(symbol: str, x: pd.DataFrame, strat: Strategy, *,
             entries: np.ndarray | None = None) -> list[Trade]:
    """Every trade the strategy would have taken in one name, one at a time.

    `entries` overrides the strategy's entry signal — used to run the same
    exits from random entry days, which is the baseline a rule must beat.
    """
    cfg = settings()
    sig = strat.rules(x)
    entry = sig["entry"].to_numpy() if entries is None else entries
    o, h, l, c = (x[k].to_numpy() for k in ("open", "high", "low", "close"))
    ex = sig["exit"].to_numpy()
    lim, stp, tgt = sig["limit"].to_numpy(), sig["stop"].to_numpy(), sig["target"].to_numpy()
    cost = cfg["round_trip_cost_pct"][market_of(symbol)] / 100
    idx = x.index
    trades, i = [], WARMUP
    while i < len(c):
        if not entry[i] or not np.isfinite(stp[i]):
            i += 1
            continue
        out = play(o, h, l, c, ex, i, lim[i], stp[i], tgt[i],
                   cfg["validity_sessions"], strat.max_days, cost)
        if out.status == "closed":
            trades.append(Trade(symbol, idx[i], idx[out.fill_i], idx[out.exit_i],
                                out.r, out.reason))
            i = out.exit_i + 1
        elif out.status in ("open", "pending"):
            break
        else:
            i += 1
    return trades


def random_entries(x: pd.DataFrame, strat: Strategy, rng: np.random.Generator) -> np.ndarray:
    """Random entry days at the strategy's own signal frequency."""
    from advisor.strategies import tradeable

    sig = strat.rules(x)
    valid = np.isfinite(sig["stop"].to_numpy()) & tradeable(x).to_numpy()
    valid[:WARMUP] = False
    p = sig["entry"].to_numpy()[valid].mean() if valid.any() else 0
    return valid & (rng.random(len(x)) < p)


# ───────────────────────────────── statistics ───────────────────────────────
def summarize(rs: list[float]) -> dict:
    if not rs:
        return {"trades": 0}
    a = np.asarray(rs)
    wins, losses = a[a > 0], a[a <= 0]
    equity = np.cumsum(a)
    drawdown = np.max(np.maximum.accumulate(np.concatenate([[0], equity]))[1:] - equity)
    return {
        "trades": int(len(a)),
        "win_rate": float(len(wins) / len(a)),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_r": float(a.mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "total_r": float(a.sum()),
        "max_drawdown_r": float(drawdown),
    }


def monte_carlo(rs: list[float], n: int = 100, runs: int = 2000, seed: int = 7) -> dict:
    """Reshuffle history into many possible next-100-trade runs.

    The order trades arrive in decides how deep the losing streaks get; the
    average does not. This asks: across 2,000 plausible orderings, how bad is
    a bad run, and how often does it end in a loss?
    """
    if len(rs) < 20:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(rs), size=(runs, n), replace=True)
    totals = draws.sum(axis=1)
    eq = np.cumsum(draws, axis=1)
    dd = (np.maximum.accumulate(np.concatenate([np.zeros((runs, 1)), eq], axis=1), axis=1)[:, 1:]
          - eq).max(axis=1)
    return {
        "p5_total_r": float(np.percentile(totals, 5)),
        "median_total_r": float(np.median(totals)),
        "p95_drawdown_r": float(np.percentile(dd, 95)),
        "chance_of_loss": float((totals < 0).mean()),
    }


def judge(older: dict, recent: dict, baseline: dict, mc: dict) -> tuple[bool, list[str]]:
    """Apply the gate. Returns the verdict and, for a failure, every reason."""
    g = settings()["gate"]
    fails = []
    if recent.get("trades", 0) < g["min_trades_recent"]:
        fails.append(f"only {recent.get('trades', 0)} recent trades (needs {g['min_trades_recent']})")
    if recent.get("expectancy_r", -1) < g["min_expectancy_r"]:
        fails.append(f"recent expectancy {recent.get('expectancy_r', 0):+.3f}R "
                     f"(needs {g['min_expectancy_r']:+.2f}R)")
    if recent.get("profit_factor", 0) < g["min_profit_factor"]:
        fails.append(f"recent profit factor {recent.get('profit_factor', 0):.2f} "
                     f"(needs {g['min_profit_factor']})")
    if g.get("both_periods_positive") and older.get("expectancy_r", -1) <= 0:
        fails.append(f"older period lost money ({older.get('expectancy_r', 0):+.3f}R)")
    if g.get("beat_random_entries") and recent.get("expectancy_r", -1) <= baseline.get("expectancy_r", 0):
        fails.append(f"no better than random entries ({baseline.get('expectancy_r', 0):+.3f}R)")
    if g.get("monte_carlo_p5_positive") and mc.get("p5_total_r", -1) <= 0:
        fails.append(f"1 in 20 reshuffled runs loses {mc.get('p5_total_r', 0):+.1f}R or worse")
    return not fails, fails


# ───────────────────────────────── the run ──────────────────────────────────
def _frames(region: str) -> dict[str, pd.DataFrame]:
    from advisor import bars

    cfg = settings()
    syms = [s for s in bars.universe() if market_of(s) in REGIONS[region]]
    out = {}
    for s in syms:
        df = bars.load(s, years=cfg["history_years"])
        if len(df) > WARMUP + 60:
            out[s] = indicators(df)
    return out


def run(regions: list[str] | None = None) -> dict:
    """Simulate every strategy on every name, store the evidence, apply the gate."""
    cfg = settings()
    report = {}
    for market in regions or list(REGIONS):
        frames = _frames(market)
        if not frames:
            continue
        last = max(df.index[-1] for df in frames.values())
        cutoff = last - pd.DateOffset(years=cfg["recent_period_years"])
        for name, strat in STRATEGIES.items():
            rng = np.random.default_rng(11)
            trades, base, rows = [], [], []
            for sym, x in frames.items():
                t = backtest(sym, x, strat)
                trades += t
                base += backtest(sym, x, strat, entries=random_entries(x, strat, rng))
                s = summarize([tr.r for tr in t])
                g = cfg["gate"]
                ok = (s.get("trades", 0) >= g["min_symbol_trades"]
                      and s.get("expectancy_r", -1) > g["min_symbol_expectancy_r"])
                rows.append((name, market, sym, ok, s))
            older = summarize([t.r for t in trades if t.filled < cutoff])
            recent = summarize([t.r for t in trades if t.filled >= cutoff])
            baseline = summarize([t.r for t in base if t.filled >= cutoff])
            mc = monte_carlo([t.r for t in trades if t.filled >= cutoff])
            passed, fails = judge(older, recent, baseline, mc)
            stats = {"all": summarize([t.r for t in trades]), "older": older, "recent": recent,
                     "random_recent": baseline, "monte_carlo": mc, "fails": fails,
                     "symbols": len(frames), "recent_from": str(cutoff.date()),
                     "through": str(last.date()), "label": strat.label}
            _store(name, market, passed, stats, rows)
            report[(name, market)] = (passed, stats)
            log.info("%s %s: %s %s", name, market, "PASS" if passed else "fail",
                     fails or f"{recent.get('expectancy_r', 0):+.3f}R")
    return report


def _store(strategy: str, market: str, passed: bool, stats: dict, rows: list) -> None:
    import json

    def clean(d):
        return json.dumps(_finite(d), default=str)

    db.execute("DELETE FROM strategy_evidence WHERE strategy = %s AND market = %s",
               (strategy, market))
    db.executemany(
        "INSERT INTO strategy_evidence (strategy, market, symbol, passed, stats) "
        "VALUES (%s,%s,%s,%s,%s)",
        [(strategy, market, "", passed, clean(stats))]
        + [(s, m, sym, ok, clean(st)) for s, m, sym, ok, st in rows])


def _finite(v):
    """JSON has no Infinity: a profit factor with no losing trades is stored as null."""
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def evidence(strategy: str, market: str, symbol: str = "") -> dict | None:
    return db.one("SELECT passed, stats, computed_at FROM strategy_evidence "
                  "WHERE strategy=%s AND market=%s AND symbol=%s", (strategy, market, symbol))

