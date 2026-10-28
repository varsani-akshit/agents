"""Entry and exit rules, written down before looking at any results.

Three long-only strategies with textbook parameters. Nothing here is tuned to
the data: a 200-day average is a 200-day average because that is what the
literature uses, not because 187 days backtested better. Tuning is how a
simulation learns the past instead of a rule — and with fixed parameters, the
older and recent halves of history are both honest tests.

Long-only on purpose: an NRI delivery account cannot short, and nobody needs
Alfred inventing reasons to.

Each strategy turns a bar frame into, per day:
    entry   the signal fired at this close
    limit   the buy-limit price for the ticket
    stop    the protective stop
    target  a profit target, or NaN for rules that exit on their own signal
    exit    the strategy's own exit fired at this close (acted on next open)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ─────────────────────────────── indicators ─────────────────────────────────
def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = _wilder(delta.clip(lower=0), n)
    down = _wilder(-delta.clip(upper=0), n)
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    return _wilder(tr, n)


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Everything any strategy reads, computed once per symbol."""
    out = df.copy()
    c = out["close"]
    out["sma5"] = c.rolling(5).mean()
    out["sma20"] = c.rolling(20).mean()
    out["sd20"] = c.rolling(20).std()
    out["sma50"] = c.rolling(50).mean()
    out["sma200"] = c.rolling(200).mean()
    out["rsi2"] = rsi(c, 2)
    out["atr14"] = atr(out, 14)
    # Prior-day channels: today's breakout is measured against the days before
    # it, or a close could never exceed a high that includes itself.
    out["hh55"] = out["high"].rolling(55).max().shift()
    out["ll20"] = out["low"].rolling(20).min().shift()
    out["hh22"] = out["high"].rolling(22).max()
    out["atr22"] = atr(out, 22)
    # Second batch: trend, momentum, and distance from the yearly high.
    out["sma100"] = c.rolling(100).mean()
    out["hh252"] = out["high"].rolling(252).max().shift()
    out["ret12"] = c / c.shift(252) - 1
    # 12-1 momentum: the year's return excluding the last month, which
    # the literature finds tends to reverse rather than continue.
    out["mom12_1"] = c.shift(21) / c.shift(252) - 1
    out["atr_pct"] = out["atr14"] / c
    out["atr_pct_med"] = out["atr_pct"].rolling(252).median()
    # Short-term channels for the third batch.
    out["hh20"] = out["high"].rolling(20).max().shift()
    out["ll10"] = out["low"].rolling(10).min().shift()
    out["hh10"] = out["high"].rolling(10).max().shift()
    # First session of each month: the monthly strategies only act on these.
    months = pd.Series(out.index.to_period("M"), index=out.index)
    out["month_start"] = months != months.shift()
    return out


def add_momentum_rank(frames: dict[str, pd.DataFrame]) -> None:
    """Rank every name's 12-1 momentum against the rest of its region, per day.

    A percentile in [0, 1]: 0.9 means stronger than 90% of the region that
    day. Written into each frame as `mom_rank`; a name with too little
    history to have momentum gets NaN and cannot enter.
    """
    if not frames:
        return
    panel = pd.DataFrame({sym: x["mom12_1"] for sym, x in frames.items()})
    ranks = panel.rank(axis=1, pct=True)
    for sym, x in frames.items():
        x["mom_rank"] = ranks[sym].reindex(x.index)


# ─────────────────────────────── strategies ─────────────────────────────────
@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    summary: str
    max_days: int
    rules: Callable[[pd.DataFrame], pd.DataFrame]
    reasons: Callable[[pd.Series], list[str]]


def tradeable(x: pd.DataFrame) -> pd.Series:
    """Days a signal may come from: the name actually traded, and its normal
    daily range is at least 0.5% of price.

    A suspended stock prints the same price for weeks with high == low, so
    its average range collapses toward zero, a stop "two ranges away" sits a
    fraction of a cent below the entry, and the next real print is a loss of
    a million R. That was OBM.AX in 2019, in the first simulation run. No
    rule should be judged on a day nobody could trade.
    """
    return (x["high"] > x["low"]) & (x["atr14"] >= 0.005 * x["close"])


def _frame(x: pd.DataFrame, entry, stop, exit_, target=None) -> pd.DataFrame:
    limit = x["close"]
    return pd.DataFrame({
        "entry": (entry & tradeable(x)).fillna(False).astype(bool),
        "limit": limit,
        "stop": stop,
        "target": target if target is not None else np.nan,
        "exit": exit_.fillna(False).astype(bool),
    }, index=x.index)


def _pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%"


# 1 · RSI(2) pullback — a short, sharp dip inside a long uptrend.
def _rsi2_rules(x):
    entry = (x["close"] > x["sma200"]) & (x["rsi2"] < 10)
    return _frame(x, entry, x["close"] - 2.5 * x["atr14"], x["close"] > x["sma5"])


def _rsi2_reasons(r):
    return [
        f"Long-term uptrend: close {r.close:,.2f} is {_pct(r.close, r.sma200)} vs its 200-day average.",
        f"Short-term washout: 2-day RSI is {r.rsi2:.1f} (below 10).",
        "Exit when it closes back above its 5-day average, after 10 sessions, or at the stop.",
    ]


# 2 · Bollinger dip — stretched two standard deviations below its 20-day mean,
# while the long trend is still up.
def _boll_rules(x):
    lower = x["sma20"] - 2 * x["sd20"]
    entry = (x["close"] > x["sma200"]) & (x["close"] < lower)
    return _frame(x, entry, x["close"] - 2.5 * x["atr14"], x["close"] > x["sma20"])


def _boll_reasons(r):
    lower = r.sma20 - 2 * r.sd20
    return [
        f"Long-term uptrend: {_pct(r.close, r.sma200)} vs its 200-day average.",
        f"Stretched: close {r.close:,.2f} is below its lower Bollinger band ({lower:,.2f}), "
        f"{_pct(r.close, r.sma20)} vs the 20-day mean.",
        "Exit when it closes back above the 20-day mean, after 15 sessions, or at the stop.",
    ]


# 3 · 55-day breakout — the classic trend-following entry, with a 20-day-low
# exit, in names already above their 200-day average.
def _breakout_rules(x):
    entry = (x["close"] > x["hh55"]) & (x["close"] > x["sma200"])
    return _frame(x, entry, x["close"] - 2 * x["atr14"], x["close"] < x["ll20"])


def _breakout_reasons(r):
    return [
        f"New 55-day high: close {r.close:,.2f} cleared the prior high of {r.hh55:,.2f}.",
        f"Trend confirmed: {_pct(r.close, r.sma200)} vs its 200-day average.",
        "Exit on a close below the 20-day low, or at the stop. No fixed target: trends are let run.",
    ]


# 4 · Quiet trend start — the 20-day average crosses above the 100-day while
# volatility is below its own one-year median. Trends that begin calm are the
# ones the breakout's large swings were missing.
def _quiet_trend_rules(x):
    cross = (x["sma20"] > x["sma100"]) & (x["sma20"].shift() <= x["sma100"].shift())
    entry = cross & (x["atr_pct"] < x["atr_pct_med"])
    return _frame(x, entry, x["close"] - 2 * x["atr14"], x["sma20"] < x["sma100"])


def _quiet_trend_reasons(r):
    return [
        f"Trend turning up: the 20-day average ({r.sma20:,.2f}) just crossed above the "
        f"100-day ({r.sma100:,.2f}).",
        f"Calm start: daily range is {r.atr_pct * 100:.2f}% of price, below its one-year "
        f"median of {r.atr_pct_med * 100:.2f}%.",
        "Exit when the 20-day average falls back below the 100-day, or at the stop.",
    ]


# 5 · Momentum rotation — on the first session of each month, buy names in the
# top 10% of their region by 12-1 momentum, above their 200-day average; sell
# at a month start once they drop out of the top 30%.
def _rotation_rules(x):
    rank = x["mom_rank"] if "mom_rank" in x else pd.Series(np.nan, index=x.index)
    entry = x["month_start"] & (rank >= 0.9) & (x["close"] > x["sma200"])
    exit_ = x["month_start"] & (rank < 0.7)
    return _frame(x, entry, x["close"] - 4 * x["atr14"], exit_)


def _rotation_reasons(r):
    return [
        f"Leader: 12-month return excluding the last month is {r.mom12_1 * 100:+.1f}%, "
        f"stronger than {r.mom_rank * 100:.0f}% of its market.",
        f"In an uptrend: {_pct(r.close, r.sma200)} vs its 200-day average.",
        "Reviewed on the first session of each month; sold once it falls out of the top 30%. "
        "Protective stop 4 daily ranges below entry.",
    ]


# 6 · Near the 52-week high — closes within 2% of its highest price of the
# past year (George & Hwang, 2004). Held six months unless the stop is hit.
def _near_high_rules(x):
    near = x["close"] >= 0.98 * x["hh252"]
    entry = near & ~near.shift(fill_value=False) & (x["close"] > x["sma200"])
    never = pd.Series(False, index=x.index)
    return _frame(x, entry, x["close"] - 3 * x["atr14"], never)


def _near_high_reasons(r):
    return [
        f"At the year's high: close {r.close:,.2f} is within 2% of its 52-week high "
        f"({r.hh252:,.2f}).",
        f"Trend confirmed: {_pct(r.close, r.sma200)} vs its 200-day average.",
        "Held for six months (126 sessions) unless the stop, 3 daily ranges below, is hit.",
    ]


# 7 · Absolute momentum — on each month's first session, own anything whose
# 12-month return is positive and which is above its 200-day average; sell at
# a month start once the 12-month return turns negative (Moskowitz, Ooi &
# Pedersen's time-series momentum, applied name by name).
def _abs_mom_rules(x):
    entry = x["month_start"] & (x["ret12"] > 0) & (x["close"] > x["sma200"])
    exit_ = x["month_start"] & (x["ret12"] < 0)
    return _frame(x, entry, x["close"] - 4 * x["atr14"], exit_)


def _abs_mom_reasons(r):
    return [
        f"Positive year: up {r.ret12 * 100:+.1f}% over 12 months.",
        f"Above its 200-day average ({_pct(r.close, r.sma200)}).",
        "Reviewed each month start; sold once the 12-month return turns negative. "
        "Protective stop 4 daily ranges below entry.",
    ]


# 8 · 20-day breakout — the Turtles' System 1: a new 20-day high in an
# uptrend, out on a 10-day low. The short-term cousin of the 55-day breakout.
def _breakout20_rules(x):
    entry = (x["close"] > x["hh20"]) & (x["close"] > x["sma200"])
    return _frame(x, entry, x["close"] - 2 * x["atr14"], x["close"] < x["ll10"])


def _breakout20_reasons(r):
    return [
        f"New 20-day high: close {r.close:,.2f} cleared the prior high of {r.hh20:,.2f}.",
        f"Uptrend: {_pct(r.close, r.sma200)} vs its 200-day average.",
        f"Exit on a close below the 10-day low ({r.ll10:,.2f} now), or at the stop. Usually a "
        "few days to a few weeks.",
    ]


# 9 · First dip to the 20-day — in a strong uptrend (50-day above 200-day,
# price above the 50-day), buy the first close back under the 20-day mean;
# sell into the next push to a 10-day high, or after ten sessions.
def _dip20_rules(x):
    strong = (x["sma50"] > x["sma200"]) & (x["close"] > x["sma50"])
    first = (x["close"] < x["sma20"]) & (x["close"].shift() >= x["sma20"].shift())
    return _frame(x, strong & first, x["close"] - 2 * x["atr14"], x["close"] > x["hh10"])


def _dip20_reasons(r):
    return [
        f"Strong uptrend: the 50-day average sits above the 200-day and price "
        f"({r.close:,.2f}) is above the 50-day ({r.sma50:,.2f}).",
        f"First dip: closed back under its 20-day average ({r.sma20:,.2f}) today.",
        f"Exit on a close above the prior 10-day high ({r.hh10:,.2f}), after 10 sessions, "
        "or at the stop.",
    ]


STRATEGIES: dict[str, Strategy] = {s.name: s for s in (
    Strategy("rsi2_pullback", "Pullback in an uptrend",
             "Buys a 2-day RSI washout above the 200-day average", 10,
             _rsi2_rules, _rsi2_reasons),
    Strategy("bollinger_dip", "Stretched below its band",
             "Buys a close under the lower Bollinger band above the 200-day average", 15,
             _boll_rules, _boll_reasons),
    Strategy("breakout_55", "55-day breakout",
             "Buys a new 55-day high above the 200-day average; exits on a 20-day low", 120,
             _breakout_rules, _breakout_reasons),
    Strategy("quiet_trend", "Calm trend start",
             "20/100-day average cross while volatility is below its yearly median", 250,
             _quiet_trend_rules, _quiet_trend_reasons),
    Strategy("momentum_rotation", "Momentum leader",
             "Top 10% of its market by 12-1 momentum, reviewed monthly", 260,
             _rotation_rules, _rotation_reasons),
    Strategy("near_52w_high", "At its 52-week high",
             "Closes within 2% of its yearly high; held six months", 126,
             _near_high_rules, _near_high_reasons),
    Strategy("abs_momentum", "Positive 12-month trend",
             "Positive 12-month return and above the 200-day, reviewed monthly", 260,
             _abs_mom_rules, _abs_mom_reasons),
    Strategy("breakout_20", "20-day breakout",
             "Short-term: a new 20-day high in an uptrend; out on a 10-day low", 40,
             _breakout20_rules, _breakout20_reasons),
    Strategy("dip_20", "First dip in a strong trend",
             "Short-term: first close under the 20-day in a strong uptrend; out on a 10-day high", 10,
             _dip20_rules, _dip20_reasons),
)}


def chandelier_stop(x: pd.DataFrame) -> pd.Series:
    """A trailing stop for things already held: 3 ATR below the 22-day high.

    Reported as a risk level, not a gated trade: it says where the trend would
    be broken, which is useful whether or not a strategy bought the name.
    """
    return x["hh22"] - 3 * x["atr22"]
