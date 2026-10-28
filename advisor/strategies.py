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
    return out


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
)}


def chandelier_stop(x: pd.DataFrame) -> pd.Series:
    """A trailing stop for things already held: 3 ATR below the 22-day high.

    Reported as a risk level, not a gated trade: it says where the trend would
    be broken, which is useful whether or not a strategy bought the name.
    """
    return x["hh22"] - 3 * x["atr22"]
