"""Same-day trades: entry next hour, stop at the first hour's low, out by the close."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from advisor.intraday import _play, _signal, backtest


def arr(*xs):
    return np.array(xs, dtype=float)


def test_exit_at_the_close_if_nothing_else_happens():
    o, h, l, c = arr(100, 101, 102, 103), arr(101, 102, 103, 104), arr(99, 100.5, 101.5, 102.5), arr(100.5, 101.5, 102.5, 103.5)
    r, why = _play(o, h, l, c, 0, stop=99, target=None, cost=0)
    assert why == "close" and r == pytest.approx((103.5 - 101) / 2)


def test_stop_on_the_entry_hour_counts_against_the_trade():
    o, h, l, c = arr(100, 101, 102), arr(101, 101.5, 103), arr(99, 98, 101), arr(100.5, 99, 102)
    r, why = _play(o, h, l, c, 0, stop=99, target=None, cost=0)
    assert why == "stop" and r == pytest.approx(-1.0)


def test_costs_are_taken_in_risk_units():
    o, h, l, c = arr(100, 100, 100), arr(101, 101, 101), arr(99.5, 99.5, 99.5), arr(100, 100, 102)
    r, _ = _play(o, h, l, c, 0, stop=99, target=None, cost=0.001)
    assert r == pytest.approx(2.0 - 0.1)


def test_signals_fire_on_their_textbook_conditions():
    # Gap and go: open 1%+ above the prior close and the first hour holds.
    o, h, l, c = arr(101.5, 102, 103), arr(102.5, 103, 104), arr(101, 101.5, 102), arr(102, 102.5, 103.5)
    assert _signal("id_gap_go", o, h, l, c, prior=100, sma20=95) == (0, 101, None)
    assert _signal("id_gap_go", o, h, l, c, prior=101, sma20=95) is None          # only a 0.5% gap
    # Gap-down reversal targets the prior close.
    o, h, l, c = arr(98, 98.8, 99.5), arr(98.9, 99.5, 100.2), arr(97.6, 98.5, 99), arr(98.7, 99.2, 100)
    assert _signal("id_gap_reversal", o, h, l, c, prior=100, sma20=95) == (0, 97.6, 100)
    # Opening-hour breakout needs the uptrend filter.
    o, h, l, c = arr(100, 100.5, 101), arr(100.8, 101.5, 102), arr(99.6, 100.3, 100.8), arr(100.5, 101.2, 101.8)
    assert _signal("id_orb", o, h, l, c, prior=100, sma20=98) == (1, 99.6, None)
    assert _signal("id_orb", o, h, l, c, prior=100, sma20=102) is None


def test_backtest_groups_hours_into_local_sessions():
    """Two sessions of seven hours; the second gaps up 2% and holds."""
    idx = pd.date_range("2026-01-05 09:15", periods=7, freq="h", tz="Asia/Kolkata").append(
        pd.date_range("2026-01-06 09:15", periods=7, freq="h", tz="Asia/Kolkata"))
    day1 = [100.0] * 7
    day2 = [102.0, 102.5, 103.0, 103.5, 104.0, 104.5, 105.0]
    close = day1 + day2
    df = pd.DataFrame({"open": close, "high": [x + 0.6 for x in close], "low": [x - 0.4 for x in close],
                       "close": [x + 0.3 for x in close]}, index=idx)
    df["session"] = df.index.date
    trades = backtest("TEST.NS", df, "id_gap_go", cost=0)
    assert len(trades) == 1 and str(trades[0].session) == "2026-01-06"
    assert trades[0].reason == "close" and trades[0].r > 0


def test_random_entries_never_precede_the_signal():
    """The first run's breakout baseline drew random hours from the whole day,
    including hours before the breakout — knowledge nobody has at that hour."""
    idx = pd.date_range("2026-01-05 09:15", periods=7, freq="h", tz="Asia/Kolkata").append(
        pd.date_range("2026-01-06 09:15", periods=7, freq="h", tz="Asia/Kolkata"))
    # First hour high 100.3; the third hour closes at 100.4 — the breakout (k=2,
    # entered at the open of hour 3).
    close = [100.0] * 7 + [100.0, 100.2, 100.4, 101.5, 101.8, 102.0, 102.2]
    df = pd.DataFrame({"open": close, "high": [x + 0.3 for x in close],
                       "low": [x - 0.3 for x in close], "close": close}, index=idx)
    df["session"] = df.index.date
    rng = np.random.default_rng(0)
    real = backtest("T.NS", df, "id_orb", cost=0)
    # The trend filter needs 20 sessions of history; force it for this check.
    from advisor import intraday
    sig = intraday._signal
    intraday._signal = lambda name, o, h, l, c, prior, sma20: sig(name, o, h, l, c, prior, prior - 1)
    try:
        for _ in range(50):
            for t in backtest("T.NS", df, "id_orb", cost=0, rng=rng):
                assert t.entry_hour >= 3, "random entry before the breakout hour"
    finally:
        intraday._signal = sig
    assert real == [] or real[0].entry_hour >= 3
