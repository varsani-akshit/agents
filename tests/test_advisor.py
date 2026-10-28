"""The advisor's arithmetic. No network, no database: hand-built bars where
the right answer is known, because every recommendation's evidence rests on
`play` getting fills, stops and costs right."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from advisor import bare, market_of, ticker_for
from advisor.simulate import judge, monte_carlo, play, summarize
from advisor.strategies import STRATEGIES, indicators, rsi


def bars(rows):
    """rows: (open, high, low, close)"""
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


NO_EXIT = np.zeros(10, dtype=bool)


def test_limit_fills_only_when_price_trades_through_it():
    o, h, l, c = bars([(100, 101, 99, 100), (101, 103, 100.5, 102), (102, 104, 101, 103),
                       (103, 105, 102, 104)])
    out = play(o, h, l, c, NO_EXIT, 0, limit=100, stop=95, target=None,
               validity=3, max_days=10, cost_frac=0)
    assert out.status == "expired", "filled at 100 although the low never got there"


def test_gap_below_the_limit_fills_at_the_open():
    o, h, l, c = bars([(100, 101, 99, 100), (98, 99, 97, 98.5), (98, 99, 97.5, 98)])
    out = play(o, h, l, c, NO_EXIT, 0, limit=100, stop=90, target=None,
               validity=3, max_days=10, cost_frac=0)
    assert out.fill == 98, "a gap down should fill at the open, not the limit"


def test_stop_and_target_in_one_day_counts_as_the_stop():
    o, h, l, c = bars([(100, 101, 99, 100), (100, 100.5, 99.5, 100),
                       (100, 115, 89, 100), (100, 100, 100, 100)])
    out = play(o, h, l, c, NO_EXIT, 0, limit=100, stop=90, target=110,
               validity=3, max_days=10, cost_frac=0)
    assert out.reason == "stop" and out.r == pytest.approx(-1.0)


def test_gap_through_the_stop_exits_at_the_open():
    o, h, l, c = bars([(100, 101, 99, 100), (100, 100.5, 99.5, 100), (85, 86, 84, 85)])
    out = play(o, h, l, c, NO_EXIT, 0, limit=100, stop=90, target=None,
               validity=3, max_days=10, cost_frac=0)
    assert out.exit == 85 and out.r == pytest.approx(-1.5), "a gap is worse than the stop"


def test_strategy_exit_is_taken_at_the_next_open():
    o, h, l, c = bars([(100, 101, 99, 100), (100, 102, 99.5, 101),
                       (101, 104, 100.5, 103), (105, 106, 104, 105)])
    rule = np.array([False, False, True, False])
    out = play(o, h, l, c, rule, 0, limit=100, stop=90, target=None,
               validity=3, max_days=10, cost_frac=0)
    assert out.exit == 105 and out.reason == "signal"


def test_costs_come_off_in_risk_units():
    o, h, l, c = bars([(100, 101, 99, 100), (100, 100.5, 99.5, 100), (100, 121, 100, 120)])
    out = play(o, h, l, c, NO_EXIT, 0, limit=100, stop=90, target=120,
               validity=3, max_days=10, cost_frac=0.01)
    # +20 on 10 risked = +2R, minus 1% of 100 = 1 = 0.1R
    assert out.r == pytest.approx(1.9)


def test_time_stop_exits_at_the_close():
    rows = [(100, 101, 99, 100)] + [(100, 100.5, 99.5, 100 + i * 0.1) for i in range(1, 6)]
    o, h, l, c = bars(rows)
    out = play(o, h, l, c, np.zeros(6, bool), 0, limit=100, stop=90, target=None,
               validity=3, max_days=3, cost_frac=0)
    assert out.reason == "time" and out.exit_i == 3


def test_summary_and_drawdown():
    s = summarize([1, -1, -1, 2, -1])
    assert s["trades"] == 5 and s["win_rate"] == pytest.approx(0.4)
    assert s["expectancy_r"] == pytest.approx(0.0)
    assert s["profit_factor"] == pytest.approx(1.0)
    assert s["max_drawdown_r"] == pytest.approx(2.0)


def test_gate_lists_every_failure():
    ok, fails = judge({"expectancy_r": -0.1},
                      {"trades": 50, "expectancy_r": 0.02, "profit_factor": 1.05},
                      {"expectancy_r": 0.03}, {"p5_total_r": -4})
    assert not ok and len(fails) == 6


def test_gate_passes_a_real_edge():
    ok, fails = judge({"expectancy_r": 0.1},
                      {"trades": 300, "expectancy_r": 0.15, "profit_factor": 1.4},
                      {"expectancy_r": 0.01}, {"p5_total_r": 3})
    assert ok, fails


def test_monte_carlo_is_reproducible_and_sane():
    rs = [1.0, -1.0] * 50 + [0.5] * 20
    a, b = monte_carlo(rs), monte_carlo(rs)
    assert a == b
    assert a["p5_total_r"] < a["median_total_r"]


def test_rsi_and_signals_on_a_known_series():
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert rsi(up, 2).iloc[-1] == pytest.approx(100)
    # 260 rising days then a sharp two-day drop: a textbook RSI(2) washout.
    close = list(np.linspace(100, 200, 260)) + [190, 180]
    df = pd.DataFrame({"open": close, "high": [c * 1.01 for c in close],
                       "low": [c * 0.99 for c in close], "close": close},
                      index=pd.date_range("2020-01-01", periods=len(close), freq="B"))
    sig = STRATEGIES["rsi2_pullback"].rules(indicators(df))
    assert sig["entry"].iloc[-1] and not sig["entry"].iloc[-10]
    assert sig["stop"].iloc[-1] < sig["limit"].iloc[-1]


def test_markets_and_brokers():
    assert market_of("RELIANCE.NS") == "NSE" and market_of("TCS.BO") == "BSE"
    assert market_of("PMGOLD.AX") == "ASX" and market_of("NVDA") == "US"
    assert bare("PMGOLD.AX") == "PMGOLD"
    assert ticker_for("TCS", "BSE") == "TCS.BO" and ticker_for("NVDA", "US") == "NVDA"


def test_a_session_in_progress_is_not_final():
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    from advisor import session_closed

    ist = ZoneInfo("Asia/Kolkata")
    today = date(2026, 10, 28)
    assert not session_closed("NSE", today, datetime(2026, 10, 28, 11, 0, tzinfo=ist))
    assert session_closed("NSE", today, datetime(2026, 10, 28, 16, 5, tzinfo=ist))
    assert session_closed("NSE", date(2026, 10, 27), datetime(2026, 10, 28, 9, 0, tzinfo=ist))


def test_no_signal_from_a_suspended_stock():
    """Frozen prices collapse the range to ~0; a stop priced off that range
    turned one ordinary move into a -995,642R trade in the first run."""
    close = list(np.linspace(1, 2, 260)) + [1.9, 1.8] + [1.8] * 30
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close},
                      index=pd.date_range("2018-01-01", periods=len(close), freq="B"))
    for strat in STRATEGIES.values():
        assert not strat.rules(indicators(df))["entry"].iloc[-25:].any(), strat.name


# ─────────────────────── against the database (real bars) ───────────────────
def _db_has_bars(symbol):
    try:
        import db
        return bool(db.one("SELECT 1 AS x FROM bars WHERE symbol=%s LIMIT 1", (symbol,)))
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _db_has_bars("GLD"), reason="no bars backfilled")
def test_ledger_settles_an_idea_by_the_same_rules_as_the_backtest():
    """An idea from three weeks ago must be filled-or-expired now, never still
    'open' — and its outcome must match `play` on the same bars."""
    import db
    from advisor import bars, ledger
    from advisor.strategies import indicators

    x = indicators(bars.load("GLD", years=1))
    i = len(x) - 15
    d, close = x.index[i].date(), float(x["close"].iloc[i])
    row = db.one("""INSERT INTO ideas (signal_date, strategy, symbol, market, broker, action,
                      currency, limit_price, stop_price, valid_until)
                    VALUES (%s,'breakout_55','GLD','US','webull','BUY','USD',%s,%s,%s)
                    RETURNING id""", (d, close * 0.5, close * 0.4, d))
    try:
        ledger.update()
        got = db.one("SELECT status FROM ideas WHERE id=%s", (row["id"],))
        assert got["status"] == "expired", "a buy limit 50% below the market filled"
    finally:
        db.execute("DELETE FROM ideas WHERE id=%s", (row["id"],))


def test_report_is_silent_on_a_quiet_day_and_labels_previews():
    from datetime import date

    from advisor import report

    quiet = {"region": "IN", "as_of": date(2026, 10, 27), "tickets": [],
             "holdings": [{"display": "TCS", "currency": "INR", "close": 2500, "stop": 2400,
                           "stop_gap_pct": -4, "below_stop": False, "newly_below": False,
                           "below_200d": False}]}
    assert report.compose(quiet) is None

    t = {"action": "BUY", "display": "GLD", "market": "US", "broker": "webull",
         "currency": "USD", "limit": 300.0, "stop": 290.0, "stop_pct": -3.3, "target": None,
         "quantity": 2.5, "risk_amount": 25.0, "valid_until": date(2026, 10, 30),
         "label": "55-day breakout", "reasons": ["New high."], "notes": [], "preview": True,
         "evidence": {"symbols": 505, "region_trades": 4841, "recent_from": "2022-10-27",
                      "expectancy_r": 0.21, "win_rate": 0.31, "profit_factor": 1.31,
                      "older_expectancy_r": 0.16, "random_expectancy_r": 0.17,
                      "chance_100_trade_loss": 0.23, "p95_drawdown_r": 33,
                      "name_trades": 12, "name_expectancy_r": 0.4}}
    subject, text, page = report.compose({**quiet, "tickets": [t]})
    assert "Advice only" in text and "never places orders" in text
    assert "PREVIEW" in text and "in Webull" in text and "US$300.00" in text
    assert "1 idea" in subject


def test_scheduler_runs_the_advisor_after_each_close():
    from scheduler import build_scheduler

    ids = {j.id for j in build_scheduler().get_jobs()}
    assert {"advise_in", "advise_asx", "advise_us", "retest"} <= ids


# ───────────────────────────── second batch ─────────────────────────────────
def _frame_from(close):
    close = list(close)
    return indicators(pd.DataFrame(
        {"open": close, "high": [c * 1.01 for c in close], "low": [c * 0.99 for c in close],
         "close": close}, index=pd.date_range("2015-01-01", periods=len(close), freq="B")))


def test_monthly_strategies_only_act_on_the_first_session_of_a_month():
    x = _frame_from(np.linspace(50, 150, 600))
    for name in ("abs_momentum", "momentum_rotation"):
        x["mom_rank"] = 0.95
        entries = STRATEGIES[name].rules(x)["entry"]
        assert entries.any(), name
        assert (x.loc[entries[entries].index, "month_start"]).all(), name


def test_rotation_needs_a_rank_and_cannot_enter_without_one():
    x = _frame_from(np.linspace(50, 150, 600))
    assert not STRATEGIES["momentum_rotation"].rules(x)["entry"].any()


def test_momentum_rank_is_relative_to_the_region():
    from advisor.strategies import add_momentum_rank

    strong = _frame_from(np.linspace(50, 200, 600))
    weak = _frame_from(np.linspace(50, 60, 600))
    frames = {"S": strong, "W": weak}
    add_momentum_rank(frames)
    assert strong["mom_rank"].iloc[-1] == 1.0 and weak["mom_rank"].iloc[-1] == 0.5


def test_near_high_fires_on_arrival_not_every_day_at_the_top():
    x = _frame_from(np.linspace(50, 150, 600))
    entries = STRATEGIES["near_52w_high"].rules(x)["entry"]
    # A steady climb sits at its high every day: that is one arrival, not 300.
    assert entries.sum() <= 2


def test_quiet_trend_needs_the_cross_itself():
    down_then_up = list(np.linspace(150, 100, 300)) + list(np.linspace(100, 160, 300))
    x = _frame_from(down_then_up)
    sig = STRATEGIES["quiet_trend"].rules(x)
    cross = (x["sma20"] > x["sma100"]) & (x["sma20"].shift() <= x["sma100"].shift())
    assert not (sig["entry"] & ~cross).any()


def test_a_scan_never_floods_the_inbox():
    """Monthly strategies fired on 142 US names on one day in the first run.
    The scan keeps the strongest few, one per name, and counts the rest."""
    from advisor.scan import select

    tickets = [{"symbol": f"S{i % 40}", "mom_rank": i / 100} for i in range(100)]
    kept, left_out = select(tickets, 5)
    assert [t["mom_rank"] for t in kept] == [0.99, 0.98, 0.97, 0.96, 0.95]
    assert len({t["symbol"] for t in kept}) == 5
    assert left_out == 35, "40 distinct names, 5 kept"
    assert select([{"symbol": "A", "mom_rank": None}], None) == ([{"symbol": "A", "mom_rank": None}], 0)


def test_bad_prints_are_dropped_but_real_crashes_are_kept():
    from advisor.bars import drop_bad_prints

    idx = pd.date_range("2019-12-10", periods=14, freq="B")
    blip = [130.0] * 6 + [13.0, 13.0] + [130.0] * 6          # NIFTYBEES, Dec 2019
    crash = [100.0] * 7 + [40.0] * 7                          # a real, lasting fall
    for close, expect in ((blip, 12), (crash, 14)):
        df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)
        assert len(drop_bad_prints(df)) == expect


# ───────────────────────────── account replay ───────────────────────────────
def _synthetic(n_names=8, days=700, seed=3):
    rng = np.random.default_rng(seed)
    frames = {}
    for k in range(n_names):
        steps = rng.normal(0.0006 * (k - 3), 0.015, days)
        close = 100 * np.exp(np.cumsum(steps))
        df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                           "close": close},
                          index=pd.date_range("2018-01-01", periods=days, freq="B"))
        frames[f"S{k}"] = indicators(df)
    from advisor.strategies import add_momentum_rank
    add_momentum_rank(frames)
    return frames


def test_account_replay_respects_cash_and_position_limits(monkeypatch):
    from advisor import portfolio_sim

    cfg = {"risk_per_trade_pct": 2.0, "max_position_pct": 40, "max_new_ideas_per_scan": 2,
           "max_open_positions": 3, "validity_sessions": 3,
           "round_trip_cost_pct": {"US": 0.6}}
    monkeypatch.setattr(portfolio_sim, "settings", lambda: cfg)
    monkeypatch.setattr("advisor.simulate.settings", lambda: cfg)
    out = portfolio_sim.replay(_synthetic(), ["abs_momentum", "quiet_trend"])
    assert out["exposure"].max() <= 1.0 + 1e-9, "spent more than the account had"
    assert out["equity"].iloc[0] == pytest.approx(1.0)
    assert out["trades"], "nothing traded on eight trending names"
    assert out["skipped"]["positions"] + out["skipped"]["cap"] > 0, "limits never bound"


def test_flat_market_cannot_make_money_from_nothing(monkeypatch):
    """Prices that never move: every trade can only lose its costs."""
    from advisor import portfolio_sim

    cfg = {"risk_per_trade_pct": 1.0, "max_position_pct": 20, "max_new_ideas_per_scan": 5,
           "max_open_positions": 10, "validity_sessions": 3,
           "round_trip_cost_pct": {"US": 0.6}}
    monkeypatch.setattr(portfolio_sim, "settings", lambda: cfg)
    monkeypatch.setattr("advisor.simulate.settings", lambda: cfg)
    frames = _synthetic()
    for x in frames.values():
        for col in ("open", "high", "low", "close"):
            x[col] = 100.0
        x["high"], x["low"] = 100.5, 99.5
    out = portfolio_sim.replay(frames, ["abs_momentum", "quiet_trend", "near_52w_high"])
    assert out["equity"].iloc[-1] <= 1.0 + 1e-9



# ───────────────────────────── account gate ─────────────────────────────────
def test_account_gate_needs_better_sharpe_and_shallower_falls(monkeypatch):
    from advisor import portfolio_sim

    monkeypatch.setattr(portfolio_sim, "settings", lambda: {"gate": {
        "account_beats_index_sharpe": True, "account_beats_index_drawdown": True}})
    idx = {"sharpe": 0.87, "max_drawdown": -0.34}
    ok, _ = portfolio_sim.judge_account({"strategy": {"sharpe": 1.01, "max_drawdown": -0.23},
                                         "index": idx})
    assert ok
    ok, fails = portfolio_sim.judge_account({"strategy": {"sharpe": 1.01, "max_drawdown": -0.40},
                                             "index": idx})
    assert not ok and "worst fall" in fails[0]
    ok, fails = portfolio_sim.judge_account({"strategy": {"sharpe": 0.80, "max_drawdown": -0.20},
                                             "index": idx})
    assert not ok and "Sharpe" in fails[0]
    ok, _ = portfolio_sim.judge_account({"strategy": {"sharpe": 2, "max_drawdown": -0.1},
                                         "index": {}, "index_symbol": "SPY"})
    assert not ok, "a missing index must fail, not pass by default"


@pytest.mark.skipif(not _db_has_bars("SPY"), reason="no bars backfilled")
def test_a_region_whose_account_fails_sends_nothing(monkeypatch):
    from advisor import portfolio_sim, scan

    monkeypatch.setattr(scan, "_store", lambda t: 0)
    monkeypatch.setattr(portfolio_sim, "account", lambda region: {"passed": False})
    res = scan.scan("US")
    assert res["tickets"] == [] and res["gate"]["account"] is False
    monkeypatch.setattr(portfolio_sim, "account", lambda region: None)
    assert scan.scan("US")["tickets"] == [], "no replay must mean no tickets"


def test_ticket_evidence_states_the_account_result_from_the_numbers():
    from advisor.report import _account_line

    a = {"since": "2015-12-08", "index": "SPY", "cagr": 0.141, "index_cagr": 0.147,
         "max_drawdown": -0.225, "index_max_drawdown": -0.337, "sharpe": 1.01,
         "index_sharpe": 0.87, "passed": True}
    line = _account_line(a)
    assert "+14.1% a year vs +14.7% for SPY" in line
    assert "About the index's return, with smaller falls." in line
    assert "more than the index" in _account_line({**a, "cagr": 0.20})


@pytest.mark.skipif(not _db_has_bars("SPY"), reason="no bars backfilled")
def test_a_dry_run_scan_stores_nothing(monkeypatch):
    """A dry run once stored three tickets as sent, so the real run that
    evening would have skipped them as duplicates and emailed nothing."""
    from advisor import scan

    stored = []
    monkeypatch.setattr(scan, "_store", lambda t: stored.append(t) or len(t))
    scan.scan("US")
    scan.scan("US", preview=True, store=True)
    assert stored == [], "a scan that was not asked to store, stored"


def test_exit_alerts_lead_the_email():
    from datetime import date

    from advisor import report

    res = {"region": "US", "as_of": date(2026, 10, 27), "tickets": [], "holdings": [],
           "exits": [{"symbol": "SPGI", "broker": "webull", "strategy": "quiet_trend",
                      "currency": "USD", "fill_price": 440.0, "last": 470.0,
                      "reason": "the strategy's exit fired at today's close"}],
           "closed": []}
    subject, text, _ = report.compose(res)
    assert "SELL SPGI" in subject
    assert text.index("SELL SPGI at the next open") < text.index("Your holdings")
    assert "+6.8% from the ticket's fill" in text


def test_ticket_says_how_much_to_add_from_the_set_aside_funds(monkeypatch):
    import numpy as np
    from advisor import scan
    from advisor.strategies import STRATEGIES, indicators

    close = list(np.linspace(100, 200, 300))
    df = pd.DataFrame({"open": close, "high": [c * 1.01 for c in close],
                       "low": [c * 0.99 for c in close], "close": close},
                      index=pd.date_range("2025-01-01", periods=300, freq="B"))
    cfg = {"validity_sessions": 3, "risk_per_trade_pct": 0.5, "max_position_pct": 10,
           "new_funds": {"kite": 100000}}
    monkeypatch.setattr(scan, "settings", lambda: cfg)
    accts = {"kite": {"value": 300000.0, "cash": 500.0, "currency": "INR"}}
    t = scan.ticket("TEST.NS", STRATEGIES["breakout_55"], indicators(df), accts, {})
    assert t["fund"] and t["fund"]["broker"] == "kite" and t["fund"]["within_reserve"]
    assert t["fund"]["amount"] % 1000 == 0
    assert any("Add about ₹" in n and "you set aside" in n for n in t["notes"])


# ──────────────────────────────── simulation runs ───────────────────────────
def test_candidate_score_prefers_better_evidence_and_nearer_triggers():
    from advisor.candidates import _score

    strong = {"stats": {"recent": {"expectancy_r": 0.8}}}
    weak = {"stats": {"recent": {"expectancy_r": 0.1}}}
    assert _score(strong, None) > _score(weak, None)
    assert _score(strong, {"stats": {"expectancy_r": 1.0}}) > _score(strong, {"stats": {"expectancy_r": -0.5}})
    assert _score(strong, None, distance=0.5) > _score(strong, None, distance=3.0)


def test_candidates_are_json_safe():
    import json

    import numpy as np

    from advisor.candidates import _clean

    out = _clean({"a": np.float64("nan"), "b": np.int64(3), "c": [np.bool_(True), float("inf")]})
    assert json.loads(json.dumps(out)) == {"a": None, "b": 3, "c": [True, None]}


def _has_runs():
    try:
        import db
        return bool(db.one("SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL LIMIT 1"))
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_runs(), reason="no stored simulation run")
def test_a_stored_run_opens_with_its_gate_and_candidates():
    from web import dashboard

    import db
    run_id = db.one("SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL "
                    "ORDER BY id DESC LIMIT 1")["id"]
    ctx = dashboard.sim_run(run_id)
    assert ctx["run"]["id"] == run_id and ctx["matrix"]
    assert set(ctx["candidates"]) == {"passed", "missed", "near_passed", "near_other"}
    # Non-passing signals are kept and labelled, not hidden.
    for row in ctx["candidates"]["missed"]["rows"]:
        assert row["passed_gate"] is False



def test_an_interrupted_run_is_closed_not_left_running():
    """A scheduler restart killed run #3 mid-way; it stayed 'running' forever,
    which disabled Run now and kept the page reloading."""
    import db
    from advisor import runs

    row = db.one("INSERT INTO sim_runs (trigger, started_at) VALUES ('pytest', now() - interval '2 hours') "
                 "RETURNING id")
    try:
        assert runs.close_interrupted() >= 1
        got = db.one("SELECT finished_at, error FROM sim_runs WHERE id=%s", (row["id"],))
        assert got["finished_at"] is not None and "interrupted" in got["error"]
    finally:
        db.execute("DELETE FROM sim_runs WHERE id=%s", (row["id"],))
