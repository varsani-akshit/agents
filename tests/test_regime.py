"""Conditions: rules learnt on the older years must hold on the recent ones
before they touch a ticket."""
from __future__ import annotations

import numpy as np
import pandas as pd

from advisor import regime
from advisor.simulate import Trade

CUTOFF = pd.Timestamp("2021-01-01")


def _states(start="2013-01-01", end="2026-01-01") -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    # Alternate calendar months between the two states.
    fear = np.where(days.month % 2 == 0, "calm", "fearful")
    return pd.DataFrame({"fear": fear}, index=days)


def _trades(rule, n=1200, seed=1) -> list[Trade]:
    """n trades spread over 2014–2025; `rule(state, recent)` gives the mean R."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2014-01-01", "2025-12-31", periods=n)
    st = _states()
    out = []
    for d in days:
        s = st.at[d.normalize(), "fear"]
        r = rng.normal(rule(s, d >= CUTOFF), 1.0)
        out.append(Trade("X", d, d, d + pd.Timedelta(days=5), float(r), "stop"))
    return out


def _random(n=1200, seed=2) -> list[Trade]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2014-01-01", "2025-12-31", periods=n)
    return [Trade("X", d, d, d, float(rng.normal(0, 1)), "stop") for d in days]


def test_state_builders():
    s = pd.Series(np.r_[np.linspace(100, 50, 300), np.linspace(50, 200, 300)],
                  index=pd.date_range("2020-01-01", periods=600))
    t = regime.trend(s)
    assert t.iloc[:199].isna().all() and t.iloc[-1] == "up" and t.iloc[299] == "down"
    d = regime.direction(s, "rising", "falling")
    assert d.iloc[-1] == "rising" and d.iloc[250] == "falling"
    th = regime.thirds(pd.Series(np.arange(1000.0), index=pd.date_range("2020-01-01", periods=1000)),
                       "low", "mid", "high")
    assert th.iloc[-1] == "high"


def test_mood_reads_a_turn_in_the_news():
    days = pd.date_range("2020-01-01", periods=800)
    tone = pd.Series(np.r_[np.zeros(700), np.full(100, -3.0)], index=days) + \
        np.random.default_rng(0).normal(0, 0.1, 800)
    m = regime.mood(tone, "upbeat", "usual", "gloomy")
    assert m.iloc[-1] == "gloomy"
    # Steady news reads as a mix of states, not as gloom.
    assert (m.iloc[400:700] == "gloomy").mean() < 0.5


def test_learn_finds_the_losing_state():
    rng = np.random.default_rng(0)
    labels = pd.DataFrame({"fear": ["fearful"] * 200 + ["calm"] * 200})
    rs = np.r_[rng.normal(-0.4, 1, 200), rng.normal(0.4, 1, 200)]
    rules = regime.learn(rs, labels)
    assert [(r["dim"], r["state"]) for r in rules["avoid"]] == [("fear", "fearful")]
    assert [(r["dim"], r["state"]) for r in rules["favour"]] == [("fear", "calm")]


def test_learn_ignores_noise_and_thin_states():
    rng = np.random.default_rng(3)
    labels = pd.DataFrame({"fear": ["fearful"] * 200 + ["calm"] * 200 + ["normal"] * 10})
    rules = regime.learn(rng.normal(0.1, 1, 410), labels)
    assert rules == {"avoid": [], "favour": []}


def test_allowed_filter_and_favour():
    labels = pd.DataFrame({"fear": ["calm", "fearful", "normal"], "trend": ["up", "up", "down"]})
    rules = {"avoid": [{"dim": "trend", "state": "down"}], "favour": [{"dim": "fear", "state": "calm"}]}
    assert regime.allowed(labels, rules, "filter").tolist() == [True, True, False]
    assert regime.allowed(labels, rules, "favour").tolist() == [True, False, False]


def test_a_rule_that_holds_out_of_sample_is_used():
    trades = _trades(lambda s, recent: 0.35 if s == "calm" else -0.35)
    out = regime.condition(trades, _random(), CUTOFF, True, _states())
    assert out["mode"] == "filter" and out["use"], out.get("fails")
    assert out["recent_after"]["expectancy_r"] > out["recent_before"]["expectancy_r"]
    assert {(r["dim"], r["state"]) for r in out["rules"]["avoid"]} == {("fear", "fearful")}


def test_a_rule_that_reverses_on_the_recent_years_is_not_used():
    # Older years: fearful months lose. Recent years: the opposite.
    trades = _trades(lambda s, recent: (0.35 if s == "calm" else -0.35) * (-1 if recent else 1))
    out = regime.condition(trades, _random(), CUTOFF, True, _states())
    assert out["rules"]["avoid"] and not out["use"]


def test_a_rule_whose_data_stops_before_the_recent_years_is_untested():
    trades = _trades(lambda s, recent: 0.35 if s == "calm" else -0.35)
    st = _states()
    st.loc[st.index >= CUTOFF, "fear"] = None        # the feed stopped in 2020
    out = regime.condition(trades, _random(), CUTOFF, True, st)
    assert out["rules"]["avoid"] and not out["use"]
    assert out["removed_recent"] == 0 and "untested" in out["fails"][0]


def test_a_failing_strategy_can_pass_only_in_its_favoured_conditions():
    trades = _trades(lambda s, recent: 0.4 if s == "calm" else -0.5)
    out = regime.condition(trades, _random(), CUTOFF, False, _states())
    assert out["mode"] == "favour" and out["use"], out.get("fails")
    assert 0.4 < out["share_kept"] < 0.6


def test_on_today():
    cond = {"use": True, "mode": "filter",
            "rules": {"avoid": [{"dim": "fear", "state": "fearful"}], "favour": []}}
    ev = {"passed": True, "stats": {"conditions": cond}}
    assert regime.on_today(ev, "US", {"fear": "calm"})[0]
    on, why = regime.on_today(ev, "US", {"fear": "fearful"})
    assert not on and why.startswith("paused")

    fav = {"use": True, "mode": "favour",
           "rules": {"avoid": [], "favour": [{"dim": "fear", "state": "calm"}]}}
    ev = {"passed": False, "stats": {"conditions": fav}}
    assert regime.on_today(ev, "US", {"fear": "calm"})[0]
    assert not regime.on_today(ev, "US", {"fear": "fearful"})[0]

    unused = {**fav, "use": False}
    assert not regime.on_today({"passed": False, "stats": {"conditions": unused}}, "US", {"fear": "calm"})[0]
    # A passing strategy whose rules did not hold out of sample is left alone.
    assert regime.on_today({"passed": True, "stats": {"conditions": {**cond, "use": False}}},
                           "US", {"fear": "fearful"})[0]


def test_labels_read_the_signal_day():
    st = pd.DataFrame({"fear": ["calm", "calm", "fearful"]},
                      index=pd.date_range("2024-01-01", periods=3))
    lab = regime.label([pd.Timestamp("2024-01-02 15:30"), pd.Timestamp("2024-01-03"),
                        pd.Timestamp("2024-02-01")], st)
    assert lab["fear"].tolist()[:2] == ["calm", "fearful"] and pd.isna(lab["fear"].iloc[2])


def test_a_quiet_feed_drops_out_instead_of_freezing(monkeypatch):
    days = pd.date_range("2024-01-01", periods=60)
    live = pd.Series(["up"] * 60, index=days)
    quiet = pd.Series(["calm"] * 10, index=days[:10])      # stopped after ten days
    dims = [regime.Dimension("trend", "Trend", lambda: live, {}),
            regime.Dimension("fear", "Fear", lambda: quiet, {})]
    monkeypatch.setattr(regime, "_dims", lambda region: dims)
    regime._history_cached.cache_clear()
    h = regime.history("US")
    regime._history_cached.cache_clear()
    assert h["fear"].iloc[9 + regime.STALE_DAYS] == "calm"
    assert pd.isna(h["fear"].iloc[-1]) and h["trend"].iloc[-1] == "up"
