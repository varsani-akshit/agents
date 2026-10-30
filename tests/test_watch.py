"""The heartbeat and the weekly review."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from advisor import watch


def _bars(times, lows, highs, closes, tz="America/New_York"):
    idx = pd.DatetimeIndex(times).tz_localize(tz)
    return pd.DataFrame({"Low": lows, "High": highs, "Close": closes}, index=idx)


def test_only_todays_session_counts():
    f = _bars(["2026-10-28 15:59", "2026-10-29 09:31", "2026-10-29 10:00"],
              [90, 101, 99], [95, 103, 104], [94, 102, 103])
    now = datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc)
    r = watch.today_range(f, "US", now)
    assert r["low"] == 99 and r["high"] == 104 and r["last"] == 103
    # Before the open, yesterday's bars are not today's.
    early = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    assert watch.today_range(f[:1], "US", early) is None


def test_ticket_alerts():
    ideas = [{"id": 1, "symbol": "DRI", "currency": "USD", "status": "open", "limit_price": 216.56,
              "stop_price": 206.44, "target_price": None},
             {"id": 2, "symbol": "MSCI", "currency": "USD", "status": "filled", "limit_price": 604.94,
              "stop_price": 576.96, "target_price": 650.0},
             {"id": 3, "symbol": "SPGI", "currency": "USD", "status": "filled", "limit_price": 448.93,
              "stop_price": 427.08, "target_price": None}]
    q = {"DRI": {"low": 215.0, "high": 219, "last": 217},
         "MSCI": {"low": 570.0, "high": 655, "last": 600},
         "SPGI": {"low": 440.0, "high": 450, "last": 445}}
    kinds = sorted((a["symbol"], a["kind"]) for a in watch.ticket_alerts(ideas, q))
    assert kinds == [("DRI", "fill"), ("MSCI", "stop"), ("MSCI", "target")]


def test_shock_alerts_need_two_and_a_half_ranges_and_spare_core_holdings():
    held = {"TCS.NS": {}, "WOW.AX": {}, "PMGOLD.AX": {}}
    q = {"TCS.NS": {"last": 2380.0}, "WOW.AX": {"last": 40.0}, "PMGOLD.AX": {"last": 54.0}}
    ranges = {"TCS.NS": (2460.0, 30.0), "WOW.AX": (39.36, 0.6), "PMGOLD.AX": (57.0, 0.6)}
    out = {a["symbol"]: a for a in watch.shock_alerts(held, q, ranges)}
    assert set(out) == {"TCS.NS", "PMGOLD.AX"}           # WOW moved about one range
    assert out["TCS.NS"]["ref"] == "down" and "-3.3%" in out["TCS.NS"]["text"]
    assert "core holding" in out["PMGOLD.AX"]["text"]


def test_alerts_are_stored_once_a_day():
    import db

    db.execute("DELETE FROM alerts WHERE symbol = 'PYTEST-W'")
    try:
        a = {"symbol": "PYTEST-W", "kind": "stop", "ref": "9", "text": "t"}
        assert len(watch.store([a])) == 1
        assert watch.store([a]) == []
    finally:
        db.execute("DELETE FROM alerts WHERE symbol = 'PYTEST-W'")


def test_the_alert_email_leads_with_stops():
    subject, text = watch.compose([
        {"id": 1, "kind": "news", "symbol": "TCS.NS", "text": "news", "detail": {"url": "https://x"}},
        {"id": 2, "kind": "stop", "symbol": "MSCI", "text": "stop hit"}])
    assert subject == "Alfred · now: STOP MSCI · news TCS"
    assert text.index("stop hit") < text.index("news") and "https://x" in text
    assert text.startswith("Advice only")


def test_the_weekly_review_composes_without_history():
    from advisor import review

    subject, text = review.compose()
    assert subject.startswith("Alfred · week to")
    for part in ("Every call, scored", "Tickets:", "Markets", "The week ahead:"):
        assert part in text


def test_a_watched_name_alerts_at_its_level_once_a_week():
    import db

    db.execute("DELETE FROM alerts WHERE symbol = 'PYTEST-G'")
    try:
        plan = [{"symbol": "PYTEST-G", "below": 255.0, "note": "second tranche"}]
        assert watch.level_alerts(plan, {"PYTEST-G": {"low": 256.0, "last": 260.0}}) == []
        hit = watch.level_alerts(plan, {"PYTEST-G": {"low": 254.5, "last": 257.0}})
        assert len(hit) == 1 and hit[0]["kind"] == "level"
        assert "at or below your US$255.00 level" in hit[0]["text"] and "second tranche" in hit[0]["text"]
        watch.store(hit)
        # Still down tomorrow: no second email within the week.
        assert watch.level_alerts(plan, {"PYTEST-G": {"low": 250.0, "last": 251.0}}) == []
    finally:
        db.execute("DELETE FROM alerts WHERE symbol = 'PYTEST-G'")


def test_a_gated_signal_on_a_watched_name_alerts(monkeypatch):
    import db

    run = db.one("INSERT INTO sim_runs (trigger, finished_at) VALUES ('pytest', now()) RETURNING id")["id"]
    try:
        db.execute("""INSERT INTO sim_candidates (run_id, region, symbol, strategy, kind, passed_gate, score, detail)
                      VALUES (%s, 'US', 'PYTEST-G', 'quiet_trend', 'signal', true, 1, %s)""",
                   (run, '{"as_of": "2026-11-03", "limit": 300.0, "stop": 285.0, "label": "Calm trend start", "currency": "USD"}'))
        out = watch.signal_alerts([{"symbol": "PYTEST-G", "signal": True, "note": "third tranche"}])
        assert len(out) == 1 and out[0]["kind"] == "signal"
        assert "Calm trend start signalled a buy" in out[0]["text"] and "limit US$300.00" in out[0]["text"]
        assert watch.signal_alerts([{"symbol": "PYTEST-G", "signal": False}]) == []
    finally:
        db.execute("DELETE FROM sim_runs WHERE id = %s", (run,))


def test_alphabet_is_on_the_watch_list():
    w = {x["symbol"]: x for x in watch.watched()}
    assert w["GOOGL"]["below"] == 255.0 and w["GOOGL"]["signal"]
