"""The Story read and the record of calls."""
from __future__ import annotations

from datetime import date

from advisor import report, story


def test_short_names_are_what_headlines_say():
    assert story.short_name("Darden Restaurants, Inc.") == "Darden Restaurants"
    assert story.short_name("Tata Consultancy Services Limited") == "Tata Consultancy Services"
    assert story.short_name("MSCI Inc.") == "MSCI"
    assert story.short_name("Auburn Holdings Group Ltd") == "Auburn"
    assert story.short_name("Coal India") == "Coal India"


def test_a_read_cites_only_articles_in_its_dossier(monkeypatch):
    from brain import decide, router

    dos = {"name": "S&P Global", "setup": {"typical_hold_days": 20}, "events_in_window": ["x"],
           "news": [{"id": 1}, {"id": 2}]}
    monkeypatch.setattr(story, "dossier", lambda call: dos)
    monkeypatch.setattr(router, "complete_json", lambda *a, **k: (
        {"direction": "supports", "strength": 1.7, "thesis": "t", "risk": "r",
         "event_risk": False, "cites": [2, 99], "flags": []}, "azure:test"))
    monkeypatch.setattr(decide, "ask", lambda *a, **k: {
        "move": {"choice": "rise", "confidence": 0.8, "probabilities": {"rise": 0.9}}})
    s = story.read({"symbol": "SPGI"})
    assert s["cites"] == [2] and s["strength"] == 1.0 and s["model"] == "azure:test"
    assert s["decision"]["choice"] == "rise" and s["news_count"] == 2


def test_a_read_stands_without_decision_one(monkeypatch):
    from brain import decide, router

    monkeypatch.setattr(story, "dossier", lambda call: {"name": "X", "setup": {"typical_hold_days": 5},
                                                        "events_in_window": [], "news": []})
    monkeypatch.setattr(router, "complete_json", lambda *a, **k: (
        {"direction": "neutral", "strength": 0.2, "thesis": "", "risk": "", "event_risk": False,
         "cites": [], "flags": ["no company news"]}, "m"))

    def down(*a, **k):
        raise decide.DecisionError("HTTP 503")
    monkeypatch.setattr(decide, "ask", down)
    s = story.read({"symbol": "X"})
    assert s["direction"] == "neutral" and "error" in s["decision"]


def test_the_email_says_what_the_news_says():
    s = {"direction": "contradicts", "strength": 0.72, "thesis": "RBA inside the hold.",
         "risk": "a dovish surprise", "event_risk": True,
         "events": ["Tue 03 Nov: RBA decision", "Fri 06 Nov: x", "y", "z"]}
    lines = report._story_lines(s)
    assert lines[0].startswith("  News: CONTRADICTS (strength 0.7)")
    assert lines[1] == "  Watch for: a dovish surprise"
    assert lines[2].count(";") == 2
    assert report._story_lines(None) == [] and report._story_lines({}) == []


def test_calls_are_recorded_once_and_scored(monkeypatch):
    import db

    sym = "PYTEST-STORY"
    db.execute("DELETE FROM calls WHERE symbol = %s", (sym,))
    try:
        t = {"symbol": sym, "strategy": "quiet_trend", "signal_date": date(2026, 10, 27),
             "limit": 100.0, "stop": 95.0, "target": None, "display": sym, "label": "Calm trend start",
             "broker": "webull", "currency": "USD", "stop_pct": -5.0, "quantity": 1,
             "reasons": ["r"], "evidence": {"median_days": 20}}
        monkeypatch.setattr(story, "read", lambda call: {"direction": "supports", "strength": 0.5})
        assert story.for_ticket(t, "US")["direction"] == "supports"
        # A second ticket for the same signal reuses the stored read.
        monkeypatch.setattr(story, "read", lambda call: (_ for _ in ()).throw(AssertionError("re-read")))
        assert story.for_ticket(t, "US")["direction"] == "supports"
        assert db.one("SELECT count(*) n FROM calls WHERE symbol=%s", (sym,))["n"] == 1

        db.execute("UPDATE calls SET status='closed', r=1.5 WHERE symbol=%s", (sym,))
        card = story.scorecard()
        assert card["by_story"]["supports"]["trades"] >= 1
        assert card["by_gate"]["passes"]["trades"] >= 1

        rows = story.attach([{"symbol": sym, "strategy": "quiet_trend",
                              "detail": {"as_of": "2026-10-27"}}])
        assert rows[0]["story"]["direction"] == "supports"
    finally:
        db.execute("DELETE FROM calls WHERE symbol = %s", (sym,))
