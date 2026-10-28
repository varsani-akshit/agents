"""The news-first brief: its format, its personal context, its story charts."""
from __future__ import annotations

import pytest


def test_format_is_built_around_stories_not_fixed_topics():
    from brain import digest

    f = digest.FORMAT
    for section in ("## Top line", "## Your money", "## The stories",
                    "## Markets at a glance", "## What's next", "## Opportunities"):
        assert section in f, section
    # The old fixed frame is gone: no mandatory topic list, no Direction table.
    for old in ("### Monetary Policy and Central Banks", "## Direction",
                "## Positioning Implications", "regime gauge under a regime call"):
        assert old not in f, old
    assert "precious\nmetals" not in digest.ROLE and "India, Australia" in digest.ROLE


def test_no_gold_worldview_is_loaded():
    from brain import agent

    text = agent.load_principles()
    assert "Lenses" in text and "Analysis Discipline" in text
    assert "Debt Cycles" not in text and "debasement trade" not in text.lower()


def test_personal_context_degrades_to_a_plain_line():
    from brain.pipeline import personal

    assert personal.as_text({}) == "(no portfolio on file)"
    text = personal.as_text({"holdings": [{"symbol": "TCS", "exchange": "NSE", "broker": "kite",
                                           "currency": "INR", "value": 9716, "pnl_pct": -46.2,
                                           "below_trailing_stop": False, "below_200d": True}],
                             "tickets": [], "near": [], "funds": {"kite": 100000}})
    assert "TCS (NSE, kite) value INR 9,716 · -46.2% vs cost · under its 200-day average" in text
    assert "kite: 100,000" in text


def _has_macro():
    try:
        from signals import storycharts
        return "OIL" in storycharts.instruments()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_macro(), reason="no macro history in the database")
def test_story_charts_build_from_data_and_skip_what_is_not_there():
    from signals import storycharts

    out = storycharts.build_many([
        {"kind": "compare", "symbols": ["OIL", "USDINR"], "days": 63, "title": "Crude and the rupee"},
        {"kind": "compare", "symbols": ["OIL", "USDINR"], "days": 63},          # duplicate
        {"kind": "performance", "symbols": ["GOLD", "SILVER", "COPPER"], "days": 21},
        {"kind": "price", "symbols": []},                                        # nothing asked
    ])
    assert list(out) == ["story_1", "story_2"]
    line, bars_ = out["story_1"], out["story_2"]
    assert line["type"] == "line" and len(line["series"]) == 2 and "rebased" in line["subtitle"]
    assert bars_["type"] == "bar_h" and len(bars_["values"]) == 3
    # Every number in a subtitle comes from the data, not from the request.
    assert "%" in line["subtitle"] and line["story"] is True


def test_a_failed_chart_never_raises(monkeypatch):
    from signals import storycharts

    monkeypatch.setattr(storycharts, "series", lambda s, d: (_ for _ in ()).throw(RuntimeError("boom")))
    assert storycharts.build({"kind": "price", "symbols": ["X"]}, "story_1") is None
