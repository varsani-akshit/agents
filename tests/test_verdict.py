"""Step 4: core holdings, where the money must be, and the news veto."""
from __future__ import annotations

import pandas as pd

from advisor import report, scan


def _frame(close: float, sma50: float, sma20: float, atr: float) -> pd.DataFrame:
    return pd.DataFrame({"close": [close], "sma50": [sma50], "sma20": [sma20], "atr22": [atr]})


def test_a_core_holding_is_never_flagged_and_trims_only_when_stretched():
    calm = scan.core_view(_frame(57.0, 60.0, 58.0, 0.6), 241)
    assert calm["core"] and not calm["below_stop"] and not calm["newly_below"] and calm["trim"] is None
    # 8 ATRs above the 50-day average: trim a fifth, buy back near the 20-day.
    hot = scan.core_view(_frame(68.0, 63.2, 64.5, 0.6), 241)
    assert hot["trim"] == {"units": 48, "rebuy": 64.5, "stretch_atr": 8.0}


def test_core_holdings_come_from_config():
    assert scan.is_core("PMGOLD.AX") and not scan.is_core("WOW.AX")


def test_the_funding_line_names_the_balance_and_the_conversion(monkeypatch):
    monkeypatch.setattr(scan, "_in_currency",
                        lambda amount, have, want: amount * (0.65 if (have, want) == ("AUD", "USD") else 1.0))
    assert scan.funding_line("kite", "INR", 45000, "INR") == "Needs ₹45,000 in Kite."
    assert scan.funding_line("webull", "AUD", 1200, "AUD") == "Needs A$1,200 in Webull."
    us = scan.funding_line("webull", "USD", 1300, "AUD")
    assert us.startswith("Needs US$1,300 in Webull's USD balance — about A$2,000")
    cb = scan.funding_line("coinbase", "USD", 65, "AUD")
    assert cb.startswith("Needs about A$100 in Coinbase")


def test_contradicted_tickets_are_held_back_and_the_next_steps_in(monkeypatch):
    from advisor import story

    reads = {"A": {"direction": "contradicts", "strength": 0.8},
             "B": {"direction": "supports", "strength": 0.7},
             "C": {"direction": "contradicts", "strength": 0.3},      # too weak to veto
             "D": {"direction": "neutral", "strength": 0.1}}
    monkeypatch.setattr(story, "for_ticket", lambda t, region: reads[t["symbol"]])
    t = lambda s: {"symbol": s}  # noqa: E731
    kept, rest, held = scan.news_check([t("A"), t("B")], [t("C"), t("D")], "US", cap=2)
    assert [x["symbol"] for x in kept] == ["B", "C"]
    assert [x["symbol"] for x in held] == ["A"]
    assert [x["symbol"] for x in rest] == ["D"]


def test_a_failed_read_is_not_a_veto(monkeypatch):
    from advisor import story

    def boom(t, region):
        raise RuntimeError("model down")
    monkeypatch.setattr(story, "for_ticket", boom)
    kept, rest, held = scan.news_check([{"symbol": "A"}], [], "US", cap=5)
    assert [x["symbol"] for x in kept] == ["A"] and not held


def test_the_email_lists_what_the_news_held_back():
    result = {"region": "US", "as_of": pd.Timestamp("2026-10-28"), "tickets": [], "holdings": [],
              "held_back": [{"display": "CSGP", "venue": "Nasdaq", "market": "US",
                             "label": "Calm trend start",
                             "story": {"thesis": "Fell 14% on earnings.", "risk": "a rebound"}}]}
    subject, text, _ = report.compose(result)
    assert "1 held back by the news" in subject
    assert "CSGP on Nasdaq (Calm trend start): Fell 14% on earnings." in text
    assert "Would change the read: a rebound" in text


def test_a_ticket_names_its_exchange_and_funding(monkeypatch):
    monkeypatch.setattr(report, "_evidence_line", lambda e: "evidence")
    t = {"action": "BUY", "display": "MSCI", "venue": "NYSE", "market": "US", "broker": "webull",
         "currency": "USD", "limit": 604.94, "stop": 576.96, "stop_pct": -4.6, "target": None,
         "quantity": 2.19, "risk_amount": 61.28, "valid_until": pd.Timestamp("2026-11-02"),
         "label": "Calm trend start", "reasons": ["r"], "notes": [], "evidence": {},
         "funding": "Needs US$1,325 in Webull's USD balance — about A$2,040, converted in Webull."}
    text = report.ticket_text(t)
    assert text.startswith("BUY MSCI on NYSE · in Webull")
    assert "  Funding: Needs US$1,325 in Webull's USD balance" in text


def test_crypto_is_a_market_of_its_own_everywhere():
    from advisor import REGIONS, region_of, settings, venue

    assert REGIONS["CRYPTO"] == ("CRYPTO",) and region_of("CRYPTO") == "CRYPTO"
    cfg = settings()
    assert cfg["benchmark"]["CRYPTO"] == "BTC-USD"
    assert cfg["round_trip_cost_pct"]["CRYPTO"] >= 1.0
    assert venue("BTC-USD") == "Coinbase" and venue("PMGOLD.AX") == "ASX"
