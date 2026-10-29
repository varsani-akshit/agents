"""The non-broker data feeds: parsing each source's shape without the network,
and the one rule that matters for Coinbase — the key must be View-only."""
from __future__ import annotations

from datetime import date, datetime, timezone

from brokers import coinbase
from data import crypto, filings, finnhub, macro, positioning, tone, twelve


def test_coinbase_refuses_a_key_that_can_trade():
    assert coinbase.view_only({"can_view": True, "can_trade": False, "can_transfer": False})
    assert not coinbase.view_only({"can_view": True, "can_trade": True, "can_transfer": False})
    assert not coinbase.view_only({"can_view": True, "can_trade": False, "can_transfer": True})
    assert not coinbase.view_only({"can_view": False})


def test_coinbase_candles_are_ordered_and_named():
    raw = [[1700003600, 1.0, 3.0, 2.0, 2.5, 10.0], [1700000000, 0.9, 2.0, 1.5, 1.0, 5.0]]
    rows = crypto.candle_rows("BTC-USD", raw)
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    sym, ts, o, h, l, c, v = rows[0]
    assert sym == "BTC-USD" and ts.tzinfo is timezone.utc
    assert (o, h, l, c, v) == (1.5, 2.0, 0.9, 1.0, 5.0)


def test_hyperliquid_rows_pair_universe_with_contexts():
    meta = {"universe": [{"name": "BTC"}, {"name": "ETH"}]}
    ctxs = [{"funding": "0.0001", "openInterest": "100", "markPx": "80000", "dayNtlVlm": "1e9",
             "premium": "0.0002"},
            {"funding": "x", "openInterest": "1", "markPx": "2"}]      # bad row skipped
    ts = datetime(2026, 10, 29, tzinfo=timezone.utc)
    rows = crypto.hyperliquid_rows(meta, ctxs, ts)
    assert rows == [("hyperliquid", "BTC", ts, 0.0001, 100.0, 80000.0, 1e9, 0.0002)]


def test_finnhub_earnings_rows_place_the_time_of_day():
    rows = finnhub.earnings_rows([
        {"symbol": "NVDA", "date": "2026-11-19", "hour": "amc", "epsEstimate": 1.2},
        {"symbol": "X", "date": "2026-11-20", "hour": "bmo"},
        {"symbol": "", "date": "2026-11-20"}])
    assert len(rows) == 2
    assert rows[0]["at"].hour == 17 and rows[0]["estimate"] == 1.2
    assert rows[1]["at"].hour == 8 and rows[0]["kind"] == "earnings"


def test_finnhub_news_becomes_documents_tagged_with_the_company():
    docs = finnhub.news_docs("MSCI", [{"headline": "MSCI falls", "url": "https://x.y/a?utm_source=z",
                                       "summary": "down 2.6%", "datetime": 1761600000}], "Finnhub · MSCI")
    assert docs[0]["symbol"] == "MSCI" and docs[0]["published_at"].tzinfo is timezone.utc
    assert "utm_source" not in docs[0]["url"]
    assert docs[0]["content_hash"]


def test_gdelt_points_parse_dates():
    pts = tone.points({"timeline": [{"series": "Average Tone", "data": [
        {"date": "20170101T000000Z", "value": -1.5}, {"date": "bad", "value": 1}]}]})
    assert pts == [(date(2017, 1, 1), -1.5)]


def test_fred_release_rows_mark_the_ones_that_move_markets():
    rows = macro.release_rows([{"release_name": "Consumer Price Index", "date": "2026-11-12",
                                "release_id": 10},
                               {"release_name": "Bankrate Monitor", "date": "2026-11-12"}],
                              {"Consumer Price Index"})
    assert [r["importance"] for r in rows] == ["high", "low"]
    assert rows[0]["at"] == datetime(2026, 11, 12, 13, 30, tzinfo=timezone.utc)


def test_nse_flows_split_fii_and_dii():
    pts = macro.flow_points([{"category": "FII/FPI *", "date": "28-Oct-2026", "netValue": "-1,234.5"},
                             {"category": "DII **", "date": "28-Oct-2026", "netValue": "2,000"}])
    assert pts["nse.fii_net_crore"] == [(date(2026, 10, 28), -1234.5)]
    assert pts["nse.dii_net_crore"] == [(date(2026, 10, 28), 2000.0)]


def test_cftc_rows_keep_only_named_markets():
    items = [{"contract_market_name": "GOLD", "report_date_as_yyyy_mm_dd": "2026-10-20T00:00:00.000",
              "noncomm_positions_long_all": "200000", "noncomm_positions_short_all": "50000",
              "open_interest_all": "400000"},
             {"contract_market_name": "WHEAT-HRW", "report_date_as_yyyy_mm_dd": "2026-10-20T00:00:00.000",
              "noncomm_positions_long_all": "1", "noncomm_positions_short_all": "1"}]
    rows = positioning.rows_from(items, {"GOLD": "gold"})
    assert rows == [("gold", date(2026, 10, 20), 200000.0, 50000.0, 400000.0)]


def test_sec_rows_build_the_document_url_and_flag_material_forms():
    recent = {"form": ["8-K", "4", "S-8"], "filingDate": ["2026-10-20", "2026-10-21", "2026-10-22"],
              "accessionNumber": ["0001045810-26-000100", "0001045810-26-000101", "0001045810-26-000102"],
              "primaryDocument": ["nvda-8k.htm", "form4.xml", "s8.htm"],
              "primaryDocDescription": ["8-K", "FORM 4", "S-8"]}
    rows = filings.sec_rows("NVDA", "0001045810", recent, days=10_000)
    assert [r["kind"] for r in rows] == ["8-K", "4"]
    assert rows[0]["url"] == "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000100/nvda-8k.htm"
    assert rows[0]["sensitive"] and not rows[1]["sensitive"]


def test_nse_rows_keep_recent_announcements_only():
    old = {"desc": "Financial Results", "attchmntFile": "https://a/b.pdf", "an_dt": "01-Jan-2020 10:00:00"}
    new = {"desc": "Financial Results", "attchmntFile": "https://a/c.pdf",
           "an_dt": datetime.now().strftime("%d-%b-%Y %H:%M:%S"), "attchmntText": "Q2 results"}
    rows = filings.nse_rows("TCS.NS", [old, new])
    assert len(rows) == 1 and rows[0]["title"] == "Q2 results" and rows[0]["sensitive"]


def test_asx_rows_carry_price_sensitivity():
    rows = filings.asx_rows("PMGOLD.AX", [{"documentKey": "k1", "headline": "Full Year Accounts",
                                           "date": "2026-09-22T08:02:50.000Z",
                                           "announcementType": "PERIODIC REPORTS",
                                           "isPriceSensitive": True}])
    assert rows[0]["sensitive"] and rows[0]["filed_at"].year == 2026 and "k1" in rows[0]["url"]


def test_twelve_symbols_and_closes():
    assert twelve.twelve_symbol("BF-B") == "BF.B"
    assert twelve.closes({"values": [{"datetime": "2026-10-28", "close": "604.94"}]}) == \
        [(date(2026, 10, 28), 604.94)]


def test_crypto_pairs_are_a_market_of_their_own():
    from advisor import BROKER, CURRENCY, market_of, ticker_for

    assert market_of("BTC-USD") == "CRYPTO" and market_of("NVDA") == "US"
    assert ticker_for("BTC", "COINBASE") == "BTC-USD"
    assert BROKER["CRYPTO"] == "coinbase" and CURRENCY["CRYPTO"] == "USD"


def test_crypto_sessions_close_at_utc_midnight():
    from datetime import date, datetime, timezone

    from advisor import session_closed

    late = datetime(2026, 10, 29, 23, 59, tzinfo=timezone.utc)
    assert session_closed("CRYPTO", date(2026, 10, 28), late)
    assert not session_closed("CRYPTO", date(2026, 10, 29), late)


def test_tone_waits_out_refusals_then_skips_the_window(monkeypatch):
    import pytest

    calls, slept = [], []

    def refuse(q, s, e):
        calls.append(1)
        raise tone.RateLimited("429")

    monkeypatch.setattr(tone, "_once", refuse)
    with pytest.raises(tone.RateLimited):
        tone.fetch("q", date(2020, 1, 1), date(2020, 12, 31), sleep=slept.append)
    assert len(calls) == len(tone.BACKOFF) + 1 and slept == list(tone.BACKOFF)


def test_tone_backfill_skips_years_already_stored(monkeypatch):
    full = {2017, 2018}
    monkeypatch.setattr(tone, "stored_days",
                        lambda name, s, e: 365 if s.year in full else 0)
    monkeypatch.setattr(tone, "last_day", lambda name: None)
    w = tone.windows("x", True, date(2020, 6, 1))
    assert [s.year for s, _ in w] == [2019, 2020]


def test_tone_feed_fails_only_if_every_window_was_refused(monkeypatch):
    import pytest

    monkeypatch.setattr(tone, "queries", lambda: {"a": "x", "b": "y"})
    monkeypatch.setattr(tone, "windows", lambda name, b, t: [(date(2026, 1, 1), date(2026, 1, 2))])
    monkeypatch.setattr(tone, "store", lambda name, pts: len(pts))

    def some(q, s, e, sleep):
        if q == "x":
            raise tone.RateLimited("429")
        return [(s, 1.0)]
    monkeypatch.setattr(tone, "fetch", some)
    assert tone.sync() == {"rows": 1, "windows": 1, "refused": 1}

    monkeypatch.setattr(tone, "fetch", lambda *a, **k: (_ for _ in ()).throw(tone.RateLimited("x")))
    with pytest.raises(RuntimeError):
        tone.sync()
