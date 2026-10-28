"""Matching articles to the reader's names with Decision-1 — no network here."""
from __future__ import annotations

import pytest


def test_ask_posts_the_systemone_shape_and_records_spend(monkeypatch):
    import httpx

    from brain import decide

    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k")
    seen, ledger = {}, []

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json)
        return httpx.Response(200, json={"answers": {"q": {"noul": 0.97, "type": "noul"}},
                                         "usage": {"input_tokens": 1000, "output_tokens": 1}})

    monkeypatch.setattr(decide.httpx, "post", fake_post)
    monkeypatch.setattr(decide.db, "execute", lambda sql, params: ledger.append(params))
    out = decide.ask("text", {"q": {"type": "noul", "instructions": "?"}}, purpose="t")
    assert out == {"q": {"noul": 0.97, "type": "noul"}}
    # The OpenAI suffix is stripped: System One lives on its own route.
    assert seen["url"] == "https://x.services.ai.azure.com/providers/microsoft/v1/systemone"
    assert seen["json"]["model"] == "Decision-1" and seen["json"]["state"] == "text"
    assert ledger and ledger[0][-2] == pytest.approx(1000 * 0.05 / 1e6)


def test_only_confident_matches_count_and_get_an_impact_read(monkeypatch):
    from brain import news_match

    calls = []

    def fake_ask(state, questions, purpose):
        calls.append(purpose)
        if purpose == "news_match.about":
            return {"n0": {"noul": 0.99}, "n1": {"noul": 0.40}}
        return {"i0": {"choice": "negative", "confidence": 0.9}}

    monkeypatch.setattr(news_match.decide, "ask", fake_ask)
    hits = news_match.judge({"title": "Telix falls 12% on FDA letter", "summary": ""},
                            {"TLX.AX": "Telix Pharmaceuticals", "WOW.AX": "Woolworths Group"})
    assert [h["symbol"] for h in hits] == ["TLX.AX"]
    assert hits[0]["impact"] == "negative" and calls == ["news_match.about", "news_match.impact"]


def test_quote_pages_are_not_news():
    from brain.news_match import _NOT_NEWS

    assert _NOT_NEWS.search("JIO PLATFORMS LIMITED Share Price - Live NSE: JPL Stock Price & Chart")
    assert _NOT_NEWS.search("Telix Pharmaceuticals Quotes 1,095 Shares Issued Under Rights Plan")
    assert not _NOT_NEWS.search("Reliance's telecom giant Jio rises in debut after India's largest IPO")
