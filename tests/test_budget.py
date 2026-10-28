"""Spend caps: every provider counts, and a cap stops a call before it is made."""
from __future__ import annotations

import pytest


def _caps(monkeypatch, *, today, month, daily=8.0, monthly=150.0, scheduled=5.0, autonomous=True):
    import config
    from brain import client

    monkeypatch.setattr(client, "spend_today", lambda: today)
    monkeypatch.setattr(client, "spend_month", lambda: month)
    monkeypatch.setattr(config, "DAILY_USD_CAP", daily)
    monkeypatch.setattr(config, "MONTHLY_USD_CAP", monthly)
    monkeypatch.setattr(config, "SCHEDULED_DAILY_USD_CAP", scheduled)
    monkeypatch.setattr(client, "AUTONOMOUS", autonomous)
    return client


def test_scheduled_jobs_stop_first_and_leave_room_for_questions(monkeypatch):
    client = _caps(monkeypatch, today=4.99, month=40)
    client.check_budget(0.005)                                  # under $5: fine
    with pytest.raises(client.BudgetExceeded, match="scheduled-job cap"):
        client.check_budget(0.02)
    client = _caps(monkeypatch, today=4.99, month=40, autonomous=False)
    client.check_budget(0.02)                                   # your own question: allowed


def test_daily_and_monthly_caps_bind_everyone(monkeypatch):
    client = _caps(monkeypatch, today=7.99, month=40, autonomous=False)
    with pytest.raises(client.BudgetExceeded, match="daily cap"):
        client.check_budget(0.02)
    client = _caps(monkeypatch, today=1, month=149.99, autonomous=False)
    with pytest.raises(client.BudgetExceeded, match="monthly cap"):
        client.check_budget(0.02)


def test_the_ledger_counts_every_provider():
    import db
    from brain import client

    db.execute("""INSERT INTO api_calls (provider, model, purpose, input_tokens, output_tokens, usd)
                  VALUES ('azure', 'gpt-5.5', 'pytest.budget', 1, 1, 0.5)""")
    try:
        before = client.spend_today()
        assert before >= 0.5, "Azure spend was not counted"
    finally:
        db.execute("DELETE FROM api_calls WHERE purpose = 'pytest.budget'")


def test_a_routed_call_is_refused_before_any_provider_is_tried(monkeypatch):
    from brain import client, llm, router

    tried = []
    monkeypatch.setattr(client, "check_budget",
                        lambda est=0.02: (_ for _ in ()).throw(client.BudgetExceeded("daily cap reached")))
    monkeypatch.setattr(llm, "_call_gemini", lambda *a, **k: tried.append(a) or ({}, 1, 1))
    with pytest.raises(client.BudgetExceeded):
        router.complete_json("deep", system="s", user="u", schema={"type": "object"},
                             purpose="pytest.budget")
    assert tried == [], "a provider was called after the cap was hit"
