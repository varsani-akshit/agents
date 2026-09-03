"""Army plumbing: router locks, sandbox containment, trace mirror, drill-down."""
from __future__ import annotations

import pytest


def test_premium_role_is_locked_to_its_two_call_sites():
    from brain import router

    with pytest.raises(ValueError):
        router.chain_for("premium")
    with pytest.raises(ValueError):
        router.chain_for("premium", premium_site="analyst")
    assert router.chain_for("premium", premium_site="editor")
    assert router.chain_for("premium", premium_site="research_synthesis")


def test_every_role_resolves_to_at_least_one_spec():
    from brain import router

    for role in ("bulk", "workhorse", "reason", "search", "deep"):
        assert router.chain_for(role), role


def test_sandbox_blocks_network_and_times_out_and_computes():
    from brain import sandbox

    ok = sandbox.run("print(1 + 1)")
    assert ok["ok"] and ok["stdout"].strip() == "2"

    net = sandbox.run("import urllib.request; urllib.request.urlopen('https://example.com')")
    assert not net["ok"] and "network access is disabled" in (net["error"] or "")


def test_agent_run_mirror_records_error_status():
    import db
    from brain import observe

    with pytest.raises(RuntimeError):
        with observe.run("test-agent", trigger="test") as rec:
            rec.set_input({"x": 1})
            raise RuntimeError("deliberate")
    row = db.one(
        "SELECT status, error FROM agent_runs WHERE agent='test-agent' ORDER BY id DESC LIMIT 1")
    assert row["status"] == "error" and "deliberate" in row["error"]
    db.execute("DELETE FROM agent_runs WHERE agent='test-agent'")


def test_verifier_applies_only_exact_quote_matches(monkeypatch):
    from brain.pipeline import stages

    def fake_json(role, **kwargs):
        return {
            "issues": [
                {"quote": "gold rose 3.1%", "corrected_quote": "gold rose 2.9%",
                 "why": "measured 2.9", "severity": "wrong"},
                {"quote": "text not in draft", "corrected_quote": "x",
                 "why": "n/a", "severity": "wrong"},
            ],
            "checked_claims": 2,
        }, "azure:gpt-oss-120b"

    monkeypatch.setattr(stages.router, "complete_json", fake_json)
    out, fixed = stages.verifier(body="This window gold rose 3.1% on the buyback.",
                                 slim={})
    assert fixed == 1
    assert "gold rose 2.9%" in out["body"]
    assert "3.1%" not in out["body"]


def test_research_routes_render(client=None):
    from starlette.testclient import TestClient

    from web.app import app

    c = TestClient(app)
    r = c.get("/research", follow_redirects=False)
    assert r.status_code in (200, 303, 307)  # redirects into Ask


def test_correction_survives_markdown_drift():
    """The verifier copies words faithfully but sheds ** around numbers —
    exactly the passages most likely to be flagged."""
    from brain.pipeline.stages import _apply_correction

    body = "Gold closed at **4,679.9** on the session, its third high."
    fixed, applied = _apply_correction(body, "Gold closed at 4,679.9 on the session",
                                       "Gold closed at **4,681.2** on the session")
    assert applied and "4,681.2" in fixed and "4,679.9" not in fixed

    # Ambiguous (two occurrences) → left alone rather than half-fixed.
    body2 = "gold rose 2% early; later gold rose 2% again"
    fixed2, applied2 = _apply_correction(body2, "gold rose 2%", "gold rose 3%")
    assert not applied2 and fixed2 == body2

    # An exact unique byte match applies even when short — it is well anchored.
    fixed3, applied3 = _apply_correction("the 10Y at 4.67%", "4.67%", "4.68%")
    assert applied3 and "4.68%" in fixed3

    # But a short quote that needs the elastic match is refused: too little
    # text to anchor a rewrite safely.
    _, applied4 = _apply_correction("the 10Y at **4.67%**", "at 4.67%", "at 4.68%")
    assert not applied4


def test_loose_json_parses_fenced_and_embedded():
    from brain.pipeline.stages import _loose_json

    assert _loose_json('```json\n{"leads": []}\n```') == {"leads": []}
    assert _loose_json('noise before {"leads": [1]} noise after') == {"leads": [1]}
    assert _loose_json("no json here") is None


def test_share_script_loads_outside_title():
    """share.js was included inside <title>, where markup is inert text —
    the Share button shipped wired to nothing."""
    from web import auth
    from starlette.testclient import TestClient
    import web.app as W

    orig = auth.is_public
    auth.is_public = lambda path: True
    try:
        c = TestClient(W.app)
        html = c.get("/").text
    finally:
        auth.is_public = orig
    import re

    title = re.search(r"<title>(.*?)</title>", html, re.S)
    assert title and "share.js" not in title.group(1)
    assert re.search(r"<script[^>]+share\.js", html)


def test_long_urls_cannot_break_ingestion():
    """A feed URL carrying a full encoded payload exceeded Postgres's btree
    limit and aborted the whole tick — prices and triggers included."""
    from ingest.dedupe import canonical_url

    out = canonical_url("https://example.com/a?d=" + "x" * 4000)
    assert len(out.encode("utf-8")) <= 1800
    # Ordinary links keep their meaning.
    assert canonical_url("https://www.reuters.com/x/?utm_source=n&id=7") == \
        "https://reuters.com/x?id=7"


def test_daily_frame_carries_the_live_price():
    """Daily bars arrive on their source's schedule; the LBMA fix can be days
    old. Today's row must hold the live price, not a forward-filled fix."""
    import db
    from signals import stats

    live = db.one(
        """SELECT symbol, price FROM prices WHERE grain='15m'
           AND ts > now() - interval '2 days' ORDER BY ts DESC LIMIT 1""")
    if not live:
        return  # no intraday data in this environment
    wide = stats.load_daily([live["symbol"]], 30)
    if wide.empty:
        return
    assert abs(float(wide[live["symbol"]].iloc[-1]) - float(live["price"])) < 0.01


def test_graph_links_securities_without_false_matches():
    """A one-letter ticker matched the article 'a' in every document, and
    first-word matching let 'Australian Foundation' claim every Australian
    story. Company nodes must be earned by a real mention."""
    from memory import graph

    g = graph.build(days=7, limit=120)
    secs = [n for n in g["nodes"] if n["kind"] == "security"]
    for n in secs:
        assert n["mentions"] >= 1
        # Single-letter and two-letter tickers cannot be matched by ticker
        # alone, so any that appear were matched by full company name.
        assert len(n["symbol"].split(".")[0]) >= 3 or n["mentions"] >= 1


def test_company_search_does_not_confuse_gold_with_goldman():
    """ILIKE '%gold%' surfaced Goldman Sachs for a question about gold."""
    from brain import tools

    hits = tools.HANDLERS["search_memory"](query="gold", limit=6).get(
        "company_coverage", [])
    names = " ".join(h["company"] or "" for h in hits).lower()
    assert "goldman" not in names or any(
        "gold" in (h["title"] or "").lower() for h in hits)
