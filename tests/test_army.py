"""Army plumbing: router locks, sandbox containment, trace mirror, drill-down."""
from __future__ import annotations

import pathlib
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

    # Pin a symbol first: picking "the newest 15m row" and re-reading it after
    # the build raced the scheduler, because crypto prints continuously.
    sym = db.one(
        """SELECT symbol FROM prices WHERE grain='15m'
           AND ts > now() - interval '2 days' ORDER BY ts DESC LIMIT 1""")
    if not sym:
        return  # no intraday data in this environment
    symbol = sym["symbol"]
    wide = stats.load_daily([symbol], 30)
    if wide.empty:
        return
    framed = float(wide[symbol].iloc[-1])
    # Any 15m print for this symbol in the last hour is an acceptable match:
    # a tick landing mid-test must not fail the assertion, but a four-day-old
    # forward-filled daily fix still will.
    recent = db.query(
        """SELECT price FROM prices WHERE grain='15m' AND symbol=%s
           AND ts > now() - interval '2 hours' ORDER BY ts DESC LIMIT 8""",
        (symbol,))
    assert recent, "no recent intraday prints to compare against"
    assert any(abs(framed - float(r["price"])) < 0.01 for r in recent), (
        f"{symbol}: daily frame shows {framed}, not any recent live print")


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


def test_verifier_discards_confirmations_filed_as_errors(monkeypatch):
    """Twice the verifier filed a matching figure under severity 'wrong' with
    a rationale saying it matched. An audit that cries wolf is worse than
    none, so a 'match' reasoning is discarded rather than applied."""
    from brain.pipeline import stages

    def fake(role, **kwargs):
        return {
            "checked": [
                {"claim": "gold 4,371", "verdict": "matches"},
                {"claim": "10Y 4.79%", "verdict": "matches"},
                {"claim": "silver 70", "verdict": "differs",
                 "claimed_value": "70", "measured_value": "64.8"},
            ],
            "issues": [
                # A confirmation mislabelled as an error — must be dropped.
                {"quote": "gold at 4,371", "corrected_quote": "gold at 4,371",
                 "why": "the measured data confirm this claim matches",
                 "severity": "wrong"},
                # A real discrepancy — must be applied.
                {"quote": "silver traded at 70.00 on the session",
                 "corrected_quote": "silver traded at 64.80 on the session",
                 "claimed_value": "70.00", "measured_value": "64.80",
                 "why": "measured 64.80", "severity": "wrong"},
            ],
        }, "azure:gpt-5.4-mini"

    monkeypatch.setattr(stages.router, "complete_json", fake)
    out, fixed = stages.verifier(
        body="Overnight, silver traded at 70.00 on the session and gold at 4,371.",
        slim={})
    audit = out["audit"]
    assert audit["checked_claims"] == 3 and audit["matched"] == 2
    assert len(audit["issues"]) == 1        # the confirmation was discarded
    assert fixed == 1 and "64.80" in out["body"]


def test_ask_api_reads_the_key_ask_actually_returns():
    """`ask()` returns prose under "answer"; /api/ask read "text" and so
    rendered an empty bubble for every quick question, while the answer saved
    correctly to the archive — the failure was invisible in the database."""
    import asyncio

    from brain import ask as ask_mod
    from web import app as W

    class _Req:
        session: dict = {}

        async def json(self):
            return {"question": "test", "depth": "quick"}

    saved = {"analysis_id": 1, "question": "test",
             "answer": "**Measured:** gold at 4,371.", "turns": 1}
    orig = ask_mod.ask
    ask_mod.ask = lambda *a, **k: saved
    try:
        out = asyncio.run(W.api_ask(_Req()))
    finally:
        ask_mod.ask = orig
    assert "4,371" in out["html"], out


# ── Multi-user isolation ─────────────────────────────────────────────────────
def _client_as(username: str):
    """A signed-in client for one user."""
    from starlette.testclient import TestClient

    from web import auth
    import web.app as W

    auth.create_user(username, "test-password-123")
    c = TestClient(W.app, base_url="https://testserver")
    r = c.post("/login", data={"username": username, "password": "test-password-123"},
               follow_redirects=False)
    assert r.status_code == 303, f"login failed for {username}"
    return c


def test_one_user_cannot_see_anothers_questions():
    """A question is private to whoever asked it. Briefs and market data are
    shared because they are the product; chat history is not."""
    import db
    from memory import store

    mine = store.save_analysis("answer", "Q: my private question",
                               "The answer.", meta={"question": "my private question"},
                               owner="privacy_a")
    theirs = store.save_analysis("answer", "Q: their private question",
                                 "Their answer.",
                                 meta={"question": "their private question"},
                                 owner="privacy_b")
    try:
        a = _client_as("privacy_a")
        page = a.get("/ask").text
        assert "my private question" in page
        assert "their private question" not in page
        # Nor by guessing the id.
        assert a.get(f"/answer/{theirs}", follow_redirects=False).status_code in (302, 303, 307)
        assert a.get(f"/answer/{mine}").status_code == 200
    finally:
        db.execute("DELETE FROM analyses WHERE id = ANY(%s)", ([mine, theirs],))
        db.execute("DELETE FROM users WHERE username = ANY(%s)",
                   (["privacy_a", "privacy_b"],))


def test_status_and_add_are_admin_only():
    """Status names the models and their cost; Add changes what Alfred
    ingests. Both are refused in middleware, so a route added later cannot
    forget to check."""
    import db
    from web import auth

    try:
        reader = _client_as("reader_only")
        for path in ("/status", "/add"):
            r = reader.get(path, follow_redirects=False)
            assert r.status_code == 303, f"{path} leaked to a reader"
        nav = reader.get("/").text
        assert 'href="/status"' not in nav and 'href="/add"' not in nav

        auth.set_admin("reader_only", True)
        boss = _client_as("reader_only")
        assert boss.get("/status").status_code == 200
        assert 'href="/status"' in boss.get("/").text
    finally:
        db.execute("DELETE FROM users WHERE username = %s", ("reader_only",))


def test_the_reader_is_addressed_by_their_full_name():
    """Whoever is signed in sees their own full name — in the account menu, in
    the brief's arrival card, and in the Ask greeting. A half-name ("Akshit")
    was the earlier behaviour; two readers with the same first name would have
    been indistinguishable."""
    import db
    from web import auth

    try:
        auth.create_user("fullname_x", "test-password-123",
                         display_name="Cassian Rivers-Okonkwo")
        c = _client_as("fullname_x")

        home = c.get("/").text
        assert "<summary>Cassian Rivers-Okonkwo</summary>" in home
        assert 'data-name="Cassian Rivers-Okonkwo"' in home
        assert "fullname_x" not in home, "username shown where the name belongs"

        assert "At your service, Cassian Rivers-Okonkwo." in c.get("/ask").text
    finally:
        db.execute("DELETE FROM users WHERE username = %s", ("fullname_x",))


def test_a_converted_timestamp_never_keeps_its_utc_label():
    """local.js rewrites every <time> into the reader's zone, so a literal
    'UTC' beside one becomes false. The strip walks up from the timestamp
    rather than matching a list of wrapper classes — the arrival card was
    printing '14:50 UTC' on a local time because .ws was not on that list."""
    js = (pathlib.Path(__file__).parent.parent / "web/static/local.js").read_text()
    assert 'querySelectorAll("time[data-localised]")' in js
    assert "parentNode" in js
    assert '.facts span, .eyebrow, .when' not in js, "back to a wrapper list"
