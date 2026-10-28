"""Unit tests for the deterministic layers. No network, no model calls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ingest import dedupe
from signals import stats, triggers


# ───────────────────────────────── dedupe ───────────────────────────────────
def test_canonical_url_strips_tracking():
    a = dedupe.canonical_url("https://WWW.Example.com/story/?utm_source=x&id=5&fbclid=z")
    assert a == "https://example.com/story?id=5"


def test_canonical_url_unwraps_google_news():
    wrapped = "https://news.google.com/rss/articles/abc?url=https://reuters.com/a/b&oc=5"
    assert dedupe.canonical_url(wrapped) == "https://reuters.com/a/b"


def test_canonical_url_trailing_slash_and_case():
    assert dedupe.canonical_url("HTTPS://Site.com/Path/") == "https://site.com/Path"


def test_content_hash_ignores_whitespace_and_case():
    a = dedupe.content_hash("Gold  Rises", "Body   text")
    b = dedupe.content_hash("gold rises", "body text")
    assert a == b


def test_content_hash_distinguishes_real_differences():
    assert dedupe.content_hash("Gold rises") != dedupe.content_hash("Silver rises")


def test_content_hash_falls_back_to_url_when_empty():
    h = dedupe.content_hash("", "", "https://example.com/x")
    assert h and h == dedupe.content_hash("", "", "https://example.com/x/")


def test_title_fingerprint_ignores_stopwords_and_order():
    a = dedupe.title_fingerprint("The Fed raises rates in March")
    b = dedupe.title_fingerprint("Fed raises rates March")
    assert a == b


# ───────────────────────────────── stats ────────────────────────────────────
def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_pct_change_basic():
    s = _series([100, 110])
    assert stats.pct_change(s, 1) == pytest.approx(10.0)


def test_pct_change_returns_none_when_too_short():
    assert stats.pct_change(_series([100]), 1) is None


def test_zscore_flags_outlier_move():
    # 60 days of ~0.1% drift then a 10% jump: must register as a large z-score.
    vals = [100 * (1.001**i) for i in range(60)]
    vals.append(vals[-1] * 1.10)
    z = stats.zscore_of_last_move(_series(vals))
    assert z is not None and z > 3


def test_zscore_none_on_constant_series():
    assert stats.zscore_of_last_move(_series([100.0] * 50)) is None


def test_rolling_corr_uses_returns_not_levels():
    """Two rising-but-unrelated series must not show near-perfect correlation."""
    rng = np.random.default_rng(0)
    a = _series(list(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 200))))
    b = _series(list(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 200))))
    corr = stats.rolling_corr(a, b, 90)
    assert corr is not None and abs(corr) < 0.5


def test_rolling_corr_detects_true_relationship():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, 200)
    a = _series(list(100 * np.cumprod(1 + base)))
    b = _series(list(100 * np.cumprod(1 - base)))  # exact inverse
    corr = stats.rolling_corr(a, b, 90)
    assert corr is not None and corr < -0.9


def test_gold_in_currencies_reading_is_labelled():
    idx = pd.date_range("2025-01-01", periods=300, freq="D")
    wide = pd.DataFrame(
        {
            "GOLD": np.linspace(2000, 4000, 300),
            "EURUSD": np.full(300, 1.1),
            "USDJPY": np.full(300, 150.0),
        },
        index=idx,
    )
    out = stats.gold_in_currencies(wide)
    assert "USD" in out and "EUR" in out and "JPY" in out
    # Monotonically rising gold with flat FX => at highs in every currency.
    assert out["_reading"].startswith("systemic")


# ──────────────────────────────── triggers ──────────────────────────────────
def _pack(symbol: str, chg_pct: float = 0.0, z: float = 0.0, bp: float | None = None):
    row = {"symbol": symbol, "last": 100.0, "chg_1d_pct": chg_pct, "z_1d_move": z}
    if bp is not None:
        row["chg_1d_bp"] = bp
    return {"performance": [row], "ratios": [], "yield_curve": {}, "correlation_flips": []}


def test_price_trigger_fires_critical_above_threshold():
    rules = triggers.load_rules()
    events = triggers.check_prices(_pack("GOLD", chg_pct=4.0), rules)
    assert any(e["severity"] == "Critical" and e["rule"] == "price_move" for e in events)


def test_price_trigger_silent_on_normal_move():
    rules = triggers.load_rules()
    events = triggers.check_prices(_pack("GOLD", chg_pct=0.4), rules)
    assert not [e for e in events if e["rule"] == "price_move"]


def test_yield_trigger_uses_basis_points_not_percent():
    """A 1% relative move in a 4% yield is 4bp — noise. Must not fire."""
    rules = triggers.load_rules()
    events = triggers.check_prices(_pack("US10Y", chg_pct=1.0, bp=4.0), rules)
    assert not [e for e in events if e["rule"] == "yield_move"]

    events = triggers.check_prices(_pack("US10Y", chg_pct=5.0, bp=20.0), rules)
    assert any(e["rule"] == "yield_move" and e["severity"] == "Critical" for e in events)


def test_zscore_trigger_fires_on_statistical_outlier():
    rules = triggers.load_rules()
    events = triggers.check_prices(_pack("DXY", chg_pct=0.5, z=3.5), rules)
    assert any(e["rule"] == "zscore_outlier" and e["severity"] == "Critical" for e in events)


def test_dedupe_key_stable_within_bucket():
    a = triggers._dedupe_key("price_move", "GOLD", "2026-08-26-0")
    b = triggers._dedupe_key("price_move", "GOLD", "2026-08-26-0")
    c = triggers._dedupe_key("price_move", "GOLD", "2026-08-26-1")
    assert a == b and a != c


# ───────────────────────────── entity hygiene ───────────────────────────────
def test_entity_aliases_canonicalise():
    from brain import extract

    for alias in ("Fed", "FOMC", "the fed", "Federal Open Market Committee"):
        assert extract.canonical(alias) == "Federal Reserve"
    assert extract.canonical("btc") == "Bitcoin"
    assert extract.canonical("treasuries") == "US Treasury Bonds"


def test_edge_rejects_self_reference():
    from brain import extract

    assert extract.upsert_edge(
        {"source": "Gold", "target": "gold", "relation": "affects",
         "direction": "positive", "strength": 0.5, "rationale": "x"}
    ) is False


def test_edge_rejects_unknown_relation():
    from brain import extract

    assert extract.upsert_edge(
        {"source": "Gold", "target": "Silver", "relation": "invented_relation",
         "direction": "positive", "strength": 0.5, "rationale": "x"}
    ) is False


# ─────────────────────────────── cost guard ─────────────────────────────────
class _Usage:
    input_tokens = 10_000
    output_tokens = 2_000
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


def test_pricing_matches_published_rates():
    from brain import client

    # Sonnet 5: $3/M in, $15/M out -> 10k in + 2k out = 0.03 + 0.03
    usd = client.price_call("claude-sonnet-5", _Usage())
    assert usd == pytest.approx(0.06, rel=1e-6)


def test_pricing_counts_web_searches():
    from brain import client

    base = client.price_call("claude-haiku-4-5", _Usage())
    with_search = client.price_call("claude-haiku-4-5", _Usage(), web_searches=10)
    assert with_search - base == pytest.approx(0.10, rel=1e-6)


def test_unknown_model_priced_conservatively():
    from brain import client

    assert client.price_call("some-future-model", _Usage()) >= client.price_call(
        "claude-sonnet-5", _Usage()
    )


def test_rate_rows_never_emit_percent_change():
    """Regression: a -1.02% change in a 4.7% yield is -4.8bp, not -102bp.

    Emitting both units invited the model to read the percent figure as basis
    points, so rate rows must carry basis points exclusively.
    """
    idx = pd.date_range("2025-01-01", periods=300, freq="D")
    wide = pd.DataFrame(
        {"US10Y": np.linspace(4.0, 4.7, 300), "GOLD": np.linspace(2000, 4000, 300)},
        index=idx,
    )
    rows = {r["symbol"]: r for r in stats.performance_table(wide)}

    rate = rows["US10Y"]
    assert rate["unit"] == "percent_yield"
    assert "chg_1d_bp" in rate
    assert not any(k.endswith("_pct") for k in rate), (
        f"rate row leaked a percent field: {sorted(rate)}"
    )

    price = rows["GOLD"]
    assert price["unit"] == "usd"
    assert "chg_1d_pct" in price
    assert not any(k.endswith("_bp") for k in price)


def test_anomalies_render_rates_in_basis_points():
    perf = [{"symbol": "US10Y", "z_1d_move": 3.0, "chg_1d_bp": -22.5},
            {"symbol": "GOLD", "z_1d_move": 2.5, "chg_1d_pct": 3.1}]
    notes = stats.anomalies(pd.DataFrame(), perf)
    joined = " ".join(notes)
    assert "-22.5bp" in joined and "+3.10%" in joined


def test_relation_vocabulary_is_sign_neutral():
    """Sign must live only in `direction`.

    Verbs that encode sign themselves (supports/pressures/suppresses) allowed
    contradictory edges like supports(negative), and let "dollar weakness lifts
    gold" become `US Dollar --supports(positive)--> Gold`, the inverse of the
    measured relationship.
    """
    from brain import extract

    for signed in ("supports", "pressures", "suppresses", "diverges_from"):
        assert signed not in extract.RELATIONS, f"{signed} re-encodes sign in the verb"
    assert "affects" in extract.RELATIONS


def test_intraday_moves_compare_to_prior_close():
    """Regression: a digest must not report the prior close as 'today'.

    Silver closed at 67.735 and traded 69.085 live (+1.99%); using the daily bar
    alone, the digest said silver *fell* 1.18% while it was actually up 2%.
    """
    idx = pd.date_range("2026-08-20", periods=6, freq="D")
    wide = pd.DataFrame({"SILVER": [66.0, 66.5, 67.0, 67.2, 67.4, 67.735]}, index=idx)

    import signals.stats as st

    real = st.latest_intraday
    st.latest_intraday = lambda: {
        "SILVER": {"ts": pd.Timestamp("2026-08-26T15:00Z"), "price": 69.085}
    }
    try:
        rows = {r["symbol"]: r for r in st.intraday_moves(wide)}
    finally:
        st.latest_intraday = real

    silver = rows["SILVER"]
    assert silver["live"] == 69.085
    assert silver["prior_close"] == 67.735
    assert silver["chg_pct"] == pytest.approx(1.99, abs=0.01)
    assert silver["chg_pct"] > 0, "direction must match the live move, not the stale bar"


def test_intraday_moves_use_basis_points_for_rates():
    idx = pd.date_range("2026-08-20", periods=3, freq="D")
    wide = pd.DataFrame({"US10Y": [4.60, 4.62, 4.656]}, index=idx)

    import signals.stats as st

    real = st.latest_intraday
    st.latest_intraday = lambda: {
        "US10Y": {"ts": pd.Timestamp("2026-08-26T04:50Z"), "price": 4.639}
    }
    try:
        row = {r["symbol"]: r for r in st.intraday_moves(wide)}["US10Y"]
    finally:
        st.latest_intraday = real

    assert row["chg_bp"] == pytest.approx(-1.7, abs=0.1)
    assert "chg_pct" not in row


def test_utc_filter_converts_from_session_timezone():
    """Regression: Postgres returns timestamptz in the session zone.

    Calling strftime directly and appending "UTC" printed Melbourne time under a
    UTC label — a 10-hour error on every timestamp in the dashboard.
    """
    from datetime import datetime, timedelta, timezone as tz
    from web.app import _utc

    melbourne = tz(timedelta(hours=10))
    dt = datetime(2026, 8, 26, 15, 57, tzinfo=melbourne)
    # The filter now emits a <time> element so the browser can re-render it in
    # the reader's own zone; the UTC instant and its fallback text must both be
    # correct, because that text is what shows without JavaScript.
    out = str(_utc(dt))
    assert 'datetime="2026-08-26T05:57:00+00:00"' in out
    assert ">26 Aug 2026, 05:57<" in out
    assert ">05:57<" in str(_utc(dt, "%H:%M"))


@pytest.mark.parametrize(
    "path", ["/", "/archive", "/ask", "/alerts", "/world", "/status", "/healthz",
             "/api/latest", "/charts", "/graph", "/add", "/api/graph", "/portfolio"]
)
def test_every_page_renders(path):
    """Smoke test across every route.

    Starlette 1.6 changed TemplateResponse to (request, name, context); the
    legacy (name, {"request": ...}) form reads the context dict as the template
    name and raises `unhashable type: 'dict'`. That bug shipped twice, because
    each route fails independently and eyeballing the pages missed /alerts
    entirely. One parametrised test covers every route for good.
    """
    from fastapi.testclient import TestClient

    from web.app import app

    with TestClient(app) as client:
        _sign_in(client)
        resp = client.get(path)
    assert resp.status_code == 200, f"{path} -> {resp.status_code}"


TEST_USER, TEST_PASS = "pytest-user", "pytest-password-1234"


@pytest.fixture(autouse=True, scope="module")
def _remove_test_user():
    """Delete the fixture login when the module finishes.

    These tests run against the real database, so without this a user with a
    password published in the repository is left able to sign in to the live
    dashboard.
    """
    yield
    import db

    db.execute("DELETE FROM users WHERE username = %s", (TEST_USER,))


def _sign_in(client):
    from web import auth

    auth.create_user(TEST_USER, TEST_PASS)
    r = client.post("/login", data={"username": TEST_USER, "password": TEST_PASS},
                    follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code}"


def test_pages_require_login():
    """Every page behind the gate; the login page itself in front of it."""
    from fastapi.testclient import TestClient

    from web.app import app

    with TestClient(app) as client:
        for path in ("/", "/archive", "/status", "/api/latest"):
            r = client.get(path, follow_redirects=False)
            assert r.status_code == 303, f"{path} was reachable without a session"
            assert r.headers["location"].startswith("/login")
        assert client.get("/login").status_code == 200
        assert client.get("/healthz").status_code == 200


def test_login_rejects_a_wrong_password():
    from fastapi.testclient import TestClient

    from web import auth
    from web.app import app

    auth.create_user(TEST_USER, TEST_PASS)
    with TestClient(app) as client:
        r = client.post("/login", data={"username": TEST_USER, "password": "not-it"},
                        follow_redirects=False)
        assert r.status_code == 401
        assert client.get("/", follow_redirects=False).status_code == 303


def test_login_does_not_redirect_off_site():
    """An open redirect would make the login page a usable phishing hop."""
    from fastapi.testclient import TestClient

    from web.app import app

    with TestClient(app) as client:
        auth_mod = __import__("web.auth", fromlist=["auth"])
        auth_mod.create_user(TEST_USER, TEST_PASS)
        r = client.post("/login",
                        data={"username": TEST_USER, "password": TEST_PASS,
                              "next": "https://evil.example.com/x"},
                        follow_redirects=False)
        assert r.headers["location"] == "/"


def test_nested_list_indent_is_normalised():
    """Regression: three-space nesting under an ordered item.

    Python-Markdown needs four spaces. With `sane_lists` the model's three-space
    bullets rendered as literal "- " text; without it they were promoted to
    top-level numbered items, silently restructuring the argument.
    """
    from web.app import render_markdown

    html = render_markdown("1. **Point**\n   - sub one\n   - sub two\n")
    assert "<ul>" in html
    assert html.count("<li>") == 3
    assert "- sub one" not in html


def test_list_normalisation_leaves_code_blocks_alone():
    from web.app import normalise_list_indent

    src = "```\n   - not a list\n```\n"
    assert normalise_list_indent(src) == src


def test_daily_frame_has_no_weekend_rows():
    """Regression: crypto's 7-day calendar leaked weekend rows into the shared
    daily frame, silently compressing every row-counted horizon by 5/7 — the
    "1y" column measured 8.3 months — and adding fake zero-return weekend rows
    to every non-crypto correlation."""
    from signals import stats

    wide = stats.load_daily(lookback_days=200)
    if wide.empty:
        import pytest as _pytest

        _pytest.skip("no price data in test database")
    assert (wide.index.dayofweek < 5).all()


def test_session_is_rolling_and_long_lived():
    """A fixed max_age expires 60 days after sign-in no matter how often you
    visit. The gate touches the session on each request so the cookie is
    re-issued, making the window roll forward with use.

    https:// base URL because production sets https_only=True, and Starlette
    will not issue a Secure cookie over plain http — on the server this test
    read as a login failure when it was actually testing the wrong scheme."""
    from fastapi.testclient import TestClient

    from web.app import app

    with TestClient(app, base_url="https://testserver") as client:
        _sign_in(client)
        resp = client.get("/status")
        assert resp.status_code == 200
        cookie = resp.headers.get("set-cookie", "")
        assert "mia_session" in cookie, "session cookie not refreshed on a normal request"
        assert "Max-Age=5184000" in cookie, f"expected a 60-day cookie, got: {cookie}"


@pytest.mark.parametrize("path", ["/login", "/", "/charts", "/graph"])
def test_no_empty_chart_bootstrap(path):
    """Regression: routes that bypass page() left `window.MIA_CHARTS = ;` in the
    HTML, a syntax error that killed every script after it on that page."""
    from fastapi.testclient import TestClient

    from web.app import app

    with TestClient(app) as client:
        if path != "/login":
            _sign_in(client)
        body = client.get(path).text
    assert "window.MIA_CHARTS = ;" not in body
    assert "window.MIA_CHARTS = " in body


def test_session_cookie_secure_flag_follows_env(monkeypatch):
    """The Secure flag has to be environment-driven: hardcoded on, local
    development over http:// cannot log in; hardcoded off, the production cookie
    crosses the internet in clear text."""
    import importlib, os

    import web.app

    monkeypatch.setenv("MIA_HTTPS_ONLY", "1")
    reloaded = importlib.reload(web.app)
    mw = [m for m in reloaded.app.user_middleware if "Session" in str(m.cls)]
    assert mw, "session middleware missing"
    assert mw[0].kwargs["https_only"] is True

    monkeypatch.delenv("MIA_HTTPS_ONLY", raising=False)
    importlib.reload(web.app)


def test_tick_syncs_instruments_before_inserting_prices():
    """Regression: prices.symbol references instruments, so on a fresh database
    every scheduled tick failed with a ForeignKeyViolation until someone ran a
    backfill. The scheduler starts before any data is loaded, so ordering cannot
    be assumed."""
    import inspect

    from ingest import prices

    src = inspect.getsource(prices.tick)
    assert "sync_instruments()" in src


def test_feed_age_window_covers_the_query_window():
    """The proxy queries ask Google News for `when:7d`. If the age filter is
    narrower than that, the fetch pulls a week and then discards most of it —
    the two settings have to move together."""
    import re

    import config
    from ingest import feeds

    sources = (config.CONFIG_DIR / "sources.yaml").read_text()
    windows = {int(d) for d in re.findall(r"when:(\d+)d", sources)}
    assert windows, "no when: windows found in sources.yaml"
    widest_query_hours = max(windows) * 24
    for tier in (2, 3, 4):
        assert feeds.age_window({"tier": tier}) >= widest_query_hours, (
            f"tier {tier} age window is narrower than the {max(windows)}-day query"
        )


def test_retention_never_touches_the_series_analysis_depends_on():
    """Daily prices, FRED series, briefs and the world model are deliberately
    exempt. The analogue engine reads twelve years of daily bars and macro
    series are meaningless without decades, so a retention rule that caught them
    would quietly hollow out the analysis rather than fail loudly."""
    import scheduler

    protected = {"fred_series", "analyses", "world_model", "users", "instruments"}
    for label, table, _ts, _window, predicate in scheduler.RETENTION:
        assert table not in protected, f"{label} would delete from {table}"
        if table == "prices":
            assert "15m" in predicate, "price retention must be limited to intraday bars"


def test_retention_rules_are_valid_sql():
    """Every rule runs as a real DELETE, so a typo in a predicate surfaces here
    rather than at 03:00 in a job nobody is watching."""
    import db
    import scheduler

    for label, table, ts, window, predicate in scheduler.RETENTION:
        rows = db.query(
            f"SELECT count(*) AS n FROM {table} "
            f"WHERE {ts} < now() - interval '{window}' AND ({predicate})"
        )
        assert rows and "n" in rows[0], f"{label} produced no result"


def test_every_document_falls_under_some_retention_rule():
    """Commentary must expire. If a tier or urgency combination matches no rule,
    those documents accumulate forever and quietly reintroduce the contradiction
    the windows exist to prevent."""
    import scheduler

    doc_rules = [(p, w) for _l, t, _ts, w, p in scheduler.RETENTION if t == "documents"]
    assert doc_rules, "no document retention at all"

    import db

    # Ask the database directly: is any document outside every rule's predicate?
    combined = " OR ".join(f"({p})" for p, _w in doc_rules)
    uncovered = db.one(f"SELECT count(*) AS n FROM documents WHERE NOT ({combined})")["n"]
    assert uncovered == 0, f"{uncovered} documents match no retention rule"


def test_share_link_shows_one_brief_and_nothing_else(monkeypatch):
    """A share recipient has no account. The token is the only credential, so
    the page must carry no route into the rest of the dashboard, and a revoked
    token must stop working immediately."""
    from fastapi.testclient import TestClient

    import config
    import db

    monkeypatch.setattr(config, "SHARING_ENABLED", True)
    from web import sharing
    from web.app import app

    row = db.one("SELECT id FROM analyses WHERE kind='digest' ORDER BY id DESC LIMIT 1")
    if not row:
        import pytest as _p

        _p.skip("no brief to share")

    token = sharing.create(row["id"], "pytest")
    with TestClient(app) as client:
        # Reachable with no session at all.
        r = client.get(f"/s/{token}")
        assert r.status_code == 200
        html = r.text
        assert 'href="/archive"' not in html, "share page links into the dashboard"
        assert 'href="/ask"' not in html
        assert "noindex" in html, "share pages must not be indexable"

        # Every other route still demands a session.
        assert client.get("/archive", follow_redirects=False).status_code == 303

        assert sharing.revoke(token) is True
        assert client.get(f"/s/{token}").status_code == 404

    db.execute("DELETE FROM brief_shares WHERE analysis_id=%s", (row["id"],))


def test_share_token_is_unguessable_and_validated():
    from web import sharing

    assert sharing.resolve("") is None
    assert sharing.resolve("short") is None
    assert sharing.resolve("z" * 32) is None


# ─────────────────────────────── private mode ───────────────────────────────
def test_disabled_account_cannot_sign_in_and_loses_open_session():
    """Disabling has to end a session already open on someone's phone, not
    just refuse the next sign-in."""
    from fastapi.testclient import TestClient

    from web import auth
    from web.app import app

    auth.create_user(TEST_USER, TEST_PASS)
    auth.set_disabled(TEST_USER, False)
    with TestClient(app) as client:
        _sign_in(client)
        assert client.get("/archive", follow_redirects=False).status_code == 200
        auth.set_disabled(TEST_USER, True)
        r = client.get("/archive", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login")
        r = client.post("/login", data={"username": TEST_USER, "password": TEST_PASS},
                        follow_redirects=False)
        assert r.status_code == 401, "a disabled account signed in"
    auth.set_disabled(TEST_USER, False)


def test_lockdown_refuses_an_owner_who_is_not_an_admin():
    """Locking everyone out, owner included, is the outcome lockdown must never
    produce — so it refuses before touching anyone."""
    from web import auth

    auth.create_user(TEST_USER, TEST_PASS)
    with pytest.raises(ValueError):
        auth.lockdown(TEST_USER)
    with pytest.raises(ValueError):
        auth.lockdown("no-such-user-xyz")


def test_share_links_are_dead_while_sharing_is_off(monkeypatch):
    from fastapi.testclient import TestClient

    import config
    import db
    from web import sharing
    from web.app import app

    row = db.one("SELECT id FROM analyses WHERE kind='digest' ORDER BY id DESC LIMIT 1")
    if not row:
        pytest.skip("no brief to share")
    token = sharing.create(row["id"], "pytest")
    monkeypatch.setattr(config, "SHARING_ENABLED", False)
    with TestClient(app) as client:
        assert client.get(f"/s/{token}").status_code == 404
        _sign_in(client)
        assert client.post(f"/digest/{row['id']}/share").status_code == 403
    db.execute("DELETE FROM brief_shares WHERE token=%s", (token,))


def test_portfolio_is_owner_only():
    """Holdings stay admin-only even if reader accounts come back."""
    from fastapi.testclient import TestClient

    from web import auth
    from web.app import app

    auth.create_user(TEST_USER, TEST_PASS)
    auth.set_admin(TEST_USER, False)
    with TestClient(app) as client:
        _sign_in(client)
        for path in ("/portfolio", "/brokers/kite/login"):
            r = client.get(path, follow_redirects=False)
            assert r.status_code == 303 and r.headers["location"] == "/", path
        auth.set_admin(TEST_USER, True)
        assert client.get("/portfolio").status_code == 200
    auth.set_admin(TEST_USER, False)


def test_kite_callback_rejects_a_foreign_state():
    """A callback link carrying someone else's request token must not connect
    their account in place of the owner's."""
    from fastapi.testclient import TestClient

    from web import auth
    from web.app import app

    auth.create_user(TEST_USER, TEST_PASS)
    auth.set_admin(TEST_USER, True)
    with TestClient(app) as client:
        _sign_in(client)
        r = client.get("/brokers/kite/callback?request_token=x&status=success&state=forged",
                       follow_redirects=False)
        assert r.headers["location"] == "/portfolio?notice=kite_state"
    auth.set_admin(TEST_USER, False)


# ───────────────────────────────── brokers ──────────────────────────────────
def test_kite_checksum_and_expiry(monkeypatch):
    import hashlib
    from datetime import datetime

    import config
    from brokers import kite

    monkeypatch.setattr(config, "KITE_API_KEY", "key")
    monkeypatch.setattr(config, "KITE_API_SECRET", "secret")
    assert kite.checksum("tok") == hashlib.sha256(b"keytoksecret").hexdigest()
    # Before 06:00 IST the token dies the same morning; after, the next one.
    early = datetime(2026, 10, 28, 5, 0, tzinfo=kite.IST)
    late = datetime(2026, 10, 28, 9, 0, tzinfo=kite.IST)
    assert kite._expiry(early) == datetime(2026, 10, 28, 6, 0, tzinfo=kite.IST)
    assert kite._expiry(late) == datetime(2026, 10, 29, 6, 0, tzinfo=kite.IST)


def test_kite_parsers_count_unsettled_shares_and_skip_closed_positions():
    from brokers import kite

    holdings = kite.parse_holdings([
        {"tradingsymbol": "INFY", "exchange": "NSE", "quantity": 10, "t1_quantity": 5,
         "average_price": 1500, "last_price": 1600, "pnl": 1500,
         "day_change_percentage": 1.2, "product": "CNC"},
        {"tradingsymbol": "SOLD", "quantity": 0, "t1_quantity": 0},
    ], "AB1234")
    assert [h.symbol for h in holdings] == ["INFY"]
    assert holdings[0].quantity == 15 and holdings[0].currency == "INR"

    positions = kite.parse_positions({"net": [
        {"tradingsymbol": "NIFTY26NOVFUT", "quantity": 50, "average_price": 24000,
         "last_price": 24100, "pnl": 5000, "product": "NRML"},
        {"tradingsymbol": "CLOSED", "quantity": 0, "pnl": -200},
    ]}, "AB1234")
    assert [p.symbol for p in positions] == ["NIFTY26NOVFUT"]

    funds = kite.parse_margins({"equity": {"net": 1000.5, "available": {"cash": 2000},
                                           "utilised": {"debits": 999.5}}})
    assert funds["available"] == 1000.5 and funds["cash"] == 2000


def test_kite_same_day_delivery_trades_are_not_double_counted():
    """From a live NRI account: holdings already drop shares sold today, and
    shares bought today appear only as a CNC position. A sale must not show
    as a short; a purchase must not vanish until settlement."""
    from brokers import kite

    lines = kite.parse_positions({"net": [
        {"tradingsymbol": "BAJFINANCE", "product": "CNC", "quantity": -20,
         "average_price": 904.1, "pnl": -64},
        {"tradingsymbol": "JPL", "product": "CNC", "quantity": 149,
         "average_price": 1357.66, "last_price": 1358.5, "pnl": 125.25},
        {"tradingsymbol": "RELIANCE", "product": "MIS", "quantity": 10},
    ]}, "AB1234")
    got = {ln.symbol: ln.kind for ln in lines}
    assert got == {"JPL": "holding", "RELIANCE": "position"}

    funds = kite.parse_margins({"equity": {"net": 471.78, "available": {
        "cash": -876.8, "opening_balance": -876.8, "live_balance": 471.78,
        "intraday_payin": 43500}, "utilised": {"debits": 42151.4}}})
    assert funds["cash"] == 471.78, "opening balance shown as cash"
    assert funds["payin_today"] == 43500


def test_webull_parsers_tolerate_wrapped_and_bare_lists():
    from brokers import webull

    rows = webull._rows({"holdings": [{"symbol": "GLD", "quantity": "3",
                                       "cost_price": "180.5", "last_price": "190",
                                       "unrealized_profit_loss": "28.5", "currency": "USD"}]})
    lines = webull.parse_positions(rows, "acct")
    assert lines[0].symbol == "GLD" and lines[0].quantity == 3 and lines[0].pnl == 28.5
    assert webull._rows([{"a": 1}]) == [{"a": 1}]
    assert lines[0].exchange is None or lines[0].exchange == "US"


def test_webull_balance_matches_a_live_au_account():
    """Shape taken from a real Webull AU response: no total field, and buying
    power listed per currency with the zero HKD entry first."""
    from brokers import webull

    bal = webull.parse_balance({
        "total_asset_currency": "AUD", "total_market_value": "15294.42",
        "total_cash_balance": "24.91", "total_unrealized_profit_loss": "-397.74",
        "account_currency_assets": [
            {"currency": "HKD", "market_value": "0.00", "cash_balance": "0.00", "buying_power": "0.00"},
            {"currency": "AUD", "market_value": "15173.68", "cash_balance": "24.68", "buying_power": "24.68"},
            {"currency": "USD", "market_value": "85.25", "cash_balance": "0.16", "buying_power": "0.16"},
            {"currency": "CNH", "market_value": "0.00", "cash_balance": "0.00", "buying_power": "0.00"},
        ]})
    assert bal["currency"] == "AUD"
    assert bal["available"] == 24.68, "buying power taken from the wrong currency"
    assert bal["total"] == pytest.approx(15319.33)
    assert set(bal["other"]) == {"USD"}
    lines = webull.parse_positions([{"symbol": "TLX", "currency": "AUD", "quantity": "44.0",
                                     "cost_price": "15.58", "last_price": "13.06"}], "a")
    assert lines[0].exchange == "ASX"


def test_broker_secrets_are_owner_only(tmp_path, monkeypatch):
    import os
    import stat

    import config
    from brokers import read_secret, write_secret

    monkeypatch.setattr(config, "SECRETS", tmp_path / "secrets")
    write_secret("t.json", {"access_token": "x"})
    mode = stat.S_IMODE(os.stat(config.SECRETS / "t.json").st_mode)
    assert mode == 0o600
    assert stat.S_IMODE(os.stat(config.SECRETS).st_mode) == 0o700
    assert read_secret("t.json") == {"access_token": "x"}


def test_brokers_expose_no_order_methods():
    """Read-only is a property of the code, not a promise: no adapter carries
    anything that could place, modify or cancel an order."""
    from brokers import kite, webull

    for mod in (kite, webull):
        names = [n.lower() for n in dir(mod)]
        assert not [n for n in names if any(w in n for w in ("order", "place", "cancel", "modify"))]


def test_nothing_in_alfred_can_place_an_order():
    """Alfred advises; people trade. This is the line that keeps it true.

    Scans every source file for the calls that would send an order to Kite or
    Webull — REST paths, SDK clients, method names. If this fails, someone has
    started teaching Alfred to trade: that is a product decision to take
    deliberately, not a diff to slip past review.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    forbidden = re.compile(
        r"""(/orders\b                  # Kite REST: POST/PUT/DELETE /orders/...
            |/gtt\b                     # Kite good-till-triggered orders
            |\.order(_v\d)?\.           # Webull SDK: client.order / order_v2 / order_v3
            |OrderOperation             # Webull SDK order classes
            |place_order|modify_order|cancel_order|place_gtt
            |trade/order                # Webull REST order paths
            )""", re.X)
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if rel.parts[0] in (".venv", "tests") or "site-packages" in rel.parts:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if forbidden.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, "order-placing code found:\n" + "\n".join(offenders)



# ────────────────────────────────── email ───────────────────────────────────
def test_email_is_silent_when_unconfigured(monkeypatch):
    import config
    from notify import email

    monkeypatch.setattr(config, "RESEND_API_KEY", "")
    assert email.send("s", "t") is None


def test_email_sends_one_message_to_the_owner_only(monkeypatch):
    """One recipient, a send-only bearer key, and an idempotency key so a
    retried job cannot deliver the same alert twice."""
    import httpx

    import config
    from notify import email

    monkeypatch.setattr(config, "RESEND_API_KEY", "re_test")
    monkeypatch.setattr(config, "EMAIL_TO", "owner@example.com")
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json={"id": "msg_1"})

    monkeypatch.setattr(email.httpx, "post", fake_post)
    assert email.send("Subject", "Body") == "msg_1"
    assert seen["json"]["to"] == ["owner@example.com"]
    assert seen["headers"]["Authorization"] == "Bearer re_test"
    first = seen["headers"]["Idempotency-Key"]
    email.send("Subject", "Body")
    assert seen["headers"]["Idempotency-Key"] == first, "same content must dedupe"


def test_email_failure_never_raises(monkeypatch):
    import httpx

    import config
    from notify import email

    monkeypatch.setattr(config, "RESEND_API_KEY", "re_test")
    monkeypatch.setattr(config, "EMAIL_TO", "owner@example.com")
    monkeypatch.setattr(email.httpx, "post", lambda *a, **k: httpx.Response(
        403, json={"message": "You can only send testing emails to your own address"}))
    assert email.send("s", "t") is None

    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(email.httpx, "post", boom)
    assert email.send("s", "t") is None


def test_webull_connect_runs_one_attempt_at_a_time(monkeypatch):
    """Every attempt texts a fresh code, and two polling at once trips Webull's
    rate limit — so a second click while one waits must do nothing."""
    import threading

    from brokers import webull

    release = threading.Event()
    calls = []

    def fake_connect(wait_seconds=300):
        calls.append(1)
        release.wait(5)
        raise webull.NotConnected("expired")

    monkeypatch.setattr(webull, "connect", fake_connect)
    monkeypatch.setattr(webull, "_attempt", {"state": None, "started": None, "error": None})
    assert webull.start_connect() is True
    assert webull.start_connect() is False, "a second attempt started"
    assert webull.attempt()["state"] == "waiting"
    release.set()
    for _ in range(50):
        if webull.attempt()["state"] != "waiting":
            break
        threading.Event().wait(0.05)
    assert webull.attempt()["state"] == "failed" and len(calls) == 1
    assert webull.start_connect() is True, "could not retry after a failure"
    release.set()
