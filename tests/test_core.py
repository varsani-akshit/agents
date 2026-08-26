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
    assert _utc(dt) == "26 Aug 2026, 05:57"
    assert _utc(dt, "%H:%M") == "05:57"


@pytest.mark.parametrize(
    "path", ["/", "/archive", "/ask", "/alerts", "/world", "/status", "/healthz", "/api/latest"]
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
