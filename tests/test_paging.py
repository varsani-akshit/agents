"""Lists that can grow must page, and must never quietly truncate."""
from __future__ import annotations

import secrets

import pytest

from web import paging


def test_a_page_reports_its_own_window():
    p = paging.Page(rows=list(range(50)), total=524, page=3, per_page=50)
    assert p.pages == 11
    assert (p.first, p.last) == (101, 150)
    assert p.has_prev and p.has_next


def test_the_window_elides_rather_than_listing_forty_numbers():
    p = paging.Page(rows=[], total=2000, page=20, per_page=50)
    w = p.window
    assert 0 in w, "no elision marker in a 40-page list"
    assert w[0] == 1 and w[-1] == p.pages, "the ends must always be reachable"
    assert 20 in w and 19 in w and 21 in w
    assert len(w) < 12, f"offered too many page numbers: {w}"


def test_a_short_list_offers_every_page_and_no_gap():
    p = paging.Page(rows=[], total=120, page=1, per_page=50)
    assert p.window == [1, 2, 3]


def test_junk_in_the_query_string_shows_the_list_rather_than_an_error():
    """A mistyped or stale URL is not an error condition — the reader should
    see page one, not a 422."""
    for junk in ("abc", "", None, "-5", "0", "1e9999"):
        page, per, offset = paging.params(junk)
        assert page >= 1 and offset >= 0, junk


def test_per_page_is_capped():
    """The query string is not a place to ask for forty thousand rows."""
    _, per, _ = paging.params(1, 10_000)
    assert per == paging.MAX_PER_PAGE


def test_a_page_past_the_end_lands_on_the_last_one():
    """Rather than an empty table under a count reading '49,901-805 of 805'."""
    import db

    db.execute("DROP TABLE IF EXISTS _page_test")
    db.execute("CREATE TABLE _page_test (i int)")
    try:
        db.executemany("INSERT INTO _page_test (i) VALUES (%s)", [(i,) for i in range(120)])
        p = paging.slice_query("SELECT i FROM _page_test ORDER BY i",
                               "SELECT count(*) AS n FROM _page_test",
                               page=999, per_page=50)
        assert p.page == 3 and p.total == 120
        assert (p.first, p.last) == (101, 120)
        assert len(p.rows) == 20
    finally:
        db.execute("DROP TABLE IF EXISTS _page_test")


def test_every_candidate_is_reachable_rather_than_capped_at_500():
    """A run holds over a thousand candidates; the old page fetched 500 and
    said nothing about the rest, so a short list and a trimmed one looked
    identical."""
    import db
    from web import dashboard

    run = db.one("""SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL
                    ORDER BY started_at DESC LIMIT 1""")
    if not run:
        pytest.skip("no completed simulation run stored")
    big = max(
        (k for k in ("passed", "missed", "near_passed", "near_other")),
        key=lambda k: dashboard.sim_run(run["id"], show=k)["candidates"][k]["total"])
    first = dashboard.sim_run(run["id"], show=big, page=1)["candidates"][big]
    if first["total"] <= first["page"].per_page:
        pytest.skip("this run's largest group fits on one page")
    last = dashboard.sim_run(run["id"], show=big,
                             page=first["page"].pages)["candidates"][big]
    assert last["page"].last == first["total"], "the final row is not reachable"
    assert {r["symbol"] for r in last["rows"]} != {r["symbol"] for r in first["rows"]}


def _owner_client():
    from starlette.testclient import TestClient
    from web import auth
    import web.app as W

    u, p = f"paging_{secrets.token_hex(4)}", secrets.token_urlsafe(16)
    auth.create_user(u, p)
    auth.set_admin(u, True)
    c = TestClient(W.app, base_url="https://testserver")
    assert c.post("/login", data={"username": u, "password": p},
                  follow_redirects=False).status_code == 303
    return c, u


def test_the_paged_pages_survive_nonsense_in_the_url():
    import db

    c, u = _owner_client()
    try:
        for path in ("/alerts?page=abc", "/ideas?page=", "/simulations?page=-1",
                     "/alerts?page=99999"):
            r = c.get(path)
            assert r.status_code == 200, f"{path} -> {r.status_code}"
            assert "Traceback" not in r.text
    finally:
        db.execute("DELETE FROM users WHERE username=%s", (u,))
