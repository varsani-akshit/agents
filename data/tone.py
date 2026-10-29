"""GDELT: the average tone of world news per day for each query in
conf/data.yaml, back to 2017. The news-tone history the regime is built from.

GDELT's documented limit is one request every five seconds; in practice it
refuses far sooner once a run is a few minutes old, and refuses for longer
the more it has refused. So this feed is patient: twenty seconds between
requests, a long back-off on a refusal, and a window that still fails is
skipped — not the whole run. The backfill is resumable: a year already
stored is not fetched again, so each run continues where the last stopped.
A full backfill (twenty queries, ten years) takes an hour or two; the daily
top-up, about seven minutes.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta

import db
from data import conf, get

log = logging.getLogger("alfred.tone")

API = "https://api.gdeltproject.org/api/v2/doc/doc"
FIRST = date(2017, 1, 1)
PAUSE = 20.0
BACKOFF = (120.0, 240.0, 480.0)
FULL = 0.9                  # a year with this share of its days stored is done


class RateLimited(RuntimeError):
    pass


def queries() -> dict[str, str]:
    return dict(conf().get("tone") or {})


def points(payload: dict) -> list[tuple[date, float]]:
    """GDELT's timeline payload → (date, tone)."""
    out = []
    for tl in payload.get("timeline") or []:
        for p in tl.get("data") or []:
            try:
                d = datetime.strptime(p["date"][:8], "%Y%m%d").date()
                out.append((d, float(p["value"])))
            except (KeyError, ValueError):
                continue
    return out


def _once(query: str, start: date, end: date) -> list[tuple[date, float]]:
    try:
        text = get(API, {"query": query, "mode": "timelinetone", "format": "json",
                         "startdatetime": start.strftime("%Y%m%d000000"),
                         "enddatetime": end.strftime("%Y%m%d235959")},
                   as_json=False, timeout=60, tries=1)
    except RuntimeError as exc:
        if "429" in str(exc):
            raise RateLimited(str(exc)) from exc
        raise
    if text.lstrip().startswith("{"):
        return points(json.loads(text))
    if "limit requests" in text:        # the rate-limit page, served as HTTP 200
        raise RateLimited("GDELT asked to slow down")
    raise RuntimeError(f"GDELT: {text[:80]}")


def fetch(query: str, start: date, end: date, sleep=time.sleep) -> list[tuple[date, float]]:
    """One window, waiting out refusals. Raises RateLimited if GDELT still
    refuses after the last back-off."""
    for wait in (*BACKOFF, None):
        try:
            pts = _once(query, start, end)
            sleep(PAUSE)
            return pts
        except RateLimited:
            if wait is None:
                raise
            log.info("GDELT refused; waiting %.0fs", wait)
            sleep(wait)
    raise RateLimited("unreachable")


def store(name: str, pts: list[tuple[date, float]]) -> int:
    return db.executemany(
        """INSERT INTO tone (query, d, tone) VALUES (%s,%s,%s)
           ON CONFLICT (query, d) DO UPDATE SET tone = EXCLUDED.tone""",
        [(name, d, t) for d, t in pts])


def stored_days(name: str, start: date, end: date) -> int:
    r = db.one("SELECT count(*) AS n FROM tone WHERE query = %s AND d BETWEEN %s AND %s",
               (name, start, end))
    return int(r["n"]) if r else 0


def last_day(name: str) -> date | None:
    r = db.one("SELECT max(d) AS d FROM tone WHERE query = %s", (name,))
    return r and r["d"]


def windows(name: str, backfill: bool, today: date, lookback_days: int = 14) -> list[tuple[date, date]]:
    """The year windows still to fetch for one query."""
    have = None if backfill else last_day(name)
    start = (have - timedelta(days=lookback_days)) if have else FIRST
    out = []
    while start < today:
        end = min(date(start.year, 12, 31), today)
        span = (end - start).days + 1
        # A past year that is already (nearly) complete needs no request.
        if not (end < today - timedelta(days=lookback_days)
                and stored_days(name, start, end) >= FULL * span):
            out.append((start, end))
        start = end + timedelta(days=1)
    return out


def sync(backfill: bool = False, sleep=time.sleep) -> dict:
    today = date.today()
    n = done = refused = 0
    for name, q in queries().items():
        for start, end in windows(name, backfill, today):
            try:
                n += store(name, fetch(q, start, end, sleep))
                done += 1
            except RateLimited:
                refused += 1        # skipped; the next run picks it up
    if refused and not done:
        raise RuntimeError(f"GDELT refused all {refused} requests")
    return {"rows": n, "windows": done, "refused": refused}
