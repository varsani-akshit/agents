"""GDELT: the average tone of world news per day for each query in
conf/data.yaml, back to 2017. The news-tone history the regime is built from.

GDELT allows one request every five seconds, so a full backfill of twenty
queries over ten years takes about twenty minutes; the daily top-up takes
two. Both are fine for a background job and far too slow for a page.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import db
from data import conf, get

API = "https://api.gdeltproject.org/api/v2/doc/doc"
FIRST = date(2017, 1, 1)
# GDELT says one request every five seconds, but answers 429 at that pace
# once a run is a few minutes old; ten seconds with a long back-off holds.
PAUSE = 10.0
BACKOFF = 90.0


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


def fetch(query: str, start: date, end: date) -> list[tuple[date, float]]:
    import json

    params = {"query": query, "mode": "timelinetone", "format": "json",
              "startdatetime": start.strftime("%Y%m%d000000"),
              "enddatetime": end.strftime("%Y%m%d235959")}
    for attempt in range(4):
        try:
            text = get(API, params, as_json=False, timeout=60, tries=1)
        except RuntimeError as exc:
            if "429" not in str(exc) or attempt == 3:
                raise
            time.sleep(BACKOFF * (attempt + 1))
            continue
        time.sleep(PAUSE)
        if text.lstrip().startswith("{"):
            return points(json.loads(text))
        if "limit requests" in text and attempt < 3:   # the rate-limit page, HTTP 200
            time.sleep(BACKOFF * (attempt + 1))
            continue
        raise RuntimeError(f"GDELT: {text[:80]}")
    raise RuntimeError("GDELT: rate-limited")


def store(name: str, pts: list[tuple[date, float]]) -> int:
    return db.executemany(
        """INSERT INTO tone (query, d, tone) VALUES (%s,%s,%s)
           ON CONFLICT (query, d) DO UPDATE SET tone = EXCLUDED.tone""",
        [(name, d, t) for d, t in pts])


def last_day(name: str) -> date | None:
    r = db.one("SELECT max(d) AS d FROM tone WHERE query = %s", (name,))
    return r and r["d"]


def sync(backfill: bool = False, lookback_days: int = 14) -> dict:
    today = date.today()
    n = 0
    done = 0
    for name, q in queries().items():
        have = None if backfill else last_day(name)
        start = (have - timedelta(days=lookback_days)) if have else FIRST
        # Yearly windows: GDELT trims very long spans to a coarser timeline.
        while start < today:
            end = min(date(start.year, 12, 31), today)
            n += store(name, fetch(q, start, end))
            start = end + timedelta(days=1)
        done += 1
    return {"rows": n, "queries": done}
