"""Data feeds that are not brokers.

Kite, Webull and Coinbase are where orders go; they are not where Alfred
thinks. Everything Alfred reasons from — prices, funding, news tone, macro
series, positioning, filings, calendars — comes from the free sources listed
in conf/data.yaml, each in its own module here.

Every module exposes `sync(...) -> dict` and is run through `run()`, which
records the outcome in `data_sources` so the Status page can say which feed
is fresh and which has gone quiet. `run()` never raises: one broken feed must
not stop the others.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
import yaml

import config
import db

log = logging.getLogger("alfred.data")
_CONF = config.CONFIG_DIR / "data.yaml"
# SEC asks for a contact in the user-agent; the others just want one that is
# not a library default.
UA = f"Alfred personal research ({config.EMAIL_TO or 'no contact set'})"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def conf() -> dict:
    return yaml.safe_load(_CONF.read_text()) or {}


def get(url: str, params: dict | None = None, headers: dict | None = None,
        timeout: float = 30, tries: int = 3, as_json: bool = True,
        method: str = "GET", body: Any = None) -> Any:
    """One HTTP call with retries on 429/5xx. Raises after the last try.

    Errors name the URL without its query string: several feeds carry their
    key as a parameter, and an error message ends up in logs and on Status.
    """
    last: str | None = None
    for i in range(tries):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True,
                              headers={"user-agent": UA, **(headers or {})}) as c:
                r = c.request(method, url, params=params, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(2.0 * (i + 1))
                continue
            if r.status_code >= 400:
                raise HTTPStatus(r.status_code, url)
            return r.json() if as_json else r.text
        except httpx.HTTPError as exc:
            last = type(exc).__name__
            time.sleep(1.0 + i)
    raise RuntimeError(f"{url}: {last}")


class HTTPStatus(RuntimeError):
    """A 4xx answer, with the status a caller can switch on."""

    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"{url}: HTTP {status}")
        self.status = status


# ───────────────────────────── freshness record ─────────────────────────────
def record(source: str, ok: bool, rows: int = 0, seconds: float | None = None,
           error: str | None = None, detail: dict | None = None) -> None:
    now = datetime.now(timezone.utc)
    db.execute(
        """INSERT INTO data_sources (source, last_run, last_ok, ok, rows, seconds, error, detail)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (source) DO UPDATE SET
             last_run = EXCLUDED.last_run,
             last_ok  = CASE WHEN EXCLUDED.ok THEN EXCLUDED.last_ok ELSE data_sources.last_ok END,
             ok = EXCLUDED.ok, rows = EXCLUDED.rows, seconds = EXCLUDED.seconds,
             error = EXCLUDED.error, detail = EXCLUDED.detail""",
        (source, now, now if ok else None, ok, rows, seconds, error,
         json.dumps(detail or {}, default=str)))


def run(source: str, fn: Callable[..., dict | None], **kw) -> dict:
    """Run one feed, record how it went, never raise."""
    t0 = time.time()
    try:
        out = fn(**kw) or {}
    except Exception as exc:  # noqa: BLE001 - recorded, then reported
        log.warning("%s failed: %s", source, exc)
        record(source, False, seconds=time.time() - t0, error=f"{type(exc).__name__}: {exc}"[:500])
        return {"source": source, "ok": False, "error": str(exc)}
    rows = int(out.get("rows", 0))
    detail = {k: v for k, v in out.items() if k != "rows"}
    record(source, True, rows, time.time() - t0, detail=detail)
    log.info("%s: %d rows in %.0fs", source, rows, time.time() - t0)
    return {"source": source, "ok": True, "rows": rows, "seconds": round(time.time() - t0),
            **detail}


def freshness() -> list[dict]:
    return db.query("SELECT * FROM data_sources ORDER BY source")


# ───────────────────────────── shared helpers ───────────────────────────────
def upsert_series(series: str, points: list[tuple]) -> int:
    """(date, value) points into macro_series."""
    return db.executemany(
        """INSERT INTO macro_series (series, d, value) VALUES (%s, %s, %s)
           ON CONFLICT (series, d) DO UPDATE SET value = EXCLUDED.value""",
        [(series, d, float(v)) for d, v in points if v is not None])


def upsert_calendar(rows: list[dict]) -> int:
    return db.executemany(
        """INSERT INTO calendar (kind, key, event, at, estimate, actual, prior, importance, raw)
           VALUES (%(kind)s, %(key)s, %(event)s, %(at)s, %(estimate)s, %(actual)s,
                   %(prior)s, %(importance)s, %(raw)s)
           ON CONFLICT (kind, key, event, at) DO UPDATE SET
             estimate = EXCLUDED.estimate, actual = EXCLUDED.actual,
             prior = EXCLUDED.prior, importance = EXCLUDED.importance, raw = EXCLUDED.raw""",
        [{**r, "raw": json.dumps(r.get("raw") or {}, default=str),
          "estimate": r.get("estimate"), "actual": r.get("actual"), "prior": r.get("prior"),
          "importance": r.get("importance")} for r in rows])


def upsert_filings(rows: list[dict]) -> int:
    return db.executemany(
        """INSERT INTO filings (source, symbol, kind, title, url, filed_at, sensitive, raw)
           VALUES (%(source)s, %(symbol)s, %(kind)s, %(title)s, %(url)s, %(filed_at)s,
                   %(sensitive)s, %(raw)s)
           ON CONFLICT (source, url) DO NOTHING""",
        [{**r, "raw": json.dumps(r.get("raw") or {}, default=str),
          "sensitive": bool(r.get("sensitive"))} for r in rows])


def in_play() -> set[str]:
    """Yahoo tickers Alfred is watching closely: holdings, open tickets, and
    the latest run's passing candidates."""
    from advisor import bars, ticker_for

    syms = set(bars.holdings())
    for r in db.query("SELECT symbol, market FROM ideas WHERE status IN ('open', 'filled')"):
        syms.add(ticker_for(r["symbol"], r["market"]))
    run_ = db.one("""SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL
                     ORDER BY started_at DESC LIMIT 1""")
    if run_:
        syms |= {r["symbol"] for r in db.query(
            "SELECT symbol FROM sim_candidates WHERE run_id = %s AND passed_gate", (run_["id"],))}
    return syms


class NSE:
    """NSE's site JSON needs the cookies its home page sets, and a browser UA."""

    def __init__(self) -> None:
        self.client = httpx.Client(timeout=20, follow_redirects=True, headers={
            "user-agent": BROWSER_UA, "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9", "referer": "https://www.nseindia.com/"})
        self.client.get("https://www.nseindia.com/")

    def get(self, path: str, params: dict | None = None) -> Any:
        r = self.client.get(f"https://www.nseindia.com{path}", params=params)
        if r.status_code != 200:
            raise RuntimeError(f"NSE {path}: HTTP {r.status_code}")
        return r.json()

    def close(self) -> None:
        self.client.close()
