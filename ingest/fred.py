"""FRED macro series ingestion — yields, balance sheet, M2, debt, RRP, breakevens."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import httpx

import config
import db
from ingest.prices import load_fred_ids, load_universe

log = logging.getLogger("mia.fred")
_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, start: date | None = None) -> list[tuple]:
    if not config.FRED_API_KEY:
        log.warning("no FRED_API_KEY; skipping %s", series_id)
        return []
    params = {
        "series_id": series_id,
        "api_key": config.FRED_API_KEY,
        "file_type": "json",
        "sort_order": "asc",
    }
    if start:
        params["observation_start"] = start.isoformat()
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(_BASE, params=params)
            r.raise_for_status()
            obs = r.json().get("observations", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("FRED %s failed: %s", series_id, exc)
        return []
    rows = []
    for o in obs:
        v = o.get("value")
        if v in (None, "", "."):
            continue
        try:
            rows.append((series_id, date.fromisoformat(o["date"]), float(v)))
        except (ValueError, KeyError):
            continue
    return rows


def store(rows: list[tuple]) -> int:
    return db.executemany(
        """INSERT INTO fred_series (series_id, ts, value) VALUES (%s,%s,%s)
           ON CONFLICT (series_id, ts) DO UPDATE SET value=EXCLUDED.value""",
        rows,
    )


def _mirror_rate_instruments() -> int:
    """Some instruments (US2Y) have FRED as their price source — mirror into prices."""
    mirrored = 0
    for inst in load_universe():
        if inst.get("source") != "fred":
            continue
        sid = str(inst["source_id"])
        rows = db.query(
            "SELECT ts, value FROM fred_series WHERE series_id=%s ORDER BY ts", (sid,)
        )
        payload = [
            (
                inst["symbol"],
                datetime.combine(r["ts"], datetime.min.time(), tzinfo=timezone.utc),
                float(r["value"]),
                None,
                None,
                None,
                None,
                "1d",
                "fred",
            )
            for r in rows
            if r["value"] and r["value"] > 0
        ]
        mirrored += db.executemany(
            """INSERT INTO prices (symbol,ts,price,open,high,low,volume,grain,source)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (symbol,ts,grain) DO UPDATE SET price=EXCLUDED.price""",
            payload,
        )
    return mirrored


def sync(lookback_days: int = 900) -> dict:
    start = date.today() - timedelta(days=lookback_days)
    ids = load_fred_ids()
    total = 0
    fetched = {}
    for sid in ids:
        rows = fetch_series(sid, start)
        n = store(rows)
        fetched[sid] = len(rows)
        total += n
    mirrored = _mirror_rate_instruments()
    return {"series": len(ids), "rows": total, "mirrored_price_rows": mirrored,
            "per_series": fetched}


def latest(series_id: str) -> dict | None:
    return db.one(
        "SELECT ts, value FROM fred_series WHERE series_id=%s ORDER BY ts DESC LIMIT 1",
        (series_id,),
    )
