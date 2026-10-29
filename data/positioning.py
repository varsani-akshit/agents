"""CFTC Commitments of Traders, legacy futures-only report: what speculators
hold in the markets that matter to Akshit's book, weekly."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import db
from data import conf, get

API = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"


def markets() -> dict[str, str]:
    return dict(conf().get("positioning") or {})


def rows_from(items: list[dict], names: dict[str, str]) -> list[tuple]:
    rows = []
    for it in items:
        short = names.get(it.get("contract_market_name") or "")
        if not short:
            continue
        try:
            rows.append((short, datetime.fromisoformat(it["report_date_as_yyyy_mm_dd"]).date(),
                         float(it["noncomm_positions_long_all"]),
                         float(it["noncomm_positions_short_all"]),
                         float(it.get("open_interest_all") or 0) or None))
        except (KeyError, ValueError, TypeError):
            continue
    return rows


def store(rows: list[tuple]) -> int:
    return db.executemany(
        """INSERT INTO positioning (market, report_date, longs, shorts, open_interest)
           VALUES (%s,%s,%s,%s,%s)
           ON CONFLICT (market, report_date) DO UPDATE SET longs = EXCLUDED.longs,
             shorts = EXCLUDED.shorts, open_interest = EXCLUDED.open_interest""", rows)


def sync(backfill: bool = False) -> dict:
    names = markets()
    since = date(1990, 1, 1) if backfill else date.today() - timedelta(days=60)
    quoted = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    items = get(API, {
        "$where": f"report_date_as_yyyy_mm_dd > '{since.isoformat()}' "
                  f"AND contract_market_name in ({quoted})",
        "$select": "contract_market_name,report_date_as_yyyy_mm_dd,"
                   "noncomm_positions_long_all,noncomm_positions_short_all,open_interest_all",
        "$limit": 50000}, timeout=90)
    return {"rows": store(rows_from(items or [], names)), "markets": len(names)}
