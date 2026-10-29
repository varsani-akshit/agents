"""Twelve Data as a second opinion on closes. The free plan covers US names
(800 requests a day, 8 a minute); the names in play on other exchanges are
skipped. A close that disagrees with Yahoo by more than a percent is flagged
in price_checks — that is how Alfred learns Yahoo printed a bad bar."""
from __future__ import annotations

import time
from datetime import date, datetime

import config
import db
from advisor import bare, market_of
from data import HTTPStatus, conf, get, in_play

API = "https://api.twelvedata.com"
FLAG_PCT = 1.0


def twelve_symbol(symbol: str) -> str:
    """Yahoo writes share classes BF-B; Twelve Data writes BF.B."""
    return bare(symbol).replace("-", ".")


def closes(payload: dict) -> list[tuple[date, float]]:
    out = []
    for v in payload.get("values") or []:
        try:
            out.append((datetime.fromisoformat(v["datetime"][:10]).date(), float(v["close"])))
        except (KeyError, ValueError):
            continue
    return out


def compare(symbol: str, twelve: list[tuple[date, float]]) -> list[tuple]:
    rows = []
    for d, tclose in twelve:
        y = db.one("SELECT close FROM bars WHERE symbol = %s AND d = %s", (symbol, d))
        yclose = y and float(y["close"])
        diff = (tclose / yclose - 1) * 100 if yclose else None
        rows.append((symbol, d, yclose, tclose, diff))
    return rows


def store(rows: list[tuple]) -> int:
    return db.executemany(
        """INSERT INTO price_checks (symbol, d, yahoo, twelve, diff_pct) VALUES (%s,%s,%s,%s,%s)
           ON CONFLICT (symbol, d) DO UPDATE SET yahoo = EXCLUDED.yahoo,
             twelve = EXCLUDED.twelve, diff_pct = EXCLUDED.diff_pct""", rows)


def sync() -> dict:
    if not config.TWELVE_DATA_API_KEY:
        raise RuntimeError("no TWELVE_DATA_API_KEY")
    cap = int(conf().get("twelve", {}).get("max_symbols_per_run", 40))
    syms = sorted(s for s in in_play() if market_of(s) == "US")[:cap]
    n = flagged = skipped = 0
    for i, sym in enumerate(syms):
        try:
            got = get(f"{API}/time_series", {"symbol": twelve_symbol(sym), "interval": "1day",
                                             "outputsize": 5, "apikey": config.TWELVE_DATA_API_KEY})
        except HTTPStatus as exc:
            if exc.status == 404:         # not a symbol Twelve Data knows
                skipped += 1
                continue
            raise
        if got.get("status") == "error":  # not on the free plan, or a bad symbol
            skipped += 1
            continue
        rows = compare(sym, closes(got))
        n += store(rows)
        flagged += sum(1 for r in rows if r[4] is not None and abs(r[4]) > FLAG_PCT)
        if (i + 1) % 8 == 0:
            time.sleep(61)        # 8 a minute
    return {"rows": n, "symbols": len(syms), "flagged": flagged, "skipped": skipped}


def disagreements(days: int = 10) -> list[dict]:
    return db.query(
        """SELECT symbol, d, yahoo, twelve, diff_pct FROM price_checks
           WHERE d > current_date - %s AND abs(diff_pct) > %s ORDER BY abs(diff_pct) DESC""",
        (days, FLAG_PCT))
