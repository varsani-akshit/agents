"""Hourly bars: fetch, store, and load as sessions in each exchange's own time.

Yahoo returns a multi-ticker download in one time zone (New York), which
splits an Indian or Australian session across two calendar dates. Bars are
stored in UTC and converted to the exchange's zone on load, so a session is
always one local trading day.
"""
from __future__ import annotations

import logging

import pandas as pd

import db
from advisor import market_of

log = logging.getLogger("alfred.hourly")

ZONE = {"NSE": "Asia/Kolkata", "BSE": "Asia/Kolkata", "ASX": "Australia/Sydney",
        "US": "America/New_York"}


def fetch(symbols: list[str], period: str = "730d", batch: int = 60) -> int:
    import yfinance as yf

    total = 0
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            data = yf.download(chunk, period=period, interval="1h", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("hourly batch %d failed: %s", i // batch, exc)
            continue
        rows = []
        for sym in chunk:
            try:
                frame = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
            if frame.empty:
                continue
            idx = frame.index.tz_convert("UTC")
            for ts, r in zip(idx, frame.itertuples()):
                rows.append((sym, ts.to_pydatetime(), float(r.Open), float(r.High), float(r.Low),
                             float(r.Close), float(r.Volume) if r.Volume == r.Volume else None))
        total += db.executemany(
            """INSERT INTO bars_1h (symbol, ts, open, high, low, close, volume)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (symbol, ts) DO UPDATE SET open = EXCLUDED.open,
                 high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close,
                 volume = EXCLUDED.volume""", rows)
        log.info("hourly %d/%d symbols", min(i + batch, len(symbols)), len(symbols))
    return total


def load(symbol: str) -> pd.DataFrame:
    """Hourly bars indexed by local exchange time, with a `session` date column."""
    rows = db.query("SELECT ts, open, high, low, close, volume FROM bars_1h "
                    "WHERE symbol = %s ORDER BY ts", (symbol,))
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "session"])
    df = pd.DataFrame(rows).set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ZONE[market_of(symbol)])
    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    df["session"] = df.index.date
    return df
