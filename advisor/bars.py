"""Daily OHLC history for everything the advisor looks at.

Adjusted for splits and dividends (yfinance auto_adjust), so a decade-old
breakout is measured on the same footing as today's price. Adjustment rewrites
the past, never the latest bar, so today's close is a price you can actually
put on a ticket.
"""
from __future__ import annotations

import logging

import pandas as pd

import db
from advisor import settings, ticker_for

log = logging.getLogger("alfred.bars")

FX_PAIRS = ["AUDUSD=X", "AUDINR=X"]
STABLECOINS = {"USDC", "USDT", "DAI", "PYUSD"}


def universe() -> list[str]:
    """Index members, your holdings, and the watchlist — de-duplicated."""
    syms = {r["symbol"] for r in db.query("SELECT symbol FROM securities")}
    syms |= set(holdings())
    # Both listings of every Indian holding, so _best_listing has a choice.
    for r in db.query("""SELECT DISTINCT l.symbol FROM portfolio_lines l
                         WHERE l.exchange IN ('NSE','BSE')"""):
        syms |= {r["symbol"] + ".NS", r["symbol"] + ".BO"}
    syms |= set(settings().get("watchlist") or [])
    syms |= set(_crypto_pairs())
    return sorted(syms)


def _crypto_pairs() -> list[str]:
    import yaml

    from config import CONFIG_DIR

    conf = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text()) or {}
    return list((conf.get("crypto") or {}).get("pairs") or [])


def holdings() -> dict[str, dict]:
    """What you hold right now, keyed by yfinance ticker, from the newest
    good snapshot of each broker."""
    rows = db.query(
        """SELECT l.symbol, l.exchange, l.quantity, l.avg_price, l.last_price,
                  l.currency, s.broker
           FROM portfolio_lines l JOIN portfolio_snapshots s ON s.id = l.snapshot_id
           WHERE s.id IN (SELECT max(id) FROM portfolio_snapshots WHERE ok GROUP BY broker)
             AND l.kind = 'holding' AND l.quantity > 0""")
    # Stablecoins are cash parked in crypto, not a position with a price to watch.
    return {_best_listing(r["symbol"], r["exchange"]): r for r in rows
            if not (r["broker"] == "coinbase" and r["symbol"] in STABLECOINS)}


def _best_listing(symbol: str, exchange: str | None) -> str:
    """The yfinance ticker with the most history for an Indian holding.

    Kite reports where you bought (often BSE), but Yahoo keeps no history
    for many BSE tickers — ASIANPAINT.BO and KOTAKBANK.BO came back with a
    single day while the NSE listings of the same companies have twelve
    years. Same company, same price within a paisa: use the deeper one.
    """
    native = ticker_for(symbol, exchange)
    if (exchange or "").upper() not in ("NSE", "BSE"):
        return native
    counts = {r["symbol"]: r["n"] for r in db.query(
        "SELECT symbol, count(*) n FROM bars WHERE symbol = ANY(%s) GROUP BY symbol",
        ([symbol + ".NS", symbol + ".BO"],))}
    return max([symbol + ".NS", symbol + ".BO"], key=lambda t: (counts.get(t, 0), t == native))


def fetch(symbols: list[str], period: str = "1mo", batch: int = 80) -> int:
    """Download and upsert bars. `period` is a yfinance period: "12y" to
    backfill, "1mo" for the daily top-up (overlap repairs revised bars)."""
    import yfinance as yf

    total = 0
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            data = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001 - one bad batch must not stop the rest
            log.warning("bars batch %d failed: %s", i // batch, exc)
            continue
        rows = []
        for sym in chunk:
            try:
                frame = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
            for ts, r in frame.iterrows():
                rows.append((sym, ts.date(), float(r["Open"]), float(r["High"]),
                             float(r["Low"]), float(r["Close"]),
                             float(r["Volume"]) if r["Volume"] == r["Volume"] else None))
        total += db.executemany(
            """INSERT INTO bars (symbol, d, open, high, low, close, volume)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (symbol, d) DO UPDATE SET open = EXCLUDED.open,
                 high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close,
                 volume = EXCLUDED.volume""", rows)
        log.info("bars %d/%d symbols", min(i + batch, len(symbols)), len(symbols))
    return total


def refresh(markets: list[str] | None = None, period: str = "1mo") -> int:
    from advisor import market_of

    # Crypto bars come from Coinbase (data/crypto.py), never from Yahoo.
    n = 0
    if markets and "CRYPTO" in markets:
        from data import crypto

        n += crypto.bars()["rows"]
    syms = [s for s in universe()
            if market_of(s) != "CRYPTO" and (not markets or market_of(s) in markets)]
    if syms or not markets:
        n += fetch(syms + FX_PAIRS, period=period)
    return n


def load(symbol: str, years: int | None = None, final_only: bool = True) -> pd.DataFrame:
    """One symbol's bars as a date-indexed frame: open, high, low, close, volume.

    `final_only` drops today's bar while its session is still trading."""
    sql = "SELECT d, open, high, low, close, volume FROM bars WHERE symbol = %s"
    params: list = [symbol]
    if years:
        sql += " AND d >= current_date - %s * interval '1 year'"
        params.append(years)
    rows = db.query(sql + " ORDER BY d", params)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    from advisor import market_of, session_closed

    if final_only and not session_closed(market_of(symbol), rows[-1]["d"]):
        rows = rows[:-1]
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).set_index("d")
    df.index = pd.to_datetime(df.index)
    return drop_bad_prints(df.astype(float))


def drop_bad_prints(df: pd.DataFrame) -> pd.DataFrame:
    """Remove isolated prices that are off by a factor and snap back.

    Yahoo recorded NIFTYBEES at 13 instead of 130 on 19–20 Dec 2019 — a split
    adjustment applied to two days only — which made the index look like it
    fell 90% and recovered. A centred 7-day median sees such a blip as an
    outlier (most of its window is at the true level), while a genuine crash
    that stays down moves the median with it and is kept.
    """
    if len(df) < 7:
        return df
    med = df["close"].rolling(7, center=True, min_periods=3).median()
    bad = (df["close"] / med - 1).abs() > 0.5
    return df[~bad]


def fx(pair: str) -> float | None:
    row = db.one("SELECT close FROM bars WHERE symbol = %s ORDER BY d DESC LIMIT 1", (pair,))
    return row and float(row["close"])
