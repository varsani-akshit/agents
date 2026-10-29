"""Crypto: Coinbase bars (the venue's own prices), perp funding and open
interest from Hyperliquid and Binance (sentiment, never traded), and the
market's size from CoinGecko."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import config
import db
from data import conf, get, upsert_series

COINBASE = "https://api.exchange.coinbase.com"
HYPERLIQUID = "https://api.hyperliquid.xyz/info"
BINANCE_FUT = "https://fapi.binance.com"
COINGECKO = "https://api.coingecko.com/api/v3"
_MAX_CANDLES = 300        # Coinbase's limit per request


def pairs() -> list[str]:
    return list(conf().get("crypto", {}).get("pairs") or [])


def perps() -> list[str]:
    return list(conf().get("crypto", {}).get("perps") or [])


# ───────────────────────────────── Coinbase bars ─────────────────────────────
def candle_rows(product: str, raw: list[list]) -> list[tuple]:
    """Coinbase candles are [time, low, high, open, close, volume], newest first."""
    rows = []
    for t, lo, hi, op, cl, vol in raw:
        ts = datetime.fromtimestamp(int(t), tz=timezone.utc)
        rows.append((product, ts, float(op), float(hi), float(lo), float(cl), float(vol)))
    rows.sort(key=lambda r: r[1])
    return rows


def fetch_candles(product: str, granularity: int, start: datetime, end: datetime) -> list[tuple]:
    rows: list[tuple] = []
    step = timedelta(seconds=granularity * _MAX_CANDLES)
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        raw = get(f"{COINBASE}/products/{product}/candles",
                  {"granularity": granularity, "start": cur.isoformat(), "end": nxt.isoformat()})
        rows += candle_rows(product, raw or [])
        cur = nxt
        time.sleep(0.15)          # 10 requests a second allowed; stay well under
    return rows


def store_daily(rows: list[tuple]) -> int:
    # The daily bar for today is still forming; the simulator's session_closed()
    # does not know crypto, so leave today out of `bars` until tomorrow.
    today = datetime.now(timezone.utc).date()
    return db.executemany(
        """INSERT INTO bars (symbol, d, open, high, low, close, volume)
           VALUES (%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (symbol, d) DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high,
             low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume""",
        [(s, ts.date(), o, h, l, c, v) for s, ts, o, h, l, c, v in rows if ts.date() < today])


def store_hourly(rows: list[tuple]) -> int:
    return db.executemany(
        """INSERT INTO bars_1h (symbol, ts, open, high, low, close, volume)
           VALUES (%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (symbol, ts) DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high,
             low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume""", rows)


def bars(backfill: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    daily_from = now - timedelta(days=365 * 10 if backfill else 10)
    hourly_from = now - timedelta(days=730 if backfill else 3)
    n_d = n_h = 0
    for p in pairs():
        n_d += store_daily(fetch_candles(p, 86400, daily_from, now))
        n_h += store_hourly(fetch_candles(p, 3600, hourly_from, now))
    return {"rows": n_d + n_h, "daily": n_d, "hourly": n_h, "pairs": len(pairs())}


# ──────────────────────────── funding and open interest ──────────────────────
def hyperliquid_rows(meta: dict, ctxs: list[dict], ts: datetime) -> list[tuple]:
    rows = []
    for asset, ctx in zip(meta.get("universe", []), ctxs):
        try:
            rows.append(("hyperliquid", asset["name"], ts, float(ctx["funding"]),
                         float(ctx["openInterest"]), float(ctx["markPx"]),
                         float(ctx.get("dayNtlVlm") or 0), float(ctx.get("premium") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def store_derivatives(rows: list[tuple]) -> int:
    return db.executemany(
        """INSERT INTO derivatives (venue, symbol, ts, funding, open_interest, mark, volume_24h, premium)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", rows)


def derivatives() -> dict:
    ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    meta, ctxs = get(HYPERLIQUID, method="POST", body={"type": "metaAndAssetCtxs"})
    n = store_derivatives(hyperliquid_rows(meta, ctxs, ts))
    try:
        b = _binance(ts)
    except Exception as exc:  # noqa: BLE001 - Binance refuses US addresses (HTTP 451)
        return {"rows": n, "hyperliquid": len(ctxs), "binance": 0, "binance_error": str(exc)[:120]}
    return {"rows": n + b, "hyperliquid": len(ctxs), "binance": b}


def _binance(ts: datetime) -> int:
    """Binance perps as a second venue, where Binance answers at all: it
    blocks US addresses, and the production server is in the US. Hyperliquid
    alone covers every coin Alfred watches."""
    # Funding for every perp in one call, open interest one call each.
    prem = {p["symbol"]: p for p in get(f"{BINANCE_FUT}/fapi/v1/premiumIndex", tries=1)}
    rows = []
    for coin in perps():
        p = prem.get(f"{coin}USDT")
        if not p:
            continue
        try:
            oi = get(f"{BINANCE_FUT}/fapi/v1/openInterest", {"symbol": f"{coin}USDT"})
            rows.append(("binance", coin, ts, float(p["lastFundingRate"]),
                         float(oi["openInterest"]), float(p["markPrice"]), None,
                         float(p["markPrice"]) / float(p["indexPrice"]) - 1
                         if float(p.get("indexPrice") or 0) else None))
        except Exception:  # noqa: BLE001 - one coin missing is not a failure
            continue
    return store_derivatives(rows)


# ─────────────────────────────── market size ────────────────────────────────
def market() -> dict:
    headers = {"x-cg-demo-api-key": config.COINGECKO_API_KEY} if config.COINGECKO_API_KEY else {}
    g = get(f"{COINGECKO}/global", headers=headers)["data"]
    today = datetime.now(timezone.utc).date()
    n = upsert_series("crypto.total_mcap_usd", [(today, g["total_market_cap"]["usd"])])
    n += upsert_series("crypto.volume_24h_usd", [(today, g["total_volume"]["usd"])])
    n += upsert_series("crypto.btc_dominance_pct", [(today, g["market_cap_percentage"]["btc"])])
    n += upsert_series("crypto.mcap_change_24h_pct", [(today, g["market_cap_change_percentage_24h_usd"])])
    return {"rows": n}


def sync(backfill: bool = False) -> dict:
    out = bars(backfill)
    d = derivatives()
    out["derivatives"] = d["rows"]
    if d.get("binance_error"):
        out["binance_skipped"] = d["binance_error"]
    out["market"] = market()["rows"]
    out["rows"] += out["derivatives"] + out["market"]
    return out
