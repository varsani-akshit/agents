"""Price ingestion: yfinance (metals, FX, rates, equities, energy) + CoinGecko (crypto).

No LLM involved. Deterministic, idempotent upserts.
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

import config
import db

warnings.filterwarnings("ignore")
log = logging.getLogger("mia.prices")

_CONF = config.CONFIG_DIR / "instruments.yaml"


def _conf() -> dict:
    return yaml.safe_load(_CONF.read_text()) or {}


def load_universe() -> list[dict]:
    return [r for r in _conf().get("instruments", []) if "symbol" in r]


def load_fred_ids() -> dict[str, str]:
    return _conf().get("fred_series", {})


def sync_instruments() -> int:
    rows = [
        (
            i["symbol"],
            i["name"],
            i["asset_class"],
            i["source"],
            str(i["source_id"]),
            bool(i.get("is_rate", False)),
        )
        for i in load_universe()
    ]
    return db.executemany(
        """INSERT INTO instruments (symbol,name,asset_class,source,source_id,is_rate)
           VALUES (%s,%s,%s,%s,%s,%s)
           ON CONFLICT (symbol) DO UPDATE SET
             name=EXCLUDED.name, asset_class=EXCLUDED.asset_class,
             source=EXCLUDED.source, source_id=EXCLUDED.source_id,
             is_rate=EXCLUDED.is_rate""",
        rows,
    )


def _sanitize(value: float, is_rate: bool) -> float | None:
    """Yahoo occasionally serves rate indices scaled by 10. Normalise to percent."""
    if value is None or value != value:  # NaN
        return None
    v = float(value)
    if is_rate:
        while v > 25:
            v /= 10
        if v <= 0:
            return None
    if v <= 0:
        return None
    return v


def _store(rows: list[tuple]) -> int:
    return db.executemany(
        """INSERT INTO prices (symbol,ts,price,open,high,low,volume,grain,source)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (symbol,ts,grain) DO UPDATE SET
             price=EXCLUDED.price, open=EXCLUDED.open, high=EXCLUDED.high,
             low=EXCLUDED.low, volume=EXCLUDED.volume""",
        rows,
    )


# ─────────────────────────────────── yfinance ───────────────────────────────
def fetch_yfinance(instruments: list[dict], period: str, interval: str) -> int:
    import yfinance as yf

    grain = "1d" if interval == "1d" else "15m"
    total = 0
    for inst in instruments:
        sym, sid = inst["symbol"], str(inst["source_id"])
        is_rate = bool(inst.get("is_rate", False))
        try:
            hist = yf.Ticker(sid).history(period=period, interval=interval, auto_adjust=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("yfinance %s (%s) failed: %s", sym, sid, exc)
            continue
        if hist is None or hist.empty:
            log.warning("yfinance %s (%s) returned no rows", sym, sid)
            continue

        rows = []
        for ts, r in hist.iterrows():
            price = _sanitize(r.get("Close"), is_rate)
            if price is None:
                continue
            pyts = ts.to_pydatetime()
            if pyts.tzinfo is None:
                pyts = pyts.replace(tzinfo=timezone.utc)
            rows.append(
                (
                    sym,
                    pyts,
                    price,
                    _sanitize(r.get("Open"), is_rate),
                    _sanitize(r.get("High"), is_rate),
                    _sanitize(r.get("Low"), is_rate),
                    float(r.get("Volume") or 0) or None,
                    grain,
                    "yfinance",
                )
            )
        total += _store(rows)
        log.info("yfinance %s: %d rows (%s)", sym, len(rows), grain)
    return total


# ─────────────────────────────────── CoinGecko ──────────────────────────────
_CG = "https://api.coingecko.com/api/v3"


def fetch_coingecko(instruments: list[dict], days: int = 365) -> int:
    total = 0
    with httpx.Client(timeout=30, headers={"accept": "application/json"}) as client:
        for inst in instruments:
            sym, sid = inst["symbol"], str(inst["source_id"])
            try:
                r = client.get(
                    f"{_CG}/coins/{sid}/market_chart",
                    params={"vs_currency": "usd", "days": days, "interval": "daily"},
                )
                r.raise_for_status()
                data = r.json()
            except Exception as exc:  # noqa: BLE001
                log.warning("coingecko %s failed: %s", sym, exc)
                continue
            rows = []
            for ms, price in data.get("prices", []):
                ts = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
                rows.append((sym, ts, float(price), None, None, None, None, "1d", "coingecko"))
            total += _store(rows)
            log.info("coingecko %s: %d rows", sym, len(rows))
    return total


def fetch_crypto_history(instruments: list[dict], period: str = "12y") -> int:
    """Deep daily crypto history from Yahoo, since CoinGecko's free tier stops at 365 days.

    That 365-day wall was silently capping everything downstream: any analysis
    joining crypto to another series — correlations, the analogue engine — is
    truncated to the shortest input, so one free-tier limit was deciding how far
    back the whole system could look. CoinGecko still serves live 15-minute
    spot, where it is the better source; this only backfills daily bars.
    """
    tickers = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD"}
    proxies = [
        {"symbol": i["symbol"], "source_id": tickers[i["symbol"]], "is_rate": False}
        for i in instruments
        if i["symbol"] in tickers
    ]
    return fetch_yfinance(proxies, period=period, interval="1d") if proxies else 0


def fetch_coingecko_spot(instruments: list[dict]) -> int:
    """Current price for the 15-minute tick — one call for all coins."""
    ids = ",".join(str(i["source_id"]) for i in instruments)
    if not ids:
        return 0
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(
                f"{_CG}/simple/price", params={"ids": ids, "vs_currencies": "usd"}
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("coingecko spot failed: %s", exc)
        return 0
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = []
    for inst in instruments:
        px = data.get(str(inst["source_id"]), {}).get("usd")
        if px:
            rows.append(
                (inst["symbol"], ts, float(px), None, None, None, None, "15m", "coingecko")
            )
    return _store(rows)


# ─────────────────────────────────── entrypoints ────────────────────────────
def backfill(days_daily: str = "2y") -> dict:
    """Historical load. Run once at setup; safe to re-run."""
    sync_instruments()
    uni = load_universe()
    yf_inst = [i for i in uni if i["source"] == "yfinance"]
    cg_inst = [i for i in uni if i["source"] == "coingecko"]
    out = {
        "daily_rows": fetch_yfinance(yf_inst, period=days_daily, interval="1d"),
        "crypto_rows": fetch_coingecko(cg_inst),
        "crypto_history_rows": fetch_crypto_history(cg_inst, period=days_daily),
        "intraday_rows": fetch_yfinance(yf_inst, period="5d", interval="15m"),
    }
    return out


def tick() -> dict:
    """15-minute incremental refresh."""
    uni = load_universe()
    yf_inst = [i for i in uni if i["source"] == "yfinance"]
    cg_inst = [i for i in uni if i["source"] == "coingecko"]
    return {
        "intraday_rows": fetch_yfinance(yf_inst, period="1d", interval="15m"),
        "crypto_rows": fetch_coingecko_spot(cg_inst),
    }


def daily_refresh() -> dict:
    uni = load_universe()
    yf_inst = [i for i in uni if i["source"] == "yfinance"]
    cg_inst = [i for i in uni if i["source"] == "coingecko"]
    return {
        "daily_rows": fetch_yfinance(yf_inst, period="1mo", interval="1d"),
        "crypto_rows": fetch_coingecko(cg_inst, days=30),
    }
