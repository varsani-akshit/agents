"""The advisor: trade ideas, proven by simulation, delivered as tickets.

Alfred recommends; it never trades. An idea is a ticket a person places by
hand — "buy X with a limit at L, stop at S, here is why and here is the
evidence" — and it is only ever sent by a strategy whose simulation across a
decade and hundreds of names has cleared the gate in conf/advisor.yaml.

    bars        daily OHLC history for the universe
    strategies  fixed, textbook entry and exit rules — nothing fitted to data
    simulate    the rules replayed over history: trades, R-multiples, the gate
    scan        today's signals from strategies that passed → tickets
    ledger      what happened to each ticket, by the same rules as simulate
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import yaml

import config

# yfinance suffix → market. US tickers carry no suffix. Crypto pairs are
# Coinbase products (BTC-USD), priced in USD whatever the account's currency.
_SUFFIX = {".NS": "NSE", ".BO": "BSE", ".AX": "ASX", "-USD": "CRYPTO"}
CURRENCY = {"NSE": "INR", "BSE": "INR", "ASX": "AUD", "US": "USD", "CRYPTO": "USD"}
# Where you would place the order. Indian listings go through Kite; ASX and
# US listings through Webull AU; crypto through Coinbase.
BROKER = {"NSE": "kite", "BSE": "kite", "ASX": "webull", "US": "webull", "CRYPTO": "coinbase"}
# Evidence is pooled per region, not per exchange: a BSE listing is an Indian
# company like any NSE one, and six BSE names alone cannot prove anything.
REGIONS = {"IN": ("NSE", "BSE"), "ASX": ("ASX",), "US": ("US",), "CRYPTO": ("CRYPTO",)}


def region_of(market: str) -> str:
    return next(r for r, ms in REGIONS.items() if market in ms)


@lru_cache(maxsize=1)
def settings() -> dict:
    return yaml.safe_load((config.CONFIG_DIR / "advisor.yaml").read_text())


def market_of(ticker: str) -> str:
    for suffix, market in _SUFFIX.items():
        if ticker.endswith(suffix):
            return market
    return "US"


def bare(ticker: str) -> str:
    """The symbol as the broker shows it: RELIANCE.NS → RELIANCE."""
    for suffix in _SUFFIX:
        if ticker.endswith(suffix):
            return ticker[: -len(suffix)]
    return ticker


def ticker_for(symbol: str, exchange: str | None) -> str:
    """A broker holding as a yfinance ticker."""
    suffix = {"NSE": ".NS", "BSE": ".BO", "ASX": ".AX",
              "COINBASE": "-USD"}.get((exchange or "").upper(), "")
    return symbol + suffix


# Each market's close, local time. A bar dated today is only final once the
# close is behind us (with a margin for the data to settle); before that it is
# a session in progress, and a signal read off it can vanish by 4pm.
_CLOSE = {
    "NSE": ("Asia/Kolkata", time(15, 30)),
    "BSE": ("Asia/Kolkata", time(15, 30)),
    "ASX": ("Australia/Sydney", time(16, 10)),
    "US": ("America/New_York", time(16, 0)),
}


def session_closed(market: str, d, now: datetime | None = None) -> bool:
    """Whether the session dated `d` in `market` has finished."""
    if market == "CRYPTO":
        # Crypto never closes; Coinbase's daily bar is the UTC day.
        utc = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo("UTC"))
        return d < utc.date()
    tz, close = _CLOSE[market]
    local = (now or datetime.now(ZoneInfo(tz))).astimezone(ZoneInfo(tz))
    if d < local.date():
        return True
    return d == local.date() and local.time() >= (
        datetime.combine(local.date(), close) + timedelta(minutes=30)).time()


# Yahoo's exchange codes for US listings, as people say them.
_US_VENUES = {"NYQ": "NYSE", "NYS": "NYSE", "NMS": "Nasdaq", "NGM": "Nasdaq", "NCM": "Nasdaq",
              "NAS": "Nasdaq", "PCX": "NYSE Arca", "ARCA": "NYSE Arca", "ASE": "NYSE American",
              "BTS": "Cboe", "BATS": "Cboe"}


def venue(ticker: str) -> str:
    """Where the order goes: NSE, BSE, ASX, Coinbase — or, for a US name, the
    exchange it actually lists on (looked up once and cached)."""
    market = market_of(ticker)
    if market != "US":
        return {"CRYPTO": "Coinbase"}.get(market, market)
    import db

    row = db.one("SELECT exchange FROM name_cache WHERE symbol = %s", (ticker,))
    if row and row["exchange"]:
        return row["exchange"]
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        code, name = info.get("exchange"), info.get("longName") or info.get("shortName")
    except Exception:  # noqa: BLE001 - an unknown venue is still a US listing
        return "NYSE/Nasdaq"
    where = _US_VENUES.get(code or "", "NYSE/Nasdaq")
    # Cached even when Yahoo knows nothing, so an unknown name is looked up once.
    db.execute("""INSERT INTO name_cache (symbol, name, exchange) VALUES (%s, %s, %s)
                  ON CONFLICT (symbol) DO UPDATE SET exchange = EXCLUDED.exchange""",
               (ticker, name or ticker, where))
    return where
