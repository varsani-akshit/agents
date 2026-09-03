"""Public equities for the three markets the reader actually trades: US,
Australia and India.

Universe is index membership — S&P 500, S&P/ASX 200, NIFTY 50, about 750
names. That is every listing liquid enough to carry a macro read, and small
enough that each one can be given real depth: daily prices, refreshed
fundamentals, and its own accumulating news file.

Deliberately separate from `ingest/prices.py`. That module feeds the macro
statistics — correlations, the analogue engine, the regime score — where 750
single names would be noise. These are read by the market beats and by the
chatbot's screens instead.
"""
from __future__ import annotations

import io
import logging
import warnings
from datetime import datetime, timezone

import httpx
import pandas as pd

import db

warnings.filterwarnings("ignore")
log = logging.getLogger("mia.equities")

_UA = {"User-Agent": "Mozilla/5.0 (compatible; AlfredResearch/1.0)"}

# Wikipedia keeps these current and is free; the exchange sites either require
# a key or block automated fetches. Each entry: page, ticker column, suffix.
INDICES = {
    "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
              "Symbol", "", "US", "USD"),
    "ASX200": ("https://en.wikipedia.org/wiki/S%26P/ASX_200",
               "Code", ".AX", "ASX", "AUD"),
    "NIFTY50": ("https://en.wikipedia.org/wiki/NIFTY_50",
                "Symbol", ".NS", "NSE", "INR"),
}

EXCHANGES = {"US": "United States", "ASX": "Australia", "NSE": "India"}


def constituents(index: str) -> list[dict]:
    """Current members of one index, as yfinance tickers."""
    url, col, suffix, exchange, ccy = INDICES[index]
    html = httpx.get(url, headers=_UA, timeout=45, follow_redirects=True).text
    table = next((t for t in pd.read_html(io.StringIO(html)) if col in t.columns), None)
    if table is None:
        log.warning("%s: no constituent table found", index)
        return []
    name_col = next((c for c in ("Security", "Company", "Company name", "Name")
                     if c in table.columns), None)
    out = []
    for _, row in table.iterrows():
        raw = str(row[col]).strip().upper().replace(".", "-") if suffix == "" \
            else str(row[col]).strip().upper()
        if not raw or raw == "NAN":
            continue
        out.append({
            "symbol": raw + suffix,
            "name": str(row[name_col]).strip() if name_col else raw,
            "exchange": exchange, "index_member": index, "currency": ccy,
        })
    return out


def sync_universe() -> dict:
    """Refresh index membership. Additions appear; departures are left in
    place, because a stock that left the index still has a news history worth
    keeping and answering questions about."""
    counts = {}
    for index in INDICES:
        try:
            rows = constituents(index)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s constituents failed: %s", index, exc)
            counts[index] = 0
            continue
        db.executemany(
            """INSERT INTO securities (symbol, name, exchange, index_member, currency)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (symbol) DO UPDATE SET
                 name = EXCLUDED.name, exchange = EXCLUDED.exchange,
                 index_member = EXCLUDED.index_member,
                 currency = EXCLUDED.currency""",
            [(r["symbol"], r["name"], r["exchange"], r["index_member"], r["currency"])
             for r in rows],
        )
        counts[index] = len(rows)
        log.info("%s: %d constituents", index, len(rows))
    return counts


def _symbols(exchange: str | None = None, limit: int | None = None) -> list[str]:
    sql = "SELECT symbol FROM securities"
    params: list = []
    if exchange:
        sql += " WHERE exchange = %s"
        params.append(exchange)
    sql += " ORDER BY symbol"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r["symbol"] for r in db.query(sql, tuple(params))]


def fetch_prices(symbols: list[str] | None = None, period: str = "1y",
                 batch: int = 60) -> int:
    """Daily closes. Batched downloads — 750 individual requests would take
    an hour and invite rate limiting; yfinance fetches them in groups."""
    import yfinance as yf

    syms = symbols or _symbols()
    total = 0
    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        try:
            data = yf.download(chunk, period=period, interval="1d",
                               group_by="ticker", auto_adjust=False,
                               progress=False, threads=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("price batch %d failed: %s", i // batch, exc)
            continue
        rows = []
        for sym in chunk:
            try:
                frame = data[sym] if len(chunk) > 1 else data
            except (KeyError, TypeError):
                continue
            if frame is None or frame.empty:
                continue
            for ts, r in frame.iterrows():
                close = r.get("Close")
                if close is None or close != close:  # NaN
                    continue
                rows.append((sym, ts.date(), float(close),
                             float(r.get("Volume") or 0) or None))
        total += db.executemany(
            """INSERT INTO security_prices (symbol, d, close, volume)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (symbol, d) DO UPDATE SET
                 close = EXCLUDED.close, volume = EXCLUDED.volume""",
            rows,
        )
        log.info("prices %d/%d symbols", min(i + batch, len(syms)), len(syms))
    return total


def fetch_fundamentals(symbols: list[str] | None = None, workers: int = 8) -> int:
    """Valuation and profile per name.

    One HTTP round trip each and about two seconds apiece, so 750 names is
    twenty-five minutes serially — long enough to hold up the whole daily job.
    Fetched in parallel and written in one batch instead.
    """
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    syms = symbols or _symbols()

    def _one(sym: str):
        try:
            info = yf.Ticker(sym).info or {}
        except Exception:  # noqa: BLE001
            return None
        if not info.get("marketCap") and not info.get("sector"):
            return None
        return (info.get("sector"), info.get("industry"), info.get("marketCap"),
                info.get("trailingPE"), info.get("forwardPE"),
                info.get("dividendYield"), info.get("fiftyTwoWeekHigh"),
                info.get("fiftyTwoWeekLow"), info.get("beta"), sym)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(_one, syms) if r]
    db.executemany(
        """UPDATE securities SET sector=%s, industry=%s, market_cap=%s,
             trailing_pe=%s, forward_pe=%s, dividend_yield=%s,
             week52_high=%s, week52_low=%s, beta=%s, fundamentals_at=now()
           WHERE symbol=%s""", rows)
    log.info("fundamentals refreshed for %d/%d", len(rows), len(syms))
    return len(rows)


def fetch_news(symbols: list[str] | None = None, per_symbol: int = 8) -> int:
    """Per-stock news, accumulated. This is the context layer: a question about
    a name reaches months of its coverage, not just today's headline."""
    import yfinance as yf

    from concurrent.futures import ThreadPoolExecutor

    syms = symbols or _symbols()

    def _news_for(sym: str):
        try:
            return sym, (yf.Ticker(sym).news or [])
        except Exception:  # noqa: BLE001
            return sym, []

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(_news_for, syms))

    rows = []
    for sym, items in fetched:
        for it in items[:per_symbol]:
            c = it.get("content") or it
            title = (c.get("title") or "").strip()
            if not title:
                continue
            pub = (c.get("provider") or {}).get("displayName") if isinstance(
                c.get("provider"), dict) else c.get("publisher")
            url = ((c.get("canonicalUrl") or {}).get("url")
                   if isinstance(c.get("canonicalUrl"), dict) else c.get("link"))
            when = c.get("pubDate") or c.get("providerPublishTime")
            published = None
            if isinstance(when, str):
                try:
                    published = datetime.fromisoformat(when.replace("Z", "+00:00"))
                except ValueError:
                    published = None
            elif isinstance(when, (int, float)):
                published = datetime.fromtimestamp(when, tz=timezone.utc)
            rows.append((sym, title[:500], pub, url,
                         (c.get("summary") or "")[:2000] or None, published))
    stored = db.executemany(
        """INSERT INTO security_news (symbol, title, publisher, url, summary, published_at)
           VALUES (%s,%s,%s,%s,%s,%s)
           ON CONFLICT (symbol, title) DO NOTHING""",
        rows,
    )
    log.info("security news: %d rows offered", len(rows))
    return stored


# ─────────────────────────────── entrypoints ────────────────────────────────
def backfill(period: str = "2y") -> dict:
    out = {"universe": sync_universe()}
    out["prices"] = fetch_prices(period=period)
    out["fundamentals"] = fetch_fundamentals()
    out["news"] = fetch_news()
    return out


def daily_refresh() -> dict:
    """Runs with the rest of the daily data job."""
    out = {"universe": sync_universe()}
    out["prices"] = fetch_prices(period="1mo")
    out["fundamentals"] = fetch_fundamentals()
    out["news"] = fetch_news()
    return out
