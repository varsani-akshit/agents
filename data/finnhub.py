"""Finnhub, free tier: US company news, the general wire, and the earnings
calendar. (The economic calendar is a paid endpoint; releases come from FRED
in data/macro.py instead.)"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import config
import db
from advisor import bare, market_of
from data import conf, get, in_play, upsert_calendar
from ingest.dedupe import canonical_url, content_hash

API = "https://finnhub.io/api/v1"


def _get(path: str, params: dict | None = None):
    if not config.FINNHUB_API_KEY:
        raise RuntimeError("no FINNHUB_API_KEY")
    return get(f"{API}{path}", {**(params or {}), "token": config.FINNHUB_API_KEY})


def us_names() -> list[str]:
    """Holdings, open tickets and the watch list always; then the latest
    run's passing candidates up to the cap. A run can pass two hundred US
    names, and one news call a second each is not an hourly job."""
    from advisor import bars

    c = conf().get("finnhub", {})
    cap = int(c.get("max_names", 40))
    first = {bare(s) for s in bars.holdings() if market_of(s) == "US"}
    first |= {r["symbol"] for r in db.query(
        "SELECT symbol FROM ideas WHERE market = 'US' AND status IN ('open', 'filled')")}
    first |= set(c.get("watch") or [])
    rest = sorted(bare(s) for s in in_play() if market_of(s) == "US")
    out = sorted(first)
    for s in rest:
        if len(out) >= cap:
            break
        if s not in first:
            out.append(s)
    return out


def news_docs(symbol: str | None, items: list[dict], source: str, tier: int = 3) -> list[dict]:
    docs = []
    for it in items:
        title = (it.get("headline") or "").strip()
        if not title:
            continue
        url = canonical_url(it.get("url") or "")
        body = (it.get("summary") or "").strip()
        published = (datetime.fromtimestamp(int(it["datetime"]), tz=timezone.utc)
                     if it.get("datetime") else None)
        docs.append({"url": url or None, "title": title[:500], "body": body, "source": source,
                     "source_tier": tier, "official": False, "published_at": published,
                     "content_hash": content_hash(title, body, url), "symbol": symbol})
    return docs


def _store_docs(docs: list[dict]) -> int:
    from ingest import feeds

    out = feeds.store(docs)
    # Tag the new rows with the company, so the Story read can find them by name.
    by_hash = {d["content_hash"]: d.get("symbol") for d in docs if d.get("symbol")}
    if out["new_ids"] and by_hash:
        for r in db.query("SELECT id, content_hash FROM documents WHERE id = ANY(%s)",
                          (out["new_ids"],)):
            sym = by_hash.get(r["content_hash"])
            if sym:
                db.execute("UPDATE documents SET entities = %s WHERE id = %s",
                           (f'["{sym}"]', r["id"]))
    return out["inserted"]


def company_news() -> dict:
    days = int(conf().get("finnhub", {}).get("news_days", 3))
    to, frm = date.today(), date.today() - timedelta(days=days)
    n = 0
    names = us_names()
    for sym in names:
        items = _get("/company-news", {"symbol": sym, "from": frm.isoformat(), "to": to.isoformat()})
        n += _store_docs(news_docs(sym, items or [], f"Finnhub · {sym}"))
        time.sleep(1.1)           # 60 calls a minute on the free plan
    return {"rows": n, "names": len(names)}


def wire() -> dict:
    items = _get("/news", {"category": "general"})
    return {"rows": _store_docs(news_docs(None, items or [], "Finnhub wire"))}


def earnings_rows(items: list[dict]) -> list[dict]:
    rows = []
    for it in items:
        if not it.get("symbol") or not it.get("date"):
            continue
        hour = {"bmo": 8, "amc": 17}.get(it.get("hour") or "", 12)
        at = datetime.fromisoformat(it["date"]).replace(hour=hour, tzinfo=timezone.utc)
        rows.append({"kind": "earnings", "key": it["symbol"], "event": "earnings", "at": at,
                     "estimate": it.get("epsEstimate"), "actual": it.get("epsActual"),
                     "prior": None, "importance": "high" if it.get("symbol") in _WATCH else "low",
                     "raw": {k: it.get(k) for k in ("hour", "quarter", "year",
                                                    "revenueEstimate", "revenueActual")}})
    return rows


_WATCH: set[str] = set()


def earnings() -> dict:
    global _WATCH
    _WATCH = set(us_names())
    ahead = int(conf().get("finnhub", {}).get("earnings_days_ahead", 21))
    frm, to = date.today() - timedelta(days=7), date.today() + timedelta(days=ahead)
    got = _get("/calendar/earnings", {"from": frm.isoformat(), "to": to.isoformat()})
    rows = earnings_rows((got or {}).get("earningsCalendar") or [])
    return {"rows": upsert_calendar(rows), "watched": sum(r["importance"] == "high" for r in rows)}


def sync(calendar: bool = True) -> dict:
    out = company_news()
    out["wire"] = wire()["rows"]
    out["rows"] += out["wire"]
    if calendar:
        e = earnings()
        out["earnings"] = e["rows"]
        out["rows"] += e["rows"]
    return out
