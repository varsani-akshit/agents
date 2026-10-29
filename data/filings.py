"""Exchange and regulator filings for the names in play: SEC EDGAR for US
names, NSE corporate announcements for Indian ones, ASX announcements for
Australian ones."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from advisor import bare, market_of
from data import NSE, BROWSER_UA, get, in_play, upsert_filings

SEC_FORMS = {"8-K", "10-Q", "10-K", "4", "4/A", "SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A",
             "6-K", "20-F", "S-3", "424B5", "DEF 14A"}
SEC_SENSITIVE = {"8-K", "10-Q", "10-K", "6-K", "20-F", "S-3", "424B5"}
_ciks: dict[str, str] = {}


# ───────────────────────────────────── SEC ──────────────────────────────────
def ciks() -> dict[str, str]:
    global _ciks
    if not _ciks:
        got = get("https://www.sec.gov/files/company_tickers.json", timeout=60)
        _ciks = {v["ticker"].upper(): f"{int(v['cik_str']):010d}" for v in got.values()}
    return _ciks


def sec_rows(symbol: str, cik: str, recent: dict, days: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    rows = []
    keys = ("form", "filingDate", "accessionNumber", "primaryDocument", "primaryDocDescription")
    for form, fdate, acc, doc, desc in zip(*(recent.get(k) or [] for k in keys)):
        if form not in SEC_FORMS:
            continue
        d = datetime.fromisoformat(fdate).date()
        if d < cutoff:
            continue
        folder = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{doc}"
        rows.append({"source": "sec", "symbol": symbol, "kind": form,
                     "title": (desc or form)[:300], "url": url,
                     "filed_at": datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                     "sensitive": form in SEC_SENSITIVE, "raw": {"accession": acc}})
    return rows


def sec(symbols: list[str], days: int = 30) -> dict:
    table = ciks()
    n = looked = 0
    for sym in symbols:
        cik = table.get(bare(sym).upper())
        if not cik:
            continue
        got = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        n += upsert_filings(sec_rows(sym, cik, (got.get("filings") or {}).get("recent") or {}, days))
        looked += 1
        time.sleep(0.2)           # SEC allows 10 requests a second
    return {"rows": n, "names": looked}


# ───────────────────────────────────── NSE ──────────────────────────────────
def nse_rows(symbol: str, items: list[dict], days: int = 60) -> list[dict]:
    """NSE answers with a company's whole announcement history (thousands of
    rows for TCS); keep the recent ones."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for it in items:
        url = it.get("attchmntFile") or ""
        title = (it.get("desc") or "").strip()
        if not url or not title:
            continue
        try:
            filed = datetime.strptime(it["an_dt"], "%d-%b-%Y %H:%M:%S").replace(
                tzinfo=timezone(timedelta(hours=5, minutes=30)))
        except (KeyError, ValueError):
            continue
        if filed < cutoff:
            continue
        body = (it.get("attchmntText") or "").strip()
        rows.append({"source": "nse", "symbol": symbol, "kind": title[:80],
                     "title": (body[:300] if body else title[:300]), "url": url,
                     "filed_at": filed, "sensitive": title.lower() in _NSE_SENSITIVE,
                     "raw": {"desc": title, "industry": it.get("smIndustry")}})
    return rows


_NSE_SENSITIVE = {"financial results", "board meeting", "acquisition", "updates",
                  "press release", "change in management", "investor presentation",
                  "resignation", "dividend", "buyback", "fund raising", "credit rating"}


def nse(symbols: list[str]) -> dict:
    client = NSE()
    n = 0
    try:
        for sym in symbols:
            items = client.get("/api/corporate-announcements", {"index": "equities", "symbol": bare(sym)})
            n += upsert_filings(nse_rows(sym, items or []))
            time.sleep(0.5)
    finally:
        client.close()
    return {"rows": n, "names": len(symbols)}


# ───────────────────────────────────── ASX ──────────────────────────────────
def asx_rows(symbol: str, items: list[dict]) -> list[dict]:
    rows = []
    code = bare(symbol)
    for it in items:
        key = it.get("documentKey")
        title = (it.get("headline") or "").strip()
        if not key or not title:
            continue
        try:
            filed = datetime.fromisoformat(it["date"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            filed = None
        rows.append({"source": "asx", "symbol": symbol, "kind": (it.get("announcementType") or "")[:80],
                     "title": title[:300],
                     "url": f"https://www.asx.com.au/markets/company/{code}#announcements?doc={key}",
                     "filed_at": filed, "sensitive": bool(it.get("isPriceSensitive")),
                     "raw": {"documentKey": key, "fileSize": it.get("fileSize")}})
    return rows


def asx(symbols: list[str], count: int = 20) -> dict:
    n = 0
    for sym in symbols:
        got = get(f"https://asx.api.markitdigital.com/asx-research/1.0/companies/{bare(sym)}/announcements",
                  {"count": count}, headers={"user-agent": BROWSER_UA})
        n += upsert_filings(asx_rows(sym, (got.get("data") or {}).get("items") or []))
        time.sleep(0.3)
    return {"rows": n, "names": len(symbols)}


def sync() -> dict:
    syms = sorted(in_play())
    by = {"US": [], "NSE": [], "BSE": [], "ASX": []}
    for s in syms:
        m = market_of(s)
        if m in by:
            by[m].append(s)
    out = {"rows": 0}
    for name, fn, group in (("sec", sec, by["US"]), ("nse", nse, by["NSE"] + by["BSE"]),
                            ("asx", asx, by["ASX"])):
        try:
            got = fn(group) if group else {"rows": 0, "names": 0}
            out[name] = got["rows"]
            out["rows"] += got["rows"]
        except Exception as exc:  # noqa: BLE001 - one exchange down is not all three
            out[f"{name}_error"] = str(exc)[:200]
    return out
