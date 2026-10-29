"""Macro: the extra FRED series, the traded regime proxies from Yahoo, the
FRED release calendar, central-bank meeting dates, and NSE's FII/DII flows."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import config
from data import NSE, conf, get, upsert_calendar, upsert_series
from ingest import fred


def fred_series(backfill: bool = False) -> dict:
    ids = conf().get("macro", {}).get("fred") or {}
    start = date.today() - timedelta(days=365 * 15 if backfill else 60)
    n = 0
    for sid in ids:
        n += fred.store(fred.fetch_series(sid, start))
    return {"rows": n, "series": len(ids)}


def proxies(backfill: bool = False) -> dict:
    from advisor import bars

    syms = conf().get("macro", {}).get("yahoo") or []
    return {"rows": bars.fetch(list(syms), period="12y" if backfill else "1mo"), "symbols": len(syms)}


def release_rows(items: list[dict], high: set[str]) -> list[dict]:
    rows = []
    for it in items:
        name, d = it.get("release_name"), it.get("date")
        if not name or not d:
            continue
        at = datetime.fromisoformat(d).replace(hour=13, minute=30, tzinfo=timezone.utc)  # 8:30 ET
        rows.append({"kind": "release", "key": "US", "event": name, "at": at,
                     "importance": "high" if name in high else "low",
                     "raw": {"release_id": it.get("release_id")}})
    return rows


def releases(days_ahead: int = 21) -> dict:
    if not config.FRED_API_KEY:
        raise RuntimeError("no FRED_API_KEY")
    high = set(conf().get("macro", {}).get("releases_high") or [])
    got = get("https://api.stlouisfed.org/fred/releases/dates", {
        "api_key": config.FRED_API_KEY, "file_type": "json",
        "realtime_start": date.today().isoformat(),
        "realtime_end": (date.today() + timedelta(days=days_ahead)).isoformat(),
        "include_release_dates_with_no_data": "true", "limit": 1000})
    rows = release_rows(got.get("release_dates") or [], high)
    return {"rows": upsert_calendar(rows), "high": sum(r["importance"] == "high" for r in rows)}


def meetings() -> dict:
    rows = []
    for m in conf().get("macro", {}).get("meetings") or []:
        for d in m.get("dates") or []:
            d = d if isinstance(d, date) else date.fromisoformat(str(d))
            rows.append({"kind": "meeting", "key": m["key"], "event": m["event"],
                         "at": datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                         "importance": "high"})
    return {"rows": upsert_calendar(rows)}


def flow_points(items: list[dict]) -> dict[str, list[tuple[date, float]]]:
    """NSE's FII/DII table: rows per category with buy, sell, net in ₹ crore."""
    out: dict[str, list[tuple[date, float]]] = {}
    for it in items:
        cat = (it.get("category") or "").upper()
        key = "nse.fii_net_crore" if cat.startswith("FII") else (
            "nse.dii_net_crore" if cat.startswith("DII") else None)
        try:
            d = datetime.strptime(it["date"], "%d-%b-%Y").date()
            out.setdefault(key, []).append((d, float(str(it["netValue"]).replace(",", ""))))
        except (KeyError, ValueError, TypeError):
            continue
    out.pop(None, None)
    return out


def flows() -> dict:
    nse = NSE()
    try:
        items = nse.get("/api/fiidiiTradeReact")
    finally:
        nse.close()
    n = 0
    for series, pts in flow_points(items or []).items():
        n += upsert_series(series, pts)
    return {"rows": n}


GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def gpr_points(df) -> dict[str, list[tuple[date, float]]]:
    """Caldara & Iacoviello's daily Geopolitical Risk index: the share of
    articles in ten newspapers about war, terrorism and geopolitical
    tension, since 1985 — total, threats, and acts."""
    out: dict[str, list] = {"gpr.daily": [], "gpr.threats": [], "gpr.acts": []}
    for r in df.itertuples():
        try:
            d = datetime.strptime(str(int(r.DAY)), "%Y%m%d").date()
        except (TypeError, ValueError):
            continue
        for key, col in (("gpr.daily", "GPRD"), ("gpr.threats", "GPRD_THREAT"),
                         ("gpr.acts", "GPRD_ACT")):
            v = getattr(r, col)
            if v == v:          # not NaN
                out[key].append((d, float(v)))
    return out


def gpr(backfill: bool = False) -> dict:
    import io

    import httpx
    import pandas as pd

    r = httpx.get(GPR_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), engine="xlrd")
    since = None if backfill else date.today() - timedelta(days=90)
    n = 0
    for series, pts in gpr_points(df).items():
        n += upsert_series(series, [(d, v) for d, v in pts if since is None or d >= since])
    return {"rows": n}


def sync(backfill: bool = False) -> dict:
    out = fred_series(backfill)
    try:
        out["gpr"] = gpr(backfill)["rows"]
    except Exception as exc:  # noqa: BLE001 - one file host down is not all macro
        out["gpr_error"] = str(exc)[:200]
        out["gpr"] = 0
    out["rows"] += out["gpr"]
    out["proxies"] = proxies(backfill)["rows"]
    out["releases"] = releases()["rows"]
    out["meetings"] = meetings()["rows"]
    try:
        out["flows"] = flows()["rows"]
    except Exception as exc:  # noqa: BLE001 - NSE's site is the flaky one
        out["flows_error"] = str(exc)[:200]
        out["flows"] = 0
    out["rows"] += out["proxies"] + out["releases"] + out["meetings"] + out["flows"]
    return out
