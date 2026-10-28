"""Charts made for a story, on request.

The brief used to choose from fifteen standing figures — the US yield curve,
real yields against gold — whatever the news was. Now the desk asks for the
chart its story needs: "Coal India against the NIFTY over six months", "the
rupee and Brent this quarter", "ASX banks over a month". This module builds
it from measured data, so the shape and every number in its subtitle come
from code, never from the model that asked.

A request names a kind, symbols and a window:
    price        one series, its level
    compare      two to five series, rebased to 0% at the window start —
                 levels in different units cannot share an axis honestly
    performance  three to twelve series, % change over the window, as bars

Symbols are Alfred's macro instruments (GOLD, US10Y, USDINR, INDIA, OIL, ...)
or any Yahoo ticker (RELIANCE.NS, BHP.AX, NVDA, GLD). A ticker Alfred has not
seen yet is fetched once, on demand.
"""
from __future__ import annotations

import logging

import pandas as pd

import db
from signals import chartdata, stats

log = logging.getLogger("mia.storycharts")

KINDS = ("price", "compare", "performance")
PALETTE = [chartdata.ORANGE, chartdata.INK, chartdata.SLATE, chartdata.TEAL, chartdata.AMBER]
MAX_DAYS = 1500


def _macro() -> pd.DataFrame:
    if not hasattr(_macro, "cache"):
        _macro.cache = stats.load_daily(lookback_days=MAX_DAYS * 2)
    return _macro.cache


def instruments() -> list[str]:
    """Macro symbols a request may name directly."""
    return sorted(_macro().columns)


def _rates() -> set[str]:
    return {r["symbol"] for r in db.query("SELECT symbol FROM instruments WHERE is_rate")}


def series(symbol: str, days: int) -> pd.Series | None:
    """Daily closes for any chartable symbol, fetching a new ticker once."""
    symbol = (symbol or "").strip()
    wide = _macro()
    if symbol in wide.columns:
        s = wide[symbol].dropna()
    else:
        from advisor import bars

        s = bars.load(symbol, years=6, final_only=False)["close"] \
            if db.one("SELECT 1 AS x FROM bars WHERE symbol=%s LIMIT 1", (symbol,)) else None
        if s is None or s.empty:
            try:
                bars.fetch([symbol], period="5y")
            except Exception as exc:  # noqa: BLE001
                log.info("could not fetch %s: %s", symbol, exc)
                return None
            got = bars.load(symbol, years=6, final_only=False)
            s = got["close"] if not got.empty else None
        if s is None:
            return None
    s = s.iloc[-days:]
    return s if len(s) >= 5 else None


def _name(symbol: str) -> str:
    row = db.one("SELECT name FROM instruments WHERE symbol=%s", (symbol,)) or \
        db.one("SELECT name FROM securities WHERE symbol=%s", (symbol,))
    return (row and row["name"]) or symbol


def build(req: dict, key: str) -> dict | None:
    """One requested chart, or None if the data is not there. Never raises."""
    try:
        return _build(req, key)
    except Exception as exc:  # noqa: BLE001 - a bad request must not sink a brief
        log.warning("story chart %s failed: %s", key, exc)
        return None


def _build(req: dict, key: str) -> dict | None:
    kind = req.get("kind") if req.get("kind") in KINDS else "compare"
    days = max(20, min(int(req.get("days") or 126), MAX_DAYS))
    symbols = [s for s in dict.fromkeys(req.get("symbols") or []) if s][:12]
    got = {s: v for s in symbols if (v := series(s, days)) is not None}
    if not got:
        return None
    title = (req.get("title") or "").strip()[:90]
    why = (req.get("why") or "").strip()[:160]
    window = f"{days} sessions"

    if kind == "performance" or (kind != "price" and len(got) > 5):
        rows = sorted(((s, (v.iloc[-1] / v.iloc[0] - 1) * 100) for s, v in got.items()),
                      key=lambda r: r[1])
        return {"key": key, "type": "bar_h", "title": title or f"Performance over {window}",
                "subtitle": f"% change over the last {window}" + (f" · {why}" if why else ""),
                "categories": [s for s, _ in rows], "values": [round(v, 2) for _, v in rows],
                "notes": [_name(s) for s, _ in rows], "unit": "%", "signColour": True,
                "story": True}

    if kind == "price" or len(got) == 1:
        sym, s = next(iter(got.items()))
        rate = sym in _rates()
        move = (f"{(s.iloc[-1] - s.iloc[0]) * 100:+.0f}bp" if rate
                else f"{(s.iloc[-1] / s.iloc[0] - 1) * 100:+.1f}%")
        frame = pd.DataFrame({sym: s})
        spec = chartdata._line(key, title or _name(sym),
                               f"last {s.iloc[-1]:,.2f} · {move} over {window}"
                               + (f" · {why}" if why else ""),
                               frame, {sym: PALETTE[0]})
        return {**spec, "story": True}

    # Rates share an axis in their own units; everything else is rebased.
    rates = _rates()
    if all(s in rates for s in got):
        frame = pd.DataFrame(got).ffill().dropna()
        sub, unit = "level, per cent", "%"
    else:
        frame = pd.DataFrame({s: (v / v.iloc[0] - 1) * 100 for s, v in got.items()}).ffill().dropna()
        sub, unit = "rebased to 0% at the window start", "%"
    if all(s in rates for s in got):
        last = ", ".join(f"{s} {frame[s].iloc[-1]:.2f}%" for s in frame.columns) + " now"
    else:
        last = ", ".join(f"{s} {frame[s].iloc[-1]:+.1f}%" for s in frame.columns)
    spec = chartdata._line(key, title or " vs ".join(frame.columns),
                           f"{sub} · {last}" + ("" if last.endswith(" now") else f" over {window}")
                           + (f" · {why}" if why else ""),
                           frame, {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(frame.columns)},
                           y_label=unit)
    spec["markZero"] = unit == "%" and not all(s in rates for s in got)
    return {**spec, "story": True}


def build_many(requests: list[dict], limit: int = 8) -> dict[str, dict]:
    """Build up to `limit` distinct requests, keyed story_1, story_2, ..."""
    out, seen = {}, set()
    for req in requests:
        sig = (req.get("kind"), tuple(req.get("symbols") or []), int(req.get("days") or 126))
        if sig in seen or not req.get("symbols"):
            continue
        seen.add(sig)
        key = f"story_{len(out) + 1}"
        spec = build(req, key)
        if spec:
            spec["for"] = (req.get("story") or req.get("why") or "")[:160]
            out[key] = spec
        if len(out) >= limit:
            break
    return out
