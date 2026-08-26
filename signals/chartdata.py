"""Chart *data*, not chart images.

Every figure the brief references is emitted here as a plain JSON spec and drawn
in the browser, so a reader can hover a line and read the value on a date rather
than squint at a baked PNG. The arithmetic is identical to the rest of the
signals package — deterministic, no model involved — only the rendering moved.

Spec shapes, all consumed by web/static/charts.js:

  line          x: ISO dates, series: [{name, data, color}]
  dual_line     same, plus axis: 0|1 per series and an optional inverted axis
  bar_h         categories + values, coloured by sign or by an explicit list
  diverging     bar_h with a weight annotation per row (the regime votes)
  heatmap       xLabels, yLabels, cells [[xi, yi, value]]
  curve         numeric x (maturities), one point per tenor

Series are downsampled to keep the payload small: a 12-year daily line is 3,000
points, which is more than any screen can resolve and enough JSON to slow the
page down noticeably on a phone.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import db
from signals import regime, stats

log = logging.getLogger("mia.chartdata")

# 300 points is finer than a phone can resolve on a chart a few hundred pixels
# wide, and keeps the whole 15-chart payload small enough to load over mobile
# data without a spinner.
MAX_POINTS = 300

# Light-theme palette. Orange is the accent and carries the subject of each
# chart; everything else is grey so the eye lands on the point being made.
INK = "#111111"
ORANGE = "#EA580C"
AMBER = "#B45309"
SLATE = "#64748B"
GREY = "#9CA3AF"
MIST = "#CBD5E1"
TEAL = "#0F766E"
UP = "#15803D"
DOWN = "#B91C1C"

SYMBOL_COLOR = {
    "GOLD": AMBER, "SILVER": SLATE, "BTC": ORANGE, "ETH": TEAL,
    "SPX": INK, "DXY": GREY, "TLT": SLATE, "US10Y": INK, "OIL": TEAL,
}


def _dates(index: pd.Index) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in index]


def _thin(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Downsample by taking every nth row, always keeping the final point."""
    if len(frame) <= MAX_POINTS:
        return frame
    step = int(np.ceil(len(frame) / MAX_POINTS))
    thinned = frame.iloc[::step]
    if not thinned.index[-1] == frame.index[-1]:
        thinned = pd.concat([thinned, frame.iloc[[-1]]])
    return thinned


def _clean(values) -> list:
    """NaN is not valid JSON; ECharts reads null as a gap, which is correct."""
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
            for v in values]


def _line(key, title, subtitle, frame: pd.DataFrame, colors: dict, y_label="") -> dict:
    frame = _thin(frame)
    return {
        "key": key, "type": "line", "title": title, "subtitle": subtitle,
        "x": _dates(frame.index), "yLabel": y_label,
        "series": [
            {"name": c, "data": _clean(frame[c]), "color": colors.get(c, GREY)}
            for c in frame.columns
        ],
    }


# ─────────────────────────────────── charts ─────────────────────────────────
def regime_gauge(_: pd.DataFrame) -> dict | None:
    rs = regime.regime_score()
    if not rs or not rs.get("components"):
        return None
    comps = rs["components"]
    rows = sorted(comps.items(), key=lambda kv: kv[1]["vote"])
    return {
        "key": "regime_gauge", "type": "diverging",
        "title": "Regime score",
        "subtitle": f"{rs['score']:+.2f} — {rs['label']}",
        "score": rs["score"], "scoreLabel": rs["label"],
        "axisLabel": "tight money  ←     → debasement",
        "categories": [k.replace("_", " ") for k, _ in rows],
        "values": [round(v["vote"], 3) for _, v in rows],
        "notes": [v["basis"] for _, v in rows],
        "weights": [rs["weights"].get(k) for k, _ in rows],
        "min": -1, "max": 1,
    }


def returns_heatmap(wide: pd.DataFrame) -> dict | None:
    horizons = [("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)]
    classes = {r["symbol"]: r["asset_class"]
               for r in db.query("SELECT symbol, asset_class FROM instruments")}
    syms, cells, flat = [], [], []
    ordered = sorted(wide.columns, key=lambda s: (classes.get(s, "zz"), s))
    for sym in ordered:
        if sym.startswith("US") and sym.endswith("Y"):
            continue
        s = wide[sym].dropna()
        if len(s) < 260:
            continue
        vals = [stats.pct_change(s, n) for _, n in horizons]
        if any(v is None for v in vals):
            continue
        syms.append(sym)
        for j, v in enumerate(vals):
            cells.append([j, len(syms) - 1, round(v, 1)])
            flat.append(abs(v))
    if len(syms) < 5:
        return None
    lim = float(np.percentile(flat, 92)) or 10.0
    return {
        "key": "returns_heatmap", "type": "heatmap",
        "title": "Total return by horizon",
        "subtitle": "Per cent. Every tracked instrument, grouped by asset class.",
        "xLabels": [h for h, _ in horizons], "yLabels": syms,
        "cells": cells, "min": round(-lim, 1), "max": round(lim, 1), "unit": "%",
    }


def cross_asset_performance(wide: pd.DataFrame) -> dict | None:
    classes = {r["symbol"]: r["asset_class"]
               for r in db.query("SELECT symbol, asset_class FROM instruments")}
    rows = []
    for sym in wide.columns:
        if sym.startswith("US") and sym.endswith("Y"):
            continue
        v = stats.pct_change(wide[sym].dropna(), 21)
        if v is not None:
            rows.append((sym, round(v, 2), classes.get(sym, "other")))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])
    return {
        "key": "cross_asset_performance", "type": "bar_h",
        "title": "Cross-asset performance",
        "subtitle": "One-month change, per cent",
        "categories": [r[0] for r in rows], "values": [r[1] for r in rows],
        "notes": [r[2] for r in rows], "unit": "%", "signColour": True,
    }


def normalised_performance(wide: pd.DataFrame, days: int = 120) -> dict | None:
    picks = ["GOLD", "SILVER", "BTC", "SPX", "DXY"]
    have = [s for s in picks if s in wide.columns]
    sub = wide[have].dropna().iloc[-days:]
    if len(sub) < 20:
        return None
    return _line("normalised_performance", "Rebased performance",
                 f"Last {len(sub)} sessions, start = 100",
                 sub / sub.iloc[0] * 100, SYMBOL_COLOR, "index")


def net_liquidity(_: pd.DataFrame) -> dict | None:
    walcl = regime.fred_frame("WALCL", 1600)
    if walcl.empty:
        return None
    tga, rrp = regime.fred_frame("WTREGEN", 1600), regime.fred_frame("RRPONTSYD", 1600)
    frame = pd.concat({"w": walcl, "t": tga, "r": rrp}, axis=1, sort=True).sort_index().ffill()
    net = ((frame["w"] - frame["t"] - frame["r"].fillna(0) * 1000).dropna() / 1e6)
    net = net.resample("W-WED").last().dropna()
    if len(net) < 12:
        return None

    wide = stats.load_daily(lookback_days=1600)
    series = [
        {"name": "Net liquidity", "data": _clean(net), "color": MIST, "axis": 0, "width": 1},
        {"name": "13-week average", "data": _clean(net.rolling(13, min_periods=4).mean()),
         "color": INK, "axis": 0, "width": 2},
    ]
    for sym, color in (("GOLD", AMBER), ("BTC", ORANGE)):
        if sym in wide.columns:
            s = wide[sym].dropna().reindex(net.index, method="ffill").dropna()
            if len(s) > 20:
                series.append({"name": f"{sym} (rebased)", "color": color, "axis": 1,
                               "data": _clean(s / s.iloc[0] * 100)})
    return {
        "key": "net_liquidity", "type": "dual_line",
        "title": "Net liquidity versus hard assets",
        "subtitle": "Fed assets less the Treasury General Account and reverse repo, weekly",
        "x": _dates(net.index), "series": series,
        "yLabel": "$tn", "y2Label": "rebased to 100",
    }


def real_yield_gold(_: pd.DataFrame) -> dict | None:
    real = regime.fred_frame("DFII10", 1600)
    if real.empty:
        return None
    wide = stats.load_daily(lookback_days=1600)
    if "GOLD" not in wide.columns:
        return None
    joint = pd.concat({"real": real, "gold": wide["GOLD"]}, axis=1, sort=True).sort_index()
    joint = _thin(joint.ffill().dropna())
    return {
        "key": "real_yield_gold", "type": "dual_line",
        "title": "Gold against the 10-year real yield",
        "subtitle": "The textbook relationship, plotted so it can be seen to break. "
                    "The real-yield axis is inverted.",
        "x": _dates(joint.index),
        "series": [
            {"name": "10y TIPS real yield", "data": _clean(joint["real"]),
             "color": SLATE, "axis": 0},
            {"name": "Gold", "data": _clean(joint["gold"]), "color": AMBER, "axis": 1},
        ],
        "yLabel": "real yield %", "y2Label": "$/oz", "invertY": True,
    }


def rolling_correlations(wide: pd.DataFrame, days: int = 400) -> dict | None:
    pairs = [("GOLD", "TLT", AMBER), ("GOLD", "DXY", SLATE),
             ("BTC", "SPX", ORANGE), ("GOLD", "US10Y", INK)]
    cols = {}
    for a, b, color in pairs:
        if a in wide.columns and b in wide.columns:
            cs = stats.corr_series(wide[a], wide[b], 30)
            if len(cs) >= 60:
                cols[f"{a}/{b}"] = (cs.iloc[-days:], color)
    if not cols:
        return None
    frame = pd.DataFrame({k: v[0] for k, v in cols.items()}).dropna(how="all")
    spec = _line("rolling_correlations", "Rolling correlations",
                 "30-day correlation of daily returns. Zero means unrelated.",
                 frame, {k: v[1] for k, v in cols.items()}, "correlation")
    spec |= {"yMin": -1, "yMax": 1, "markZero": True}
    return spec


def correlation_heatmap(wide: pd.DataFrame, window: int = 30) -> dict | None:
    cols = [c for c in stats.CORE if c in wide.columns]
    if len(cols) < 4:
        return None
    rets = wide[cols].pct_change().dropna().iloc[-window:]
    if len(rets) < window // 2:
        return None
    m = rets.corr()
    cells = [[j, i, round(float(m.values[i, j]), 2)]
             for i in range(len(cols)) for j in range(len(cols))]
    return {
        "key": "correlation_heatmap", "type": "heatmap",
        "title": f"{window}-day return correlations",
        "subtitle": "Computed on daily returns, never on price levels.",
        "xLabels": cols, "yLabels": cols, "cells": cells,
        "min": -1, "max": 1, "unit": "",
    }


def fx_performance(wide: pd.DataFrame) -> dict | None:
    board = stats.fx_board(wide)
    if not board:
        return None
    board = sorted(board, key=lambda r: r["currency_1m_vs_usd_pct"])
    return {
        "key": "fx_performance", "type": "bar_h",
        "title": "Currencies against the dollar",
        "subtitle": "One month, per cent. Positive means the currency gained on the dollar.",
        "categories": [r["currency"] for r in board],
        "values": [r["currency_1m_vs_usd_pct"] for r in board],
        "notes": [r["pair"] for r in board], "unit": "%", "signColour": True,
    }


def global_equities(wide: pd.DataFrame, days: int = 252) -> dict | None:
    picks = {"SPX": INK, "EUROPE": AMBER, "JAPAN": DOWN, "CHINA": ORANGE,
             "INDIA": TEAL, "EM": GREY}
    have = [s for s in picks if s in wide.columns]
    if len(have) < 3:
        return None
    sub = wide[have].dropna().iloc[-days:]
    if len(sub) < 40:
        return None
    return _line("global_equities", "Regional equities",
                 f"Rebased, last {len(sub)} sessions. Is this a US story or a global one?",
                 sub / sub.iloc[0] * 100, picks, "index")


def credit_spreads(_: pd.DataFrame) -> dict | None:
    spec = {"BAMLH0A0HYM2": ("US high yield", DOWN),
            "BAMLC0A0CM": ("US investment grade", SLATE),
            "BAMLEMCBPIOAS": ("EM corporate", ORANGE)}
    cols, colors = {}, {}
    for sid, (label, color) in spec.items():
        s = regime.fred_frame(sid, 1600)
        if len(s) >= 30:
            cols[label] = s
            colors[label] = color
    if not cols:
        return None
    frame = pd.concat(cols, axis=1, sort=True).sort_index().ffill().dropna(how="all")
    return _line("credit_spreads", "Credit spreads",
                 "Option-adjusted spread, per cent. The deflationary counterweight "
                 "to the debasement trade.", frame, colors, "spread %")


def yield_curve(_: pd.DataFrame) -> dict | None:
    tenors = [("US2Y", 2), ("US5Y", 5), ("US10Y", 10), ("US30Y", 30)]
    now_pts, then_pts = [], []
    for sym, yrs in tenors:
        rows = db.query(
            "SELECT ts, price FROM prices WHERE symbol=%s AND grain='1d' "
            "ORDER BY ts DESC LIMIT 25", (sym,))
        if not rows:
            continue
        now_pts.append([yrs, round(float(rows[0]["price"]), 3)])
        if len(rows) >= 21:
            then_pts.append([yrs, round(float(rows[20]["price"]), 3)])
    if len(now_pts) < 3:
        return None
    series = [{"name": "Today", "data": now_pts, "color": ORANGE}]
    if len(then_pts) >= 3:
        series.append({"name": "One month ago", "data": then_pts, "color": GREY, "dashed": True})
    return {
        "key": "yield_curve", "type": "curve",
        "title": "US Treasury curve",
        "subtitle": "Shape, not just level.",
        "series": series, "xLabel": "maturity (years)", "yLabel": "yield %",
    }


def ratios(wide: pd.DataFrame, days: int = 500) -> dict | None:
    if "GOLD" not in wide.columns or "SILVER" not in wide.columns:
        return None
    r = (wide["GOLD"] / wide["SILVER"]).dropna().iloc[-days:]
    if len(r) < 60:
        return None
    mean, sd = float(r.mean()), float(r.std())
    thin = _thin(r)
    return {
        "key": "ratios", "type": "line",
        "title": "Gold / silver ratio",
        "subtitle": f"Last {r.iloc[-1]:.1f}. Mean {mean:.1f} over the window, "
                    f"±1σ band shaded.",
        "x": _dates(thin.index), "yLabel": "ratio",
        "series": [{"name": "Gold / silver", "data": _clean(thin), "color": AMBER}],
        "band": {"mean": round(mean, 2), "upper": round(mean + sd, 2),
                 "lower": round(mean - sd, 2)},
    }


def drawdowns(wide: pd.DataFrame) -> dict | None:
    rows = [r for r in stats.drawdown_table(wide) if r["pct_from_high"] is not None]
    if len(rows) < 5:
        return None
    rows = sorted(rows, key=lambda r: r["pct_from_high"])
    return {
        "key": "drawdowns", "type": "bar_h",
        "title": "Distance from the 52-week high",
        "subtitle": "Per cent below the high. Momentum's missing half.",
        "categories": [r["symbol"] for r in rows],
        "values": [r["pct_from_high"] for r in rows],
        "notes": [f"high {r['high_52w']}, low {r['low_52w']}" for r in rows],
        "unit": "%", "signColour": False, "colour": SLATE,
    }


def macro_panel(_: pd.DataFrame) -> dict | None:
    """Term premium, real yield, TGA and high-yield spread as one small-multiple."""
    spec = [("THREEFYTP10", "10y term premium (ACM)"), ("DFII10", "10y real yield %"),
            ("WTREGEN", "Treasury General Account ($mn)"), ("BAMLH0A0HYM2", "HY spread %")]
    panels = []
    for sid, label in spec:
        s = regime.fred_frame(sid, 500)
        if len(s) < 10:
            continue
        s = _thin(s)
        panels.append({"name": label, "x": _dates(s.index), "data": _clean(s),
                       "last": round(float(s.iloc[-1]), 2), "color": ORANGE})
    if not panels:
        return None
    return {
        "key": "macro_panel", "type": "small_multiple",
        "title": "Fiscal and monetary plumbing",
        "subtitle": "The four series the repression thesis actually rests on.",
        "panels": panels,
    }


# ─────────────────────────────────── the pack ───────────────────────────────
BUILDERS = {
    "regime_gauge": regime_gauge,
    "returns_heatmap": returns_heatmap,
    "cross_asset_performance": cross_asset_performance,
    "normalised_performance": normalised_performance,
    "net_liquidity": net_liquidity,
    "real_yield_gold": real_yield_gold,
    "rolling_correlations": rolling_correlations,
    "correlation_heatmap": correlation_heatmap,
    "fx_performance": fx_performance,
    "global_equities": global_equities,
    "credit_spreads": credit_spreads,
    "yield_curve": yield_curve,
    "ratios": ratios,
    "drawdowns": drawdowns,
    "macro_panel": macro_panel,
}


def build_pack() -> dict[str, dict]:
    """Every chart as JSON. A failure skips that chart rather than the cycle."""
    wide = stats.load_daily(lookback_days=4000)
    if wide.empty:
        return {}
    out: dict[str, dict] = {}
    for key, fn in BUILDERS.items():
        try:
            spec = fn(wide)
            if spec:
                out[key] = spec
        except Exception as exc:  # noqa: BLE001
            log.warning("chart %s failed: %s", key, exc)
    log.info("built %d chart specs", len(out))
    return out


def save_pack(analysis_id: int, pack: dict) -> None:
    import json

    db.execute(
        """INSERT INTO chart_packs (analysis_id, payload) VALUES (%s, %s)
           ON CONFLICT (analysis_id) DO UPDATE SET payload = EXCLUDED.payload""",
        (analysis_id, json.dumps(pack, default=str)),
    )


def load_pack(analysis_id: int) -> dict:
    row = db.one("SELECT payload FROM chart_packs WHERE analysis_id=%s", (analysis_id,))
    return row["payload"] if row else {}


def latest_pack() -> dict:
    """Freshest pack, for pages that are not tied to one digest."""
    row = db.one("SELECT payload FROM chart_packs ORDER BY created_at DESC LIMIT 1")
    return row["payload"] if row else {}


def in_display_order(pack: dict) -> list[dict]:
    """Specs ordered as BUILDERS declares them.

    Postgres JSONB stores object keys sorted by length then bytewise, so a pack
    round-tripped through the database comes back in an order that has nothing to
    do with how the charts should be read — the regime gauge, which is the
    summary figure, lands somewhere in the middle.
    """
    ordered = [pack[k] for k in BUILDERS if k in pack]
    extra = [v for k, v in pack.items() if k not in BUILDERS]
    return ordered + extra
