"""Deterministic statistics. No LLM anywhere in this module.

This is where the *numerical* pattern finding happens: rolling correlations and
their regime flips, z-scored moves, cross-asset ratios, divergence from
historical relationships, realised vol, and lead/lag hints. The output — one
JSON "stats pack" — is what grounds the model's semantic reasoning later, so
that correlations are measured rather than imagined.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import db

log = logging.getLogger("mia.stats")

WINDOWS = (30, 90, 180)
CORE = ["GOLD", "SILVER", "BTC", "DXY", "US10Y", "US2Y", "US30Y", "SPX", "OIL", "ETH", "COPPER"]
RATIOS = {
    "gold_silver": ("GOLD", "SILVER"),
    "gold_spx": ("GOLD", "SPX"),
    "btc_gold": ("BTC", "GOLD"),
    "gold_oil": ("GOLD", "OIL"),
    "copper_gold": ("COPPER", "GOLD"),
}


# ───────────────────────────────── data loading ─────────────────────────────
def load_daily(symbols: list[str] | None = None, lookback_days: int = 800) -> pd.DataFrame:
    """Wide frame of daily closes, forward-filled onto a common calendar."""
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = db.query(
        """SELECT symbol, ts::date AS d, price FROM prices
           WHERE grain='1d' AND ts >= %s
           ORDER BY symbol, d""",
        (since,),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="d", columns="symbol", values="price", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().ffill()
    if symbols:
        wide = wide[[s for s in symbols if s in wide.columns]]
    return wide


def latest_intraday() -> dict[str, dict]:
    """Most recent 15m print per symbol, with the prior session close for context."""
    rows = db.query(
        """SELECT DISTINCT ON (symbol) symbol, ts, price
           FROM prices WHERE grain='15m' AND ts > now() - interval '3 days'
           ORDER BY symbol, ts DESC"""
    )
    return {r["symbol"]: {"ts": r["ts"], "price": float(r["price"])} for r in rows}


# ───────────────────────────────── primitives ───────────────────────────────
def pct_change(series: pd.Series, periods: int = 1) -> float | None:
    s = series.dropna()
    if len(s) <= periods:
        return None
    prev, last = s.iloc[-1 - periods], s.iloc[-1]
    if prev == 0:
        return None
    return float((last - prev) / prev * 100)


def zscore_of_last_move(series: pd.Series, window: int = 90) -> float | None:
    """How unusual is today's return against the last `window` daily returns?"""
    rets = series.dropna().pct_change().dropna()
    if len(rets) < 20:
        return None
    hist = rets.iloc[-window:]
    sd = hist.std()
    if not sd or sd == 0 or np.isnan(sd):
        return None
    return float((rets.iloc[-1] - hist.mean()) / sd)


def realised_vol(series: pd.Series, window: int = 30) -> float | None:
    rets = series.dropna().pct_change().dropna()
    if len(rets) < window:
        return None
    return float(rets.iloc[-window:].std() * np.sqrt(252) * 100)


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> float | None:
    """Correlation of daily *returns* — never of levels (levels give spurious ~1.0)."""
    ra, rb = a.pct_change(), b.pct_change()
    joint = pd.concat([ra, rb], axis=1).dropna()
    if len(joint) < window:
        return None
    val = joint.iloc[-window:, 0].corr(joint.iloc[-window:, 1])
    return None if val is None or np.isnan(val) else float(val)


def corr_series(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    ra, rb = a.pct_change(), b.pct_change()
    joint = pd.concat([ra, rb], axis=1).dropna()
    if len(joint) < window + 5:
        return pd.Series(dtype=float)
    return joint.iloc[:, 0].rolling(window).corr(joint.iloc[:, 1]).dropna()


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int = 5, window: int = 120) -> dict | None:
    """Which lag of `a` best explains `b`? Positive lag => a leads b."""
    ra, rb = a.pct_change(), b.pct_change()
    joint = pd.concat([ra, rb], axis=1).dropna().iloc[-window:]
    if len(joint) < 40:
        return None
    best = {"lag": 0, "corr": 0.0}
    for lag in range(-max_lag, max_lag + 1):
        shifted = joint.iloc[:, 0].shift(lag)
        val = shifted.corr(joint.iloc[:, 1])
        if val is not None and not np.isnan(val) and abs(val) > abs(best["corr"]):
            best = {"lag": int(lag), "corr": float(val)}
    return best if abs(best["corr"]) > 0.15 else None


# ───────────────────────────── composite detectors ──────────────────────────
def correlation_flips(wide: pd.DataFrame, window: int = 30, lookback: int = 45) -> list[dict]:
    """Pairs whose 30d return-correlation changed sign or moved sharply.

    This is the single highest-value signal in the pack: a relationship that
    *used to hold and stopped* is usually where the real macro story is.
    """
    out = []
    cols = [c for c in CORE if c in wide.columns]
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            cs = corr_series(wide[a], wide[b], window)
            if len(cs) < lookback + 1:
                continue
            now_c, then_c = float(cs.iloc[-1]), float(cs.iloc[-lookback])
            delta = now_c - then_c
            flipped = (now_c > 0.15 and then_c < -0.15) or (now_c < -0.15 and then_c > 0.15)
            if flipped or abs(delta) > 0.55:
                out.append(
                    {
                        "pair": f"{a}/{b}",
                        "corr_now": round(now_c, 3),
                        "corr_prior": round(then_c, 3),
                        "delta": round(delta, 3),
                        "sign_flip": bool(flipped),
                        "window_days": window,
                        "compared_to_days_ago": lookback,
                    }
                )
    return sorted(out, key=lambda d: -abs(d["delta"]))[:10]


def ratio_divergence(wide: pd.DataFrame) -> list[dict]:
    """Cross-asset ratios vs their own 1-year history, in z-score terms."""
    out = []
    for name, (a, b) in RATIOS.items():
        if a not in wide.columns or b not in wide.columns:
            continue
        ratio = (wide[a] / wide[b]).dropna()
        if len(ratio) < 90:
            continue
        hist = ratio.iloc[-252:] if len(ratio) >= 252 else ratio
        sd = hist.std()
        z = float((ratio.iloc[-1] - hist.mean()) / sd) if sd else None
        out.append(
            {
                "ratio": name,
                "value": round(float(ratio.iloc[-1]), 4),
                "z_vs_1y": round(z, 2) if z is not None else None,
                "pct_1w": round(pct_change(ratio, 5) or 0, 2),
                "pct_1m": round(pct_change(ratio, 21) or 0, 2),
                "pct_3m": round(pct_change(ratio, 63) or 0, 2),
            }
        )
    return out


def performance_table(wide: pd.DataFrame) -> list[dict]:
    out = []
    for sym in [c for c in CORE if c in wide.columns]:
        s = wide[sym].dropna()
        if s.empty:
            continue
        is_rate = sym.startswith("US") and sym.endswith("Y")
        # For yields, absolute basis-point change is the meaningful unit.
        def bp(days: int) -> float | None:
            if len(s) <= days:
                return None
            return round(float((s.iloc[-1] - s.iloc[-1 - days]) * 100), 1)

        row = {
            "symbol": sym,
            "last": round(float(s.iloc[-1]), 4),
            "unit": "percent_yield" if is_rate else "usd",
            "z_1d_move": round(zscore_of_last_move(s) or 0, 2),
            "realised_vol_30d": round(realised_vol(s) or 0, 1),
        }
        if is_rate:
            # Percent change of a yield is a meaningless quantity to a reader and
            # is easily misread as basis points (a -1.02% change in a 4.7% yield
            # is -4.8bp, not -102bp). Rates therefore carry basis points only.
            row |= {
                "chg_1d_bp": bp(1),
                "chg_1w_bp": bp(5),
                "chg_1m_bp": bp(21),
                "chg_1y_bp": bp(252),
            }
        else:
            row |= {
                "chg_1d_pct": round(pct_change(s, 1) or 0, 2),
                "chg_1w_pct": round(pct_change(s, 5) or 0, 2),
                "chg_1m_pct": round(pct_change(s, 21) or 0, 2),
                "chg_3m_pct": round(pct_change(s, 63) or 0, 2),
                "chg_1y_pct": round(pct_change(s, 252) or 0, 2),
            }
        out.append(row)
    return out


def correlation_matrix(wide: pd.DataFrame) -> dict:
    cols = [c for c in CORE if c in wide.columns]
    matrix: dict[str, dict] = {}
    for w in WINDOWS:
        m = {}
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                c = rolling_corr(wide[a], wide[b], w)
                if c is not None:
                    m[f"{a}/{b}"] = round(c, 3)
        matrix[f"{w}d"] = m
    return matrix


def anomalies(wide: pd.DataFrame, perf: list[dict]) -> list[str]:
    """Plain-language flags. These are *observations*, never explanations."""
    notes = []
    for row in perf:
        z = row.get("z_1d_move") or 0
        if abs(z) >= 2.0:
            if row.get("chg_1d_bp") is not None:
                move = f"{row['chg_1d_bp']:+.1f}bp"
            else:
                move = f"{row.get('chg_1d_pct', 0):+.2f}%"
            notes.append(
                f"{row['symbol']} 1-day move is {z:+.1f} sigma vs its 90-day "
                f"distribution ({move})"
            )
    gs = next((r for r in ratio_divergence(wide) if r["ratio"] == "gold_silver"), None)
    if gs and gs.get("z_vs_1y") is not None and abs(gs["z_vs_1y"]) > 1.5:
        direction = "stretched high" if gs["z_vs_1y"] > 0 else "compressed low"
        notes.append(
            f"gold/silver ratio {direction} at {gs['value']:.1f} "
            f"({gs['z_vs_1y']:+.1f} sigma vs 1y)"
        )
    return notes


# ─────────────────────────────── macro context ──────────────────────────────
def gold_in_currencies(wide: pd.DataFrame) -> dict:
    """Gold priced in non-USD currencies.

    The debasement framework's cleanest test is whether gold is at records in
    several currencies at once (systemic fiat debasement) or only in USD (a
    dollar story). Derived from existing series — no extra data source needed.
    """
    if "GOLD" not in wide.columns:
        return {}
    gold = wide["GOLD"].dropna()
    out: dict = {}

    variants = {"USD": gold}
    if "EURUSD" in wide.columns:
        variants["EUR"] = (gold / wide["EURUSD"]).dropna()
    if "USDJPY" in wide.columns:
        variants["JPY"] = (gold * wide["USDJPY"]).dropna()
    if "DXY" in wide.columns:
        # Gold deflated by the broad dollar — a trade-weighted real-terms proxy.
        variants["DXY_adj"] = (gold * wide["DXY"] / 100).dropna()

    for name, series in variants.items():
        if len(series) < 60:
            continue
        window = series.iloc[-252:] if len(series) >= 252 else series
        peak = float(window.max())
        last = float(series.iloc[-1])
        out[name] = {
            "last": round(last, 2),
            "pct_1m": round(pct_change(series, 21) or 0, 2),
            "pct_from_1y_high": round((last - peak) / peak * 100, 2),
            "at_1y_high": bool(last >= peak * 0.999),
        }
    highs = [k for k, v in out.items() if v.get("at_1y_high")]
    out["_reading"] = (
        "systemic fiat debasement signature (gold at 1y highs in multiple currencies)"
        if len(highs) >= 2
        else ("dollar-specific move" if highs == ["USD"] else "no currency at 1y high")
    )
    return out


def macro_snapshot() -> dict:
    """Latest FRED levels plus their own recent deltas."""
    ids = [
        "WALCL", "M2SL", "GFDEBTN", "RRPONTSYD", "T10YIE",
        "DFII10", "DTWEXBGS", "WTREGEN", "CPIAUCSL", "UNRATE", "FEDFUNDS",
        # Term premium decomposition: without it you cannot tell "buybacks are
        # suppressing yields against a rising term premium" from "buybacks are
        # doing nothing" — and that distinction is the repression thesis.
        "THREEFYTP10", "THREEFYTP5", "DFII5", "T5YIE", "SOFR", "DGS20",
    ]
    out = {}
    for sid in ids:
        rows = db.query(
            "SELECT ts, value FROM fred_series WHERE series_id=%s ORDER BY ts DESC LIMIT 60",
            (sid,),
        )
        if not rows:
            continue
        latest = rows[0]
        entry = {"value": float(latest["value"]), "as_of": latest["ts"].isoformat()}
        if len(rows) > 1:
            prev = float(rows[min(4, len(rows) - 1)]["value"])
            if prev:
                entry["chg_recent_pct"] = round((entry["value"] - prev) / abs(prev) * 100, 2)
        out[sid] = entry
    return out


def yield_curve() -> dict:
    """2s10s and 2s30s in basis points — regime-defining for the debt cycle view."""
    def last(sym: str) -> float | None:
        r = db.one(
            "SELECT price FROM prices WHERE symbol=%s AND grain='1d' ORDER BY ts DESC LIMIT 1",
            (sym,),
        )
        return float(r["price"]) if r else None

    y2, y10, y30 = last("US2Y"), last("US10Y"), last("US30Y")
    out: dict = {"US2Y": y2, "US10Y": y10, "US30Y": y30}
    if y2 and y10:
        out["2s10s_bp"] = round((y10 - y2) * 100, 1)
        out["curve_state"] = "inverted" if y10 < y2 else "positive"
    if y2 and y30:
        out["2s30s_bp"] = round((y30 - y2) * 100, 1)
    return out


# ─────────────────────────────────── the pack ───────────────────────────────
def build(persist: bool = True) -> dict:
    wide = load_daily()
    if wide.empty:
        raise RuntimeError("no daily price data — run `mia backfill` first")

    perf = performance_table(wide)
    pack = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "symbols": list(wide.columns),
            "first_date": str(wide.index.min().date()),
            "last_date": str(wide.index.max().date()),
            "rows": int(len(wide)),
        },
        "performance": perf,
        "correlations": correlation_matrix(wide),
        "correlation_flips": correlation_flips(wide),
        "ratios": ratio_divergence(wide),
        "gold_in_currencies": gold_in_currencies(wide),
        "yield_curve": yield_curve(),
        "macro": macro_snapshot(),
        "anomalies": anomalies(wide, perf),
        "lead_lag": {
            f"{a}->{b}": lead_lag(wide[a], wide[b])
            for a, b in [("DXY", "GOLD"), ("US10Y", "GOLD"), ("BTC", "GOLD"), ("GOLD", "SILVER")]
            if a in wide.columns and b in wide.columns
        },
        "intraday_latest": {
            k: {"price": v["price"], "ts": v["ts"].isoformat()}
            for k, v in latest_intraday().items()
        },
    }
    if persist:
        import json

        db.execute("INSERT INTO stats_packs (payload) VALUES (%s)", (json.dumps(pack, default=str),))
    return pack


def latest_pack() -> dict | None:
    row = db.one("SELECT payload, created_at FROM stats_packs ORDER BY created_at DESC LIMIT 1")
    return row["payload"] if row else None
