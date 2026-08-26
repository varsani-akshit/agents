"""Derived macro state: liquidity, a scored regime reading, and historical analogues.

Everything here is arithmetic on stored series. No LLM. The point is to hand the
model quantities it cannot eyeball from a price table — net liquidity is three
series differenced with unit conversions, the regime score is a weighted vote of
eight components, and the analogue engine is a nearest-neighbour search over
several hundred trading days. A model asked to "consider liquidity" without these
will confabulate; given them it has something to reason about.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import db
from signals import stats

log = logging.getLogger("mia.regime")


# ─────────────────────────────── FRED helpers ───────────────────────────────
def fred_frame(series_id: str, days: int = 1200) -> pd.Series:
    rows = db.query(
        """SELECT ts, value FROM fred_series
           WHERE series_id=%s AND ts > now() - make_interval(days => %s)
           ORDER BY ts""",
        (series_id, days),
    )
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(
        [float(r["value"]) for r in rows],
        index=pd.to_datetime([r["ts"] for r in rows]),
        name=series_id,
    )


def _z(series: pd.Series, window: int = 252) -> float | None:
    s = series.dropna()
    if len(s) < 30:
        return None
    hist = s.iloc[-window:]
    sd = hist.std()
    if not sd:
        return None
    return float((s.iloc[-1] - hist.mean()) / sd)


# ─────────────────────────────── net liquidity ──────────────────────────────
def net_liquidity(days: int = 500) -> dict:
    """Fed balance sheet less the Treasury's cash box and the RRP facility.

    WALCL on its own is not the liquidity that reaches markets: cash parked in
    the TGA or the reverse repo facility is drained from the banking system even
    though the balance sheet still carries it. The difference is the series that
    actually tracks the debasement complex.

    Unit trap: FRED publishes WALCL and WTREGEN in millions and RRPONTSYD in
    billions. Adding them raw understates the RRP drain by a factor of 1000.
    """
    walcl = fred_frame("WALCL", days)
    tga = fred_frame("WTREGEN", days)
    rrp = fred_frame("RRPONTSYD", days)
    if walcl.empty:
        return {}

    frame = pd.concat(
        {"walcl": walcl, "tga": tga, "rrp_bn": rrp}, axis=1, sort=True
    ).sort_index().ffill().dropna(subset=["walcl", "tga"])
    frame["rrp"] = frame["rrp_bn"] * 1000  # billions -> millions
    frame["net"] = frame["walcl"] - frame["tga"] - frame["rrp"].fillna(0)

    # Resample to the Wednesday grid the Fed actually publishes on. WALCL and
    # WTREGEN are weekly; only RRP is daily. Carrying the pair forward across a
    # week and calling the result a daily series is fake precision — it invents
    # day-to-day movement in two of the three inputs that was never measured.
    net = frame["net"].resample("W-WED").last().dropna()
    if len(net) < 12:
        return {}

    def chg(weeks: int) -> float | None:
        if len(net) <= weeks:
            return None
        return round(float(net.iloc[-1] - net.iloc[-1 - weeks]) / 1000, 1)  # $bn

    return {
        "net_liquidity_usd_bn": round(float(net.iloc[-1]) / 1000, 1),
        "as_of": str(net.index[-1].date()),
        "frequency": "weekly (Wednesday), matching the Fed H.4.1 release",
        "chg_1w_bn": chg(1),
        "chg_1m_bn": chg(4),
        "chg_3m_bn": chg(13),
        "z_vs_1y": round(_z(net, window=52) or 0, 2),
        "components_usd_bn": {
            "fed_assets": round(float(frame["walcl"].iloc[-1]) / 1000, 1),
            "treasury_general_account": round(float(frame["tga"].iloc[-1]) / 1000, 1)
            if not np.isnan(frame["tga"].iloc[-1]) else None,
            "reverse_repo": round(float(frame["rrp"].iloc[-1]) / 1000, 1)
            if not np.isnan(frame["rrp"].iloc[-1]) else None,
        },
        "note": "net = Fed assets - TGA - RRP; rising is easier financial conditions",
    }


def credit_conditions() -> dict:
    """Spreads and financial-conditions indices with their own 1-year z-scores.

    A spread level means nothing without its distribution: 3.2% high-yield is
    calm in 2009 terms and tight in 2021 terms.
    """
    out: dict = {}
    spec = {
        "BAMLH0A0HYM2": "us_high_yield_spread_pct",
        "BAMLC0A0CM": "us_investment_grade_spread_pct",
        "BAMLEMCBPIOAS": "em_corporate_spread_pct",
        "NFCI": "chicago_fed_financial_conditions",
    }
    for sid, label in spec.items():
        s = fred_frame(sid)
        if s.empty:
            continue
        out[label] = {
            "value": round(float(s.iloc[-1]), 3),
            "z_vs_1y": round(_z(s) or 0, 2),
            "chg_1m": round(float(s.iloc[-1] - s.iloc[-22]), 3) if len(s) > 22 else None,
            "as_of": str(s.index[-1].date()),
        }
    if "chicago_fed_financial_conditions" in out:
        v = out["chicago_fed_financial_conditions"]["value"]
        out["chicago_fed_financial_conditions"]["reading"] = (
            "looser than average" if v < 0 else "tighter than average"
        )
    return out


# ──────────────────────────────── regime score ──────────────────────────────
# Each component votes in [-1, +1]. Positive means "debasement / hard-asset
# regime"; negative means "cash and duration are being rewarded". The weights
# are judgement, but they are fixed, visible, and identical every cycle — which
# is what makes the score comparable across digests rather than a vibe.
COMPONENTS = (
    ("gold_vs_spx_3m", 0.18),
    ("real_yield_10y", 0.16),
    ("net_liquidity_trend", 0.14),
    ("dollar_trend", 0.12),
    ("crypto_vs_equity_3m", 0.12),
    ("term_premium", 0.10),
    ("credit_stress", 0.10),
    ("copper_gold", 0.08),
)


def _clip(x: float, scale: float) -> float:
    return float(np.clip(x / scale, -1.0, 1.0))


def regime_score(wide: pd.DataFrame | None = None) -> dict:
    """A single number for the regime, decomposed into its votes.

    The model is asked to name a regime every cycle. Without an anchor that
    label drifts with the news flow. This makes the call auditable: the score is
    reproducible from the components, and the components are reproducible from
    the series.
    """
    wide = stats.load_daily() if wide is None else wide
    if wide.empty:
        return {}
    votes: dict[str, dict] = {}

    def vote(name: str, value: float | None, detail: str) -> None:
        if value is not None:
            votes[name] = {"vote": round(float(value), 3), "basis": detail}

    # Hard assets versus equities over a quarter.
    if {"GOLD", "SPX"} <= set(wide.columns):
        g, s = stats.pct_change(wide["GOLD"], 63), stats.pct_change(wide["SPX"], 63)
        if g is not None and s is not None:
            vote("gold_vs_spx_3m", _clip(g - s, 15.0),
                 f"gold {g:+.1f}% vs SPX {s:+.1f}% over 3m")

    # Negative real yields are the definition of repression; positive ones argue
    # against it however loud the fiscal noise.
    dfii = fred_frame("DFII10")
    if not dfii.empty:
        v = float(dfii.iloc[-1])
        vote("real_yield_10y", _clip(-v, 2.0), f"10y TIPS real yield {v:.2f}%")

    nl = net_liquidity()
    if nl.get("chg_3m_bn") is not None:
        vote("net_liquidity_trend", _clip(nl["chg_3m_bn"], 400.0),
             f"net liquidity {nl['chg_3m_bn']:+.0f}bn over 3m")

    if "DXY" in wide.columns:
        d = stats.pct_change(wide["DXY"], 63)
        if d is not None:
            vote("dollar_trend", _clip(-d, 5.0), f"DXY {d:+.1f}% over 3m")

    if {"BTC", "SPX"} <= set(wide.columns):
        b, s = stats.pct_change(wide["BTC"], 63), stats.pct_change(wide["SPX"], 63)
        if b is not None and s is not None:
            vote("crypto_vs_equity_3m", _clip(b - s, 30.0),
                 f"BTC {b:+.1f}% vs SPX {s:+.1f}% over 3m")

    tp = fred_frame("THREEFYTP10")
    if not tp.empty:
        v = float(tp.iloc[-1])
        # A rising term premium is the market charging for fiscal risk.
        vote("term_premium", _clip(v, 1.5), f"ACM 10y term premium {v:.2f}")

    hy = fred_frame("BAMLH0A0HYM2")
    if not hy.empty:
        z = _z(hy) or 0.0
        # Widening credit is a deflationary shock, which cuts against the
        # debasement trade even though both are "bad macro".
        vote("credit_stress", _clip(-z, 2.0), f"HY spread {float(hy.iloc[-1]):.2f}% ({z:+.1f}σ)")

    if {"COPPER", "GOLD"} <= set(wide.columns):
        ratio = (wide["COPPER"] / wide["GOLD"]).dropna()
        r = stats.pct_change(ratio, 63)
        if r is not None:
            # Copper losing to gold = monetary demand beating industrial demand.
            vote("copper_gold", _clip(-r, 15.0), f"copper/gold {r:+.1f}% over 3m")

    if not votes:
        return {}
    total_w = sum(w for k, w in COMPONENTS if k in votes)
    score = sum(votes[k]["vote"] * w for k, w in COMPONENTS if k in votes) / (total_w or 1)

    if score > 0.45:
        label = "debasement — hard assets rewarded, fiat and cash penalised"
    elif score > 0.15:
        label = "tilting to debasement — hard assets leading but not uncontested"
    elif score > -0.15:
        label = "mixed — no dominant monetary regime in the price data"
    elif score > -0.45:
        label = "tilting to disinflation — cash and duration competitive"
    else:
        label = "disinflation / tight money — real assets penalised"

    return {
        "score": round(float(score), 3),
        "scale": "-1 (tight money, real assets penalised) to +1 (debasement)",
        "label": label,
        "components": votes,
        "weights": {k: w for k, w in COMPONENTS if k in votes},
    }


# ───────────────────────────── historical analogues ─────────────────────────
FEATURES = (
    "gold_silver_z", "copper_gold_z", "curve_2s10s", "dxy_3m", "spx_3m",
    "gold_3m", "vix_level", "corr_gold_rates", "corr_btc_spx",
)


def _feature_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-date feature matrix, each column using only data up to that date."""
    f = pd.DataFrame(index=wide.index)
    have = set(wide.columns)

    def roll_z(series: pd.Series, window: int = 252) -> pd.Series:
        m, sd = series.rolling(window, min_periods=60).mean(), series.rolling(
            window, min_periods=60).std()
        return (series - m) / sd

    if {"GOLD", "SILVER"} <= have:
        f["gold_silver_z"] = roll_z(wide["GOLD"] / wide["SILVER"])
    if {"COPPER", "GOLD"} <= have:
        f["copper_gold_z"] = roll_z(wide["COPPER"] / wide["GOLD"])
    if {"US10Y", "US2Y"} <= have:
        f["curve_2s10s"] = (wide["US10Y"] - wide["US2Y"]) * 100
    if "DXY" in have:
        f["dxy_3m"] = wide["DXY"].pct_change(63) * 100
    if "SPX" in have:
        f["spx_3m"] = wide["SPX"].pct_change(63) * 100
    if "GOLD" in have:
        f["gold_3m"] = wide["GOLD"].pct_change(63) * 100
    if "VIX" in have:
        f["vix_level"] = wide["VIX"]
    if {"GOLD", "US10Y"} <= have:
        f["corr_gold_rates"] = (
            wide["GOLD"].pct_change().rolling(30).corr(wide["US10Y"].pct_change()))
    if {"BTC", "SPX"} <= have:
        f["corr_btc_spx"] = (
            wide["BTC"].pct_change().rolling(30).corr(wide["SPX"].pct_change()))
    return f.dropna(how="all")


def historical_analogues(
    wide: pd.DataFrame | None = None,
    top_n: int = 4,
    min_separation_days: int = 45,
    exclude_recent_days: int = 60,
) -> dict:
    """Past dates whose macro fingerprint most resembles today, and what followed.

    Deliberately modest in its claims. With a few years of daily history this
    finds *rhymes*, not laws, and the sample size is reported alongside every
    result so the model can weight it honestly. Matches are forced apart by
    `min_separation_days` because adjacent days are near-duplicates and would
    otherwise fill the whole list with one episode.
    """
    wide = stats.load_daily(lookback_days=4000) if wide is None else wide
    if wide.empty or len(wide) < 200:
        return {"available": False, "reason": "insufficient history"}

    feats = _feature_frame(wide)
    cols = [c for c in FEATURES if c in feats.columns]
    if len(cols) < 4:
        return {"available": False, "reason": "insufficient feature coverage"}

    # Drop thin features before aligning. A single short series would otherwise
    # decide the sample: joining a 12-year frame to one 300-day column and then
    # dropping NaNs leaves 300 rows, and the search quietly becomes a search over
    # the last year — while still reporting itself as a historical analogue.
    coverage = feats[cols].notna().sum()
    keep = [c for c in cols if coverage[c] >= 0.5 * coverage.max()]
    dropped = sorted(set(cols) - set(keep))

    matrix = feats[keep].dropna()
    if len(matrix) < 150:
        return {"available": False, "reason": f"only {len(matrix)} usable days"}

    # Standardise so basis points and correlations carry comparable weight.
    norm = (matrix - matrix.mean()) / matrix.std().replace(0, np.nan)
    norm = norm.dropna(axis=1, how="all").dropna()
    today = norm.iloc[-1]
    cutoff = norm.index[-1] - pd.Timedelta(days=exclude_recent_days)
    candidates = norm.loc[:cutoff]
    if len(candidates) < 100:
        return {"available": False, "reason": "not enough history before the exclusion window"}

    dist = ((candidates - today) ** 2).sum(axis=1) ** 0.5
    picks: list[pd.Timestamp] = []
    for ts in dist.sort_values().index:
        if all(abs((ts - p).days) >= min_separation_days for p in picks):
            picks.append(ts)
        if len(picks) >= top_n:
            break

    forward_syms = [s for s in ("GOLD", "SILVER", "BTC", "SPX", "TLT", "DXY", "OIL")
                    if s in wide.columns]
    matches = []
    for ts in picks:
        entry = {
            "date": str(ts.date()),
            "distance": round(float(dist.loc[ts]), 2),
            "conditions": {c: round(float(matrix.loc[ts, c]), 2) for c in norm.columns},
            "forward_returns_pct": {},
        }
        pos = wide.index.get_loc(ts)
        for horizon, label in ((21, "1m"), (63, "3m")):
            if pos + horizon >= len(wide):
                continue
            for sym in forward_syms:
                series = wide[sym]
                a, b = series.iloc[pos], series.iloc[pos + horizon]
                if pd.isna(a) or pd.isna(b) or a == 0:
                    continue
                entry["forward_returns_pct"].setdefault(label, {})[sym] = round(
                    float((b - a) / a * 100), 1)
        matches.append(entry)

    # Median forward return across matches — the only aggregate worth quoting.
    summary: dict[str, dict] = {}
    for label in ("1m", "3m"):
        vals: dict[str, list[float]] = {}
        for m in matches:
            for sym, v in (m["forward_returns_pct"].get(label) or {}).items():
                vals.setdefault(sym, []).append(v)
        if vals:
            summary[label] = {
                sym: {"median_pct": round(float(np.median(v)), 1), "n": len(v)}
                for sym, v in sorted(vals.items())
            }

    return {
        "available": True,
        "today": {c: round(float(matrix.iloc[-1][c]), 2) for c in norm.columns},
        "features_used": list(norm.columns),
        "history_days": int(len(matrix)),
        "history_from": str(matrix.index[0].date()),
        "matches": matches,
        "median_forward_returns_pct": summary,
        "features_dropped_for_short_history": dropped,
        "caveat": (
            f"nearest-neighbour over {len(matrix)} trading days from "
            f"{matrix.index[0].date()}. {len(matches)} matches is a small sample: "
            "treat as a rhyme worth checking, never as a forecast or a base rate. "
            "Forward returns are unadjusted price changes, and a symbol missing "
            "from a match simply had no data at that date."
        ),
    }


def build() -> dict:
    """Everything in this module, as one block for the stats pack."""
    wide = stats.load_daily(lookback_days=4000)
    out: dict = {"generated_at": datetime.now(timezone.utc).isoformat()}
    for name, fn in (
        ("net_liquidity", lambda: net_liquidity()),
        ("credit_conditions", lambda: credit_conditions()),
        ("regime_score", lambda: regime_score(wide)),
        ("historical_analogues", lambda: historical_analogues(wide)),
    ):
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001
            log.warning("regime block %s failed: %s", name, exc)
            out[name] = {}
    return out
