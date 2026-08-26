"""Replay historical prices through the trigger thresholds.

Purpose is calibration, not prediction: if a threshold would have fired 40 times
a day over the last two years, it is noise and will train the user to ignore
alerts. Target is roughly 0-3 Critical events per day.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from signals import stats, triggers


def replay(lookback_days: int = 730) -> dict:
    wide = stats.load_daily(lookback_days=lookback_days)
    if wide.empty:
        raise RuntimeError("no price history to backtest")

    rules = triggers.load_rules()
    price_specs = {s["symbol"]: s for s in rules.get("price_moves", [])}
    yield_specs = {s["symbol"]: s for s in rules.get("yield_moves", [])}
    zconf = rules.get("zscore", {})

    returns = wide.pct_change() * 100
    fired: list[tuple[pd.Timestamp, str, str, str]] = []

    for sym in wide.columns:
        r = returns[sym].dropna()
        if r.empty:
            continue

        spec = price_specs.get(sym)
        if spec:
            for ts, val in r.items():
                mag = abs(val)
                if mag >= spec.get("critical_pct", 1e9):
                    fired.append((ts, sym, "price_move", "Critical"))
                elif mag >= spec.get("high_pct", 1e9):
                    fired.append((ts, sym, "price_move", "High"))

        yspec = yield_specs.get(sym)
        if yspec:
            bp = wide[sym].diff() * 100
            for ts, val in bp.dropna().items():
                mag = abs(val)
                if mag >= yspec.get("critical_bp", 1e9):
                    fired.append((ts, sym, "yield_move", "Critical"))
                elif mag >= yspec.get("high_bp", 1e9):
                    fired.append((ts, sym, "yield_move", "High"))

        if sym in zconf.get("symbols", []):
            roll_mean = r.rolling(90).mean()
            roll_sd = r.rolling(90).std()
            z = ((r - roll_mean) / roll_sd).dropna()
            for ts, val in z.items():
                mag = abs(val)
                if mag >= zconf.get("critical", 1e9):
                    fired.append((ts, sym, "zscore_outlier", "Critical"))
                elif mag >= zconf.get("high", 1e9):
                    fired.append((ts, sym, "zscore_outlier", "High"))

    if not fired:
        return {"days": len(wide), "events": 0}

    df = pd.DataFrame(fired, columns=["ts", "symbol", "rule", "severity"])
    # One alert per symbol/rule/day — mirrors the live dedupe window.
    df = df.drop_duplicates(subset=["ts", "symbol", "rule", "severity"])
    days = max(len(wide), 1)
    crit = df[df.severity == "Critical"]
    high = df[df.severity == "High"]

    per_day_crit = crit.groupby(crit.ts.dt.date).size()
    per_day_all = df.groupby(df.ts.dt.date).size()

    return {
        "days": days,
        "trading_days_covered": int(wide.index.nunique()),
        "total_events": int(len(df)),
        "critical": int(len(crit)),
        "high": int(len(high)),
        "critical_per_day_mean": round(float(len(crit)) / days, 2),
        "all_per_day_mean": round(float(len(df)) / days, 2),
        "critical_busiest_day": (
            {"date": str(per_day_crit.idxmax()), "count": int(per_day_crit.max())}
            if len(per_day_crit)
            else None
        ),
        "days_with_zero_critical_pct": round(
            100 * (1 - len(per_day_crit) / days), 1
        ),
        "p95_alerts_in_a_day": (
            int(np.percentile(per_day_all.values, 95)) if len(per_day_all) else 0
        ),
        "by_rule": dict(Counter(df.rule)),
        "by_symbol_critical": dict(Counter(crit.symbol).most_common(10)),
    }


def report() -> str:
    r = replay()
    lines = [
        f"Backtest over {r['trading_days_covered']} trading days",
        f"  Critical:  {r['critical']:>4}  ({r['critical_per_day_mean']}/day)",
        f"  High:      {r['high']:>4}",
        f"  All rules: {r['total_events']:>4}  ({r['all_per_day_mean']}/day)",
        f"  Quiet days (no Critical): {r['days_with_zero_critical_pct']}%",
        f"  95th-percentile alerts in one day: {r['p95_alerts_in_a_day']}",
    ]
    if r.get("critical_busiest_day"):
        b = r["critical_busiest_day"]
        lines.append(f"  Busiest day: {b['date']} with {b['count']} Critical")
    lines.append(f"  By rule: {r['by_rule']}")
    lines.append(f"  Critical by symbol: {r['by_symbol_critical']}")
    return "\n".join(lines)
