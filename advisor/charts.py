"""Chart specs for the simulation pages: price, and the record behind a setup.

Same contract as signals/chartdata.py — plain JSON specs drawn in the browser
by web/static/charts.js — so hovering a candle reads the day's prices instead
of squinting at a picture.

Why a candle rather than the line used elsewhere: a setup is a statement about
where a day traded, not where it closed. "Buy at 243.41, stop 189.54" can only
be judged against the day's range — whether the limit sat inside recent bars or
above everything, how far the stop is in the body of the move. A close-only
line hides exactly the part being decided.

The specs are built on demand, one setup at a time, rather than for every
candidate in a run: a run holds a thousand, and nobody reads a thousand charts.
"""
from __future__ import annotations

import db
from advisor import bare, market_of
from advisor.strategies import STRATEGIES, indicators

# Enough history to see the setup in context, short enough to read. A year of
# daily bars at 3px a candle is about the width of the page.
CANDLE_DAYS = 180


def setup(symbol: str, strategy: str, *, levels: dict | None = None,
          days: int = CANDLE_DAYS) -> dict | None:
    """One name's recent price as candles, with the setup's levels drawn on.

    `levels` carries limit/stop/target when the caller already has a ticket;
    without it the strategy's own rules are read off the latest bar, which is
    what the scan would do today.
    """
    from advisor import bars

    df = bars.load(symbol, years=2)
    if df.empty or len(df) < 30:
        return None
    x = indicators(df)
    strat = STRATEGIES.get(strategy)
    if levels is None and strat is not None:
        try:
            sig = strat.rules(x).iloc[-1]
            levels = {"limit": float(sig["limit"]), "stop": float(sig["stop"]),
                      "target": None if sig["target"] != sig["target"] else float(sig["target"])}
        except Exception:  # noqa: BLE001 - a chart must not need the rules
            levels = None

    tail = x.tail(days)
    marks = []
    for key, label, kind in (("limit", "Buy limit", "entry"),
                             ("stop", "Stop", "stop"),
                             ("target", "Target", "target")):
        v = (levels or {}).get(key)
        if v is not None and v == v:  # not NaN
            marks.append({"key": kind, "label": label, "value": round(float(v), 2)})

    return {
        "type": "candle",
        "key": f"setup:{symbol}:{strategy}",
        "title": f"{bare(symbol)} — {strat.label if strat else strategy}",
        "subtitle": f"Daily bars, last {len(tail)} sessions. "
                    f"{market_of(symbol)}.",
        "x": [d.strftime("%Y-%m-%d") for d in tail.index],
        # ECharts candlesticks take [open, close, low, high], in that order.
        "candles": [[round(float(r.open), 4), round(float(r.close), 4),
                     round(float(r.low), 4), round(float(r.high), 4)]
                    for r in tail.itertuples()],
        "volume": [None if r.volume != r.volume else round(float(r.volume))
                   for r in tail.itertuples()],
        "overlays": [
            {"name": "50-day average",
             "data": [_num(v) for v in tail["sma50"]]},
            {"name": "200-day average",
             "data": [_num(v) for v in tail["sma200"]]},
        ],
        "marks": marks,
    }


def _num(v) -> float | None:
    return None if v != v else round(float(v), 4)


def record(strategy: str, region: str, run_id: int | None = None) -> dict | None:
    """What this strategy actually did: the shape of its results, not a verdict.

    Two panels of the same evidence the gate reads — how the recent period
    compares with the older one, and where the trades ended up. A pass/fail
    badge says a strategy works; this says how, and how reliably.
    """
    row = db.one(
        """SELECT stats FROM sim_results
           WHERE strategy=%s AND region=%s AND (%s::bigint IS NULL OR run_id=%s)
           ORDER BY run_id DESC LIMIT 1""",
        (strategy, region, run_id, run_id))
    if not row or not row["stats"]:
        return None
    st = row["stats"]
    periods = [(k, st.get(k) or {}) for k in ("older", "recent", "all")
               if isinstance(st.get(k), dict) and st[k].get("trades")]
    if not periods:
        return None

    return {
        "type": "diverging",
        "key": f"record:{strategy}:{region}",
        "title": f"{st.get('label') or strategy} — expectancy by period",
        "subtitle": "R per trade. One R is the distance from entry to stop, so "
                    "+0.1R means a tenth of what you risked, per trade.",
        "unit": "R",
        "categories": [{"older": "Older half", "recent": "Recent half",
                        "all": "All history"}.get(k, k) for k, _ in periods],
        "values": [round(float(p.get("expectancy_r") or 0), 3) for _, p in periods],
        "weights": [f"{int(p.get('trades') or 0):,} trades · "
                    f"{(p.get('win_rate') or 0) * 100:.0f}% won" for _, p in periods],
    }


def exits(strategy: str, region: str, run_id: int | None = None) -> dict | None:
    """How trades ended: stopped out, timed out, or the signal said leave.

    Worth its own figure because two strategies with the same expectancy and
    very different exit mixes are not the same strategy to live with.
    """
    row = db.one(
        """SELECT stats FROM sim_results
           WHERE strategy=%s AND region=%s AND (%s::bigint IS NULL OR run_id=%s)
           ORDER BY run_id DESC LIMIT 1""",
        (strategy, region, run_id, run_id))
    mix = (row or {}).get("stats", {}).get("exits") if row else None
    if not mix:
        return None
    label = {"stop": "Stopped out", "time": "Held to the limit", "signal": "Signal exit",
             "target": "Hit target"}
    items = sorted(mix.items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in items) or 1
    return {
        "type": "bar_h",
        "key": f"exits:{strategy}:{region}",
        "title": "How trades ended",
        "subtitle": f"{total:,} trades.",
        "unit": "%",
        "categories": [label.get(k, k) for k, _ in items],
        "values": [round(v / total * 100, 1) for _, v in items],
    }
