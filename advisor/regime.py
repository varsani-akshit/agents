"""Conditions: what kind of world it is, and whether a strategy works in it.

A strategy that passes the gate on average may still lose money in some
conditions (when fear is high, when the dollar is rising), and one that fails
on average may work in some. This module measures that — honestly.

**The conditions.** Each market gets about ten dimensions, each a small set
of named states computed day by day from data that existed on that day:
the market's own trend, fear (VIX, India VIX), the dollar, US yields, credit
spreads, oil, the local currency, China, and the tone of world and local news
(GDELT, since 2017). Published-later data is lagged: FRED and GDELT by a day,
CFTC positioning by four.

**The rules are learnt on the older years and judged on the recent ones.**
For each strategy, every trade is labelled with the states on its signal
day. On the *older* period only, a state becomes a rule when trades in it
differ from trades outside it by more than two standard errors:

- *avoid* — the strategy lost money in this state;
- *favour* — it made clearly more in this state.

The rules are then applied, unchanged, to the *recent* period, which they
never saw. A rule set is used live only if it survives that:

- for a strategy that already passes: the filtered recent trades must do at
  least as well as the unfiltered ones and still pass the gate;
- for a strategy that fails: the recent trades taken only in its favoured
  states must pass the whole gate on their own, against random entries
  filtered the same way.

Twenty-odd states per strategy will throw up a pattern or two by chance; the
out-of-sample test is what stops those reaching a ticket.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

import db
from advisor import settings

log = logging.getLogger("alfred.regime")

MIN_LEARN = 40      # trades on each side of a state, older period, to form a rule
MIN_SEP = 2.0       # standard errors between trades in the state and the rest
MIN_TEST = 30       # recent trades a filter must set aside to count as tested
LOOKBACK = 63       # three months, for "rising / falling"
PCT_WINDOW = 756    # three years, for "high / low compared with recently"
STALE_DAYS = 14     # a state is carried over gaps this long, no longer


# ───────────────────────────────── raw series ───────────────────────────────
def _close(symbol: str) -> pd.Series:
    """Finished sessions only — a bar still trading is not a condition yet."""
    from advisor import bars

    df = bars.load(symbol)
    return df["close"].astype(float) if not df.empty else pd.Series(dtype=float)


def _fred(series_id: str) -> pd.Series:
    rows = db.query("SELECT ts, value FROM fred_series WHERE series_id = %s ORDER BY ts",
                    (series_id,))
    s = pd.Series([float(r["value"]) for r in rows],
                  index=pd.DatetimeIndex([r["ts"] for r in rows]), dtype=float)
    return _lag(s, 1)           # FRED posts most daily series the next morning


def _tone(query: str) -> pd.Series:
    rows = db.query("SELECT d, tone FROM tone WHERE query = %s ORDER BY d", (query,))
    s = pd.Series([float(r["tone"]) for r in rows],
                  index=pd.DatetimeIndex([r["d"] for r in rows]), dtype=float)
    return _lag(s, 1)


def _cftc(market: str) -> pd.Series:
    """Speculators' net position as a share of open interest. Reported as of
    Tuesday, published Friday: usable from the following Monday at best."""
    rows = db.query("""SELECT report_date, longs, shorts, open_interest FROM positioning
                       WHERE market = %s AND open_interest > 0 ORDER BY report_date""", (market,))
    s = pd.Series([(float(r["longs"]) - float(r["shorts"])) / float(r["open_interest"])
                   for r in rows], index=pd.DatetimeIndex([r["report_date"] for r in rows]),
                  dtype=float)
    return _lag(s, 4)


def _lag(s: pd.Series, days: int) -> pd.Series:
    if s.empty:
        return s
    out = s.copy()
    out.index = out.index + timedelta(days=days)
    return out


# ───────────────────────────────── states ───────────────────────────────────
def trend(s: pd.Series) -> pd.Series:
    sma = s.rolling(200, min_periods=200).mean()
    return pd.Series(np.where(s > sma, "up", "down"), index=s.index).where(sma.notna())


def direction(s: pd.Series, up: str, down: str, *, level: bool = False) -> pd.Series:
    """Rising or falling over three months. `level` compares differences
    (yields, spreads) rather than percentage changes."""
    ch = (s - s.shift(LOOKBACK)) if level else (s / s.shift(LOOKBACK) - 1)
    return pd.Series(np.where(ch > 0, up, down), index=s.index).where(ch.notna())


def thirds(s: pd.Series, low: str, mid: str, high: str) -> pd.Series:
    """Where today's value sits among the last three years'."""
    pct = s.rolling(PCT_WINDOW, min_periods=PCT_WINDOW // 2).rank(pct=True)
    return pd.Series(np.select([pct < 1 / 3, pct > 2 / 3], [low, high], mid),
                     index=s.index).where(pct.notna())


def mood(s: pd.Series, good: str, usual: str, bad: str) -> pd.Series:
    """News tone: the last month's average against the past year's."""
    m = s.rolling(30, min_periods=20).mean()
    base = m.rolling(365, min_periods=180)
    z = (m - base.mean()) / base.std()
    return pd.Series(np.select([z > 0.5, z < -0.5], [good, bad], usual),
                     index=s.index).where(z.notna())


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    build: Callable[[], pd.Series]
    states: dict[str, str]          # state → plain words for the page and the email


def _dims(region: str) -> list[Dimension]:
    index = {"IN": "^NSEI", "ASX": "^AXJO", "US": "^GSPC"}[region]
    vix = "^INDIAVIX" if region == "IN" else "^VIX"
    market = {"IN": "Nifty 50", "ASX": "ASX 200", "US": "S&P 500"}[region]
    fear = "India VIX" if region == "IN" else "VIX"
    dims = [
        Dimension("trend", "Market trend", lambda: trend(_close(index)),
                  {"up": f"{market} above its 200-day average",
                   "down": f"{market} below its 200-day average"}),
        Dimension("fear", "Fear", lambda: thirds(_close(vix), "calm", "normal", "fearful"),
                  {"calm": f"{fear} in its lowest third of three years",
                   "normal": f"{fear} in its middle third",
                   "fearful": f"{fear} in its highest third of three years"}),
        Dimension("dollar", "US dollar", lambda: direction(_close("DX-Y.NYB"), "rising", "falling"),
                  {"rising": "dollar up over three months", "falling": "dollar down over three months"}),
        Dimension("yields", "US 10-year yield",
                  lambda: direction(_close("^TNX"), "rising", "falling", level=True),
                  {"rising": "US 10-year yield up over three months",
                   "falling": "US 10-year yield down over three months"}),
        Dimension("credit", "Credit spreads",
                  lambda: direction(_fred("BAMLH0A0HYM2"), "widening", "tightening", level=True),
                  {"widening": "high-yield spreads wider than three months ago (stress rising)",
                   "tightening": "high-yield spreads tighter than three months ago"}),
        Dimension("oil", "Oil", lambda: direction(_close("CL=F"), "rising", "falling"),
                  {"rising": "oil up over three months", "falling": "oil down over three months"}),
        Dimension("world_news", "World news",
                  lambda: mood(_tone("geopolitics"), "calmer", "usual", "tenser"),
                  {"calmer": "war and sanctions coverage calmer than the past year",
                   "usual": "war and sanctions coverage as usual",
                   "tenser": "war and sanctions coverage tenser than the past year"}),
    ]
    if region == "IN":
        dims += [
            Dimension("currency", "Rupee",
                      lambda: direction(_close("INR=X"), "weakening", "strengthening"),
                      {"weakening": "rupee weaker against the dollar than three months ago",
                       "strengthening": "rupee stronger against the dollar than three months ago"}),
            Dimension("local_news", "India news",
                      lambda: mood(_tone("india_economy"), "upbeat", "usual", "gloomy"),
                      {"upbeat": "Indian economic news more upbeat than the past year",
                       "usual": "Indian economic news as usual",
                       "gloomy": "Indian economic news gloomier than the past year"}),
        ]
    if region == "ASX":
        dims += [
            Dimension("currency", "Australian dollar",
                      lambda: direction(_close("AUDUSD=X"), "strengthening", "weakening"),
                      {"strengthening": "Australian dollar up against the US dollar over three months",
                       "weakening": "Australian dollar down against the US dollar over three months"}),
            Dimension("china", "China", lambda: trend(_close("000001.SS")),
                      {"up": "Shanghai Composite above its 200-day average",
                       "down": "Shanghai Composite below its 200-day average"}),
            Dimension("local_news", "Australia news",
                      lambda: mood(_tone("australia_economy"), "upbeat", "usual", "gloomy"),
                      {"upbeat": "Australian economic news more upbeat than the past year",
                       "usual": "Australian economic news as usual",
                       "gloomy": "Australian economic news gloomier than the past year"}),
        ]
    if region == "US":
        dims += [
            Dimension("positioning", "Speculators",
                      lambda: thirds(_cftc("us_equity"), "short", "neutral", "crowded"),
                      {"short": "speculators' S&P futures position in its lowest third of three years",
                       "neutral": "speculators' S&P futures position in its middle third",
                       "crowded": "speculators' S&P futures position in its highest third (crowded long)"}),
            Dimension("local_news", "US news",
                      lambda: mood(_tone("us_economy"), "upbeat", "usual", "gloomy"),
                      {"upbeat": "US economic news more upbeat than the past year",
                       "usual": "US economic news as usual",
                       "gloomy": "US economic news gloomier than the past year"}),
        ]
    return dims


def dimensions(region: str) -> dict[str, Dimension]:
    return {d.key: d for d in _dims(region)}


@lru_cache(maxsize=8)
def _history_cached(region: str, stamp: str) -> pd.DataFrame:
    cols = {}
    for d in _dims(region):
        try:
            s = d.build()
        except Exception as exc:  # noqa: BLE001 - a missing feed drops one dimension
            log.warning("regime %s/%s unavailable: %s", region, d.key, exc)
            continue
        if s is not None and not s.dropna().empty:
            cols[d.key] = s
    if not cols:
        return pd.DataFrame()
    # One row per calendar day, carried forward: weekends, holidays and the
    # weekly CFTC report keep their last known state until the next one — but
    # for two weeks at most. A feed that has gone quiet drops out rather than
    # freezing a stale reading into today.
    start = min(s.dropna().index.min() for s in cols.values())
    end = max(s.index.max() for s in cols.values())
    days = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame({k: s[~s.index.duplicated(keep="last")].reindex(days).ffill(limit=STALE_DAYS)
                          for k, s in cols.items()})
    return frame


def history(region: str) -> pd.DataFrame:
    """Every dimension's state, day by day. Cached per region per hour."""
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H")
    return _history_cached(region, stamp)


def now(region: str) -> dict:
    """Today's conditions: each dimension's state, since when, in words."""
    h = history(region)
    if h.empty:
        return {"region": region, "as_of": None, "states": []}
    dims = dimensions(region)
    out, stale = [], []
    for key in h.columns:
        col = h[key]
        state = col.iloc[-1]
        if not isinstance(state, str):
            # No fresh reading: say so, rather than show an old one as today's.
            known = col.dropna()
            stale.append({"key": key, "label": dims[key].label,
                          "last": known.index[-1].date() if not known.empty else None})
            continue
        changed = col[col != state]
        since = (changed.index[-1] + timedelta(days=1)) if not changed.empty else col.index[0]
        out.append({"key": key, "label": dims[key].label, "state": state,
                    "words": dims[key].states.get(state, state), "since": since.date()})
    return {"region": region, "as_of": h.index[-1].date(), "states": out, "stale": stale}


# ─────────────────────────────── trade labelling ────────────────────────────
def label(dates: list[pd.Timestamp], states: pd.DataFrame) -> pd.DataFrame:
    """The state of every dimension on each date (the signal day)."""
    if states.empty or not dates:
        return pd.DataFrame(index=range(len(dates)))
    idx = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])
    # `states` already has every calendar day; a date past its end is unknown.
    return states.reindex(idx).reset_index(drop=True)


def _mean_se(a: np.ndarray) -> tuple[float, float]:
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else math.inf


def learn(rs: np.ndarray, labels: pd.DataFrame) -> dict:
    """Rules from one period's trades: states that clearly hurt or help.

    `rs` are the trades' R, `labels` their states (same order)."""
    avoid, favour = [], []
    for dim in labels.columns:
        col = labels[dim]
        known = col.notna().to_numpy()
        for state in sorted(col.dropna().unique()):
            inside = known & (col == state).to_numpy()
            outside = known & ~inside
            n_in, n_out = int(inside.sum()), int(outside.sum())
            if n_in < MIN_LEARN or n_out < MIN_LEARN:
                continue
            m_in, se_in = _mean_se(rs[inside])
            m_out, se_out = _mean_se(rs[outside])
            sep = (m_in - m_out) / math.sqrt(se_in ** 2 + se_out ** 2)
            rule = {"dim": dim, "state": state, "trades": n_in,
                    "expectancy_r": round(m_in, 4), "rest_expectancy_r": round(m_out, 4),
                    "separation": round(sep, 2)}
            if sep <= -MIN_SEP and m_in < 0:
                avoid.append(rule)
            elif sep >= MIN_SEP and m_in > 0:
                favour.append(rule)
    return {"avoid": avoid, "favour": favour}


def allowed(labels: pd.DataFrame, rules: dict, mode: str) -> np.ndarray:
    """Which labelled trades the rules let through.

    `filter`: everything except trades in an avoided state.
    `favour`: only trades in a favoured state, and none in an avoided one."""
    n = len(labels)
    blocked = np.zeros(n, dtype=bool)
    for r in rules.get("avoid", []):
        if r["dim"] in labels:
            blocked |= (labels[r["dim"]] == r["state"]).to_numpy()
    if mode == "filter":
        return ~blocked
    liked = np.zeros(n, dtype=bool)
    for r in rules.get("favour", []):
        if r["dim"] in labels:
            liked |= (labels[r["dim"]] == r["state"]).to_numpy()
    return liked & ~blocked


def condition(trades: list, base: list, cutoff: pd.Timestamp, passed: bool,
              states: pd.DataFrame) -> dict:
    """Learn rules on the older trades, test them on the recent ones."""
    from advisor.simulate import judge, monte_carlo, summarize

    if states.empty or not trades:
        return {"rules": {"avoid": [], "favour": []}, "mode": None, "use": False,
                "verdict": "no condition history yet"}
    rs = np.array([t.r for t in trades])
    lab = label([t.signal for t in trades], states)
    old = np.array([t.signal < cutoff for t in trades])
    rules = learn(rs[old], lab[old].reset_index(drop=True))
    mode = "filter" if passed else "favour"
    out = {"rules": rules, "mode": mode, "use": False}
    if (mode == "filter" and not rules["avoid"]) or (mode == "favour" and not rules["favour"]):
        out["verdict"] = ("no condition it clearly loses in" if passed
                          else "no condition it clearly works in")
        return out

    keep = allowed(lab, rules, mode)
    b_rs = np.array([t.r for t in base])
    b_lab = label([t.signal for t in base], states)
    b_keep = allowed(b_lab, rules, mode) if len(base) else np.zeros(0, dtype=bool)
    b_recent = np.array([t.signal >= cutoff for t in base], dtype=bool)

    before = summarize(list(rs[~old]))
    after = summarize(list(rs[~old & keep]))
    older_after = summarize(list(rs[old & keep]))
    baseline = summarize(list(b_rs[b_recent & b_keep])) if len(base) else {"trades": 0}
    mc = monte_carlo(list(rs[~old & keep]))
    ok, fails = judge(older_after, after, baseline, mc)
    if mode == "filter":
        # A rule that removed (almost) nothing on the recent years was not
        # tested at all — typically its data does not reach that far.
        removed = int((~old & ~keep).sum())
        helped = after.get("expectancy_r", -1) >= before.get("expectancy_r", 0)
        if removed < MIN_TEST:
            fails = [f"untested: the rule set aside only {removed} recent trades "
                     f"(needs {MIN_TEST})"] + fails
        elif not helped:
            fails = [f"skipping those conditions did worse on the recent years "
                     f"({after.get('expectancy_r', 0):+.3f}R vs {before.get('expectancy_r', 0):+.3f}R)"] + fails
        ok = ok and helped and removed >= MIN_TEST
        out["removed_recent"] = removed
        out["removed_after"] = summarize(list(rs[~old & ~keep]))
    out.update({"use": bool(ok), "recent_before": before, "recent_after": after,
                "older_after": older_after, "random_after": baseline, "monte_carlo": mc,
                "fails": fails, "share_kept": round(float(keep[~old].mean()), 3) if (~old).any() else None})
    if ok and mode == "filter":
        out["verdict"] = (f"skipping its bad conditions lifted the recent years from "
                          f"{before.get('expectancy_r', 0):+.3f}R to {after.get('expectancy_r', 0):+.3f}R")
    elif ok:
        out["verdict"] = (f"in its favoured conditions it passes the gate on the recent years, "
                          f"{after.get('expectancy_r', 0):+.3f}R over {after.get('trades', 0)} trades")
    elif mode == "filter" and out.get("removed_recent", 0) < MIN_TEST:
        out["verdict"] = "its conditions could not be tested on the recent years"
    else:
        out["verdict"] = "the conditions found on the older years did not hold on the recent ones"
    return out


# ──────────────────────────────── live verdict ──────────────────────────────
def today_states(region: str) -> dict[str, str]:
    h = history(region)
    if h.empty:
        return {}
    last = h.iloc[-1]
    return {k: v for k, v in last.items() if isinstance(v, str)}


def describe(rule: dict, region: str) -> str:
    d = dimensions(region).get(rule["dim"])
    return d.states.get(rule["state"], rule["state"]) if d else f"{rule['dim']} {rule['state']}"


def on_today(ev: dict | None, region: str, states: dict[str, str] | None = None,
             account: dict | None = None) -> tuple[bool, str]:
    """May this strategy send tickets today? Its gate verdict, adjusted by the
    conditions it was tested in — but only where that test held out of sample,
    and only in a market whose account replay did at least as well with them
    (`account` is that replay's stats)."""
    if not ev:
        return False, "not simulated"
    stats = ev.get("stats") or {}
    cond = stats.get("conditions") or {}
    passed = bool(ev.get("passed"))
    if account is not None and not account.get("conditions_used", True):
        reason = (account.get("conditions_check") or {}).get("reason") or "not used in this market"
        if passed:
            return True, "passes the gate"
        if cond.get("use"):
            return False, "passes in some conditions, but left out: " + reason
        return False, "fails the gate"
    states = today_states(region) if states is None else states
    rules = cond.get("rules") or {}
    active_avoid = [r for r in rules.get("avoid", []) if states.get(r["dim"]) == r["state"]]
    active_favour = [r for r in rules.get("favour", []) if states.get(r["dim"]) == r["state"]]
    if passed:
        if cond.get("use") and active_avoid:
            return False, "paused: it has lost money when " + describe(active_avoid[0], region)
        return True, "passes the gate"
    if cond.get("use") and cond.get("mode") == "favour":
        if active_favour and not active_avoid:
            return True, "passes in today's conditions: " + describe(active_favour[0], region)
        return False, "passes only when " + describe(rules["favour"][0], region)
    return False, "fails the gate"


def allowed_days(ev: dict | None, region: str) -> pd.DatetimeIndex | None:
    """The days a strategy would have been allowed to trade under its live
    rule — for the account replay. None means every day."""
    if not ev:
        return None
    cond = (ev.get("stats") or {}).get("conditions") or {}
    if not cond.get("use"):
        return None if ev.get("passed") else pd.DatetimeIndex([])
    h = history(region)
    if h.empty:
        return None
    keep = allowed(h.reset_index(drop=True), cond["rules"], cond["mode"])
    return h.index[keep]
