"""The advisor replayed as one account.

The per-strategy simulation answers "does this rule make money per trade?"
by taking every signal on every name independently. Nobody trades like that.
This replays what following the emails would actually have been: one account,
the strategies that pass the gate, at most `max_new_ideas_per_scan` new
tickets per day (strongest momentum first), at most `max_open_positions` held
at once, each sized to risk `risk_per_trade_pct` of the account, and cash that
can only be spent once. Every ticket's fill and exit come from the same
`simulate.play` as everything else.

Because it is one account marked to market daily, the thing the per-trade
Monte Carlo cannot see — momentum and trend trades losing together when a
market turns — shows up here as the account's real drawdown.

Three yardsticks, each answering a different doubt:
  index      the region's index ETF, failed companies included. Did following
             Alfred beat simply owning the market?
  survivors  equal weight in every name in today's universe. The universe is
             today's index members, so it is missing every company that fell
             out since 2014; owning all the survivors shows how much that
             alone flatters a backtest. Beating the index but not the
             survivors means the "edge" may be survivorship.
  random     the same system choosing its five daily tickets at random rather
             than by momentum. Tests whether the ranking adds anything.

Not tested here, and stated rather than hidden: the per-name filter the live
scan applies (a name's own record under a strategy) is computed over all of
history, so using it in a replay would let the past see the future. The
replay therefore takes signals on every name in the region.
"""
from __future__ import annotations

import json
import logging
import math

import numpy as np
import pandas as pd

import db
from advisor import market_of, settings
from advisor.simulate import WARMUP, _frames, evidence, play
from advisor.strategies import STRATEGIES

log = logging.getLogger("alfred.portfolio_sim")


def _passing(region: str, conditions: bool = True) -> dict[str, pd.DatetimeIndex | None]:
    """The strategies the live scan would use, each with the days its
    conditions allow it to trade (None: every day). A strategy that passes
    the gate only in certain conditions trades only on those days — so the
    account is replayed under the same rule the scan applies.

    `conditions=False` is the plain gate: passing strategies, every day."""
    from advisor import regime

    out = {}
    for n in STRATEGIES:
        ev = evidence(n, region)
        if not ev:
            continue
        cond = (ev["stats"] or {}).get("conditions") or {}
        if ev["passed"]:
            out[n] = regime.allowed_days(ev, region) if conditions else None
        elif conditions and cond.get("use"):
            out[n] = regime.allowed_days(ev, region)
    return out


def _sharpe_from(r: dict) -> float:
    eq = r["equity"]
    start = eq.index[min(WARMUP + 60, len(eq) - 1)]
    return metrics(eq.loc[start:] / eq.loc[start]).get("sharpe") or 0.0


def _choose(frames: dict, region: str) -> tuple[dict, dict | None, dict]:
    """Conditions or the plain gate, whichever the account does better with.

    A strategy can pass on its own and still make the account worse — its
    trades take position slots from better ones. So the condition rules are
    used in a market only if replaying the account with them is at least as
    good, on Sharpe, as replaying it without."""
    plain = _passing(region, conditions=False)
    cond = _passing(region)
    if set(cond) == set(plain) and all(d is None for d in cond.values()):
        return plain, None, {"used": False, "reason": "no condition rule applies here"}
    r_cond = replay(frames, cond, rank="momentum") if cond else None
    r_plain = replay(frames, plain, rank="momentum") if plain else None
    s_cond = _sharpe_from(r_cond) if r_cond else None
    s_plain = _sharpe_from(r_plain) if r_plain else None
    check = {"sharpe_with": s_cond, "sharpe_without": s_plain,
             "switched_on": [n for n in cond if n not in plain],
             "filtered": [n for n in cond if n in plain and cond[n] is not None]}
    if r_plain is None or (r_cond is not None and s_cond >= s_plain):
        return cond, r_cond, {**check, "used": True}
    return plain, r_plain, {**check, "used": False,
                            "reason": f"the account did worse with them (Sharpe {s_cond:.2f} "
                                      f"vs {s_plain:.2f} without)"}


def replay(frames: dict[str, pd.DataFrame], strategies: list[str] | dict, *,
           rank: str = "momentum", seed: int = 0) -> dict:
    """Run the account. Returns the daily equity curve and trade log.

    `strategies` is a list, or a dict of name → the days it may trade."""
    allow = strategies if isinstance(strategies, dict) else {n: None for n in strategies}
    allow_sets = {n: (None if d is None else set(d.normalize())) for n, d in allow.items()}
    strategies = list(allow)
    cfg = settings()
    risk_pct = cfg["risk_per_trade_pct"] / 100
    max_pos = cfg["max_position_pct"] / 100
    cap = cfg["max_new_ideas_per_scan"]
    max_open = cfg["max_open_positions"]
    validity = cfg["validity_sessions"]
    rng = np.random.default_rng(seed)

    calendar = sorted(set().union(*(x.index for x in frames.values())))
    calendar = pd.DatetimeIndex(calendar)
    closes = pd.DataFrame({s: x["close"] for s, x in frames.items()}).reindex(calendar).ffill()

    # Per symbol: arrays for play(), and each strategy's signal frame.
    arrays, signals, where = {}, {}, {}
    candidates: dict[pd.Timestamp, list[tuple[str, str]]] = {}
    for sym, x in frames.items():
        arrays[sym] = tuple(x[k].to_numpy() for k in ("open", "high", "low", "close"))
        where[sym] = {d: i for i, d in enumerate(x.index)}
        for name in strategies:
            sig = STRATEGIES[name].rules(x)
            signals[(sym, name)] = sig
            ok = sig["entry"].to_numpy() & np.isfinite(sig["stop"].to_numpy())
            ok[:WARMUP] = False
            days = allow_sets[name]
            for i in np.flatnonzero(ok):
                if days is not None and x.index[i].normalize() not in days:
                    continue
                candidates.setdefault(x.index[i], []).append((sym, name))

    cash, equity = 1.0, 1.0
    held: dict[str, dict] = {}       # filled positions by symbol
    pending: dict[str, dict] = {}    # tickets waiting to fill, cash reserved
    fills: dict[pd.Timestamp, list[str]] = {}
    exits: dict[pd.Timestamp, list[str]] = {}
    curve, trades = [], []
    skipped = {"cap": 0, "positions": 0, "cash": 0, "held": 0, "unfilled": 0}

    for d in calendar:
        # 1 · fills: reserved cash becomes a position
        for sym in fills.pop(d, []):
            p = pending.pop(sym, None)
            if p is None:
                continue
            cash += p["reserved"] - p["qty"] * p["fill"]
            held[sym] = p
        # 2 · exits: sell, pay the round-trip cost
        for sym in exits.pop(d, []):
            p = held.pop(sym, None)
            if p is None:
                continue
            cash += p["qty"] * p["exit"] - p["cost"] * p["qty"] * p["fill"]
            trades.append({"symbol": sym, "strategy": p["strategy"], "r": p["r"],
                           "filled": p["fill_date"], "exited": d})
        # 3 · mark to market at the close
        marked = sum(p["qty"] * closes.at[d, s] for s, p in held.items()
                     if not math.isnan(closes.at[d, s]))
        reserved = sum(p["reserved"] for p in pending.values())
        equity = cash + reserved + marked
        curve.append((d, equity, marked / equity if equity else 0))

        # 4 · today's signals → at most `cap` new tickets
        todays = candidates.get(d, [])
        if not todays:
            continue
        if rank == "momentum":
            todays = sorted(todays, key=lambda c: -np.nan_to_num(
                frames[c[0]]["mom_rank"].get(d, np.nan) if "mom_rank" in frames[c[0]] else 0))
        else:
            todays = [todays[i] for i in rng.permutation(len(todays))]
        taken, seen = 0, set()
        for sym, name in todays:
            if sym in seen:
                continue
            seen.add(sym)
            if sym in held or sym in pending:
                skipped["held"] += 1
                continue
            if taken >= cap:
                skipped["cap"] += 1
                continue
            if len(held) + len(pending) >= max_open:
                skipped["positions"] += 1
                continue
            sig = signals[(sym, name)]
            i = where[sym][d]
            limit, stop = float(sig["limit"].iat[i]), float(sig["stop"].iat[i])
            target = sig["target"].iat[i]
            per_share = limit - stop
            if per_share <= 0:
                continue
            qty = min(equity * risk_pct / per_share, equity * max_pos / limit)
            need = qty * limit
            if need > cash:
                skipped["cash"] += 1
                continue
            o, h, l, c = arrays[sym]
            cost = cfg["round_trip_cost_pct"][market_of(sym)] / 100
            out = play(o, h, l, c, sig["exit"].to_numpy(), i, limit, stop,
                       None if math.isnan(target) else float(target),
                       validity, STRATEGIES[name].max_days, cost)
            if out.status in ("expired", "pending"):
                skipped["unfilled"] += 1
                continue
            idx = frames[sym].index
            p = {"strategy": name, "qty": qty, "reserved": need, "fill": out.fill,
                 "fill_date": idx[out.fill_i], "cost": cost, "r": out.r,
                 "exit": out.exit}
            pending[sym] = p
            cash -= need
            fills.setdefault(idx[out.fill_i], []).append(sym)
            if out.status == "closed":
                exits.setdefault(idx[out.exit_i], []).append(sym)
            taken += 1

    eq = pd.Series([e for _, e, _ in curve], index=[d for d, _, _ in curve])
    exposure = pd.Series([x for _, _, x in curve], index=eq.index)
    return {"equity": eq, "exposure": exposure, "trades": trades, "skipped": skipped,
            "open_at_end": len(held)}


# ───────────────────────────────── yardsticks ───────────────────────────────
def benchmark(symbol: str, calendar: pd.DatetimeIndex) -> pd.Series | None:
    from advisor import bars

    b = bars.load(symbol, years=settings()["history_years"])
    if b.empty:
        return None
    s = b["close"].reindex(calendar).ffill().dropna()
    return s / s.iloc[0]


def survivors(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.Series:
    """Equal weight in every name of today's universe, rebalanced daily."""
    rets = pd.DataFrame({s: x["close"].pct_change() for s, x in frames.items()}).reindex(calendar)
    # One bad print can show as +500% in a day; clip so a data error cannot
    # carry the yardstick.
    daily = rets.clip(-0.5, 0.5).mean(axis=1, skipna=True).fillna(0)
    return (1 + daily).cumprod()


def metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 60:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    daily = eq.pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
        "total_return": float(eq.iloc[-1] - 1),
        "max_drawdown": float(dd),
        "volatility": float(daily.std() * math.sqrt(252)),
        "sharpe": float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() else None,
        "worst_year": float(eq.resample("YE").last().pct_change().min()),
        "years": round(years, 1),
    }


def _period(eq: pd.Series, start=None, end=None) -> dict:
    part = eq.loc[start:end]
    return metrics(part / part.iloc[0]) if len(part) > 60 else {}


# ─────────────────────────────────── run ────────────────────────────────────
def run(region: str, *, random_seeds: int = 5) -> dict | None:
    cfg = settings()
    frames = _frames(region)
    strategies, main, conditions = _choose(frames, region) if frames else ({}, None, {})
    if not frames or not strategies:
        db.execute("DELETE FROM portfolio_backtests WHERE region = %s", (region,))
        return None

    main = main or replay(frames, strategies, rank="momentum")
    eq = main["equity"]
    # Start the clock when the first strategy could act: before that the
    # account is cash by construction, which would flatter its drawdown.
    start = eq.index[min(WARMUP + 60, len(eq) - 1)]
    eq = eq.loc[start:] / eq.loc[start]
    cal = eq.index

    rand = [replay(frames, strategies, rank="random", seed=s)["equity"] for s in range(1, random_seeds + 1)]
    rand = [(r.loc[start:] / r.loc[start]) for r in rand]
    rand_mean = pd.concat(rand, axis=1).mean(axis=1)
    index = benchmark(cfg["benchmark"][region], cal)
    surv = survivors(frames, cal)
    surv = surv / surv.iloc[0]

    cutoff = cal[-1] - pd.DateOffset(years=cfg["recent_period_years"])
    from advisor.simulate import summarize

    rs = [t["r"] for t in main["trades"] if t["filled"] >= start]
    stats = {
        "strategies": list(strategies),
        "conditional": [n for n, d in strategies.items() if d is not None],
        "conditions_used": bool(conditions.get("used")), "conditions_check": conditions,
        "strategy": metrics(eq), "older": _period(eq, end=cutoff), "recent": _period(eq, start=cutoff),
        "random_picks": metrics(rand_mean),
        "random_picks_each": [metrics(r).get("cagr") for r in rand],
        "index": metrics(index) if index is not None else {},
        "index_symbol": cfg["benchmark"][region],
        "survivors": metrics(surv),
        "trades": summarize(rs), "skipped": main["skipped"], "open_at_end": main["open_at_end"],
        "avg_exposure": float(main["exposure"].loc[start:].mean()),
        "from": str(cal[0].date()), "through": str(cal[-1].date()), "symbols": len(frames),
        "recent_from": str(cutoff.date()),
    }
    stats["passed"], stats["fails"] = judge_account(stats)
    weekly = pd.DataFrame({"strategy": eq, "random": rand_mean, "survivors": surv,
                           "index": index if index is not None else np.nan}).resample("W").last()
    curve = {"dates": [str(d.date()) for d in weekly.index],
             **{k: [None if pd.isna(v) else round(float(v), 4) for v in weekly[k]]
                for k in weekly.columns}}
    db.execute(
        """INSERT INTO portfolio_backtests (region, stats, curve, computed_at)
           VALUES (%s, %s, %s, now())
           ON CONFLICT (region) DO UPDATE SET stats = EXCLUDED.stats,
             curve = EXCLUDED.curve, computed_at = now()""",
        (region, json.dumps(_finite(stats), default=str), json.dumps(curve)))
    return stats


def judge_account(stats: dict) -> tuple[bool, list[str]]:
    """The region-level gate: did following Alfred beat holding the index?

    Not on return — on risk. The account replays roughly matched each index's
    return; what they can claim is getting it with less risk, so that is the
    claim tested. A missing index is a failure, never a pass by default.
    """
    g = settings()["gate"]
    me, idx = stats.get("strategy") or {}, stats.get("index") or {}
    fails = []
    if not idx:
        return False, [f"no data for the index {stats.get('index_symbol')}"]
    if g.get("account_beats_index_sharpe") and (me.get("sharpe") or 0) <= (idx.get("sharpe") or 0):
        fails.append(f"Sharpe {me.get('sharpe') or 0:.2f} vs index {idx.get('sharpe') or 0:.2f}")
    if g.get("account_beats_index_drawdown") and (me.get("max_drawdown") or -1) <= (idx.get("max_drawdown") or -1):
        fails.append(f"worst fall {(me.get('max_drawdown') or 0) * 100:.0f}% vs index "
                     f"{(idx.get('max_drawdown') or 0) * 100:.0f}%")
    return not fails, fails


def account(region: str) -> dict | None:
    """The stored replay for a region, or None if there is none."""
    row = db.one("SELECT stats, computed_at FROM portfolio_backtests WHERE region=%s", (region,))
    return row and {**row["stats"], "computed_at": row["computed_at"]}


def _finite(v):
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
