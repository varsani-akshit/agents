"""Every candidate a simulation run surfaces — passed or not.

The ticket scanner only listens to strategies that passed the gate, on names
with a clean record, five at a time. That is right for the inbox. This module
is the wider view behind it, for the person who wants to look: every signal
that fired on the last close under ANY strategy, and every name close to
firing, each with the evidence its strategy has — so a strong-looking name a
strategy just missed the gate with is visible, and plainly labelled as not
having passed.

Score, for ordering only: the strategy's recent per-trade expectancy in the
region, nudged by the name's own record under it, less a little for each
percent a near-trigger still has to travel. It is not a forecast.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from advisor import BROKER, CURRENCY, REGIONS, bare, bars, market_of
from advisor.scan import _attach_evidence, _rank, accounts, near_trigger, ticket
from advisor.simulate import WARMUP, evidence
from advisor.strategies import STRATEGIES, add_momentum_rank, indicators

log = logging.getLogger("alfred.candidates")


def _score(pooled: dict | None, own: dict | None, distance: float = 0.0) -> float:
    rec = ((pooled or {}).get("stats") or {}).get("recent") or {}
    exp = rec.get("expectancy_r") or 0.0
    name = ((own or {}).get("stats") or {}).get("expectancy_r")
    name = 0.0 if name is None or (isinstance(name, float) and math.isnan(name)) else name
    return round(exp + 0.3 * max(-1.0, min(2.0, name)) - 0.05 * distance, 4)


def collect(region: str) -> list[dict]:
    """Signals and near-triggers in one region, under every strategy."""
    from advisor import portfolio_sim

    accts = accounts()
    held = bars.holdings()
    acct = portfolio_sim.account(region)
    account_ok = bool(acct and acct.get("passed"))
    pooled = {name: evidence(name, region) for name in STRATEGIES}
    from advisor import regime

    states = regime.today_states(region)
    live = {name: regime.on_today(ev, region, states, acct or {}) for name, ev in pooled.items()}

    frames = {}
    for sym in (s for s in bars.universe() if market_of(s) in REGIONS[region]):
        df = bars.load(sym, years=2)
        if len(df) >= WARMUP:
            frames[sym] = indicators(df)
    add_momentum_rank(frames)

    out, seen = [], set()
    for sym, x in frames.items():
        last = x.iloc[-1]
        for name, strat in STRATEGIES.items():
            ev = pooled.get(name)
            if not ev:
                continue
            own = evidence(name, region, sym)
            on, why = live[name]
            gate = bool(on and account_ok)
            name_ok = bool(own and own["passed"])
            base = {
                "region": region, "symbol": sym, "display": bare(sym), "strategy": name,
                "label": strat.label, "market": market_of(sym), "broker": BROKER[market_of(sym)],
                "currency": CURRENCY[market_of(sym)], "close": float(last["close"]),
                "as_of": str(x.index[-1].date()), "mom_rank": _rank(x),
                "strategy_passed": bool(ev["passed"]), "market_sending": account_ok,
                "regime_on": on, "regime_why": why,
                "name_record_ok": name_ok,
                "name_trades": (own or {}).get("stats", {}).get("trades"),
                "name_expectancy_r": (own or {}).get("stats", {}).get("expectancy_r"),
                "fails": (ev["stats"] or {}).get("fails") or [],
                "held": sym in held,
            }
            if strat.rules(x)["entry"].iloc[-1]:
                key = (bare(sym), name, "signal")
                if key in seen:
                    continue
                seen.add(key)
                t = ticket(sym, strat, x, accts, held)
                _attach_evidence(t, ev, own, acct)
                out.append({**base, "kind": "signal", "passed_gate": gate and name_ok,
                            "score": _score(ev, own),
                            "limit": t["limit"], "stop": t["stop"], "target": t["target"],
                            "stop_pct": round(t["stop_pct"], 2), "quantity": t["quantity"],
                            "risk_amount": t["risk_amount"], "notes": t["notes"],
                            "reasons": t["reasons"], "evidence": t["evidence"]})
                continue
            n = near_trigger(name, x)
            if n:
                key = (bare(sym), name, "near")
                if key in seen:
                    continue
                seen.add(key)
                out.append({**base, "kind": "near", "passed_gate": gate and name_ok,
                            "score": _score(ev, own, n["distance_pct"]),
                            "waiting": n["waiting"], "level": n["level"],
                            "distance_pct": round(n["distance_pct"], 2)})
    out.sort(key=lambda c: (not c["passed_gate"], c["kind"] != "signal", -c["score"]))
    return out


def _clean(v):
    """JSON-safe: NaN and numpy scalars out."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v
