"""Today's tickets, and where your holdings' stops are.

Runs after a market closes. For every name in the region it asks each
strategy whether its entry fired on the final bar — and only listens to
strategies whose simulation passed the gate, on names whose own record under
that strategy is not a losing one. A fired signal becomes a ticket: limit,
stop, size, validity, the reasons in numbers, and the evidence behind it.

Holdings get a separate section: where the trend would break (a trailing
stop you could set), and a warning when a holding has just closed below it.
Those are risk levels, not gated trades, and are labelled as such.

`preview=True` ignores the gate, so you can see what a ticket looks like
before anything has earned the right to send one. Preview tickets are never
stored and never emailed.
"""
from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np

import db
from advisor import BROKER, CURRENCY, REGIONS, bare, bars, market_of, settings
from advisor.simulate import WARMUP, evidence
from advisor.strategies import STRATEGIES, add_momentum_rank, chandelier_stop, indicators

log = logging.getLogger("alfred.scan")


# ───────────────────────────────── accounts ─────────────────────────────────
def accounts() -> dict[str, dict]:
    """Each broker's value and free cash from its newest good snapshot."""
    out = {}
    for broker in ("kite", "webull"):
        snap = db.one("""SELECT id, funds FROM portfolio_snapshots WHERE broker=%s AND ok
                         ORDER BY taken_at DESC LIMIT 1""", (broker,))
        if not snap:
            continue
        funds = snap["funds"] or {}
        held = db.one("""SELECT coalesce(sum(quantity * coalesce(last_price, avg_price)), 0) v
                         FROM portfolio_lines WHERE snapshot_id=%s AND kind='holding'
                           AND currency=%s""", (snap["id"], funds.get("currency")))
        value = funds.get("total") or (float(held["v"]) + float(funds.get("cash") or 0))
        out[broker] = {"value": float(value), "cash": float(funds.get("available") or 0),
                       "currency": funds.get("currency")}
    return out


def _in_currency(amount: float, have: str, want: str) -> float | None:
    """Convert an account amount into the instrument's currency."""
    if have == want:
        return amount
    if (have, want) == ("AUD", "USD"):
        rate = bars.fx("AUDUSD=X")
        return amount * rate if rate else None
    if (have, want) == ("AUD", "INR"):
        rate = bars.fx("AUDINR=X")
        return amount * rate if rate else None
    return None


# ───────────────────────────────── tickets ──────────────────────────────────
def ticket(sym: str, strat, x, accts: dict, held: dict) -> dict:
    cfg = settings()
    market = market_of(sym)
    ccy = CURRENCY[market]
    broker = BROKER[market]
    sig = strat.rules(x).iloc[-1]
    last = x.iloc[-1]
    limit = float(sig["limit"])
    stop = float(sig["stop"])
    target = None if math.isnan(sig["target"]) else float(sig["target"])
    risk_per_share = limit - stop

    t = {
        "symbol": sym, "display": bare(sym), "market": market, "broker": broker,
        "strategy": strat.name, "label": strat.label, "action": "BUY", "currency": ccy,
        "signal_date": x.index[-1].date(), "limit": round(limit, 2), "stop": round(stop, 2),
        "target": target and round(target, 2),
        "valid_until": np.busday_offset(x.index[-1].date(), cfg["validity_sessions"],
                                        roll="forward").astype(date),
        "reasons": strat.reasons(last), "notes": [],
        "stop_pct": (stop / limit - 1) * 100,
    }

    acct = accts.get(broker)
    equity = acct and _in_currency(acct["value"], acct["currency"], ccy)
    cash = acct and _in_currency(acct["cash"], acct["currency"], ccy)
    if not equity:
        t["notes"].append(f"No {broker} account value on file, so no size is suggested.")
        t["quantity"] = None
        t["risk_amount"] = None
    else:
        risk_amount = equity * cfg["risk_per_trade_pct"] / 100
        qty = risk_amount / risk_per_share
        qty = min(qty, equity * cfg["max_position_pct"] / 100 / limit)
        # Webull AU buys fractional US shares; Indian and ASX shares are whole.
        qty = math.floor(qty * 100) / 100 if market == "US" else math.floor(qty)
        t["quantity"] = qty
        t["risk_amount"] = round(qty * risk_per_share, 2)
        cost = qty * limit
        if qty <= 0:
            t["notes"].append("The stop is too wide for this account at "
                              f"{cfg['risk_per_trade_pct']}% risk — skip, or size it yourself.")
        elif cash is not None and cash < cost:
            t["notes"].append(f"Needs about {ccy} {cost:,.0f}; {broker} shows {ccy} {cash:,.0f} "
                              "available — add funds or free up cash first.")
    if sym in held:
        h = held[sym]
        t["notes"].append(f"You already hold {float(h['quantity']):g} at "
                          f"{float(h['avg_price'] or 0):,.2f}; this would add to it.")
    return t


def _attach_evidence(t: dict, pooled: dict, own: dict | None, acct: dict | None = None) -> None:
    rec, mc = pooled["stats"]["recent"], pooled["stats"]["monte_carlo"]
    t["evidence"] = {
        "region_trades": rec.get("trades"), "win_rate": rec.get("win_rate"),
        "expectancy_r": rec.get("expectancy_r"), "profit_factor": rec.get("profit_factor"),
        "random_expectancy_r": pooled["stats"]["random_recent"].get("expectancy_r"),
        "older_expectancy_r": pooled["stats"]["older"].get("expectancy_r"),
        "chance_100_trade_loss": mc.get("chance_of_loss"),
        "p95_drawdown_r": mc.get("p95_drawdown_r"),
        "recent_from": pooled["stats"].get("recent_from"),
        "symbols": pooled["stats"].get("symbols"),
        "name_trades": own and own["stats"].get("trades"),
        "name_expectancy_r": own and own["stats"].get("expectancy_r"),
        "passed": bool(pooled["passed"]),
    }
    if acct:
        me, idx = acct.get("strategy") or {}, acct.get("index") or {}
        t["evidence"]["account"] = {
            "since": acct.get("from"), "index": acct.get("index_symbol"),
            "cagr": me.get("cagr"), "index_cagr": idx.get("cagr"),
            "max_drawdown": me.get("max_drawdown"), "index_max_drawdown": idx.get("max_drawdown"),
            "sharpe": me.get("sharpe"), "index_sharpe": idx.get("sharpe"),
            "passed": bool(acct.get("passed")),
        }


# ───────────────────────────────── holdings ─────────────────────────────────
def holding_risk(sym: str, x, h: dict) -> dict:
    stop = chandelier_stop(x)
    last, prev = x.iloc[-1], x.iloc[-2]
    level, prev_level = float(stop.iloc[-1]), float(stop.iloc[-2])
    close = float(last["close"])
    return {
        "symbol": sym, "display": bare(sym), "currency": CURRENCY[market_of(sym)],
        "broker": BROKER[market_of(sym)], "quantity": float(h["quantity"]),
        "avg_price": float(h["avg_price"] or 0), "close": close,
        "stop": round(level, 2), "stop_gap_pct": (level / close - 1) * 100,
        "below_stop": close < level,
        "newly_below": close < level and float(prev["close"]) >= prev_level,
        "below_200d": bool(close < last["sma200"]) if not math.isnan(last["sma200"]) else None,
    }


def _too_new(sym: str, h: dict, n: int) -> dict:
    """A holding without the 200+ days a trend reading needs: say so, rather
    than leave it off the list as if it were not held."""
    return {"symbol": sym, "display": bare(sym), "currency": CURRENCY[market_of(sym)],
            "broker": BROKER[market_of(sym)], "quantity": float(h["quantity"]),
            "avg_price": float(h["avg_price"] or 0), "close": float(h["last_price"] or 0),
            "stop": None, "stop_gap_pct": 0.0, "below_stop": False, "newly_below": False,
            "below_200d": None, "note": f"only {n} days of history — no trend reading yet"}


# ─────────────────────────────────── scan ───────────────────────────────────
def scan(region: str, *, preview: bool = False, store: bool = False) -> dict:
    """Signals for one region.

    Stores new ideas only when `store` is set — which only the scheduled
    cycle does, right before emailing them. A stored idea counts as sent and
    is not repeated, so a dry run that stored would swallow the real email.
    """
    held = bars.holdings()
    accts = accounts()
    tickets, risks = [], []
    open_ideas = {(r["strategy"], r["symbol"]) for r in db.query(
        "SELECT strategy, symbol FROM ideas WHERE status IN ('open','filled')")}
    pooled = {name: evidence(name, region) for name in STRATEGIES}
    from advisor import portfolio_sim

    # The region-level gate: no replay, or a replay that did not beat holding
    # the index on risk, and the region sends nothing.
    acct = portfolio_sim.account(region)
    account_ok = bool(acct and acct.get("passed"))
    syms = [s for s in bars.universe() if market_of(s) in REGIONS[region]]
    as_of = None

    # Load the whole region first: momentum rank compares every name with
    # every other on the same day, so no single name can be judged alone.
    frames = {}
    for sym in syms:
        df = bars.load(sym, years=2)
        if len(df) < WARMUP:
            if sym in held:
                risks.append(_too_new(sym, held[sym], len(df)))
            continue
        frames[sym] = indicators(df)
    add_momentum_rank(frames)

    for sym, x in frames.items():
        as_of = max(as_of or x.index[-1], x.index[-1])
        if sym in held:
            risks.append(holding_risk(sym, x, held[sym]))
        for name, strat in STRATEGIES.items():
            ev = pooled.get(name)
            if not ev or not ((ev["passed"] and account_ok) or preview):
                continue
            own = evidence(name, region, sym)
            if not preview and not (own and own["passed"]):
                continue
            if (name, sym) in open_ideas:
                continue
            if not strat.rules(x)["entry"].iloc[-1]:
                continue
            t = ticket(sym, strat, x, accts, held)
            _attach_evidence(t, ev, own, acct)
            t["preview"] = preview and not (ev["passed"] and account_ok)
            rank = x["mom_rank"].iloc[-1] if "mom_rank" in x else np.nan
            t["mom_rank"] = None if np.isnan(rank) else float(rank)
            tickets.append(t)

    tickets, left_out = select(tickets, settings().get("max_new_ideas_per_scan"))

    stored = _store(tickets) if store and not preview else 0
    risks.sort(key=lambda r: (not r["newly_below"], not r["below_stop"], r["stop_gap_pct"]))
    return {"region": region, "as_of": as_of and as_of.date(), "tickets": tickets,
            "left_out": left_out, "holdings": risks, "stored": stored,
            "gate": {**{n: bool(e and e["passed"]) for n, e in pooled.items()},
                     "account": account_ok}}


def select(tickets: list[dict], cap: int | None) -> tuple[list[dict], int]:
    """The strongest few tickets by momentum, one per name.

    Applied after the simulation, which took every signal — so the email
    says how many were left out rather than pretending there were no others.
    """
    seen, kept = set(), []
    for t in sorted(tickets, key=lambda t: -(t.get("mom_rank") or 0)):
        if t["symbol"] not in seen:
            seen.add(t["symbol"])
            kept.append(t)
    cap = cap or len(kept)
    return kept[:cap], max(0, len(kept) - cap)


def _store(tickets: list[dict]) -> int:
    import json

    n = 0
    for t in tickets:
        row = db.one(
            """INSERT INTO ideas (signal_date, strategy, symbol, market, broker, action,
                 currency, limit_price, stop_price, target_price, quantity, risk_amount,
                 valid_until, reasons, evidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (strategy, symbol, signal_date) DO NOTHING RETURNING id""",
            (t["signal_date"], t["strategy"], t["symbol"], t["market"], t["broker"],
             t["action"], t["currency"], t["limit"], t["stop"], t["target"], t["quantity"],
             t["risk_amount"], t["valid_until"], json.dumps(t["reasons"] + t["notes"]),
             json.dumps(t["evidence"], default=str)))
        if row:
            t["id"] = row["id"]
            n += 1
    return n


def regime() -> str | None:
    row = db.one("SELECT regime FROM world_model ORDER BY version DESC LIMIT 1")
    return row and row["regime"]


