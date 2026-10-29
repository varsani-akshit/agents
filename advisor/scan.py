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
import pandas as pd

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
    reserve = float((cfg.get("new_funds") or {}).get(broker) or 0)
    equity = acct and _in_currency(acct["value"] + reserve, acct["currency"], ccy)
    cash = acct and _in_currency(acct["cash"], acct["currency"], ccy)
    reserve_here = acct and _in_currency(reserve, acct["currency"], ccy)
    t["fund"] = None
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
            # How much to move in, in the account's own currency, rounded up
            # to something you would actually transfer.
            short = (cost - cash) / (_in_currency(1.0, acct["currency"], ccy) or 1)
            step = 1000 if acct["currency"] == "INR" else 50
            add = math.ceil(short * 1.02 / step) * step
            t["fund"] = {"broker": broker, "currency": acct["currency"], "amount": add,
                         "within_reserve": add <= reserve}
            within = (f"within the {_cur(acct['currency'], reserve)} you set aside"
                      if add <= reserve else
                      f"more than the {_cur(acct['currency'], reserve)} you set aside — size it down or skip")
            t["notes"].append(f"Add about {_cur(acct['currency'], add)} to "
                              f"{'Kite' if broker == 'kite' else 'Webull'} first ({within}).")
    if sym in held:
        h = held[sym]
        t["notes"].append(f"You already hold {float(h['quantity']):g} at "
                          f"{float(h['avg_price'] or 0):,.2f}; this would add to it.")
    return t


def _cur(ccy: str, v: float) -> str:
    sym = {"INR": "₹", "AUD": "A$", "USD": "US$"}.get(ccy, ccy + " ")
    return f"{sym}{v:,.0f}"


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
        "median_days": pooled["stats"].get("median_days"),
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
    tickets, risks, near = [], [], []
    open_ideas = {(r["strategy"], r["symbol"]) for r in db.query(
        "SELECT strategy, symbol FROM ideas WHERE status IN ('open','filled')")}
    pooled = {name: evidence(name, region) for name in STRATEGIES}
    from advisor import portfolio_sim, regime

    # The region-level gate: no replay, or a replay that did not beat holding
    # the index on risk, and the region sends nothing.
    acct = portfolio_sim.account(region)
    account_ok = bool(acct and acct.get("passed"))

    # Each strategy's verdict today: the gate, adjusted by the conditions it
    # was tested in where that test held on years it never saw — and where
    # the market's account did at least as well with them.
    states = regime.today_states(region)
    live = {name: regime.on_today(ev, region, states, acct or {}) for name, ev in pooled.items()}
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
            on, why = live[name]
            if not ev or not ((on and account_ok) or preview):
                continue
            own = evidence(name, region, sym)
            if not preview and not (own and own["passed"]):
                continue
            if (name, sym) in open_ideas:
                continue
            if not strat.rules(x)["entry"].iloc[-1]:
                if on and account_ok:
                    n = near_trigger(name, x)
                    if n:
                        near.append({"symbol": sym, "strategy": name, "label": strat.label,
                                     "mom_rank": _rank(x), "close": float(x["close"].iloc[-1]),
                                     "currency": CURRENCY[market_of(sym)],
                                     "broker": BROKER[market_of(sym)], **n})
                continue
            t = ticket(sym, strat, x, accts, held)
            _attach_evidence(t, ev, own, acct)
            t["conditions"] = {"on": on, "why": why}
            if on and not ev["passed"]:
                t["reasons"].append(f"Conditions: this strategy {why}.")
            t["preview"] = preview and not (on and account_ok)
            t["mom_rank"] = _rank(x)
            tickets.append(t)

    tickets, also = select(tickets, settings().get("max_new_ideas_per_scan"), keep_rest=True)
    left_out = len(also)
    near = merge_near(near)

    stored = _store(tickets) if store and not preview else 0
    risks.sort(key=lambda r: (not r["newly_below"], not r["below_stop"], r["stop_gap_pct"]))
    return {"region": region, "as_of": as_of and as_of.date(), "tickets": tickets,
            "left_out": left_out, "also": also[:15], "near": near[:20],
            "holdings": risks, "stored": stored,
            "gate": {**{n: live[n][0] for n in pooled},
                     "account": account_ok}}


def select(tickets: list[dict], cap: int | None, keep_rest: bool = False):
    """The strongest few tickets by momentum, one per name.

    Applied after the simulation, which took every signal — so the email
    says how many were left out rather than pretending there were no others.
    Returns (kept, number left out), or (kept, the rest) with `keep_rest`.
    """
    seen, kept = set(), []
    for t in sorted(tickets, key=lambda t: -(t.get("mom_rank") or 0)):
        # One company once: an Indian name can be listed on NSE and BSE.
        if bare(t["symbol"]) not in seen:
            seen.add(bare(t["symbol"]))
            kept.append(t)
    cap = cap or len(kept)
    if keep_rest:
        return kept[:cap], kept[cap:]
    return kept[:cap], max(0, len(kept) - cap)


def _rank(x) -> float | None:
    r = x["mom_rank"].iloc[-1] if "mom_rank" in x else np.nan
    return None if np.isnan(r) else float(r)


def near_trigger(name: str, x) -> dict | None:
    """How close a name is to firing a strategy that passed — the watchlist.

    Distances, not forecasts: "1.2% under its 55-day high" is measured; whether
    it gets there is not. Monthly strategies report the next month start.
    """
    r = x.iloc[-1]
    up = r.close > r.sma200 if not np.isnan(r.sma200) else False

    def gap(level):
        return (level / r.close - 1) * 100

    if name in ("breakout_55", "breakout_20"):
        level = r.hh55 if name == "breakout_55" else r.hh20
        if up and not np.isnan(level) and 0 < gap(level) <= 3:
            return {"distance_pct": gap(level), "level": float(level),
                    "waiting": f"a close above {level:,.2f} ({gap(level):.1f}% away)"}
    if name == "near_52w_high" and up and not np.isnan(r.hh252):
        level = 0.98 * r.hh252
        if 0 < gap(level) <= 4:
            return {"distance_pct": gap(level), "level": float(level),
                    "waiting": f"a close above {level:,.2f}, within 2% of its yearly high"}
    if name == "quiet_trend" and not np.isnan(r.sma100) and r.sma20 < r.sma100:
        g = (r.sma100 / r.sma20 - 1) * 100
        if g <= 1 and r.atr_pct < r.atr_pct_med:
            return {"distance_pct": g, "level": float(r.sma100),
                    "waiting": f"its 20-day average to cross the 100-day ({g:.1f}% apart)"}
    if name in ("abs_momentum", "momentum_rotation") and up:
        ok = (r.ret12 > 0) if name == "abs_momentum" else ((x["mom_rank"].iloc[-1] or 0) >= 0.9
                                                           if "mom_rank" in x else False)
        if ok:
            first = (x.index[-1] + pd.offsets.MonthBegin(1)).date()
            nxt = np.busday_offset(first, 0, roll="forward").astype(date)
            return {"distance_pct": 0.0, "level": None,
                    "waiting": f"the next month start ({nxt:%a %d %b}); qualifies today"}
    return None


def merge_near(near: list[dict]) -> list[dict]:
    """One row per company, strategies listed together; price triggers first
    (nearest first), then names that qualify at the next month start
    (strongest momentum first)."""
    by: dict[str, dict] = {}
    for n in near:
        key = bare(n["symbol"])
        if key in by:
            if n["label"] not in by[key]["labels"]:
                by[key]["labels"].append(n["label"])
            continue
        by[key] = {**n, "display": key, "labels": [n["label"]]}
    rows = list(by.values())
    price = sorted((r for r in rows if r["level"] is not None), key=lambda r: r["distance_pct"])
    monthly = sorted((r for r in rows if r["level"] is None), key=lambda r: -(r["mom_rank"] or 0))
    return price + monthly


def save_candidates(result: dict, exits: list[dict] | None = None) -> int:
    """Keep what this scan saw beyond its tickets, for the Today page."""
    import json

    region = result["region"]
    rows = ([("also", t["symbol"], t["strategy"],
              {"label": t["label"], "limit": t["limit"], "stop": t["stop"],
               "currency": t["currency"], "broker": t["broker"], "mom_rank": t.get("mom_rank")})
             for t in result.get("also") or []]
            + [("near", n["symbol"], n["strategy"], n) for n in result.get("near") or []]
            + [("exit", e["symbol"], e["strategy"], e) for e in exits or []])
    db.execute("DELETE FROM scan_candidates WHERE region = %s", (region,))
    db.executemany(
        "INSERT INTO scan_candidates (region, kind, symbol, strategy, as_of, detail) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        [(region, k, sym, strat, result.get("as_of"), json.dumps(d, default=str))
         for k, sym, strat, d in rows])
    return len(rows)


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


