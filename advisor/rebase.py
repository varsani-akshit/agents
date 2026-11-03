"""Bring standing tickets onto the latest close.

A ticket that has not filled is a standing order: "buy SYMBOL at this limit,
stop there". Its numbers came from the strategy's rules applied to one
particular close. Two closes later those rules give different numbers — the
pullback level has moved, the ATR that set the stop has changed — but the
ticket on the dashboard still showed the old ones, because the scan skips any
symbol that already has an open idea.

That is the dangerous kind of stale: not obviously missing, just quietly
wrong. A limit from three sessions ago can sit above the current price (so it
fills instantly at a level the strategy never intended) or far below it (so it
never fills and the idea silently rots).

So after every run, each unfilled ticket is re-derived from the newest bars:

- The setup is still there and still passing → update the numbers, and keep
  the original so the drift is visible rather than silently applied.
- The strategy stopped passing its gate, or conditions switched it off, or the
  entry no longer fires → withdraw it, with the reason. A withdrawn ticket is
  not a failure to record against the strategy; it is an order that was never
  placed and should not be.

What is deliberately NOT rebased: a *filled* idea. Its entry is history — the
price was paid. Moving its stop here would rewrite the record of a real
position. Stops on filled positions move through the ledger's own replay,
which is the accounting the backtest shares.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np

import db
from advisor import region_of, settings
from advisor.strategies import STRATEGIES, add_momentum_rank, indicators

log = logging.getLogger("alfred.rebase")

# A level that has barely moved is not worth reporting; re-stating the same
# number as "changed" would train the eye to ignore the notice.
MATERIAL_PCT = 0.25


def rebase(region: str | None = None, *, dry_run: bool = False) -> dict:
    """Re-derive every unfilled ticket against the latest close.

    Returns what moved and what was withdrawn, for the run record and for the
    Today page's "these numbers moved" line.
    """
    from advisor import scan
    from advisor.simulate import evidence

    sql = "SELECT * FROM ideas WHERE status = 'open'"
    params: list = []
    if region:
        sql += " AND market = %s"
        params.append(region)
    live = db.query(sql + " ORDER BY created_at", params)
    if not live:
        return {"checked": 0, "moved": [], "withdrawn": [], "unchanged": 0}

    accts = scan.accounts()
    from advisor import bars as bars_mod

    held = bars_mod.holdings()
    moved, withdrawn, unchanged = [], [], 0

    # Momentum rotation ranks a name against its whole region, so the frames
    # have to be built per region rather than per symbol. `ideas.market` holds
    # the exchange (NSE, BSE, ASX, US, CRYPTO) while evidence is pooled per
    # region (IN groups NSE and BSE) — reading one as the other silently finds
    # no evidence and withdraws perfectly good tickets.
    by_region: dict[str, list[dict]] = {}
    for idea in live:
        try:
            reg = region_of(idea["market"])
        except StopIteration:
            # An exchange code we no longer recognise. We cannot judge the
            # ticket, so we leave it alone and say so: withdrawing on the
            # strength of our own blind spot would be worse than stale.
            log.warning("idea %s has unknown market %r — left untouched",
                        idea["id"], idea["market"])
            continue
        by_region.setdefault(reg, []).append(idea)

    for reg, ideas in by_region.items():
        frames = {}
        for idea in ideas:
            df = bars_mod.load(idea["symbol"], years=2)
            if not df.empty:
                frames[idea["symbol"]] = indicators(df)
        if any(i["strategy"] == "momentum_rotation" for i in ideas):
            add_momentum_rank(frames)

        for idea in ideas:
            strat = STRATEGIES.get(idea["strategy"])
            x = frames.get(idea["symbol"])
            if strat is None or x is None or x.empty:
                # No bars to judge it by. Leave it exactly as it is: silence is
                # the honest answer, not a withdrawal we cannot justify.
                unchanged += 1
                continue

            why = _no_longer_valid(idea, strat, x, reg, evidence)
            if why:
                withdrawn.append({**_brief(idea), "reason": why})
                if not dry_run:
                    db.execute(
                        """UPDATE ideas SET status='withdrawn', withdrawn_reason=%s,
                             rebased_at=now() WHERE id=%s""", (why, idea["id"]))
                continue

            t = scan.ticket(idea["symbol"], strat, x, accts, held)
            change = _diff(idea, t)
            if not change:
                unchanged += 1
                if not dry_run:
                    db.execute("UPDATE ideas SET rebased_at=now() WHERE id=%s", (idea["id"],))
                continue

            moved.append({**_brief(idea), **change})
            if not dry_run:
                _apply(idea, t, change)

    log.info("rebase: %d checked, %d moved, %d withdrawn, %d unchanged",
             len(live), len(moved), len(withdrawn), unchanged)
    return {"checked": len(live), "moved": moved, "withdrawn": withdrawn,
            "unchanged": unchanged}


def _no_longer_valid(idea: dict, strat, x, region: str, evidence) -> str | None:
    """Why this ticket should not stand any more, or None if it still should.

    Each reason is phrased as something that happened, because a withdrawal
    notice that says only "no longer valid" leaves you unable to tell a
    strategy being switched off from a setup simply passing by.
    """
    from advisor import regime
    from advisor.portfolio_sim import account

    ev = evidence(strat.name, region)
    if not ev or not ev["passed"]:
        return f"{strat.label} no longer passes its test in this market."

    own = evidence(strat.name, region, idea["symbol"])
    if not (own and own["passed"]):
        return f"{strat.label} no longer passes on this name."

    try:
        states = {s["key"]: s["state"] for s in regime.now(region)["states"]}
        on, cond_why = regime.on_today(ev, region, states, account(region) or {})
        if not on:
            return f"Conditions switched this strategy off: {cond_why}."
    except Exception:  # noqa: BLE001 - conditions are advisory here
        pass

    if not bool(strat.rules(x)["entry"].iloc[-1]):
        return "The setup is no longer there on the latest close."

    # A ticket whose validity window has run out is the ledger's business, not
    # ours: it expires with its record intact rather than being withdrawn.
    return None


def _diff(idea: dict, t: dict) -> dict | None:
    """What materially moved between the stored ticket and today's."""
    out = {}
    for key, col in (("limit", "limit_price"), ("stop", "stop_price")):
        was, now = float(idea[col]), float(t[key])
        if was and abs(now / was - 1) * 100 >= MATERIAL_PCT:
            out[key] = {"was": round(was, 2), "now": round(now, 2),
                        "pct": round((now / was - 1) * 100, 2)}
    if idea["quantity"] and t.get("quantity") and \
            abs(float(t["quantity"]) / float(idea["quantity"]) - 1) >= 0.05:
        out["quantity"] = {"was": float(idea["quantity"]), "now": float(t["quantity"])}
    return out or None


def _apply(idea: dict, t: dict, change: dict) -> None:
    cfg = settings()
    market = idea["market"]
    as_of = t["signal_date"]
    valid_until = (as_of + timedelta(days=cfg["validity_sessions"]) if market == "CRYPTO"
                   else np.busday_offset(as_of, cfg["validity_sessions"],
                                         roll="forward").astype(date))
    note = "; ".join(
        f"{k} {v['was']:,.2f} → {v['now']:,.2f}" if k in ("limit", "stop")
        else f"size {v['was']:g} → {v['now']:g}"
        for k, v in change.items())
    db.execute(
        """UPDATE ideas SET limit_price=%s, stop_price=%s, target_price=%s, quantity=%s,
             risk_amount=%s, signal_date=%s, valid_until=%s, reasons=%s, evidence=%s,
             rebased_at=now(), rebase_note=%s,
             first_limit=coalesce(first_limit, %s), first_stop=coalesce(first_stop, %s)
           WHERE id=%s""",
        (t["limit"], t["stop"], t["target"], t.get("quantity"), t.get("risk_amount"),
         as_of, valid_until, json.dumps(t["reasons"] + t["notes"]),
         json.dumps(t.get("evidence") or {}, default=str), note,
         float(idea["limit_price"]), float(idea["stop_price"]), idea["id"]))


def _brief(idea: dict) -> dict:
    return {"id": idea["id"], "symbol": idea["symbol"], "strategy": idea["strategy"],
            "market": idea["market"], "broker": idea["broker"],
            "currency": idea["currency"]}
