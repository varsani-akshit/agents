"""Pull every connected broker and keep what came back.

Each pull is stored as a snapshot, failures included. The page reads the
newest *good* snapshot per broker and says how old it is, so a broker that has
stopped answering shows as stale rather than quietly passing off yesterday's
numbers as live.
"""
from __future__ import annotations

import json
import logging

import db
from brokers import NotConnected, available, coinbase, kite, webull

log = logging.getLogger("alfred.portfolio")

ADAPTERS = {"kite": kite, "webull": webull, "coinbase": coinbase}


def pull(broker: str) -> dict:
    """One snapshot from one broker, recorded whatever the outcome."""
    mod = ADAPTERS[broker]
    try:
        snap = mod.snapshot()
    except NotConnected as exc:
        # Not an error worth a stored row every 15 minutes: nobody has logged in.
        return {"broker": broker, "ok": False, "connected": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - recorded, then reported
        log.warning("%s pull failed: %s", broker, exc)
        db.execute(
            "INSERT INTO portfolio_snapshots (broker, ok, error) VALUES (%s, false, %s)",
            (broker, f"{type(exc).__name__}: {exc}"[:1000]))
        return {"broker": broker, "ok": False, "connected": True, "error": str(exc)}

    row = db.one(
        "INSERT INTO portfolio_snapshots (broker, ok, funds) VALUES (%s, true, %s) RETURNING id",
        (broker, json.dumps(snap.funds, default=str)))
    db.executemany(
        """INSERT INTO portfolio_lines
           (snapshot_id, kind, account, symbol, exchange, quantity, avg_price,
            last_price, pnl, day_change_pct, currency, product, raw)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        [(row["id"], ln.kind, ln.account, ln.symbol, ln.exchange, ln.quantity,
          ln.avg_price, ln.last_price, ln.pnl, ln.day_change_pct, ln.currency,
          ln.product, json.dumps(ln.raw, default=str)) for ln in snap.lines])
    return {"broker": broker, "ok": True, "connected": True, "lines": len(snap.lines)}


def pull_all() -> list[dict]:
    return [pull(b) for b in available()]


def latest(broker: str) -> dict | None:
    """The newest good snapshot for a broker, with its lines and the last failure."""
    snap = db.one(
        """SELECT id, taken_at, funds FROM portfolio_snapshots
           WHERE broker = %s AND ok ORDER BY taken_at DESC LIMIT 1""", (broker,))
    failed = db.one(
        """SELECT taken_at, error FROM portfolio_snapshots
           WHERE broker = %s AND NOT ok ORDER BY taken_at DESC LIMIT 1""", (broker,))
    if not snap:
        return {"snapshot": None, "lines": [], "failed": failed} if failed else None
    lines = db.query(
        """SELECT kind, account, symbol, exchange, quantity, avg_price, last_price,
                  pnl, day_change_pct, currency, product
           FROM portfolio_lines WHERE snapshot_id = %s
           ORDER BY kind, abs(coalesce(last_price, avg_price, 0) * quantity) DESC""",
        (snap["id"],))
    for ln in lines:
        price = ln["last_price"] if ln["last_price"] is not None else ln["avg_price"]
        # Value only for holdings. A futures position's price × quantity is
        # notional exposure, not money you have — counting it would put a lot
        # of silver in the INR total that was only ever margin. Positions
        # report P&L, which is the number that is actually yours.
        ln["value"] = (float(price) * float(ln["quantity"])
                       if price is not None and ln["kind"] == "holding" else None)
    if failed and failed["taken_at"] < snap["taken_at"]:
        failed = None
    return {"snapshot": snap, "lines": lines, "failed": failed}


def totals(lines: list[dict]) -> dict:
    """Market value and unrealised P&L per currency. No FX conversion: an INR
    total next to a USD total is honest; a blended figure needs a rate and a
    timestamp this page does not have."""
    out: dict[str, dict] = {}
    for ln in lines:
        t = out.setdefault(ln["currency"] or "?", {"value": 0.0, "pnl": 0.0})
        t["value"] += ln["value"] or 0.0
        t["pnl"] += float(ln["pnl"] or 0)
    return out


def book_summary(max_lines: int = 25) -> str:
    """The owner's book as plain text, for models that should reason about it."""
    parts = []
    for broker in ADAPTERS:
        got = latest(broker)
        if not got or not got["snapshot"]:
            continue
        parts.append(f"{broker} (as of {got['snapshot']['taken_at']:%Y-%m-%d %H:%M %Z}):")
        for ln in got["lines"][:max_lines]:
            parts.append(
                f"  {ln['kind']:<8} {ln['symbol']:<14} qty {float(ln['quantity']):g} "
                f"avg {ln['avg_price']} last {ln['last_price']} {ln['currency']} "
                f"pnl {ln['pnl']}")
    return "\n".join(parts)
