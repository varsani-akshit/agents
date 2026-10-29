"""The heartbeat: every five minutes, what you hold and the tickets you placed.

The strategies decide at the close, on finished bars, and that does not
change here. What changes is how fast Alfred tells you something happened
in between:

- a ticket's limit was reached (it has probably filled);
- a filled ticket's stop or target was touched;
- something you hold moved hard — 2.5 or more of its typical daily ranges
  from yesterday's close — in either direction;
- the news matcher found news on a name you hold or have a ticket on, and
  read it as bad for a shareholder.

Each alert is stored once (alerts table) and all new ones go out in one
email. Prices are Yahoo's one-minute bars for the exchange's own session
today, and Coinbase's for crypto; a market that is closed is skipped, so
yesterday's session is never mistaken for today's. The ideas ledger still
settles every ticket at the close, by the same rules as the backtest —
these alerts only say it sooner.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import db
from advisor import CURRENCY, bare, bars, market_of, venue

log = logging.getLogger("alfred.watch")

SHOCK_ATR = 2.5
NEWS_MIN_P = 0.7
_ZONE = {"NSE": "Asia/Kolkata", "BSE": "Asia/Kolkata", "ASX": "Australia/Sydney",
         "US": "America/New_York", "CRYPTO": "UTC"}


# ───────────────────────────────── prices ───────────────────────────────────
def today_range(frame, market: str, now: datetime | None = None) -> dict | None:
    """Today's session in the exchange's own time: last, high, low. None if
    the bars are from another day (the market has not opened yet)."""
    if frame is None or frame.empty:
        return None
    tz = ZoneInfo(_ZONE[market])
    local = frame.copy()
    local.index = local.index.tz_convert(tz)
    today = (now or datetime.now(tz)).astimezone(tz).date()
    local = local[local.index.date == today]
    if local.empty:
        return None
    return {"last": float(local["Close"].iloc[-1]), "high": float(local["High"].max()),
            "low": float(local["Low"].min()), "at": local.index[-1].to_pydatetime()}


def quotes(symbols: list[str]) -> dict[str, dict]:
    """Today's range for each symbol whose market is trading today."""
    out: dict[str, dict] = {}
    stocks = [s for s in symbols if market_of(s) != "CRYPTO"]
    if stocks:
        import pandas as pd
        import yfinance as yf

        try:
            data = yf.download(stocks, period="1d", interval="1m", group_by="ticker",
                               progress=False, prepost=False, threads=True)
        except Exception as exc:  # noqa: BLE001 - one missed beat is not an outage
            log.warning("quotes failed: %s", exc)
            data = None
        for s in stocks:
            if data is None:
                break
            try:
                f = data[s] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            r = today_range(f.dropna(subset=["Close"]), market_of(s))
            if r:
                out[s] = r
    for s in (s for s in symbols if market_of(s) == "CRYPTO"):
        r = _coinbase_today(s)
        if r:
            out[s] = r
    return out


def _coinbase_today(product: str) -> dict | None:
    from data import crypto

    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        rows = crypto.fetch_candles(product, 300, start, now)
    except Exception as exc:  # noqa: BLE001
        log.info("coinbase %s: %s", product, exc)
        return None
    if not rows:
        return None
    return {"last": rows[-1][5], "high": max(r[3] for r in rows), "low": min(r[4] for r in rows),
            "at": rows[-1][1]}


# ───────────────────────────────── checks ───────────────────────────────────
def ticket_alerts(ideas: list[dict], q: dict[str, dict]) -> list[dict]:
    """Limit reached, stop or target touched — for tickets you were sent."""
    out = []
    for i in ideas:
        r = q.get(i["symbol"])
        if not r:
            continue
        ccy, name = i["currency"], bare(i["symbol"])
        base = {"symbol": i["symbol"], "ref": str(i["id"])}
        if i["status"] == "open" and r["low"] <= i["limit_price"]:
            out.append({**base, "kind": "fill",
                        "text": f"{name}: the limit {_m(ccy, i['limit_price'])} was reached "
                                f"(low {_m(ccy, r['low'])}) — if your order is in, it has probably filled. "
                                f"Place the stop at {_m(ccy, i['stop_price'])} if you have not."})
        if i["status"] == "filled":
            if r["low"] <= i["stop_price"]:
                out.append({**base, "kind": "stop",
                            "text": f"{name}: the stop {_m(ccy, i['stop_price'])} was touched "
                                    f"(low {_m(ccy, r['low'])}, now {_m(ccy, r['last'])}). If your stop "
                                    "order was in, you are out; if not, this is the exit the ticket set."})
            if i["target_price"] and r["high"] >= i["target_price"]:
                out.append({**base, "kind": "target",
                            "text": f"{name}: the target {_m(ccy, i['target_price'])} was reached "
                                    f"(high {_m(ccy, r['high'])}). Take the profit as the ticket set."})
    return out


def shock_alerts(held: dict[str, dict], q: dict[str, dict], ranges: dict[str, tuple]) -> list[dict]:
    """A holding moving 2.5 typical days or more from yesterday's close."""
    from advisor.scan import is_core

    out = []
    for sym in held:
        r, base = q.get(sym), ranges.get(sym)
        if not r or not base:
            continue
        prev, atr = base
        if not atr:
            continue
        move = (r["last"] - prev) / atr
        if abs(move) < SHOCK_ATR:
            continue
        pct = (r["last"] / prev - 1) * 100
        ccy = CURRENCY[market_of(sym)]
        tail = (" It is a core holding: no action is suggested on a move alone."
                if is_core(sym) else
                " Check the news before acting; the trailing stop is judged at the close.")
        out.append({"symbol": sym, "ref": "up" if move > 0 else "down", "kind": "shock",
                    "text": f"{bare(sym)} on {venue(sym)}: {pct:+.1f}% today at {_m(ccy, r['last'])} — "
                            f"{abs(move):.1f} times its usual daily range.{tail}"})
    return out


def news_alerts(symbols: set[str], since_minutes: int = 30) -> list[dict]:
    """Bad news the matcher found on a name you hold or have a ticket on."""
    if not symbols:
        return []
    rows = db.query(
        """SELECT m.doc_id, m.symbol, m.name, m.impact, m.impact_p, d.title, d.source, d.url
           FROM news_matches m JOIN documents d ON d.id = m.doc_id
           WHERE m.created_at > now() - make_interval(mins => %s)
             AND m.symbol = ANY(%s) AND m.impact = 'negative' AND m.impact_p >= %s""",
        (since_minutes, sorted(symbols), NEWS_MIN_P))
    return [{"symbol": r["symbol"], "ref": str(r["doc_id"]), "kind": "news",
             "text": f"{r['name']}: \"{r['title']}\" ({r['source']}) reads as bad for shareholders "
                     f"({r['impact_p']:.0%}).", "detail": {"url": r["url"]}} for r in rows]


def _m(ccy: str, v: float) -> str:
    sym = {"INR": "₹", "AUD": "A$", "USD": "US$"}.get(ccy, ccy + " ")
    return f"{sym}{v:,.2f}"


def _ranges(symbols: list[str]) -> dict[str, tuple]:
    """Yesterday's close and the 22-day ATR, from finished daily bars."""
    from advisor.strategies import indicators

    out = {}
    for s in symbols:
        x = bars.load(s, years=1)
        if len(x) < 30:
            continue
        x = indicators(x)
        out[s] = (float(x["close"].iloc[-1]), float(x["atr22"].iloc[-1]))
    return out


# ───────────────────────────────── the beat ─────────────────────────────────
def store(found: list[dict]) -> list[dict]:
    """Keep the ones not seen before today; return those."""
    new = []
    for a in found:
        day = datetime.now(timezone.utc).date()
        row = db.one(
            """INSERT INTO alerts (kind, symbol, ref, day, text, detail)
               VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id""",
            (a["kind"], a["symbol"], a.get("ref", ""), day, a["text"],
             json.dumps(a.get("detail") or {}, default=str)))
        if row:
            new.append({**a, "id": row["id"]})
    return new


SUBJECT = {"stop": "STOP", "target": "TARGET", "fill": "filled", "shock": "big move", "news": "news"}
ORDER = ["stop", "target", "fill", "shock", "news"]


def compose(new: list[dict]) -> tuple[str, str]:
    from advisor.report import HEADER

    new = sorted(new, key=lambda a: ORDER.index(a["kind"]))
    bits = []
    for a in new:
        b = f"{SUBJECT[a['kind']]} {bare(a['symbol'])}"
        if b not in bits:
            bits.append(b)
    subject = "Alfred · now: " + " · ".join(bits[:4]) + (" …" if len(bits) > 4 else "")
    lines = [HEADER, ""] + [f"• {a['text']}" + (f"\n  {a['detail']['url']}" if (a.get("detail") or {}).get("url") else "")
                            for a in new]
    return subject, "\n".join(lines)


def beat() -> dict:
    """One heartbeat. Returns what it found and whether it emailed."""
    from notify import email

    ideas = db.query("SELECT * FROM ideas WHERE status IN ('open', 'filled')")
    held = bars.holdings()
    symbols = sorted({i["symbol"] for i in ideas} | set(held))
    q = quotes(symbols)
    found = ticket_alerts(ideas, q)
    found += shock_alerts(held, q, _ranges([s for s in held if s in q]))
    found += news_alerts(set(held) | {i["symbol"] for i in ideas})
    new = store(found)
    sent = None
    if new:
        subject, text = compose(new)
        sent = email.send(subject, text, key="alerts-" + "-".join(str(a["id"]) for a in new))
        if sent:
            db.execute("UPDATE alerts SET emailed_at = now() WHERE id = ANY(%s)", ([a["id"] for a in new],))
    return {"watched": len(symbols), "trading": len(q), "found": len(found), "new": len(new),
            "emailed": bool(sent)}
