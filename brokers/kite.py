"""Zerodha Kite Connect — holdings, positions and margins, read-only.

Kite's session model drives the design. There is no refresh token: a person
logs in through Zerodha's own page, Zerodha redirects back with a one-time
`request_token`, and that is exchanged (with a checksum signed by the API
secret) for an access token that dies at about 06:00 IST the next morning.
So connecting is a daily tap, by design — Zerodha does not permit automating
its login, and Alfred does not try.

Plain REST over httpx rather than the `kiteconnect` package: three GETs and one
POST do not justify a dependency, and keeping the client this small makes it
obvious that nothing here can place an order.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, time, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

import config
from brokers import Line, NotConnected, Snapshot, num, read_secret, write_secret

log = logging.getLogger("alfred.kite")

NAME = "kite"
API = "https://api.kite.trade"
LOGIN = "https://kite.zerodha.com/connect/login"
IST = ZoneInfo("Asia/Kolkata")
_TOKEN_FILE = "kite_session.json"


def configured() -> bool:
    return bool(config.KITE_API_KEY and config.KITE_API_SECRET)


def login_url() -> str:
    """Zerodha's login page. It redirects to the URL registered on the Kite app."""
    return f"{LOGIN}?{urlencode({'v': 3, 'api_key': config.KITE_API_KEY})}"


def _expiry(now: datetime | None = None) -> datetime:
    """The next 06:00 IST — when Kite invalidates every access token."""
    now = (now or datetime.now(IST)).astimezone(IST)
    six = datetime.combine(now.date(), time(6, 0), tzinfo=IST)
    return six if now < six else six + timedelta(days=1)


def checksum(request_token: str) -> str:
    """SHA-256 of api_key + request_token + api_secret, as Kite specifies."""
    raw = config.KITE_API_KEY + request_token + config.KITE_API_SECRET
    return hashlib.sha256(raw.encode()).hexdigest()


def exchange(request_token: str) -> dict:
    """Swap the one-time request token for today's access token and store it."""
    if not configured():
        raise RuntimeError("KITE_API_KEY and KITE_API_SECRET are not set")
    r = httpx.post(
        f"{API}/session/token",
        data={"api_key": config.KITE_API_KEY, "request_token": request_token,
              "checksum": checksum(request_token)},
        headers={"X-Kite-Version": "3"}, timeout=20,
    )
    body = r.json()
    if r.status_code != 200 or body.get("status") != "success":
        raise RuntimeError(f"Kite refused the login: {body.get('message') or r.status_code}")
    data = body["data"]
    session = {
        "access_token": data["access_token"],
        "user_id": data.get("user_id"),
        "user_name": data.get("user_name"),
        "connected_at": datetime.now(IST).isoformat(),
        "expires_at": _expiry().isoformat(),
    }
    write_secret(_TOKEN_FILE, session)
    log.info("kite connected for %s until %s", session["user_id"], session["expires_at"])
    return {k: v for k, v in session.items() if k != "access_token"}


def session() -> dict | None:
    """Today's session if there is one and it has not passed its expiry."""
    s = read_secret(_TOKEN_FILE)
    if not s or not s.get("access_token"):
        return None
    try:
        if datetime.fromisoformat(s["expires_at"]) <= datetime.now(IST):
            return None
    except (KeyError, ValueError):
        return None
    return s


def status() -> dict:
    s = session()
    return {
        "broker": NAME, "label": "Zerodha Kite", "configured": configured(),
        "connected": bool(s), "account": s and s.get("user_id"),
        "expires_at": s and s.get("expires_at"),
        "connect_hint": "Log in with Zerodha once each morning — Kite tokens end at 06:00 IST.",
    }


def _get(path: str) -> dict | list:
    s = session()
    if not s:
        raise NotConnected("Kite is not connected today — log in from the Portfolio page")
    r = httpx.get(
        f"{API}{path}", timeout=20,
        headers={"X-Kite-Version": "3",
                 "Authorization": f"token {config.KITE_API_KEY}:{s['access_token']}"},
    )
    body = r.json()
    if r.status_code == 403 or body.get("error_type") == "TokenException":
        # Revoked early (logged out elsewhere, or Zerodha's nightly flush).
        # Forget it, so the page asks for a login instead of retrying a corpse.
        write_secret(_TOKEN_FILE, {})
        raise NotConnected("Kite session expired — log in again")
    if r.status_code != 200 or body.get("status") != "success":
        raise RuntimeError(f"Kite {path}: {body.get('message') or r.status_code}")
    return body["data"]


def parse_holdings(rows: list[dict], account: str | None) -> list[Line]:
    out = []
    for h in rows:
        # T1 shares were bought yesterday and have not settled: still yours.
        qty = (num(h.get("quantity")) or 0) + (num(h.get("t1_quantity")) or 0)
        if not qty:
            continue
        out.append(Line(
            kind="holding", symbol=h["tradingsymbol"], exchange=h.get("exchange"),
            account=account, quantity=qty, avg_price=num(h.get("average_price")),
            last_price=num(h.get("last_price")), pnl=num(h.get("pnl")),
            day_change_pct=num(h.get("day_change_percentage")), currency="INR",
            product=h.get("product"), raw=h,
        ))
    return out


def parse_positions(data: dict, account: str | None) -> list[Line]:
    """Open risk that is not already a holding.

    Delivery (CNC) trades appear here on the day they happen, and Kite's
    holdings already account for one side of them, so they need care:
      - sold today: holdings already show the reduced quantity (a full exit
        reads 0), so the -20 here is the same sale again — skip it, or it
        reads as a short an NRI delivery account cannot even hold;
      - bought today: not in holdings until it settles (not even as T1), so
        it is counted as a holding here, or it would vanish for a day.
    Intraday and F&O positions are genuine open positions and stay as such.
    """
    out = []
    # `net` is the running position; `day` is today's activity only. Closed
    # positions stay in the list with quantity 0 — those are history, not risk.
    for p in (data or {}).get("net") or []:
        qty = num(p.get("quantity")) or 0
        if not qty:
            continue
        delivery = p.get("product") == "CNC"
        if delivery and qty < 0:
            continue
        out.append(Line(
            kind="holding" if delivery else "position",
            symbol=p["tradingsymbol"], exchange=p.get("exchange"),
            account=account, quantity=qty, avg_price=num(p.get("average_price")),
            last_price=num(p.get("last_price")), pnl=num(p.get("pnl")),
            currency="INR", product="CNC · bought today" if delivery else p.get("product"),
            raw=p,
        ))
    return out


def parse_margins(data: dict) -> dict:
    eq = (data or {}).get("equity") or {}
    co = (data or {}).get("commodity") or {}
    avail = eq.get("available") or {}
    return {
        "currency": "INR",
        # `cash` in Kite's response is the opening balance: it ignores money
        # added today. `live_balance` includes today's pay-in and trades.
        "cash": num(avail.get("live_balance")) if avail.get("live_balance") is not None
        else num(avail.get("cash")),
        "opening_balance": num(avail.get("opening_balance")),
        "payin_today": num(avail.get("intraday_payin")),
        "available": num(eq.get("net")),
        "used": num((eq.get("utilised") or {}).get("debits")),
        "commodity_available": num(co.get("net")),
    }


def snapshot() -> Snapshot:
    account = (session() or {}).get("user_id")
    lines = parse_holdings(_get("/portfolio/holdings"), account)
    lines += parse_positions(_get("/portfolio/positions"), account)
    return Snapshot(NAME, lines, parse_margins(_get("/user/margins")))
