"""Webull OpenAPI — accounts, balances and positions, read-only.

Uses Webull's official SDK, because every request carries an HMAC signature
whose construction Webull documents only through that SDK; hand-rolling it
would be guesswork against a live brokerage account.

Session model: the SDK mints an access token and Webull holds it PENDING
until you approve it in the Webull app, once. After that the token is reused
from disk. `connect()` waits for that approval and is meant to be run by a
person at a terminal (`mia broker connect webull`). Scheduled pulls wait only
briefly, so an unapproved token surfaces as "not connected" instead of
stalling the scheduler for five minutes.

Webull's reference pages render their response schemas client-side, so the
field names below accept the plausible spellings and every line keeps its raw
payload. `mia broker show webull --raw` prints exactly what came back.
"""
from __future__ import annotations

import logging
import time

import config
from brokers import Line, NotConnected, Snapshot, num, secret_path

log = logging.getLogger("alfred.webull")

NAME = "webull"
HOSTS = {"au": "api.webull.com.au", "us": "api.webull.com"}


def configured() -> bool:
    return bool(config.WEBULL_APP_KEY and config.WEBULL_APP_SECRET)


def _token_dir() -> str:
    path = secret_path("webull")
    path.mkdir(mode=0o700, exist_ok=True)
    return str(path)


def _client(wait_seconds: int):
    """A TradeClient. Constructing it is what creates or checks the token."""
    if not configured():
        raise RuntimeError("WEBULL_APP_KEY and WEBULL_APP_SECRET are not set")
    from webull.core.client import ApiClient
    from webull.core.exception.exceptions import ClientException
    from webull.trade.trade_client import TradeClient

    region = config.WEBULL_REGION
    api = ApiClient(config.WEBULL_APP_KEY, config.WEBULL_APP_SECRET, region,
                    token_check_duration_seconds=wait_seconds,
                    token_check_interval_seconds=5)
    if region in HOSTS:
        api.add_endpoint(region, HOSTS[region])
    api.set_token_dir(_token_dir())
    try:
        return TradeClient(api)
    except ClientException as exc:
        raise NotConnected(
            "Webull token is not approved — run `mia broker connect webull` and "
            f"approve it in the Webull app ({exc})") from exc


def connect(wait_seconds: int = 300) -> list[dict]:
    """Create the token and wait for approval in the Webull app."""
    client = _client(wait_seconds)
    return _accounts(client)


def _token_file() -> dict | None:
    """The SDK's token file: token, expiry, status — one per line."""
    for f in secret_path("webull").glob("*"):
        try:
            token, expires, status = (f.read_text().splitlines() + ["", "", ""])[:3]
        except OSError:
            continue
        if token:
            exp = int(expires) if expires.isdigit() else 0
            if exp > 1e12:  # milliseconds
                exp //= 1000
            return {"status": status, "expires": exp}
    return None


def status() -> dict:
    tok = _token_file() if configured() else None
    live = bool(tok and tok["status"] == "NORMAL"
                and (not tok["expires"] or tok["expires"] > time.time()))
    return {
        "broker": NAME, "label": f"Webull ({config.WEBULL_REGION.upper()})",
        "configured": configured(), "connected": live, "account": None,
        "expires_at": tok and tok["expires"] and time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(tok["expires"])),
        "connect_hint": "Run `mia broker connect webull` on the server, then approve in the Webull app.",
    }


def _ok(resp, what: str):
    if resp.status_code != 200:
        raise RuntimeError(f"Webull {what}: HTTP {resp.status_code} {resp.text[:200]}")
    return resp.json()


def _rows(payload) -> list[dict]:
    """Lists arrive bare or wrapped, depending on the endpoint and region."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("holdings", "positions", "data", "list", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _pick(d: dict, *keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _accounts(client) -> list[dict]:
    return _rows(_ok(client.account_v2.get_account_list(), "account list"))


def parse_positions(rows: list[dict], account: str | None) -> list[Line]:
    out = []
    for p in rows:
        qty = num(_pick(p, "quantity", "qty", "position", "total_quantity"))
        sym = _pick(p, "symbol", "ticker", "instrument_symbol")
        if not qty or not sym:
            continue
        out.append(Line(
            kind="holding", symbol=str(sym), account=account, quantity=qty,
            exchange=_pick(p, "exchange", "exchange_code"),
            avg_price=num(_pick(p, "cost_price", "avg_price", "average_price", "unit_cost")),
            last_price=num(_pick(p, "last_price", "price", "market_price")),
            pnl=num(_pick(p, "unrealized_profit_loss", "unrealized_pnl", "profit_loss")),
            day_change_pct=num(_pick(p, "day_change_pct", "day_profit_loss_rate")),
            currency=_pick(p, "currency") or "USD",
            product=_pick(p, "instrument_type", "category"), raw=p,
        ))
    return out


def parse_balance(b: dict) -> dict:
    assets = _rows(b.get("account_currency_assets")) or [{}]
    first = assets[0] if isinstance(assets[0], dict) else {}
    return {
        "currency": _pick(b, "total_asset_currency", "currency") or first.get("currency") or "USD",
        "cash": num(_pick(b, "total_cash_balance", "cash_balance") or first.get("cash_balance")),
        "available": num(_pick(b, "buying_power", "total_buying_power")
                         or first.get("buying_power")),
        "total": num(_pick(b, "total_asset", "total_net_liquidation_value", "net_liquidation")),
        "market_value": num(_pick(b, "total_market_value", "market_value")),
    }


def snapshot(wait_seconds: int = 10) -> Snapshot:
    client = _client(wait_seconds)
    lines, funds = [], {}
    for acct in _accounts(client):
        aid = acct.get("account_id")
        if not aid:
            continue
        bal = _ok(client.account_v2.get_account_balance(aid), "balance")
        funds[aid] = parse_balance(bal if isinstance(bal, dict) else {})
        pos = _ok(client.account_v2.get_account_position(aid), "positions")
        lines += parse_positions(_rows(pos), aid)
    # One account is the common case; flatten so the page reads the same as Kite.
    if len(funds) == 1:
        funds = next(iter(funds.values()))
    return Snapshot(NAME, lines, funds)


def raw(wait_seconds: int = 10) -> dict:
    """Unparsed responses, for checking the field mapping against a live account."""
    client = _client(wait_seconds)
    out = {"accounts": _ok(client.account_v2.get_account_list(), "account list")}
    for acct in _rows(out["accounts"]):
        aid = acct.get("account_id")
        if aid:
            out[aid] = {
                "balance": _ok(client.account_v2.get_account_balance(aid), "balance"),
                "positions": _ok(client.account_v2.get_account_position(aid), "positions"),
            }
    return out
