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
import sys
import threading
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
    from webull.core.exception.exceptions import ClientException, ServerException
    from webull.trade.trade_client import TradeClient

    region = config.WEBULL_REGION
    # Checking every 5s (the SDK default) trips Webull's rate limit while you
    # are still reaching for your phone, and a 429 aborts the wait with the
    # pending token unsaved. 15s is slow enough to stay under it.
    api = ApiClient(config.WEBULL_APP_KEY, config.WEBULL_APP_SECRET, region,
                    token_check_duration_seconds=wait_seconds,
                    token_check_interval_seconds=15)
    # Left alone, the SDK writes webull_trade_sdk.log into whatever directory it
    # runs from, world-readable, and logs whole requests on error — headers,
    # signature and the token in the body. Registering a logger of our own is
    # what stops it creating that file; this one passes only CRITICAL, since
    # errors reach us as exceptions anyway.
    api.set_stream_logger(log_level=logging.CRITICAL, logger_name="webull",
                          stream=sys.stderr)
    if region in HOSTS:
        api.add_endpoint(region, HOSTS[region])
    api.set_token_dir(_token_dir())
    try:
        return TradeClient(api)
    except ClientException as exc:
        raise NotConnected(
            "Webull token is not approved — run `mia broker connect webull` and "
            f"approve it in the Webull app ({exc})") from exc
    except ServerException as exc:
        # Usually TOO_MANY_REQUESTS. Message only: str(exc) carries no token,
        # unlike the SDK's own error log.
        raise NotConnected(f"Webull refused the request ({exc.get_error_code()}): "
                           "wait a minute and try again") from exc


def connect(wait_seconds: int = 300) -> list[dict]:
    """Create the token and wait for approval in the Webull app."""
    client = _client(wait_seconds)
    _lock_token_files()
    return _accounts(client)


def _lock_token_files() -> None:
    """The SDK writes its token file 0644. The directory is already 0700, but
    a token should not depend on one permission being right."""
    for f in secret_path("webull").glob("*"):
        f.chmod(0o600)


# One approval request at a time, started from the Portfolio page. Each
# attempt sends a fresh code to the phone, and two attempts polling at once
# is what tripped Webull's rate limit — so a second click while one is
# waiting does nothing.
_attempt: dict = {"state": None, "started": None, "error": None}
_attempt_lock = threading.Lock()


def start_connect(wait_seconds: int = 300) -> bool:
    """Begin an approval request in the background. False if one is running."""
    with _attempt_lock:
        if _attempt["state"] == "waiting":
            return False
        _attempt.update(state="waiting", started=time.time(), error=None)

    def run():
        try:
            connect(wait_seconds)
        except Exception as exc:  # noqa: BLE001 - shown on the page
            _attempt.update(state="failed", error=str(exc)[:300])
            return
        _attempt.update(state="connected")
        from brokers import portfolio

        portfolio.pull(NAME)

    threading.Thread(target=run, name="webull-connect", daemon=True).start()
    return True


def attempt() -> dict:
    a = dict(_attempt)
    a["waited"] = int(time.time() - a["started"]) if a["started"] else 0
    return a


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
        "connect_hint": "Connect, then approve the request in the Webull app.",
        "attempt": attempt(),
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
            # Positions carry no exchange; the currency says which market.
            exchange=_pick(p, "exchange", "exchange_code")
            or {"AUD": "ASX", "USD": "US", "HKD": "HKEX"}.get(_pick(p, "currency") or ""),
            avg_price=num(_pick(p, "cost_price", "avg_price", "average_price", "unit_cost")),
            last_price=num(_pick(p, "last_price", "price", "market_price")),
            pnl=num(_pick(p, "unrealized_profit_loss", "unrealized_pnl", "profit_loss")),
            day_change_pct=num(_pick(p, "day_change_pct", "day_profit_loss_rate")),
            currency=_pick(p, "currency") or "USD",
            product=_pick(p, "instrument_type", "category"), raw=p,
        ))
    return out


def parse_balance(b: dict) -> dict:
    """Account totals, in the account's base currency.

    Checked against a live Webull AU account: there is no single "total"
    field, and buying power is only given per currency in
    `account_currency_assets` — which lists every currency the account could
    hold, HKD and CNH included, zeros and all. So "available" is the entry for
    the base currency, and the total is holdings plus cash.
    """
    base = _pick(b, "total_asset_currency", "currency") or "USD"
    per_ccy = [a for a in _rows(b.get("account_currency_assets")) if isinstance(a, dict)]
    home = next((a for a in per_ccy if a.get("currency") == base), {})
    market = num(_pick(b, "total_market_value", "market_value"))
    cash = num(_pick(b, "total_cash_balance", "cash_balance"))
    total = num(_pick(b, "total_asset", "total_net_liquidation_value", "net_liquidation"))
    if total is None and market is not None:
        total = market + (cash or 0)
    return {
        "currency": base,
        "cash": cash,
        "available": num(home.get("buying_power")),
        "total": total,
        "market_value": market,
        "unrealised": num(_pick(b, "total_unrealized_profit_loss")),
        # Other currencies that actually hold something, e.g. USD for US stocks.
        "other": {a["currency"]: {"cash": num(a.get("cash_balance")),
                                  "market_value": num(a.get("market_value"))}
                  for a in per_ccy
                  if a.get("currency") != base and (num(a.get("market_value"))
                                                    or num(a.get("cash_balance")))},
    }


def snapshot(wait_seconds: int = 10) -> Snapshot:
    client = _client(wait_seconds)
    _lock_token_files()
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
