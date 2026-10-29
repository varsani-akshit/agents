"""Coinbase — read-only balances through the Advanced Trade API.

The key is a CDP key (Ed25519) that must have been created with the View
permission only. `status()` asks Coinbase what the key may do and reports a
key that can trade or transfer as *not* connected, so a wrongly-made key is
refused rather than quietly used. Nothing here places an order.
"""
from __future__ import annotations

import base64
import secrets
import time
from datetime import datetime, timedelta, timezone

import httpx

import config
from brokers import Line, NotConnected, Snapshot

NAME = "coinbase"
API = "https://api.coinbase.com"
PUBLIC = "https://api.exchange.coinbase.com"
QUOTE = "USD"                     # crypto is priced in USD; fiat balances keep their own
STABLE = {"USDC", "USDT", "DAI", "PYUSD"}
_perm_cache: dict = {}            # {"at": datetime, "perms": dict}


def configured() -> bool:
    return bool(config.COINBASE_API_KEY_ID and config.COINBASE_API_PRIVATE_KEY)


def _pem() -> bytes:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    raw = base64.b64decode(config.COINBASE_API_PRIVATE_KEY)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32])
    return key.private_bytes(serialization.Encoding.PEM,
                             serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _token(method: str, path: str) -> str:
    import jwt

    now = int(time.time())
    kid = config.COINBASE_API_KEY_ID
    return jwt.encode(
        {"sub": kid, "iss": "cdp", "nbf": now, "exp": now + 120,
         "uri": f"{method} api.coinbase.com{path}"},
        _pem(), algorithm="EdDSA", headers={"kid": kid, "nonce": secrets.token_hex(16)})


def _get(path: str, params: dict | None = None) -> dict:
    if not configured():
        raise NotConnected("Coinbase keys are not in .env")
    r = httpx.get(f"{API}{path}", params=params, timeout=20,
                  headers={"Authorization": f"Bearer {_token('GET', path)}"})
    if r.status_code == 401:
        raise NotConnected("Coinbase rejected the key — make a new View-only key")
    if r.status_code != 200:
        raise RuntimeError(f"Coinbase {path}: HTTP {r.status_code} {r.text[:120]}")
    return r.json()


def permissions(max_age: timedelta = timedelta(minutes=10)) -> dict:
    """What the key may do, cached briefly — the page asks on every load."""
    now = datetime.now(timezone.utc)
    if _perm_cache and now - _perm_cache["at"] < max_age:
        return _perm_cache["perms"]
    perms = _get("/api/v3/brokerage/key_permissions")
    _perm_cache.update(at=now, perms=perms)
    return perms


def view_only(perms: dict) -> bool:
    return bool(perms.get("can_view")) and not perms.get("can_trade") and not perms.get("can_transfer")


def status() -> dict:
    out = {"broker": NAME, "label": "Coinbase", "configured": configured(),
           "connected": False, "account": None, "expires_at": None,
           "connect_hint": "Make a View-only API key at portal.cdp.coinbase.com and add it to .env."}
    if not configured():
        return out
    try:
        perms = permissions()
    except Exception as exc:  # noqa: BLE001 - the page shows why
        out["error"] = str(exc)
        return out
    if not view_only(perms):
        out["error"] = "this key can trade or transfer — Alfred will not use it; make a View-only key"
        return out
    out["connected"] = True
    out["account"] = (perms.get("portfolio_type") or "").lower() or None
    return out


# ─────────────────────────────── public prices ──────────────────────────────
def ticker(product: str) -> dict:
    """Last price and 24h open from the public exchange feed (no key)."""
    r = httpx.get(f"{PUBLIC}/products/{product}/stats", timeout=15,
                  headers={"user-agent": "alfred"})
    return r.json() if r.status_code == 200 else {}


def snapshot() -> Snapshot:
    perms = permissions()
    if not view_only(perms):
        raise NotConnected("Coinbase key is not View-only")
    accounts: list[dict] = []
    cursor = None
    while True:
        page = _get("/api/v3/brokerage/accounts",
                    {"limit": 250, **({"cursor": cursor} if cursor else {})})
        accounts += page.get("accounts", [])
        cursor = page.get("cursor") if page.get("has_next") else None
        if not cursor:
            break

    lines: list[Line] = []
    funds: dict = {"currency": None, "available": 0.0, "by_currency": {}}
    for a in accounts:
        cur = a.get("currency")
        qty = float(a.get("available_balance", {}).get("value") or 0) + \
            float(a.get("hold", {}).get("value") or 0)
        if not cur or qty <= 0:
            continue
        if a.get("type") == "ACCOUNT_TYPE_FIAT":
            funds["by_currency"][cur] = qty
            continue
        if cur in STABLE:
            last, day = 1.0, 0.0
        else:
            st = ticker(f"{cur}-{QUOTE}")
            last = float(st["last"]) if st.get("last") else None
            day = ((last / float(st["open"]) - 1) * 100
                   if last and st.get("open") and float(st["open"]) else None)
        lines.append(Line(kind="holding", symbol=cur, quantity=qty, exchange="COINBASE",
                          account=a.get("uuid"), last_price=last, day_change_pct=day,
                          currency=QUOTE, product="CRYPTO", raw={"type": a.get("type")}))
    # Free cash: the fiat balance in the account's own currency, else stablecoins.
    if funds["by_currency"]:
        cur, amt = max(funds["by_currency"].items(), key=lambda kv: kv[1])
        funds.update(currency=cur, available=amt)
    else:
        stable = sum(ln.quantity for ln in lines if ln.symbol in STABLE)
        funds.update(currency=QUOTE, available=stable)
    return Snapshot(NAME, lines, funds)
