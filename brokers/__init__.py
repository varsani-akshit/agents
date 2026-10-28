"""Broker connections — read-only.

Each adapter answers three questions about a real account: what is held, what
is open, and how much cash is free. None of them can place, modify or cancel
an order. That is not an oversight to be patched in an adapter later: when
Alfred trades, orders will go through a single risk gate in code that checks
limits before any broker sees them, and the adapters stay read-only underneath.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config


class NotConnected(RuntimeError):
    """The broker is configured but has no live session — a person must log in."""


@dataclass
class Line:
    """One holding or open position, normalised across brokers."""
    kind: str                      # "holding" | "position"
    symbol: str
    quantity: float
    exchange: str | None = None
    account: str | None = None
    avg_price: float | None = None
    last_price: float | None = None
    pnl: float | None = None
    day_change_pct: float | None = None
    currency: str | None = None
    product: str | None = None
    raw: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class Snapshot:
    broker: str
    lines: list[Line]
    funds: dict


# ─────────────────────────────── token files ────────────────────────────────
def secret_path(name: str) -> Path:
    """A file under SECRETS, with the directory created owner-only."""
    config.SECRETS.mkdir(mode=0o700, exist_ok=True)
    os.chmod(config.SECRETS, 0o700)
    return config.SECRETS / name


def write_secret(name: str, payload: dict) -> None:
    """Write a token file readable by this user alone.

    Opened with 0600 from the start rather than chmod-ed afterwards, so there
    is no moment where the token sits on disk world-readable.
    """
    path = secret_path(name)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)


def read_secret(name: str) -> dict | None:
    path = secret_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def num(v) -> float | None:
    """Brokers send numbers as numbers, strings, or empty strings."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def available() -> list[str]:
    """Brokers with credentials configured, connected or not."""
    out = []
    if config.KITE_API_KEY and config.KITE_API_SECRET:
        out.append("kite")
    if config.WEBULL_APP_KEY and config.WEBULL_APP_SECRET:
        out.append("webull")
    return out
