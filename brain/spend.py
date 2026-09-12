"""Who a model call is charged to, and whether they may make it.

Briefs, prices, the graph and the ingestion cycle are shared infrastructure:
they cost the same whoever is reading, and no reader triggers them. Ask is
different — a reader spends real money on demand, and deep research spends it
several times over in parallel. So Ask is the one surface with a cap.

Attribution is a contextvar rather than an argument threaded through six call
sites, because the thing being attributed is "whatever this request causes",
including calls made deep inside a research fan-out that nobody passed a
username to. `charged_to()` marks the request; every api_calls row written
while it is active carries the owner.

Two deliberate choices:

- The month is counted from api_calls, not from the saved answer's cost. A
  question that errors halfway still burned tokens, and a reader who retried a
  failing question five times has spent that money whether or not an answer
  exists to show for it.

- An unset budget means the default, never unlimited. A permissive NULL is how
  budget systems quietly stop working the day someone adds a user by hand.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone

import config
import db

# None means the work is Alfred's own — a scheduled brief, an ingestion cycle —
# and is charged to nobody.
_OWNER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "alfred_spend_owner", default=None
)


@contextmanager
def charged_to(username: str | None):
    """Attribute every model call made in this context to one reader.

    Threads started with `asyncio.to_thread` or `observe.ctx_submit` inherit
    the context, so a research fan-out is attributed without any of its stages
    knowing a user exists.
    """
    token = _OWNER.set((username or "").strip().lower() or None)
    try:
        yield
    finally:
        _OWNER.reset(token)


def owner() -> str | None:
    """The reader this call is charged to, if any."""
    return _OWNER.get()


def month_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def spent_this_month(username: str) -> float:
    row = db.one(
        "SELECT coalesce(sum(usd), 0) s FROM api_calls "
        "WHERE owner = %s AND created_at >= %s",
        ((username or "").lower(), month_start()),
    )
    return float(row["s"] if row else 0.0)


def budget_for(username: str) -> float | None:
    """The reader's monthly cap in USD, or None for unlimited.

    Admins are unlimited here rather than by carrying a huge number in their
    row, so revoking someone's admin restores their cap in the same action
    instead of leaving them silently uncapped.
    """
    row = db.one(
        "SELECT is_admin, monthly_usd FROM users WHERE username = %s",
        ((username or "").lower(),),
    )
    if not row:
        # Not a known user: the caller has no business spending. Zero, not the
        # default, because this should never happen behind the auth gate.
        return 0.0
    if row["is_admin"]:
        return None
    if row["monthly_usd"] is not None:
        return float(row["monthly_usd"])
    return float(config.READER_MONTHLY_USD)


def status(username: str) -> dict:
    """Spent, allowed and remaining for one reader this month."""
    limit = budget_for(username)
    spent = spent_this_month(username)
    return {
        "username": (username or "").lower(),
        "spent": round(spent, 4),
        "limit": limit,
        "remaining": None if limit is None else round(max(0.0, limit - spent), 4),
        "unlimited": limit is None,
        "exhausted": limit is not None and spent >= limit,
        "resets": next_reset().isoformat(),
        "resets_label": next_reset().strftime("%-d %B"),
    }


def next_reset() -> datetime:
    s = month_start()
    return s.replace(year=s.year + 1, month=1) if s.month == 12 \
        else s.replace(month=s.month + 1)


class OverBudget(Exception):
    """Raised when a reader has spent their month. Carries the numbers so the
    caller can tell them what they spent and when it resets, rather than
    refusing without explanation."""

    def __init__(self, state: dict):
        self.state = state
        limit = state["limit"]
        super().__init__(
            f"monthly limit reached: ${state['spent']:.2f} of ${limit:.2f} used"
        )


def guard(username: str) -> dict:
    """Refuse the request if this reader has spent their month.

    Checked before the work starts, not during: a single deep-research run can
    cost more than the remainder, and stopping it halfway would bill for a
    fan-out whose answer was never written. The overshoot within one question
    is accepted and counted, so a reader can exceed their cap by at most the
    cost of the question that crossed it.
    """
    state = status(username)
    if state["exhausted"]:
        raise OverBudget(state)
    return state
