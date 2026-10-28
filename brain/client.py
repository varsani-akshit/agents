"""Anthropic client wrapper: cost ledger, hard budget guard, retries.

Every call is priced and recorded before it is allowed to happen again. The guard
is deliberately strict — this runs unattended against a small prepaid balance, and
an unbounded agent loop is the realistic way to burn it.
"""
from __future__ import annotations

import logging
from typing import Any

import anthropic

import config
import db
from brain import spend

log = logging.getLogger("mia.brain")


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would breach a spend cap."""


_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        # An explicit ceiling matters more here than in an interactive tool: a
        # hung request in a scheduled cycle stalls that cycle silently, and the
        # next thing anyone notices is a missing digest.
        _client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY, max_retries=3, timeout=1800.0
        )
    return _client


# ─────────────────────────────── cost accounting ────────────────────────────
def spend_today() -> float:
    """Rolling 24-hour spend, every provider.

    A calendar-day cap on a continuously-running system behaves badly: spend
    early in the day blocks the rest of it, and the limiter resets on an
    arbitrary boundary rather than relative to actual usage. A rolling window is
    what "don't burn more than X per day" actually means.
    """
    return _spend_since("24 hours")


def spend_month() -> float:
    """Rolling 30-day spend, every provider."""
    return _spend_since("30 days")


def spend_total() -> float:
    """All-time spend, every provider. Reported, not capped."""
    row = db.one("SELECT COALESCE(SUM(usd),0) AS s FROM api_calls")
    return float(row["s"]) if row else 0.0


def _spend_since(window: str) -> float:
    # Every provider counts. The caps once covered Anthropic alone, to guard a
    # prepaid balance; when the work moved to Gemini and Azure that left every
    # call the system actually makes uncapped.
    row = db.one("SELECT COALESCE(SUM(usd),0) AS s FROM api_calls "
                 f"WHERE created_at > now() - interval '{window}'")
    return float(row["s"]) if row else 0.0


def spend_by_provider() -> list[dict]:
    return db.query(
        """SELECT provider, count(*) AS calls, round(sum(usd)::numeric, 4) AS usd
           FROM api_calls GROUP BY provider ORDER BY sum(usd) DESC"""
    )


def price_call(model: str, usage: Any, web_searches: int = 0) -> float:
    inp, out = config.price_for(model)
    it = getattr(usage, "input_tokens", 0) or 0
    ot = getattr(usage, "output_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        it * inp / 1e6
        + ot * out / 1e6
        + cr * inp * 0.1 / 1e6      # cache reads bill at ~10% of input
        + cw * inp * 1.25 / 1e6     # cache writes bill at ~125% of input
        + web_searches * config.WEB_SEARCH_USD
    )


def record(model: str, purpose: str, usage: Any, web_searches: int = 0) -> float:
    usd = price_call(model, usage, web_searches)
    db.execute(
        """INSERT INTO api_calls
             (model,purpose,input_tokens,output_tokens,cache_read,cache_write,web_searches,usd,owner)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            model,
            purpose,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            web_searches,
            usd,
            spend.owner(),
        ),
    )
    from brain import observe

    observe.record_llm(
        spec=f"anthropic:{model}", purpose=purpose,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0, usd=usd,
    )
    return usd


# Set False by interactive entrypoints so a user's own question can draw on the
# reserve that scheduled jobs are not allowed to touch.
AUTONOMOUS = True


def check_budget(estimated: float = 0.02) -> None:
    """Refuse a call that would cross a cap. Raised before money is spent."""
    today, month = spend_today(), spend_month()
    if today + estimated > config.DAILY_USD_CAP:
        raise BudgetExceeded(
            f"daily cap reached: ${today:.2f} spent in 24h, cap ${config.DAILY_USD_CAP:.2f}")
    if month + estimated > config.MONTHLY_USD_CAP:
        raise BudgetExceeded(
            f"monthly cap reached: ${month:.2f} spent in 30 days, cap ${config.MONTHLY_USD_CAP:.2f}")
    if AUTONOMOUS and today + estimated > config.SCHEDULED_DAILY_USD_CAP:
        raise BudgetExceeded(
            f"scheduled-job cap reached: ${today:.2f} spent in 24h, scheduled jobs stop at "
            f"${config.SCHEDULED_DAILY_USD_CAP:.2f} — the rest of the "
            f"${config.DAILY_USD_CAP:.2f} daily cap is kept for your own questions")


def budget_status() -> dict:
    today, month = spend_today(), spend_month()
    return {
        "spent_today_usd": round(today, 4),
        "daily_cap_usd": config.DAILY_USD_CAP,
        "daily_remaining_usd": round(max(0.0, config.DAILY_USD_CAP - today), 4),
        "scheduled_daily_cap_usd": config.SCHEDULED_DAILY_USD_CAP,
        "interactive_reserve_usd": round(
            max(0.0, config.DAILY_USD_CAP - max(today, config.SCHEDULED_DAILY_USD_CAP)), 4),
        "spent_month_usd": round(month, 4),
        "monthly_cap_usd": config.MONTHLY_USD_CAP,
        "monthly_remaining_usd": round(max(0.0, config.MONTHLY_USD_CAP - month), 4),
        "spent_total_usd": round(spend_total(), 4),
    }


# ──────────────────────────────── call helpers ──────────────────────────────
def _count_web_searches(response: Any) -> int:
    try:
        su = getattr(response.usage, "server_tool_use", None)
        return int(getattr(su, "web_search_requests", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def complete(
    *,
    model: str,
    purpose: str,
    system: Any,
    messages: list[dict],
    max_tokens: int = 4000,
    tools: list[dict] | None = None,
    estimated_usd: float = 0.02,
    _skip_ledger: bool = False,
    **kwargs,
) -> Any:
    """One call, priced and recorded. Raises BudgetExceeded first.

    Streamed rather than blocking: adaptive thinking on a long brief can run past
    the ten-minute ceiling for a non-streamed request, and the failure only shows
    up on the hardest cycles. Nothing is consumed incrementally —
    `get_final_message()` returns the same Message a blocking call would.
    """
    check_budget(estimated_usd)
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if tools:
        params["tools"] = tools
    params.update(kwargs)

    with client().messages.stream(**params) as stream:
        resp = stream.get_final_message()
    if _skip_ledger:      # caller (brain/llm.py router) records it instead
        return resp
    usd = record(model, purpose, resp.usage, _count_web_searches(resp))
    log.info("%s via %s: $%.5f (in=%s out=%s)", purpose, model, usd,
             resp.usage.input_tokens, resp.usage.output_tokens)
    return resp


def text_of(response: Any) -> str:
    return "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
