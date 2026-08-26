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

log = logging.getLogger("mia.brain")


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would breach a spend cap."""


_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3)
    return _client


# ─────────────────────────────── cost accounting ────────────────────────────
def spend_today() -> float:
    row = db.one(
        "SELECT COALESCE(SUM(usd),0) AS s FROM api_calls WHERE created_at::date = CURRENT_DATE"
    )
    return float(row["s"]) if row else 0.0


def spend_total() -> float:
    row = db.one("SELECT COALESCE(SUM(usd),0) AS s FROM api_calls")
    return float(row["s"]) if row else 0.0


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
             (model,purpose,input_tokens,output_tokens,cache_read,cache_write,web_searches,usd)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            model,
            purpose,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            web_searches,
            usd,
        ),
    )
    return usd


def check_budget(estimated: float = 0.02) -> None:
    today, total = spend_today(), spend_total()
    if today + estimated > config.DAILY_USD_CAP:
        raise BudgetExceeded(
            f"daily cap reached: ${today:.4f} spent, cap ${config.DAILY_USD_CAP:.2f}"
        )
    if total + estimated > config.TOTAL_USD_CAP:
        raise BudgetExceeded(
            f"total cap reached: ${total:.4f} spent, cap ${config.TOTAL_USD_CAP:.2f}"
        )


def budget_status() -> dict:
    today, total = spend_today(), spend_total()
    return {
        "spent_today_usd": round(today, 4),
        "daily_cap_usd": config.DAILY_USD_CAP,
        "daily_remaining_usd": round(max(0.0, config.DAILY_USD_CAP - today), 4),
        "spent_total_usd": round(total, 4),
        "total_cap_usd": config.TOTAL_USD_CAP,
        "total_remaining_usd": round(max(0.0, config.TOTAL_USD_CAP - total), 4),
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
    **kwargs,
) -> Any:
    """One non-streaming call, priced and recorded. Raises BudgetExceeded first."""
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

    resp = client().messages.create(**params)
    usd = record(model, purpose, resp.usage, _count_web_searches(resp))
    log.info("%s via %s: $%.5f (in=%s out=%s)", purpose, model, usd,
             resp.usage.input_tokens, resp.usage.output_tokens)
    return resp


def text_of(response: Any) -> str:
    return "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
