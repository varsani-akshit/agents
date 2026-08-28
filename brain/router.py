"""Role-based model routing for the agent army.

Agents ask for a *role*, never a model. Each role is an escalation chain: the
first spec is the assignment, the rest are what a failure escalates to — a
provider error or malformed JSON moves one step up the chain rather than
silently degrading the answer. Chains deliberately cross providers, so no
single outage (Gemini quota, Azure hiccup) can take a pipeline down.

The premium tier is locked by call site, not convention: gpt-5.4 exists for
exactly the two single-writer moments a reader experiences — the Editor of a
brief and the Deep Researcher's synthesis. Any other caller asking for it gets
a ValueError at development time, which is how "use it sparingly" survives
future edits.
"""
from __future__ import annotations

import logging
import os

from brain import llm

log = logging.getLogger("mia.router")


def _azure(env_key: str, default: str) -> str:
    return f"azure:{os.getenv(env_key, default)}"


def _roles() -> dict[str, list[str]]:
    return {
        # Mechanical bulk: triage, classification, dedupe. Cheapest first.
        "bulk": [
            _azure("AZURE_DEPLOY_NANO", "gpt-4.1-nano"),
            "gemini:gemini-flash-latest",
        ],
        # High-volume comprehension: article compression, entity extraction.
        "workhorse": [
            _azure("AZURE_DEPLOY_MAVERICK", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
            "gemini:gemini-flash-latest",
        ],
        # Strict structured reasoning: the Verifier's claim audit. gpt-5.4-mini
        # leads: gpt-oss-120b is the cheaper reasoner but times out on
        # audit-sized prompts (measured: >180s on a 60KB draft+data payload),
        # so it serves as the fallback rather than the first choice.
        "reason": [
            _azure("AZURE_DEPLOY_GPT54_MINI", "gpt-5.4-mini"),
            _azure("AZURE_DEPLOY_OSS", "gpt-oss-120b"),
        ],
        # Grounded web search is a Gemini-only capability; no cross-provider
        # escape exists, so the chain stays within the family.
        "search": [
            "gemini:gemini-flash-latest",
            "gemini:gemini-flash-lite-latest",
        ],
        # Interpretation: Analysts, Marshal, sub-researchers.
        "deep": [
            "gemini:gemini-pro-latest",
            _azure("AZURE_DEPLOY_GPT54_MINI", "gpt-5.4-mini"),
            "gemini:gemini-flash-latest",
        ],
        # The two single-writer moments. See PREMIUM_SITES.
        "premium": [
            _azure("AZURE_DEPLOY_GPT54", "gpt-5.4"),
            "gemini:gemini-pro-latest",
        ],
    }


PREMIUM_SITES = {"editor", "research_synthesis"}


def chain_for(role: str, *, premium_site: str | None = None) -> list[str]:
    roles = _roles()
    if role not in roles:
        raise ValueError(f"unknown role '{role}' (have {sorted(roles)})")
    if role == "premium" and premium_site not in PREMIUM_SITES:
        raise ValueError(
            f"premium role is restricted to call sites {sorted(PREMIUM_SITES)}; "
            f"got {premium_site!r}. Use 'deep' instead."
        )
    return [s for s in roles[role] if llm.available(s)] or roles[role][:1]


def complete_json(role: str, *, system: str, user: str, schema: dict,
                  purpose: str, max_tokens: int = 3000,
                  premium_site: str | None = None) -> tuple[dict, str]:
    """Schema-constrained completion on the role's chain.

    Returns (payload, spec_used) so callers can record which model answered.
    """
    last: Exception | None = None
    for spec in chain_for(role, premium_site=premium_site):
        # Two attempts per spec: a dropped connection is transient, and moving
        # straight to a different (often weaker) model over a network blip
        # trades answer quality for nothing.
        for attempt in range(2):
            try:
                payload = llm.complete_json(
                    spec, system=system, user=user, schema=schema,
                    purpose=purpose, max_tokens=max_tokens, fallback=spec,
                )
                return payload, spec
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("%s: role=%s %s attempt %d failed (%s)",
                            purpose, role, spec, attempt + 1, str(exc)[:160])
    raise llm.ProviderError(f"{purpose}: role '{role}' exhausted its chain ({last})")


def complete_text(role: str, *, system: str, user: str, purpose: str,
                  max_tokens: int = 2000, premium_site: str | None = None,
                  ) -> tuple[str, str, float]:
    """Free-text completion on the role's chain.

    Returns (text, spec_used, usd).
    """
    last: Exception | None = None
    for spec in chain_for(role, premium_site=premium_site):
        for attempt in range(2):
            try:
                text, usd = llm.complete_text(
                    spec, system=system, user=user, purpose=purpose,
                    max_tokens=max_tokens, fallback=spec, with_cost=True,
                )
                return text, spec, usd
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("%s: role=%s %s attempt %d failed (%s)",
                            purpose, role, spec, attempt + 1, str(exc)[:160])
    raise llm.ProviderError(f"{purpose}: role '{role}' exhausted its chain ({last})")
