"""Microsoft Decision-1: a model that judges rather than writes.

Given a piece of text (the "state") and named questions, it returns numbers,
not prose: the probability a yes/no condition holds (`noul`), the chosen
option with a probability for each (`choice`), or a position on a scale
(`score`). One call can carry many questions. At about $0.04 per million
input tokens and one output token, it is the cheapest way to ask the same
narrow question of every article in the corpus — which is what news_match
does with the reader's holdings.

The route is not the OpenAI one: POST {resource}/providers/microsoft/v1/systemone
with {"model", "state", "questions": {id: {"type", "instructions", "criteria"?}}}.
That shape was established by probing the live endpoint; Microsoft had not
published a schema the docs search could find.
"""
from __future__ import annotations

import logging
import os

import httpx

import db
from brain import spend

log = logging.getLogger("mia.decide")

DEPLOYMENT = os.getenv("AZURE_DEPLOY_DECISION", "Decision-1")
USD_PER_M_INPUT = 0.05   # list price ~$0.04; rounded up so the caps stay safe


class DecisionError(RuntimeError):
    pass


def available() -> bool:
    return bool(os.getenv("AZURE_FOUNDRY_ENDPOINT") and os.getenv("AZURE_FOUNDRY_API_KEY"))


def _url() -> str:
    ep = os.environ["AZURE_FOUNDRY_ENDPOINT"].rstrip("/")
    for tail in ("/openai/v1", "/openai"):
        if ep.endswith(tail):
            ep = ep[: -len(tail)]
    if "/api/projects/" in ep:
        ep = ep.split("/api/projects/")[0]
    return f"{ep}/providers/microsoft/v1/systemone"


def ask(state: str, questions: dict[str, dict], *, purpose: str, timeout: float = 60) -> dict:
    """Ask Decision-1 several questions about one text. Returns {id: answer}.

    A `noul` answer is {"noul": p}; a `choice` answer carries "choice",
    "confidence" and "probabilities". Raises DecisionError on any failure, so
    callers decide whether a missing judgement matters.
    """
    if not available():
        raise DecisionError("Azure is not configured")
    try:
        r = httpx.post(_url(), timeout=timeout,
                       headers={"api-key": os.environ["AZURE_FOUNDRY_API_KEY"]},
                       json={"model": DEPLOYMENT, "state": state[:60000], "questions": questions})
    except httpx.HTTPError as exc:
        raise DecisionError(f"{purpose}: {exc}") from exc
    if r.status_code != 200:
        raise DecisionError(f"{purpose}: HTTP {r.status_code} {r.text[:200]}")
    body = r.json()
    usage = body.get("usage") or {}
    itok, otok = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    usd = itok * USD_PER_M_INPUT / 1e6
    db.execute(
        """INSERT INTO api_calls (provider, model, purpose, input_tokens, output_tokens, usd, owner)
           VALUES ('azure', %s, %s, %s, %s, %s, %s)""",
        (DEPLOYMENT, purpose, itok, otok, usd, spend.owner()))
    return body.get("answers") or {}
