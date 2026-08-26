"""The living World Model — a versioned view of the current macro regime.

Each deep-analysis cycle reads its predecessor and must reconcile new evidence
against it. That forced reconciliation is what produces continuity of thought
across cycles instead of a series of disconnected daily takes.
"""
from __future__ import annotations

import db

SEED = """# World Model — initial state

**Regime label:** unclassified (bootstrapping)

This is the seed world model. It has not yet been informed by a full analysis
cycle. The first deep-analysis run should replace it entirely with a view built
from the measured data.

## What we believe about the current regime
Nothing yet established. Prior cycles: none.

## Key relationships currently holding
To be established from measured correlations.

## Relationships currently broken or unusual
To be established.

## Open questions
- Which of Dalio's four levers is currently dominant in US policy?
- Is the metals bid driven by real rates, official-sector buying, or confidence?
- Is BTC trading in its risk-on regime or its debasement regime?

## Confidence
Low across the board — no cycles completed.
"""


def latest() -> dict | None:
    return db.one("SELECT * FROM world_model ORDER BY version DESC LIMIT 1")


def current_body() -> str:
    row = latest()
    return row["body"] if row else SEED


def save(body: str, regime: str | None = None, analysis_id: int | None = None) -> int:
    row = db.one(
        """INSERT INTO world_model (body, regime, source_analysis_id)
           VALUES (%s,%s,%s) RETURNING version""",
        (body, regime, analysis_id),
    )
    return row["version"]


def history(limit: int = 10) -> list[dict]:
    return db.query(
        "SELECT version, regime, created_at, left(body, 400) AS preview "
        "FROM world_model ORDER BY version DESC LIMIT %s",
        (limit,),
    )


def ensure_seeded() -> None:
    if latest() is None:
        save(SEED, regime="bootstrapping")
