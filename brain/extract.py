"""Entity canonicalisation and relationship-edge extraction.

The graph's value depends entirely on hygiene: one canonical id per entity, typed
directional relations, and every edge citing the documents that evidence it. A
graph full of duplicate nodes and unsourced edges is worse than no graph, whatever
database it lives in.
"""
from __future__ import annotations

import json
import logging
import re

import config
import db
from brain import client

log = logging.getLogger("mia.extract")

# Alias table for the entities that dominate this domain. Applied before the
# model sees anything, so "Fed", "the Fed", and "FOMC" never become three nodes.
ALIASES = {
    "fed": "Federal Reserve",
    "the fed": "Federal Reserve",
    "us fed": "Federal Reserve",
    "federal reserve board": "Federal Reserve",
    "fomc": "Federal Reserve",
    "federal open market committee": "Federal Reserve",
    "ecb": "European Central Bank",
    "boe": "Bank of England",
    "boj": "Bank of Japan",
    "pboc": "People's Bank of China",
    "snb": "Swiss National Bank",
    "treasury": "US Treasury",
    "us treasury department": "US Treasury",
    "treasury department": "US Treasury",
    "imf": "International Monetary Fund",
    "bis": "Bank for International Settlements",
    "xau": "Gold",
    "xauusd": "Gold",
    "gold price": "Gold",
    "bullion": "Gold",
    "xag": "Silver",
    "btc": "Bitcoin",
    "eth": "Ethereum",
    "us dollar": "US Dollar",
    "usd": "US Dollar",
    "dxy": "US Dollar",
    "greenback": "US Dollar",
    "ust": "US Treasury Bonds",
    "treasuries": "US Treasury Bonds",
    "us treasuries": "US Treasury Bonds",
    "10-year treasury": "US Treasury Bonds",
    "cpi": "Inflation",
    "us cpi": "Inflation",
}

# Relation verbs are deliberately sign-neutral. Sign lives in `direction` alone.
# Earlier the set included "supports"/"pressures"/"suppresses", which encode sign
# in the verb as well — that double-encoding produced meaningless combinations
# like supports(negative), and let the extractor emit
# `US Dollar --supports(positive)--> Gold`, contradicting the measured -0.54
# dollar/gold correlation because "dollar *weakness* supports gold" lost its
# negation when collapsed to the entity "US Dollar".
RELATIONS = [
    "affects", "funds", "buys", "sells", "regulates", "issues",
    "correlates_with", "hedges", "competes_with", "depends_on", "signals",
]

SYSTEM = """You extract a relationship graph from macro-financial news for a
research system tracking gold, silver, fiat, sovereign debt, and crypto.

For each document, extract only relationships the text actually asserts or
directly implies. Do not add relationships from your background knowledge — an
edge that the document does not support is worse than a missing edge.

Rules:
- Use canonical entity names: "Federal Reserve", "US Treasury", "Gold",
  "Silver", "Bitcoin", "US Dollar", "US Treasury Bonds", "European Central Bank",
  "Inflation", "China", "Real Yields". Never abbreviations.
- relation must come from this set: affects, funds, buys, sells, regulates,
  issues, correlates_with, hedges, competes_with, depends_on, signals
  These verbs are sign-neutral on purpose. Use `affects` for any causal
  influence and let `direction` carry the sign.
- direction encodes the sign of the effect when the SOURCE RISES:
    "positive"  = source rises -> target rises
    "negative"  = source rises -> target falls
    "ambiguous" = the document does not establish a direction
  Worked example: an article saying "dollar weakness lifted gold" describes
  US Dollar rising -> Gold falling, so the edge is
  {source: "US Dollar", target: "Gold", relation: "affects", direction: "negative"}.
  Do not drop the negation when you collapse a phrase like "dollar weakness"
  into the entity "US Dollar" — invert the direction instead.
- strength: 0.0-1.0, how strongly the document supports this edge
- rationale: one short clause, max 15 words, grounded in the document

Return at most 3 edges per document, and none at all if the document asserts no
real relationship. Most routine news yields zero edges. Return only JSON."""

SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "integer"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string", "enum": RELATIONS},
                    "direction": {
                        "type": "string",
                        "enum": ["positive", "negative", "ambiguous"],
                    },
                    "strength": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "doc_id", "source", "target", "relation", "direction",
                    "strength", "rationale",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edges"],
    "additionalProperties": False,
}


def canonical(name: str) -> str:
    n = re.sub(r"\s+", " ", (name or "").strip())
    if not n:
        return ""
    return ALIASES.get(n.lower(), n.title() if n.islower() else n)


def upsert_entity(name: str, kind: str = "unknown") -> None:
    if not name:
        return
    db.execute(
        """INSERT INTO entities (canonical, kind, mention_count)
           VALUES (%s,%s,1)
           ON CONFLICT (canonical) DO UPDATE
             SET last_seen=now(), mention_count = entities.mention_count + 1""",
        (name, kind),
    )


def upsert_edge(e: dict) -> bool:
    src, tgt = canonical(e.get("source", "")), canonical(e.get("target", ""))
    if not src or not tgt or src == tgt:
        return False
    if e.get("relation") not in RELATIONS:
        return False

    upsert_entity(src)
    upsert_entity(tgt)
    doc_id = e.get("doc_id")
    docs = [int(doc_id)] if isinstance(doc_id, int) else []

    db.execute(
        """INSERT INTO edges (source_entity,target_entity,relation,direction,
                              strength,evidence_doc_ids,rationale)
           VALUES (%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (source_entity,target_entity,relation) DO UPDATE SET
             last_confirmed = now(),
             confirm_count  = edges.confirm_count + 1,
             -- Repeated observation raises confidence, with a ceiling.
             strength = LEAST(1.0, (edges.strength * edges.confirm_count
                                    + EXCLUDED.strength) / (edges.confirm_count + 1) + 0.05),
             evidence_doc_ids = (
                 SELECT ARRAY(SELECT DISTINCT unnest(
                     edges.evidence_doc_ids || EXCLUDED.evidence_doc_ids) LIMIT 12)
             ),
             rationale = COALESCE(edges.rationale, EXCLUDED.rationale)""",
        (
            src, tgt, e["relation"], e.get("direction", "ambiguous"),
            max(0.0, min(1.0, float(e.get("strength", 0.5)))), docs,
            (e.get("rationale") or "")[:300],
        ),
    )
    return True


def run(hours: int = 6, limit: int = 30) -> int:
    """Extract edges from recent, non-Low documents. Returns edges written."""
    docs = db.query(
        """SELECT id, title, summary, left(body, 700) AS body, source
           FROM documents
           WHERE fetched_at > now() - make_interval(hours => %s)
             AND urgency IN ('Critical','High','Medium')
           ORDER BY urgency_score DESC, source_tier LIMIT %s""",
        (hours, limit),
    )
    if not docs:
        return 0

    payload = [
        {
            "doc_id": d["id"],
            "title": d["title"],
            "summary": d.get("summary"),
            "excerpt": (d["body"] or "")[:400],
        }
        for d in docs
    ]
    try:
        resp = client.complete(
            model=config.CLASSIFY_MODEL,
            purpose="extract_edges",
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            max_tokens=3000,
            estimated_usd=0.01,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        edges = json.loads(client.text_of(resp)).get("edges", [])
    except client.BudgetExceeded as exc:
        log.warning("edge extraction skipped: %s", exc)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.error("edge extraction failed: %s", exc)
        return 0

    return sum(1 for e in edges if upsert_edge(e))


def hygiene() -> dict:
    """Nightly pass: prune weak, stale, single-sighting edges."""
    pruned = db.execute(
        """DELETE FROM edges
           WHERE confirm_count = 1
             AND strength < 0.35
             AND last_confirmed < now() - interval '14 days'"""
    )
    orphans = db.execute(
        """DELETE FROM entities e
           WHERE NOT EXISTS (
             SELECT 1 FROM edges g
             WHERE g.source_entity = e.canonical OR g.target_entity = e.canonical)
             AND e.mention_count <= 1
             AND e.last_seen < now() - interval '14 days'"""
    )
    return {"edges_pruned": pruned, "entities_pruned": orphans}


def graph_stats() -> dict:
    return {
        "entities": db.one("SELECT count(*) n FROM entities")["n"],
        "edges": db.one("SELECT count(*) n FROM edges")["n"],
        "top_entities": db.query(
            """SELECT canonical, mention_count FROM entities
               ORDER BY mention_count DESC LIMIT 10"""
        ),
        "strongest_edges": db.query(
            """SELECT source_entity, relation, target_entity, direction,
                      round(strength::numeric,2) AS strength, confirm_count
               FROM edges ORDER BY strength DESC, confirm_count DESC LIMIT 10"""
        ),
    }
