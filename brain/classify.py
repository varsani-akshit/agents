"""Cheap-model triage: urgency, entities, themes, one-line summary.

Runs on Haiku over batches of headlines. This is the highest-volume LLM call in
the system, so it is deliberately narrow: strict JSON out, no reasoning prose,
no tools. Structured outputs guarantee the shape so nothing downstream parses
free text.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import db
from brain import client, llm

log = logging.getLogger("mia.classify")

URGENCY = ["Critical", "High", "Medium", "Low"]

SYSTEM = """You triage financial news for a macro research system focused on gold,
silver, fiat currencies, sovereign debt, central bank policy, and crypto.

For each item assign:

urgency — how much this should interrupt a serious macro investor right now:
  Critical: a market-moving official action or shock with direct implications for
    metals, the dollar, sovereign debt, or crypto. Rate decisions, emergency
    actions, intervention, default/restructuring, major geopolitical rupture,
    debt-ceiling resolution, large surprise data.
  High: material policy signal or a significant move//development that changes the
    picture but is not a shock. Official speeches with new content, notable
    positioning shifts, meaningful data.
  Medium: relevant context, incremental commentary, sector news.
  Low: noise, promotional content, routine recaps, price-summary churn,
    listicles, anything with no informational content.

Be strict. Most items are Medium or Low. Reserve Critical for genuine events —
in a normal day there are none. A headline that merely reports a price level is
Low. Price-forecast and "analyst says" pieces are Low unless they contain new
information.

entities — canonical names of institutions, people, assets, policies, countries
  actually central to the item. Use standard forms: "Federal Reserve" not "Fed",
  "US Treasury", "European Central Bank", "Gold", "Silver", "Bitcoin",
  "US Treasury Bonds". Empty list if none. Max 6.

themes — from this fixed set only, those that genuinely apply:
  monetary_policy, fiscal_policy, debasement, inflation, real_rates,
  sovereign_debt, central_bank_gold, industrial_demand, geopolitics,
  crypto_adoption, liquidity, banking_stress, trade_policy, market_structure
  Note: `central_bank_gold` means gold specifically — official-sector gold
  buying, selling, or repatriation. A central bank holding government bonds is
  `sovereign_debt` or `monetary_policy`, not `central_bank_gold`.

summary — one factual sentence, max 25 words, stating what happened. No
  speculation, no adjectives, no "this could mean".

Return only the JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "urgency": {"type": "string", "enum": URGENCY},
                    "summary": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "themes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "urgency", "summary", "entities", "themes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


# Cheap relevance gate. A document with no macro-relevant term anywhere in it
# cannot be Critical or High by construction, so spending an LLM call on it is
# waste. Official (tier-1) sources always go to the model regardless.
_RELEVANT = re.compile(
    r"\b(gold|silver|bullion|platinum|palladium|copper"
    r"|fed(eral reserve)?|fomc|powell|ecb|boj|bank of (england|japan)|central bank"
    r"|treasur(y|ies)|yield|bond|debt|deficit|auction|buyback|issuance"
    r"|inflation|cpi|pce|deflation|debasement|devalu|repress"
    r"|dollar|usd|dxy|currency|fx|yuan|yen|euro"
    r"|bitcoin|btc|ethereum|crypto|stablecoin"
    r"|rate (cut|hike|decision)|monetary|fiscal|qe|quantitative|liquidity"
    r"|sanction|tariff|geopolit|war|default|restructur)\b",
    re.I,
)


def prefilter() -> int:
    """Mark obviously-irrelevant, non-official documents Low without an LLM call."""
    rows = db.query(
        """SELECT id, title, left(body, 500) AS body FROM documents
           WHERE classified_at IS NULL AND source_tier >= 3"""
    )
    dead = [r["id"] for r in rows if not _RELEVANT.search(f"{r['title']} {r['body'] or ''}")]
    if not dead:
        return 0
    return db.execute(
        """UPDATE documents SET urgency='Low', urgency_score=1, classified_at=now(),
             summary='Filtered: no macro-relevant content detected.'
           WHERE id = ANY(%s)""",
        (dead,),
    )


def pending(limit: int = 40) -> list[dict]:
    return db.query(
        """SELECT id, title, left(body, 600) AS body, source, source_tier
           FROM documents WHERE classified_at IS NULL
           ORDER BY source_tier, published_at DESC NULLS LAST LIMIT %s""",
        (limit,),
    )


def _apply(items: list[dict]) -> int:
    n = 0
    with db.conn() as c, c.cursor() as cur:
        for it in items:
            urgency = it.get("urgency")
            if urgency not in URGENCY:
                urgency = "Low"
            cur.execute(
                """UPDATE documents
                   SET urgency=%s, urgency_score=%s, summary=%s,
                       entities=%s, themes=%s, classified_at=now()
                   WHERE id=%s""",
                (
                    urgency,
                    {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}[urgency],
                    (it.get("summary") or "")[:600],
                    json.dumps(it.get("entities", [])[:6]),
                    json.dumps(it.get("themes", [])[:6]),
                    it["id"],
                ),
            )
            n += cur.rowcount
    return n


def _classify_batch(docs: list[dict]) -> list[dict]:
    """One model call over a batch of documents. Returns parsed items or []."""
    payload = [
        {
            "id": d["id"],
            "title": d["title"],
            "source": d["source"],
            "tier": d["source_tier"],
            "excerpt": (d["body"] or "")[:400],
        }
        for d in docs
    ]
    return llm.complete_json(
        config.CLASSIFY_SPEC,
        system=SYSTEM,
        user=json.dumps(payload, ensure_ascii=False),
        schema=SCHEMA,
        purpose="classify",
        max_tokens=3000,
    ).get("items", [])


def run(batch: int = 20, max_batches: int = 8, workers: int = 3) -> dict:
    """Classify pending documents. Prefilters first, then batches in parallel.

    Never raises on budget exhaustion — it stops early and reports what it did,
    so a scheduled run degrades gracefully instead of failing the whole tick.
    """
    spent_start = client.spend_total()
    filtered = prefilter()

    docs = pending(batch * max_batches)
    if not docs:
        return {"classified": 0, "prefiltered": filtered, "batches": 0, "usd": 0.0}

    chunks = [docs[i : i + batch] for i in range(0, len(docs), batch)]
    total, done, stopped = 0, 0, None

    from brain import observe

    def _traced_batch(chunk):
        with observe.stage(f"classify.batch({len(chunk)})", kind="generic",
                           input={"doc_ids": [d["id"] for d in chunk]}) as sp:
            items = _classify_batch(chunk)
            sp.set_output(items)
            sp.set_attribute("classified", len(items or []))
            return items

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {observe.ctx_submit(pool, _traced_batch, c): c for c in chunks}
        for fut in as_completed(futures):
            try:
                items = fut.result()
            except client.BudgetExceeded as exc:
                stopped = str(exc)
                continue
            except llm.ProviderError as exc:
                log.error("classification batch failed: %s", exc)
                continue
            except Exception as exc:  # noqa: BLE001
                log.error("classification batch failed: %s", exc)
                continue
            total += _apply(items)
            done += 1

    if stopped:
        log.warning("classification stopped early: %s", stopped)
    return {
        "classified": total,
        "prefiltered": filtered,
        "batches": done,
        "stopped": stopped,
        "usd": round(client.spend_total() - spent_start, 5),
    }


def urgent_since(hours: int = 6, min_urgency: str = "High") -> list[dict]:
    order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    keep = [k for k, v in order.items() if v >= order[min_urgency]]
    return db.query(
        """SELECT id,title,summary,source,source_tier,url,urgency,themes,entities,published_at
           FROM documents
           WHERE urgency = ANY(%s) AND fetched_at > now() - make_interval(hours => %s)
           ORDER BY urgency_score DESC, source_tier, published_at DESC NULLS LAST""",
        (keep, hours),
    )
