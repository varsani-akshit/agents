"""The knowledge graph: every stored document, linked to everything it relates to.

Three kinds of edge, none of which costs a model call:

  document ── document   semantic similarity, from the embeddings already stored
  document ── entity     the entities and themes the classifier tagged
  entity   ── entity     the typed relations the extractor asserted

The document-to-document edges are the point. Entity nodes alone give a dozen
topic bubbles; linking the documents themselves is what makes this a knowledge
base you can wander through — the same shape Obsidian draws over a vault of
notes, except the links are computed from meaning rather than typed by hand.

Similarity is cosine distance over the pgvector column, so building the edge set
is one indexed query rather than 1,762 comparisons in Python.
"""
from __future__ import annotations

import json
import re

import logging

import db

log = logging.getLogger("alfred.graph")

# Cosine scales are NOT comparable across embedding models, so the threshold is
# per model. On gemini-embedding-001 two entirely unrelated documents already
# score ~0.72, and 99% of random pairs fall under 0.851 — carrying over the
# OpenAI-era 0.55 would have linked every document to every other and turned the
# graph into a single hairball.
#
# Each figure is measured on this corpus by comparing the random-pair
# distribution against the top-5 nearest-neighbour distribution, then choosing
# the point that keeps most genuine neighbours while admitting few random pairs.
# gemini-embedding-001 at 0.82: keeps 81% of true neighbours, admits 2.7% of
# random pairs. Re-measure if the embedding model changes again.
_THRESHOLDS = {
    "gemini-embedding-001": 0.82,
    "text-embedding-3-small": 0.55,
}
DEFAULT_MIN_SIMILARITY = 0.75
NEIGHBOURS_PER_DOC = 5


def min_similarity_for_active(model: str | None = None) -> float:
    """The similarity floor calibrated for the active embedding model."""
    from memory import embed

    name = model or embed.active_model()
    return _THRESHOLDS.get(name, DEFAULT_MIN_SIMILARITY)


def rebuild_links(days: int = 30, per_doc: int = NEIGHBOURS_PER_DOC,
                  min_similarity: float | None = None) -> dict:
    """Recompute semantic edges for recently-fetched documents.

    Embeddings from different models are not comparable, so neighbours are only
    ever drawn from documents sharing the source document's `embed_model` — the
    same rule the search path applies.
    """
    if min_similarity is None:
        min_similarity = min_similarity_for_active()
    rows = db.query(
        """
        WITH recent AS (
          SELECT id, embedding, embed_model FROM documents
          WHERE embedding IS NOT NULL
            AND fetched_at > now() - make_interval(days => %(days)s)
        ),
        pairs AS (
          -- Similarity is symmetric, so A picking B and B picking A yield the
          -- same normalised pair. Postgres refuses to ON CONFLICT DO UPDATE the
          -- same row twice within one command, so collapse duplicates first.
          SELECT DISTINCT ON (LEAST(r.id, n.id), GREATEST(r.id, n.id))
                 LEAST(r.id, n.id) AS src, GREATEST(r.id, n.id) AS dst, n.sim
          FROM recent r
          CROSS JOIN LATERAL (
            SELECT d.id, 1 - (r.embedding <=> d.embedding) AS sim
            FROM documents d
            WHERE d.id <> r.id
              AND d.embedding IS NOT NULL
              AND d.embed_model = r.embed_model
            ORDER BY r.embedding <=> d.embedding
            LIMIT %(per_doc)s
          ) n
          WHERE n.sim >= %(min_sim)s
          ORDER BY LEAST(r.id, n.id), GREATEST(r.id, n.id), n.sim DESC
        )
        INSERT INTO doc_links (src_id, dst_id, similarity)
        SELECT src, dst, sim FROM pairs
        ON CONFLICT (src_id, dst_id) DO UPDATE SET similarity = EXCLUDED.similarity
        RETURNING src_id
        """,
        {"days": days, "per_doc": per_doc, "min_sim": min_similarity},
    )
    total = db.one("SELECT count(*) AS c FROM doc_links")["c"]
    log.info("doc links: %d written, %d total", len(rows), total)
    return {"written": len(rows), "total": total}


def build(days: int = 7, limit: int = 220, min_urgency: str = "Medium",
          query: str = "", concept: str = "") -> dict:
    """Nodes and edges for the graph view.

    Scoped rather than exhaustive: 1,700 documents at once is a grey cloud, not
    a map. The window, the urgency floor and an optional text filter decide which
    slice is drawn, and every node carries enough to render its detail panel
    without a second request.
    """
    order = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
    floor = order.get(min_urgency, 1)
    allowed = [u for u, rank in order.items() if rank >= floor]

    params: dict = {"days": days, "limit": limit, "allowed": allowed}
    where_extra = ""
    if query:
        where_extra = " AND (d.title ILIKE %(q)s OR d.summary ILIKE %(q)s)"
        params["q"] = f"%{query}%"
    if concept:
        # A concept is matched against the extracted entity and theme arrays,
        # not the prose. Filtering a concept by text search returned nothing
        # whenever the entity was named differently in the copy than in the
        # extraction — which blanked the graph on a click that should have
        # focused it.
        # entities/themes are jsonb arrays, so containment (@>) is the test —
        # and it uses the GIN index rather than scanning every row.
        where_extra += (" AND (d.entities @> %(concept)s::jsonb"
                        " OR d.themes @> %(concept)s::jsonb)")
        params["concept"] = json.dumps([concept])

    docs = db.query(
        f"""SELECT d.id, d.title, d.url, d.source, d.source_tier, d.urgency,
                   d.urgency_score, d.summary, d.entities, d.themes, d.fetched_at
            FROM documents d
            WHERE d.fetched_at > now() - make_interval(days => %(days)s)
              AND coalesce(d.urgency, 'Low') = ANY(%(allowed)s)
              {where_extra}
            ORDER BY d.urgency_score DESC NULLS LAST, d.fetched_at DESC
            LIMIT %(limit)s""",
        params,
    )
    if not docs:
        return {"nodes": [], "edges": [], "stats": {"documents": 0}}

    ids = [d["id"] for d in docs]
    id_set = set(ids)

    nodes = [
        {
            "id": f"d{d['id']}",
            "kind": "document",
            "label": d["title"][:90],
            "url": d["url"],
            "source": d["source"],
            "tier": d["source_tier"],
            "urgency": d["urgency"] or "Low",
            "summary": (d["summary"] or "")[:400],
            "when": d["fetched_at"].isoformat(),
            "weight": 1 + (d["urgency_score"] or 0) / 40,
        }
        for d in docs
    ]

    edges = [
        {"source": f"d{r['src_id']}", "target": f"d{r['dst_id']}",
         "kind": "similar", "weight": round(float(r["similarity"]), 3)}
        for r in db.query(
            """SELECT src_id, dst_id, similarity FROM doc_links
               WHERE src_id = ANY(%s) AND dst_id = ANY(%s)""",
            (ids, ids),
        )
    ]

    # Entity and theme nodes, attached to the documents that mention them. Only
    # those appearing more than once earn a node — a tag used by a single
    # document adds a leaf and no structure.
    tallies: dict[tuple[str, str], list[int]] = {}
    for d in docs:
        for kind, field in (("entity", "entities"), ("theme", "themes")):
            for raw in (d.get(field) or []):
                name = str(raw).strip()
                if name:
                    tallies.setdefault((kind, name), []).append(d["id"])

    for (kind, name), doc_ids in tallies.items():
        if len(doc_ids) < 2:
            continue
        nodes.append({
            "id": f"{kind[0]}:{name}", "kind": kind, "label": name,
            "weight": 1 + len(doc_ids) / 3, "mentions": len(doc_ids),
        })
        for doc_id in doc_ids:
            edges.append({"source": f"{kind[0]}:{name}", "target": f"d{doc_id}",
                          "kind": "mentions", "weight": 0.35})

    # Securities: the reader's investable ground joined to the news that moves
    # it. A company earns a node when this window's coverage names it — by
    # ticker or by company name — so the map runs from a macro development
    # through the concepts it touches to the listed names that carry it.
    secs = db.query(
        """SELECT symbol, name, exchange, sector, market_cap
           FROM securities WHERE name IS NOT NULL""")
    # Cased text, because a ticker is only a ticker in capitals: lowercasing
    # first made "A" (Agilent) match the article "a" in every document, and
    # first-word matching made "Australian Foundation Investment" claim every
    # story containing "Australian". A company is matched by its full name, or
    # by a ticker of three characters or more appearing in capitals.
    doc_text = [(d["id"], f"{d['title']} {d.get('summary') or ''}") for d in docs]
    for sec in secs:
        name = (sec["name"] or "").strip()
        if len(name) < 4:
            continue
        base = sec["symbol"].split(".")[0]
        needle = name.lower()
        ticker_re = (re.compile(rf"\b{re.escape(base)}\b")
                     if len(base) >= 3 else None)
        hits = [
            doc_id for doc_id, text in doc_text
            if needle in text.lower() or (ticker_re and ticker_re.search(text))
        ]
        if not hits:
            continue
        nodes.append({
            "id": f"s:{sec['symbol']}", "kind": "security",
            "label": f"{name} ({sec['symbol']})",
            "symbol": sec["symbol"], "exchange": sec["exchange"],
            "sector": sec["sector"],
            "weight": 1.6 + len(hits) / 3, "mentions": len(hits),
        })
        for doc_id in hits[:12]:
            edges.append({"source": f"s:{sec['symbol']}", "target": f"d{doc_id}",
                          "kind": "covers", "weight": 0.5})

    present = {n["id"] for n in nodes}
    for e in db.query(
        """SELECT source_entity, target_entity, relation, direction, strength
           FROM edges ORDER BY strength DESC LIMIT 400"""
    ):
        src, tgt = f"e:{e['source_entity']}", f"e:{e['target_entity']}"
        if src in present and tgt in present:
            edges.append({
                "source": src, "target": tgt, "kind": "relation",
                "weight": round(float(e["strength"]), 2),
                "label": f"{e['relation']} ({e['direction']})",
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "documents": len(docs),
            "concepts": sum(1 for n in nodes if n["kind"] in ("entity", "theme")),
            "securities": sum(1 for n in nodes if n["kind"] == "security"),
            "links": len(edges),
            "window_days": days,
        },
    }


def neighbours(doc_id: int, limit: int = 8) -> list[dict]:
    """The documents most related to one document, for its detail panel."""
    return db.query(
        """SELECT d.id, d.title, d.url, d.source, d.fetched_at, l.similarity
           FROM doc_links l
           JOIN documents d
             ON d.id = CASE WHEN l.src_id = %(id)s THEN l.dst_id ELSE l.src_id END
           WHERE %(id)s IN (l.src_id, l.dst_id)
           ORDER BY l.similarity DESC LIMIT %(limit)s""",
        {"id": doc_id, "limit": limit},
    )
