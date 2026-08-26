"""Vector memory over documents and past analyses.

Search blends semantic similarity with recency and source credibility, because
in macro research a highly-similar three-month-old blog post is worth less than
a moderately-similar Fed release from this morning.
"""
from __future__ import annotations

import logging

from pgvector.psycopg import register_vector

import db
from memory import embed

log = logging.getLogger("mia.memory")


def _vec_conn():
    """Connection with pgvector adapters registered."""
    ctx = db.conn()
    conn = ctx.__enter__()
    register_vector(conn)
    return ctx, conn


# ─────────────────────────────── backfilling ────────────────────────────────
def embed_documents(limit: int = 200) -> int:
    model = embed.active_model()
    rows = db.query(
        """SELECT id, title, body FROM documents
           WHERE embedding IS NULL OR embed_model IS DISTINCT FROM %s
           ORDER BY published_at DESC NULLS LAST LIMIT %s""",
        (model, limit),
    )
    if not rows:
        return 0
    texts = [f"{r['title']}\n\n{(r['body'] or '')[:4000]}" for r in rows]
    try:
        vecs = embed.embed(texts, input_type="document")
    except embed.EmbeddingUnavailable as exc:
        log.warning("embedding unavailable: %s", exc)
        return 0

    ctx, conn = _vec_conn()
    try:
        with conn.cursor() as cur:
            for r, v in zip(rows, vecs):
                cur.execute(
                    "UPDATE documents SET embedding=%s, embed_model=%s WHERE id=%s",
                    (v, model, r["id"]),
                )
    finally:
        ctx.__exit__(None, None, None)
    return len(rows)


def embed_analysis(analysis_id: int, text: str) -> bool:
    try:
        vec = embed.embed_one(text, input_type="document")
    except embed.EmbeddingUnavailable:
        return False
    ctx, conn = _vec_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE analyses SET embedding=%s, embed_model=%s WHERE id=%s",
                (vec, embed.active_model(), analysis_id),
            )
    finally:
        ctx.__exit__(None, None, None)
    return True


# ───────────────────────────────── searching ────────────────────────────────
def search_documents(
    query: str, limit: int = 12, days: int | None = None, min_tier: int = 4
) -> list[dict]:
    """Semantic search over the news corpus, re-ranked by recency and credibility."""
    try:
        qv = embed.embed_one(query, input_type="query")
    except embed.EmbeddingUnavailable:
        return _keyword_fallback(query, limit)

    where = ["embedding IS NOT NULL", "embed_model = %s", "source_tier <= %s"]
    if days:
        where.append("published_at > now() - make_interval(days => %s)")
    sql = f"""
        SELECT id, title, left(body, 900) AS body, source, source_tier, url,
               published_at, urgency, summary,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE {' AND '.join(where)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [qv, embed.active_model(), min_tier] + ([days] if days else []) + [qv, limit * 3]
    ctx, conn = _vec_conn()
    try:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    finally:
        ctx.__exit__(None, None, None)

    return _rerank(rows, limit)


def _rerank(rows: list[dict], limit: int) -> list[dict]:
    """Blend similarity with recency decay and a credibility bonus."""
    import math
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for r in rows:
        sim = float(r.get("similarity") or 0)
        pub = r.get("published_at")
        age_days = (now - pub).total_seconds() / 86400 if pub else 30.0
        recency = math.exp(-age_days / 21)          # ~3-week half-life
        credibility = {1: 1.15, 2: 1.0, 3: 0.92, 4: 0.75}.get(r.get("source_tier", 3), 0.9)
        r["score"] = round(sim * (0.55 + 0.45 * recency) * credibility, 4)
    rows.sort(key=lambda r: -r["score"])
    return rows[:limit]


def _keyword_fallback(query: str, limit: int) -> list[dict]:
    terms = [t for t in query.lower().split() if len(t) > 3][:6]
    if not terms:
        return []
    pattern = "|".join(terms)
    return db.query(
        """SELECT id,title,left(body,900) AS body,source,source_tier,url,
                  published_at,urgency,summary, 0.0 AS similarity
           FROM documents WHERE lower(title) ~ %s OR lower(body) ~ %s
           ORDER BY published_at DESC NULLS LAST LIMIT %s""",
        (pattern, pattern, limit),
    )


def search_analyses(query: str, limit: int = 5) -> list[dict]:
    """Recall MIA's own past digests and answers."""
    try:
        qv = embed.embed_one(query, input_type="query")
    except embed.EmbeddingUnavailable:
        return db.query(
            "SELECT id,kind,title,body,created_at FROM analyses ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
    ctx, conn = _vec_conn()
    try:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id,kind,title,body,created_at,
                          1 - (embedding <=> %s::vector) AS similarity
                   FROM analyses WHERE embedding IS NOT NULL AND embed_model = %s
                   ORDER BY embedding <=> %s::vector LIMIT %s""",
                (qv, embed.active_model(), qv, limit),
            )
            return cur.fetchall()
    finally:
        ctx.__exit__(None, None, None)


def near_duplicate(text: str, threshold: float = 0.94) -> dict | None:
    """Is this story already in the corpus under a different headline?"""
    try:
        qv = embed.embed_one(text, input_type="document")
    except embed.EmbeddingUnavailable:
        return None
    ctx, conn = _vec_conn()
    try:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id,title, 1 - (embedding <=> %s::vector) AS similarity
                   FROM documents WHERE embedding IS NOT NULL AND embed_model = %s
                   ORDER BY embedding <=> %s::vector LIMIT 1""",
                (qv, embed.active_model(), qv),
            )
            row = cur.fetchone()
    finally:
        ctx.__exit__(None, None, None)
    if row and float(row["similarity"]) >= threshold:
        return row
    return None


def recent_documents(hours: int = 6, limit: int = 60, min_urgency: str | None = None) -> list[dict]:
    sql = """SELECT id,title,left(body,700) AS body,source,source_tier,url,
                    published_at,urgency,summary,themes
             FROM documents WHERE fetched_at > now() - make_interval(hours => %s)"""
    params: list = [hours]
    if min_urgency:
        order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        keep = [k for k, v in order.items() if v >= order.get(min_urgency, 0)]
        sql += " AND urgency = ANY(%s)"
        params.append(keep)
    sql += " ORDER BY source_tier, published_at DESC NULLS LAST LIMIT %s"
    params.append(limit)
    return db.query(sql, params)


def save_analysis(kind: str, title: str, body: str, meta: dict | None = None) -> int:
    import json

    row = db.one(
        "INSERT INTO analyses (kind,title,body,meta) VALUES (%s,%s,%s,%s) RETURNING id",
        (kind, title, body, json.dumps(meta or {}, default=str)),
    )
    aid = row["id"]
    embed_analysis(aid, f"{title}\n\n{body[:6000]}")
    return aid
