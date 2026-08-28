"""Tool surface shared by the digest agent and the interactive `ask` agent.

Deliberately one toolbox for both. A follow-up question and a scheduled digest
need the same capabilities — measured prices, computed statistics, semantic
recall over stored news and past analyses, the relationship graph, and fresh web
search. Building them separately would guarantee they drift apart.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import db
from memory import store, world_model
from signals import stats

log = logging.getLogger("mia.tools")

# Anthropic's server-side web search. Runs on their infrastructure; results come
# back in the same response, so there is no client-side execution for this one.
WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 4,
}

DEFINITIONS = [
    {
        "name": "query_prices",
        "description": (
            "Historical and current prices for tracked instruments. Use for any "
            "question about levels or moves over a window. Symbols: GOLD, SILVER, "
            "PLATINUM, COPPER, BTC, ETH, DXY, EURUSD, USDJPY, SPX, OIL, and yields "
            "US2Y, US5Y, US10Y, US30Y (values are percent yields, not prices). "
            "Returns measured values — always prefer this over recalling a number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}},
                "days": {"type": "integer", "description": "lookback window, default 30"},
                "granularity": {"type": "string", "enum": ["1d", "15m"]},
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_stats_pack",
        "description": (
            "The full computed statistics pack: performance table, rolling "
            "correlation matrix (30/90/180-day), correlation flips, cross-asset "
            "ratios with z-scores, yield curve, FRED macro levels, anomalies, and "
            "lead/lag estimates. All values are computed deterministically from "
            "stored prices. This is the authoritative source for every number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": (
                        "Optional single section: performance, correlations, "
                        "correlation_flips, ratios, yield_curve, macro, anomalies, "
                        "lead_lag. Omit for everything."
                    ),
                }
            },
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Semantic search over stored news documents and Alfred's own past "
            "analyses. Use to recall what was reported or concluded previously — "
            "this is the system's long-term memory. Results are ranked by "
            "similarity, recency, and source credibility (tier 1 = official)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": ["documents", "analyses", "both"],
                    "description": "default both",
                },
                "days": {"type": "integer", "description": "restrict to last N days"},
                "limit": {"type": "integer", "description": "default 8"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_relationships",
        "description": (
            "Traverse the entity relationship graph built from analysed news. "
            "Given an entity, returns connected entities, the relation type, "
            "direction, and the documents that evidence each edge. Multi-hop: set "
            "depth 2-3 to find indirect connections, e.g. from 'US Treasury' to "
            "'Gold'. Use canonical names like 'Federal Reserve', 'Gold', 'Bitcoin'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "depth": {"type": "integer", "description": "1-3, default 2"},
                "limit": {"type": "integer", "description": "default 25"},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "get_recent_news",
        "description": (
            "Documents ingested in the last N hours, optionally filtered by "
            "minimum urgency. Use to see what is new since the last cycle rather "
            "than searching by topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "description": "default 6"},
                "min_urgency": {
                    "type": "string",
                    "enum": ["Critical", "High", "Medium", "Low"],
                },
                "limit": {"type": "integer", "description": "default 40"},
            },
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch a web page and return its readable article text. Use after a "
            "search surfaces a promising source: the search snippet is a "
            "sentence, the article is the evidence. Returns title and text; "
            "quote from it rather than paraphrasing from the snippet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute short Python analysis code in a sandbox (pandas/numpy, no "
            "network, 12s limit). Request the price/FRED series you need via "
            "series_names, then access them in code with series('SYMBOL') — a "
            "pandas Series of daily values. print() the result. Use this for "
            "any number not already in the stats pack: custom-window "
            "correlations, drawdowns, spreads, rebased comparisons. Never "
            "state a computed figure without computing it here or reading it "
            "from a tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "series_names": {
                    "type": "array", "items": {"type": "string"},
                    "description": "symbols/FRED ids to load, e.g. GOLD, DXY, WALCL",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_world_model",
        "description": (
            "The current living world model — Alfred's standing view of the macro "
            "regime, written by the previous deep-analysis cycle. Read this to "
            "know what was believed before, so new evidence can be reconciled "
            "against it rather than analysed in isolation."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ─────────────────────────────── implementations ────────────────────────────
def _query_prices(symbols: list[str], days: int = 30, granularity: str = "1d") -> dict:
    out: dict = {}
    for sym in [s.upper() for s in symbols][:12]:
        rows = db.query(
            """SELECT ts, price FROM prices
               WHERE symbol=%s AND grain=%s AND ts > now() - make_interval(days => %s)
               ORDER BY ts""",
            (sym, granularity, days),
        )
        if not rows:
            out[sym] = {"error": "no data for symbol/granularity"}
            continue
        first, last = float(rows[0]["price"]), float(rows[-1]["price"])
        series = [
            {"t": r["ts"].strftime("%Y-%m-%d" if granularity == "1d" else "%Y-%m-%d %H:%M"),
             "p": round(float(r["price"]), 4)}
            for r in rows
        ]
        # Keep payloads small: thin long series rather than flooding the context.
        if len(series) > 60:
            step = len(series) // 60 + 1
            series = series[::step] + [series[-1]]
        out[sym] = {
            "first": round(first, 4),
            "last": round(last, 4),
            "change_pct": round((last - first) / first * 100, 3) if first else None,
            "high": round(max(float(r["price"]) for r in rows), 4),
            "low": round(min(float(r["price"]) for r in rows), 4),
            "points": len(rows),
            "series": series,
        }
    return out


def _get_stats_pack(section: str | None = None) -> dict:
    pack = stats.latest_pack()
    if not pack:
        pack = stats.build()
    if section:
        if section not in pack:
            return {"error": f"unknown section '{section}'", "available": list(pack.keys())}
        return {section: pack[section], "generated_at": pack.get("generated_at")}
    return pack


def _search_memory(query: str, scope: str = "both", days: int | None = None,
                   limit: int = 8) -> dict:
    out: dict = {}
    if scope in ("documents", "both"):
        docs = store.search_documents(query, limit=limit, days=days)
        out["documents"] = [
            {
                "id": d["id"],
                "title": d["title"],
                "summary": d.get("summary") or (d.get("body") or "")[:220],
                "source": d["source"],
                "tier": d["source_tier"],
                "urgency": d.get("urgency"),
                "published": d["published_at"].isoformat() if d.get("published_at") else None,
                "url": d.get("url"),
                "relevance": d.get("score"),
            }
            for d in docs
        ]
    if scope in ("analyses", "both"):
        past = store.search_analyses(query, limit=max(3, limit // 2))
        out["past_analyses"] = [
            {
                "id": a["id"],
                "kind": a["kind"],
                "title": a["title"],
                "created_at": a["created_at"].isoformat(),
                "excerpt": (a["body"] or "")[:600],
            }
            for a in past
        ]
    return out


def _query_relationships(entity: str, depth: int = 2, limit: int = 25) -> dict:
    """Multi-hop traversal via a recursive CTE — the SQL equivalent of a graph query."""
    depth = max(1, min(int(depth), 3))
    rows = db.query(
        """
        WITH RECURSIVE walk(source_entity, target_entity, relation, direction,
                            strength, rationale, evidence_doc_ids, hop, path) AS (
            SELECT e.source_entity, e.target_entity, e.relation, e.direction,
                   e.strength, e.rationale, e.evidence_doc_ids, 1,
                   ARRAY[e.source_entity, e.target_entity]
            FROM edges e
            WHERE lower(e.source_entity) = lower(%s) OR lower(e.target_entity) = lower(%s)
          UNION ALL
            SELECT e.source_entity, e.target_entity, e.relation, e.direction,
                   e.strength, e.rationale, e.evidence_doc_ids, w.hop + 1,
                   w.path || e.target_entity
            FROM edges e
            JOIN walk w ON (e.source_entity = w.target_entity
                            OR e.target_entity = w.target_entity)
            WHERE w.hop < %s AND NOT (e.target_entity = ANY(w.path))
        )
        SELECT DISTINCT source_entity, target_entity, relation, direction,
                        strength, rationale, evidence_doc_ids, hop
        FROM walk ORDER BY hop, strength DESC LIMIT %s
        """,
        (entity, entity, depth, limit),
    )
    if not rows:
        return {
            "entity": entity,
            "edges": [],
            "note": "No relationships recorded yet for this entity. The graph is "
                    "built incrementally from analysed news; try search_memory.",
        }

    doc_ids = sorted({i for r in rows for i in (r.get("evidence_doc_ids") or [])})[:20]
    evidence = {}
    if doc_ids:
        for d in db.query(
            "SELECT id,title,source,url FROM documents WHERE id = ANY(%s)", (doc_ids,)
        ):
            evidence[d["id"]] = {"title": d["title"], "source": d["source"], "url": d.get("url")}

    return {
        "entity": entity,
        "depth": depth,
        "edges": [
            {
                "from": r["source_entity"],
                "to": r["target_entity"],
                "relation": r["relation"],
                "direction": r["direction"],
                "strength": round(float(r["strength"]), 2),
                "hops": r["hop"],
                "rationale": r.get("rationale"),
                "evidence": [evidence[i] for i in (r.get("evidence_doc_ids") or [])
                             if i in evidence][:3],
            }
            for r in rows
        ],
    }


def _get_recent_news(hours: int = 6, min_urgency: str | None = None, limit: int = 40) -> dict:
    docs = store.recent_documents(hours=hours, limit=limit, min_urgency=min_urgency)
    return {
        "window_hours": hours,
        "count": len(docs),
        "documents": [
            {
                "id": d["id"],
                "title": d["title"],
                "summary": d.get("summary"),
                "source": d["source"],
                "tier": d["source_tier"],
                "urgency": d.get("urgency"),
                "themes": d.get("themes"),
                "url": d.get("url"),
                "published": d["published_at"].isoformat() if d.get("published_at") else None,
            }
            for d in docs
        ],
    }


def _fetch_url(url: str) -> dict:
    import httpx
    from bs4 import BeautifulSoup

    if not url.startswith(("http://", "https://")):
        return {"error": "only http(s) URLs"}
    try:
        r = httpx.get(url, timeout=25, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (compatible; AlfredResearch/1.0)"})
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"fetch failed: {type(exc).__name__}: {exc}"}
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    node = soup.find("article") or soup.find("main") or soup.body or soup
    text = " ".join(node.get_text(" ", strip=True).split())
    return {
        "url": str(r.url),
        "title": (soup.title.get_text(strip=True) if soup.title else None),
        "text": text[:18000],
        "truncated": len(text) > 18000,
    }


def _run_python(code: str, series_names: list[str] | None = None) -> dict:
    from brain import sandbox

    return sandbox.run(code, series_names=series_names)


def _get_world_model() -> dict:
    row = world_model.latest()
    if not row:
        return {"body": world_model.SEED, "version": 0, "regime": "bootstrapping"}
    return {
        "version": row["version"],
        "regime": row.get("regime"),
        "written_at": row["created_at"].isoformat(),
        "body": row["body"],
    }


# ───────────────────────── grounding-link resolution ────────────────────────
_GROUNDING_RE = None


def resolve_grounding_links(text: str, citations: list[dict] | None = None) -> str:
    """Replace Gemini grounding redirect URLs with the pages they lead to.

    Grounded search returns every source as an opaque
    vertexaisearch.cloud.google.com/grounding-api-redirect/… URL. Those links
    work, but a reader cannot see what they are, and they die if Google retires
    the redirector. Resolved once here, in parallel, after generation — a URL
    that fails to resolve keeps its redirect form rather than breaking.
    Mutates the url field of `citations` in place; returns the rewritten text.
    """
    import re
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    global _GROUNDING_RE
    if _GROUNDING_RE is None:
        _GROUNDING_RE = re.compile(
            r"https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[\w\-=]+")

    urls = set(_GROUNDING_RE.findall(text))
    for c in citations or []:
        if c.get("url") and _GROUNDING_RE.fullmatch(c["url"]):
            urls.add(c["url"])
    if not urls:
        return text

    def _resolve(url: str) -> tuple[str, str | None]:
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=12,
                              headers={"User-Agent": "Mozilla/5.0 (AlfredResearch)"}) as r:
                final = str(r.url)
            return url, (final if final != url else None)
        except Exception:  # noqa: BLE001
            return url, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        resolved = dict(ex.map(_resolve, urls))
    for src, dst in resolved.items():
        if dst:
            text = text.replace(src, dst)
    for c in citations or []:
        dst = resolved.get(c.get("url"))
        if dst:
            c["url"] = dst
    return text


HANDLERS = {
    "query_prices": _query_prices,
    "get_stats_pack": _get_stats_pack,
    "search_memory": _search_memory,
    "query_relationships": _query_relationships,
    "get_recent_news": _get_recent_news,
    "get_world_model": _get_world_model,
    "fetch_url": _fetch_url,
    "run_python": _run_python,
}

# Span kinds for the trace tree: lookups that feed the model context are
# retrieval, actions are tools.
_SPAN_KIND = {
    "search_memory": "retrieval",
    "get_recent_news": "retrieval",
    "query_relationships": "retrieval",
    "get_world_model": "retrieval",
}


def dispatch(name: str, args: dict) -> str:
    """Execute a client-side tool call, returning a JSON string for the model.

    Each dispatch emits one tool/retrieval span under whatever agent run is
    active (a no-op outside a run), so the trace tree shows the real steps.
    """
    from brain import observe

    fn = HANDLERS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool '{name}'"})
    with observe.stage(name, kind=_SPAN_KIND.get(name, "tool"), input=args) as span:
        try:
            result = fn(**args)
        except TypeError as exc:
            result = {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s failed", name)
            result = {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
        span.set_output(result)
        if isinstance(result, dict) and result.get("error"):
            span.set_error(str(result["error"])[:500])
    return json.dumps(result, default=str)[:60000]
