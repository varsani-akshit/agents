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
            "Semantic search over stored news documents, Alfred's own past "
            "analyses, and the per-company news files of the 750 covered "
            "listed names. Use to recall what was reported or concluded "
            "previously — this is the system's long-term memory. Results are "
            "ranked by similarity, recency, and source credibility."
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
        "name": "screen_stocks",
        "description": (
            "Screen the covered equity universe — S&P 500, S&P/ASX 200 and "
            "NIFTY 50 — on measured criteria. Filter by exchange (US, ASX, "
            "NSE), sector, valuation and momentum; sort by any of them. Use "
            "this to find candidates rather than recalling names: it returns "
            "live figures with the move over the chosen window."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exchange": {"type": "string", "enum": ["US", "ASX", "NSE"]},
                "sector": {"type": "string", "description": "e.g. Energy, Financial Services"},
                "sort_by": {
                    "type": "string",
                    "enum": ["change_pct", "market_cap", "trailing_pe",
                             "dividend_yield", "from_52w_high"],
                    "description": "default change_pct",
                },
                "direction": {"type": "string", "enum": ["desc", "asc"]},
                "days": {"type": "integer", "description": "move window, default 30"},
                "max_pe": {"type": "number"},
                "min_market_cap": {"type": "number"},
                "limit": {"type": "integer", "description": "default 20, max 60"},
            },
        },
    },
    {
        "name": "stock_profile",
        "description": (
            "One company in full: profile, valuation, the price path over a "
            "window, distance from its 52-week range, and how it has moved "
            "against its own index. Accepts a ticker (BHP.AX, RELIANCE.NS, "
            "AAPL) or a company name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "days": {"type": "integer", "description": "default 180"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "stock_news",
        "description": (
            "The accumulated news file for one company, or the most recent "
            "coverage across a whole market. This is Alfred's per-stock "
            "context layer: it reaches back through stored coverage rather "
            "than only today's tape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "omit to see the whole market"},
                "exchange": {"type": "string", "enum": ["US", "ASX", "NSE"]},
                "days": {"type": "integer", "description": "default 14"},
                "limit": {"type": "integer", "description": "default 25"},
            },
        },
    },
    {
        "name": "market_snapshot",
        "description": (
            "The state of one market at a glance: index level and move, "
            "breadth (how many names are up), the strongest and weakest "
            "sectors by median move, and the biggest movers. Use this before "
            "screening, to know what kind of tape you are reading."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exchange": {"type": "string", "enum": ["US", "ASX", "NSE"]},
                "days": {"type": "integer", "description": "default 5"},
            },
            "required": ["exchange"],
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
    # Company coverage is part of the memory, not a separate silo: a question
    # about a macro theme should surface the listed names whose own news file
    # touches it, which is where a stock-level insight usually starts.
    if scope in ("documents", "both"):
        # Full-text rather than ILIKE: '%gold%' matches "Goldman Sachs", which
        # is precisely the wrong company to surface for a question about gold.
        # to_tsquery stems and respects word boundaries, and ranks the hits.
        sec_rows = db.query(
            """SELECT n.symbol, s.name, s.exchange, n.title, n.publisher,
                      n.url, n.published_at
               FROM security_news n JOIN securities s ON s.symbol = n.symbol
               WHERE to_tsvector('english', n.title || ' ' || coalesce(n.summary, ''))
                     @@ plainto_tsquery('english', %s)
               ORDER BY n.published_at DESC NULLS LAST LIMIT %s""",
            (query, max(4, limit // 2)))
        if sec_rows:
            out["company_coverage"] = [
                {"symbol": r["symbol"], "company": r["name"],
                 "exchange": r["exchange"], "title": r["title"],
                 "publisher": r["publisher"], "url": r["url"],
                 "published": r["published_at"].isoformat() if r["published_at"] else None}
                for r in sec_rows
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


# ───────────────────────────── public equities ──────────────────────────────
def _resolve_symbol(symbol: str) -> str | None:
    """Accept a ticker in any of the three markets, or a company name."""
    s = (symbol or "").strip()
    if not s:
        return None
    row = db.one("SELECT symbol FROM securities WHERE upper(symbol) = upper(%s)", (s,))
    if row:
        return row["symbol"]
    # Bare ticker without its exchange suffix, then name match.
    for suffix in (".AX", ".NS"):
        row = db.one("SELECT symbol FROM securities WHERE upper(symbol) = upper(%s)",
                     (s + suffix,))
        if row:
            return row["symbol"]
    row = db.one(
        """SELECT symbol FROM securities WHERE name ILIKE %s
           ORDER BY market_cap DESC NULLS LAST LIMIT 1""", (f"%{s}%",))
    return row["symbol"] if row else None


_MOVE_CTE = """
    WITH win AS (
      SELECT symbol,
             (array_agg(close ORDER BY d DESC))[1] AS last_close,
             (array_agg(close ORDER BY d ASC))[1]  AS first_close,
             max(d) AS last_day
      FROM security_prices
      WHERE d >= current_date - %(days)s
      GROUP BY symbol
    )
"""


def _screen_stocks(exchange: str | None = None, sector: str | None = None,
                   sort_by: str = "change_pct", direction: str = "desc",
                   days: int = 30, max_pe: float | None = None,
                   min_market_cap: float | None = None, limit: int = 20) -> dict:
    limit = max(1, min(int(limit or 20), 60))
    sort_col = {
        "change_pct": "change_pct", "market_cap": "s.market_cap",
        "trailing_pe": "s.trailing_pe", "dividend_yield": "s.dividend_yield",
        "from_52w_high": "from_52w_high",
    }.get(sort_by, "change_pct")
    order = "ASC" if str(direction).lower() == "asc" else "DESC"

    where = ["w.first_close > 0"]
    params: dict = {"days": int(days or 30)}
    if exchange:
        where.append("s.exchange = %(exchange)s")
        params["exchange"] = exchange.upper()
    if sector:
        where.append("s.sector ILIKE %(sector)s")
        params["sector"] = f"%{sector}%"
    if max_pe is not None:
        where.append("s.trailing_pe IS NOT NULL AND s.trailing_pe <= %(max_pe)s")
        params["max_pe"] = float(max_pe)
    if min_market_cap is not None:
        where.append("s.market_cap >= %(min_mcap)s")
        params["min_mcap"] = float(min_market_cap)

    rows = db.query(
        _MOVE_CTE + f"""
        SELECT s.symbol, s.name, s.exchange, s.sector, s.currency,
               round(w.last_close, 2) AS last,
               round(((w.last_close / w.first_close) - 1) * 100, 2) AS change_pct,
               s.market_cap, round(s.trailing_pe, 1) AS trailing_pe,
               round(s.dividend_yield, 2) AS dividend_yield,
               CASE WHEN s.week52_high > 0
                    THEN round(((w.last_close / s.week52_high) - 1) * 100, 1) END
                 AS from_52w_high,
               w.last_day
        FROM win w JOIN securities s ON s.symbol = w.symbol
        WHERE {' AND '.join(where)}
        ORDER BY {sort_col} {order} NULLS LAST
        LIMIT {limit}""",
        params,
    )
    return {"window_days": int(days or 30), "sorted_by": sort_by,
            "count": len(rows), "results": rows}


def _stock_profile(symbol: str, days: int = 180) -> dict:
    sym = _resolve_symbol(symbol)
    if not sym:
        return {"error": f"'{symbol}' is not in the covered universe "
                         "(S&P 500, ASX 200, NIFTY 50)"}
    sec = db.one("SELECT * FROM securities WHERE symbol = %s", (sym,))
    prices = db.query(
        """SELECT d, close FROM security_prices WHERE symbol = %s
           AND d >= current_date - %s ORDER BY d""", (sym, int(days or 180)))
    if not prices:
        return {"security": sec, "note": "no stored price history yet"}
    first, last = float(prices[0]["close"]), float(prices[-1]["close"])
    series = [{"d": str(p["d"]), "c": round(float(p["close"]), 2)} for p in prices]
    if len(series) > 80:  # thin for the context window, keep the endpoints
        step = len(series) // 80 + 1
        series = series[::step] + [series[-1]]
    peers = db.query(
        """SELECT symbol, name FROM securities
           WHERE sector = %s AND exchange = %s AND symbol <> %s
           ORDER BY market_cap DESC NULLS LAST LIMIT 6""",
        (sec.get("sector"), sec.get("exchange"), sym))
    return {
        "security": {k: v for k, v in sec.items() if k != "created_at"},
        "window_days": int(days or 180),
        "last": round(last, 2), "change_pct": round((last / first - 1) * 100, 2),
        "high": round(max(float(p["close"]) for p in prices), 2),
        "low": round(min(float(p["close"]) for p in prices), 2),
        "from_52w_high": (round((last / float(sec["week52_high"]) - 1) * 100, 1)
                          if sec.get("week52_high") else None),
        "sector_peers": peers,
        "series": series,
    }


def _stock_news(symbol: str | None = None, exchange: str | None = None,
                days: int = 14, limit: int = 25) -> dict:
    limit = max(1, min(int(limit or 25), 80))
    where, params = ["n.published_at > now() - make_interval(days => %(days)s)"], \
        {"days": int(days or 14)}
    if symbol:
        sym = _resolve_symbol(symbol)
        if not sym:
            return {"error": f"'{symbol}' is not in the covered universe"}
        where.append("n.symbol = %(sym)s")
        params["sym"] = sym
    if exchange:
        where.append("s.exchange = %(ex)s")
        params["ex"] = exchange.upper()
    rows = db.query(
        f"""SELECT n.symbol, s.name, s.exchange, n.title, n.publisher, n.url,
                   n.summary, n.published_at
            FROM security_news n JOIN securities s ON s.symbol = n.symbol
            WHERE {' AND '.join(where)}
            ORDER BY n.published_at DESC NULLS LAST
            LIMIT {limit}""", params)
    return {"count": len(rows), "window_days": int(days or 14), "items": rows}


def _market_snapshot(exchange: str, days: int = 5) -> dict:
    ex = (exchange or "").upper()
    days = int(days or 5)
    index = {"US": "SPX", "ASX": None, "NSE": "INDIA"}.get(ex)
    rows = db.query(
        _MOVE_CTE + """
        SELECT s.symbol, s.name, s.sector,
               round(((w.last_close / w.first_close) - 1) * 100, 2) AS change_pct
        FROM win w JOIN securities s ON s.symbol = w.symbol
        WHERE s.exchange = %(ex)s AND w.first_close > 0""",
        {"days": days, "ex": ex})
    if not rows:
        return {"error": f"no stored prices for {ex} yet"}
    ups = sum(1 for r in rows if (r["change_pct"] or 0) > 0)
    by_sector: dict[str, list] = {}
    for r in rows:
        by_sector.setdefault(r["sector"] or "Unclassified", []).append(
            float(r["change_pct"] or 0))
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    sectors = sorted(({"sector": k, "median_pct": round(med(v), 2), "n": len(v)}
                      for k, v in by_sector.items()),
                     key=lambda x: x["median_pct"], reverse=True)
    ranked = sorted(rows, key=lambda r: float(r["change_pct"] or 0), reverse=True)
    out = {
        "exchange": ex, "market": {"US": "United States", "ASX": "Australia",
                                   "NSE": "India"}.get(ex, ex),
        "window_days": days, "covered": len(rows),
        "advancing": ups, "declining": len(rows) - ups,
        "breadth_pct": round(ups / len(rows) * 100, 1),
        "sectors_strongest": sectors[:4], "sectors_weakest": sectors[-4:],
        "top_movers": ranked[:8], "worst_movers": ranked[-8:],
    }
    if index:
        idx = _query_prices([index], days=max(days, 7))
        out["index"] = idx.get(index)
    return out


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
    "screen_stocks": _screen_stocks,
    "stock_profile": _stock_profile,
    "stock_news": _stock_news,
    "market_snapshot": _market_snapshot,
}

# Span kinds for the trace tree: lookups that feed the model context are
# retrieval, actions are tools.
_SPAN_KIND = {
    "search_memory": "retrieval",
    "get_recent_news": "retrieval",
    "query_relationships": "retrieval",
    "get_world_model": "retrieval",
    "screen_stocks": "retrieval",
    "stock_profile": "retrieval",
    "stock_news": "retrieval",
    "market_snapshot": "retrieval",
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
