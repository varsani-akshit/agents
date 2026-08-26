"""MIA dashboard — the reading surface for digests, alerts, and Q&A.

Chat apps mangle tables and force charts into separate messages, so the
comprehensive brief needs a page. This serves rendered digests with their charts
inline, the world-model history, fired alerts, and an ask box so the whole
two-way loop lives in one place.

Reads Postgres directly — no API layer between, because there is exactly one
consumer.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import markdown as md
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
import db

BASE = Path(__file__).resolve().parent
app = FastAPI(title="MIA", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(BASE / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
# Charts live beside the digests they belong to.
app.mount("/outbox", StaticFiles(directory=str(config.OUTBOX)), name="outbox")

_MD = md.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists"])


_LIST_ITEM = re.compile(r"^(\s+)([-*+]|\d+[.)])\s")


def normalise_list_indent(text: str) -> str:
    """Round sub-list indentation up to a multiple of four spaces.

    Python-Markdown needs four spaces to nest a list under an ordered item, and
    the model writes three (aligning under "1. "). With `sane_lists` the nested
    bullets then render as literal "- " inside the parent paragraph; without it
    they are silently promoted into the numbered list, which is worse — the
    sub-points become top-level findings and the structure of the argument is
    quietly rewritten.

    Fenced code blocks are left untouched, since indentation is content there.
    """
    out, in_fence = [], False
    for line in (text or "").split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        m = None if in_fence else _LIST_ITEM.match(line)
        if m:
            depth = -(-len(m.group(1)) // 4)  # ceil to the next multiple of 4
            line = " " * (4 * depth) + line.lstrip()
        out.append(line)
    return "\n".join(out)


def render_markdown(text: str, day: str | None = None) -> str:
    """Markdown to HTML, rewriting relative chart paths to served URLs."""
    if day:
        text = re.sub(r"\]\((charts/[^)]+)\)", rf"](/outbox/{day}/\1)", text)
    _MD.reset()
    return _MD.convert(normalise_list_indent(text))


def _day_of(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _ago(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt.astimezone(timezone.utc)
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins}m ago"
    if mins < 1440:
        return f"{mins // 60}h ago"
    return f"{mins // 1440}d ago"


def _utc(dt: datetime, fmt: str = "%d %b %Y, %H:%M") -> str:
    """Format in real UTC.

    Postgres returns timestamptz in the session timezone (here
    Australia/Melbourne), so calling strftime directly and appending "UTC"
    printed local time under a UTC label — a 10-hour lie on every timestamp.
    """
    return dt.astimezone(timezone.utc).strftime(fmt)


templates.env.filters["ago"] = _ago
templates.env.filters["utc"] = _utc


# ──────────────────────────────── data access ───────────────────────────────
def latest_digest() -> dict | None:
    return db.one(
        "SELECT id, title, body, meta, created_at FROM analyses "
        "WHERE kind='digest' ORDER BY created_at DESC LIMIT 1"
    )


def digest_list(limit: int = 60) -> list[dict]:
    return db.query(
        """SELECT id, title, created_at, meta->>'regime' AS regime,
                  meta->>'model' AS model, meta->>'usd' AS usd
           FROM analyses WHERE kind='digest'
           ORDER BY created_at DESC LIMIT %s""",
        (limit,),
    )


def system_summary() -> dict:
    from brain import client

    counts = db.one(
        """SELECT (SELECT count(*) FROM documents) AS docs,
                  (SELECT count(*) FROM documents
                    WHERE fetched_at > now() - interval '24 hours') AS docs_24h,
                  (SELECT count(*) FROM analyses WHERE kind='digest') AS digests,
                  (SELECT count(*) FROM trigger_events
                    WHERE created_at > now() - interval '24 hours') AS triggers_24h,
                  (SELECT count(*) FROM entities) AS entities,
                  (SELECT count(*) FROM edges) AS edges"""
    )
    last_tick = db.one(
        "SELECT started_at, ok FROM job_runs WHERE job='tick' ORDER BY started_at DESC LIMIT 1"
    )
    return {
        "counts": counts,
        "last_tick": last_tick,
        "budget": client.budget_status(),
        "providers": client.spend_by_provider(),
    }


# ───────────────────────────────── routes ───────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    d = latest_digest()
    body_html = ""
    if d:
        body_html = render_markdown(d["body"], _day_of(d["created_at"]))
    return templates.TemplateResponse(
        request,
        "digest.html",
        {
            "digest": d,
            "body_html": body_html,
            "recent": digest_list(12),
            "summary": system_summary(),
            "active": "latest",
        },
    )


@app.get("/digest/{digest_id}", response_class=HTMLResponse)
async def one_digest(request: Request, digest_id: int):
    d = db.one(
        "SELECT id,title,body,meta,created_at FROM analyses WHERE id=%s AND kind='digest'",
        (digest_id,),
    )
    if not d:
        return RedirectResponse("/")
    return templates.TemplateResponse(
        request,
        "digest.html",
        {
            "digest": d,
            "body_html": render_markdown(d["body"], _day_of(d["created_at"])),
            "recent": digest_list(12),
            "summary": system_summary(),
            "active": "latest",
        },
    )


@app.get("/archive", response_class=HTMLResponse)
async def archive(request: Request):
    return templates.TemplateResponse(
        request,
        "archive.html",
        {
            "digests": digest_list(100),
            "answers": db.query(
                """SELECT id,title,created_at,meta->>'usd' AS usd FROM analyses
                   WHERE kind='answer' ORDER BY created_at DESC LIMIT 40"""
            ),
            "summary": system_summary(),
            "active": "archive",
        },
    )


@app.get("/answer/{answer_id}", response_class=HTMLResponse)
async def one_answer(request: Request, answer_id: int):
    a = db.one("SELECT id,title,body,meta,created_at FROM analyses WHERE id=%s", (answer_id,))
    if not a:
        return RedirectResponse("/ask")
    return templates.TemplateResponse(
        request,
        "ask.html",
        {
            "answer": a,
            "answer_html": render_markdown(a["body"]),
            "question": (a["meta"] or {}).get("question", a["title"]),
            "history": db.query(
                """SELECT id,title,created_at FROM analyses WHERE kind='answer'
                   ORDER BY created_at DESC LIMIT 25"""
            ),
            "summary": system_summary(),
            "active": "ask",
        },
    )


@app.get("/ask", response_class=HTMLResponse)
async def ask_form(request: Request):
    return templates.TemplateResponse(
        request,
        "ask.html",
        {
            "answer": None,
            "history": db.query(
                """SELECT id,title,created_at FROM analyses WHERE kind='answer'
                   ORDER BY created_at DESC LIMIT 25"""
            ),
            "summary": system_summary(),
            "active": "ask",
        },
    )


@app.post("/ask")
async def ask_submit(question: str = Form(...), model: str = Form("")):
    """Run a question through the agent. Blocking calls go to a worker thread so
    the event loop stays responsive while the model thinks."""
    from brain import ask as ask_mod
    from brain import client

    client.AUTONOMOUS = False  # interactive: may use the reserved budget

    def _run():
        return ask_mod.ask(question.strip(), model=model or None)

    result = await asyncio.to_thread(_run)
    if result.get("analysis_id"):
        return RedirectResponse(f"/answer/{result['analysis_id']}", status_code=303)
    return JSONResponse({"error": result.get("stopped") or "no answer produced"}, 500)


@app.get("/alerts", response_class=HTMLResponse)
async def alerts(request: Request):
    events = db.query(
        """SELECT id, rule, severity, symbol, detail, created_at, notified_at
           FROM trigger_events ORDER BY created_at DESC LIMIT 100"""
    )
    written = {
        (a["meta"] or {}).get("event_id"): a
        for a in db.query(
            "SELECT title, body, meta, created_at FROM analyses WHERE kind='alert' "
            "ORDER BY created_at DESC LIMIT 100"
        )
    }
    for e in events:
        e["text"] = (written.get(e["id"]) or {}).get("body")
    return templates.TemplateResponse(
        request,
        "alerts.html",
        {"events": events, "summary": system_summary(), "active": "alerts"},
    )


@app.get("/world", response_class=HTMLResponse)
async def world(request: Request):
    versions = db.query(
        "SELECT version, regime, created_at FROM world_model ORDER BY version DESC LIMIT 40"
    )
    current = db.one("SELECT * FROM world_model ORDER BY version DESC LIMIT 1")
    return templates.TemplateResponse(
        request,
        "world.html",
        {
            "current": current,
            "current_html": render_markdown(current["body"]) if current else "",
            "versions": versions,
            "summary": system_summary(),
            "active": "world",
        },
    )


@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    from brain import extract, llm
    from ingest import feeds

    return templates.TemplateResponse(
        request,
        "status.html",
        {
            "summary": system_summary(),
            "sources": feeds.source_health(),
            "routing": llm.routing_table(),
            "graph": extract.graph_stats(),
            "jobs": db.query(
                """SELECT job, started_at, finished_at, ok, left(coalesce(error,''),160) AS error
                   FROM job_runs ORDER BY started_at DESC LIMIT 30"""
            ),
            "spend": db.query(
                """SELECT provider, model,
                          regexp_replace(purpose,':turn[0-9]+','') AS purpose,
                          count(*) AS calls, round(sum(usd)::numeric,4) AS usd
                   FROM api_calls GROUP BY 1,2,3 ORDER BY sum(usd) DESC LIMIT 20"""
            ),
            "active": "status",
        },
    )


@app.get("/api/latest")
async def api_latest():
    """JSON for whatever notifier ends up delivering the ping."""
    d = latest_digest()
    if not d:
        return JSONResponse({"error": "no digest yet"}, 404)
    meta = d["meta"] or {}
    return {
        "id": d["id"],
        "title": d["title"],
        "created_at": d["created_at"].isoformat(),
        "regime": meta.get("regime"),
        "model": meta.get("model"),
        "usd": meta.get("usd"),
        "body": d["body"],
    }


@app.get("/healthz")
async def healthz():
    try:
        db.one("SELECT 1 AS ok")
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, 503)
