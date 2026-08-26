"""MIA dashboard — the reading surface for digests, alerts, and Q&A.

Chat apps mangle tables and force charts into separate messages, so the
comprehensive brief needs a page. This serves rendered digests with their charts
drawn live in the browser, the world-model history, fired alerts, and an ask box,
so the whole two-way loop lives in one place.

Reads Postgres directly — no API layer between, because there is exactly one
consumer.
"""
from __future__ import annotations

import asyncio
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import markdown as md
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import config
import db
from signals import chartdata
from web import auth

BASE = Path(__file__).resolve().parent
app = FastAPI(title="MIA", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(BASE / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

_MD = md.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists"])

_LIST_ITEM = re.compile(r"^(\s+)([-*+]|\d+[.)])\s")
# The model writes figures as image references; they become live chart mounts.
_CHART_REF = re.compile(r"!\[([^\]]*)\]\((?:\./)?charts/([A-Za-z0-9_\-]+)\.png\)")


# ──────────────────────────────── rendering ─────────────────────────────────
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


def mount_charts(text: str, pack: dict) -> tuple[str, list[str]]:
    """Swap chart image references for mount points the browser draws into.

    Titles come from the pack rather than the model's alt text, so a figure is
    always labelled with what it actually plots. A reference to a chart that was
    not rendered this cycle is dropped rather than left as a broken image.
    """
    used: list[str] = []

    def sub(m: re.Match) -> str:
        key = m.group(2)
        spec = pack.get(key)
        if not spec:
            return ""
        used.append(key)
        title = html.escape(spec.get("title") or m.group(1) or key)
        subtitle = html.escape(spec.get("subtitle") or "")
        return (
            f'<figure class="chart" data-chart="{html.escape(key)}">'
            f'<figcaption class="chart-head"><div class="t">{title}</div>'
            + (f'<div class="s">{subtitle}</div>' if subtitle else "")
            + '</figcaption><div class="chart-canvas"></div></figure>'
        )

    return _CHART_REF.sub(sub, text or ""), used


def wrap_tables(html_text: str) -> str:
    """Let wide tables scroll inside their own box instead of the page."""
    return html_text.replace("<table>", '<div class="tablewrap"><table>').replace(
        "</table>", "</table></div>")


def render_markdown(text: str, pack: dict | None = None) -> str:
    text, _ = mount_charts(text or "", pack or {})
    _MD.reset()
    return wrap_tables(_MD.convert(normalise_list_indent(text)))


# ──────────────────────────────── formatting ────────────────────────────────
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


def page(request: Request, name: str, ctx: dict) -> HTMLResponse:
    """Render with the context every template's chrome expects."""
    ctx.setdefault("summary", system_summary())
    ctx.setdefault("charts_json", "{}")
    ctx["user"] = auth.current_user(request)
    return templates.TemplateResponse(request, name, ctx)


# ─────────────────────────────── auth gate ──────────────────────────────────
@app.middleware("http")
async def require_login(request: Request, call_next):
    if auth.is_public(request.url.path) or auth.current_user(request):
        return await call_next(request)
    nxt = request.url.path
    if request.url.query:
        nxt += "?" + request.url.query
    return RedirectResponse(f"/login?next={nxt}", status_code=303)


# Registered *after* the gate above, deliberately. Starlette wraps the most
# recently added middleware outermost, so the session must be added last for
# `request.session` to exist by the time the gate reads it. Added first, the gate
# runs outside the session middleware and every request dies on
# "SessionMiddleware must be installed".
app.add_middleware(
    SessionMiddleware,
    secret_key=auth.session_secret(),
    session_cookie="mia_session",
    max_age=60 * 60 * 24 * 30,   # a month; this is a personal dashboard
    same_site="lax",
    https_only=False,            # the host terminates TLS; set true behind one you control
)


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/"):
    if auth.current_user(request):
        return RedirectResponse(next or "/", status_code=303)
    return templates.TemplateResponse(
        request, "login.html", {"next": next, "error": None})


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    user = await asyncio.to_thread(auth.verify, username, password)
    if not user:
        return templates.TemplateResponse(
            request, "login.html",
            {"next": next, "error": "That username and password do not match."},
            status_code=401,
        )
    auth.login_session(request, user)
    # Only ever redirect within this site: an open redirect turns the login page
    # into a credible-looking way to bounce someone somewhere else.
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    return RedirectResponse(target, status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ───────────────────────────────── routes ───────────────────────────────────
def _digest_page(request: Request, d: dict | None) -> HTMLResponse:
    pack = chartdata.load_pack(d["id"]) if d else {}
    # Briefs written before charts were stored fall back to the newest pack so
    # the figures still draw — but that pack is today's data under an older
    # brief, so the page says so rather than letting it pass as contemporaneous.
    stale = bool(d and not pack)
    if stale:
        pack = chartdata.latest_pack()
    return page(request, "digest.html", {
        "digest": d,
        "body_html": render_markdown(d["body"], pack) if d else "",
        "recent": digest_list(12),
        "charts_json": json.dumps(pack, default=str),
        "charts_stale": stale,
        "active": "latest",
    })


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return _digest_page(request, latest_digest())


@app.get("/digest/{digest_id}", response_class=HTMLResponse)
async def one_digest(request: Request, digest_id: int):
    d = db.one(
        "SELECT id,title,body,meta,created_at FROM analyses WHERE id=%s AND kind='digest'",
        (digest_id,),
    )
    if not d:
        return RedirectResponse("/")
    return _digest_page(request, d)


@app.get("/archive", response_class=HTMLResponse)
async def archive(request: Request):
    return page(request, "archive.html", {
        "digests": digest_list(100),
        "answers": db.query(
            """SELECT id,title,created_at,meta->>'usd' AS usd FROM analyses
               WHERE kind='answer' ORDER BY created_at DESC LIMIT 40"""
        ),
        "active": "archive",
    })


@app.get("/answer/{answer_id}", response_class=HTMLResponse)
async def one_answer(request: Request, answer_id: int):
    a = db.one("SELECT id,title,body,meta,created_at FROM analyses WHERE id=%s", (answer_id,))
    if not a:
        return RedirectResponse("/ask")
    pack = chartdata.latest_pack()
    return page(request, "ask.html", {
        "answer": a,
        "answer_html": render_markdown(a["body"], pack),
        "question": (a["meta"] or {}).get("question", a["title"]),
        "history": db.query(
            """SELECT id,title,created_at FROM analyses WHERE kind='answer'
               ORDER BY created_at DESC LIMIT 25"""
        ),
        "charts_json": json.dumps(pack, default=str),
        "active": "ask",
    })


@app.get("/ask", response_class=HTMLResponse)
async def ask_form(request: Request):
    return page(request, "ask.html", {
        "answer": None,
        "history": db.query(
            """SELECT id,title,created_at FROM analyses WHERE kind='answer'
               ORDER BY created_at DESC LIMIT 25"""
        ),
        "active": "ask",
    })


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


@app.get("/charts", response_class=HTMLResponse)
async def charts_page(request: Request):
    """Every figure on one page, independent of any brief."""
    pack = chartdata.latest_pack()
    return page(request, "charts.html", {
        "specs": [{"key": s.get("key"), "title": s.get("title", ""),
                   "subtitle": s.get("subtitle", "")}
                  for s in chartdata.in_display_order(pack)],
        "charts_json": json.dumps(pack, default=str),
        "active": "charts",
    })


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
    return page(request, "alerts.html", {"events": events, "active": "alerts"})


@app.get("/world", response_class=HTMLResponse)
async def world(request: Request):
    versions = db.query(
        "SELECT version, regime, created_at FROM world_model ORDER BY version DESC LIMIT 40"
    )
    current = db.one("SELECT * FROM world_model ORDER BY version DESC LIMIT 1")
    return page(request, "world.html", {
        "current": current,
        "current_html": render_markdown(current["body"], {}) if current else "",
        "versions": versions,
        "active": "world",
    })


@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    from brain import extract, llm
    from ingest import feeds

    return page(request, "status.html", {
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
    })


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
