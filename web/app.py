"""Alfred dashboard — the reading surface for briefs, alerts, and Q&A.

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
import logging
import os
import re
from contextlib import asynccontextmanager
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
log = logging.getLogger("mia.web")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Optionally run the scheduler in-process, so one service does both jobs.

    Off by default: on a host that runs two services, or locally where launchd
    owns the scheduler, starting a second copy here would double every tick and
    write duplicate briefs.
    """
    from brain import observe

    observe.init("web")
    sched = None
    if os.getenv("MIA_EMBEDDED_SCHEDULER", "").lower() in ("1", "true", "yes"):
        from scheduler import build_background_scheduler

        sched = build_background_scheduler()
        sched.start()
        log.info("embedded scheduler started: %s",
                 ", ".join(j.id for j in sched.get_jobs()))
    try:
        yield
    finally:
        if sched:
            sched.shutdown(wait=False)


app = FastAPI(title="Alfred", docs_url=None, redoc_url=None, lifespan=lifespan)
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


_DEV_TITLE = re.compile(r"<p><strong>((?:(?!</?strong>).)+)</strong></p>")


def render_markdown(text: str, pack: dict | None = None) -> str:
    text, _ = mount_charts(text or "", pack or {})
    _MD.reset()
    html_text = wrap_tables(_MD.convert(normalise_list_indent(text)))
    # A bold line alone in its own paragraph is a development title — the
    # brief format's convention. Tagged here, deterministically, so CSS can
    # promote it without also catching paragraphs that merely open bold.
    return _DEV_TITLE.sub(r'<p class="devtitle"><strong>\1</strong></p>', html_text)


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


def asset(path: str) -> str:
    """Static URL stamped with the file's modification time.

    Without this the browser keeps serving the JavaScript and CSS it cached
    before a change — locally that wastes debugging time chasing behaviour that
    is already fixed, and on a deploy it ships new HTML against old scripts,
    which is worse because the two disagree. The stamp changes only when the
    file does, so caching stays aggressive and correctness is not negotiable.
    """
    file = BASE / "static" / path.lstrip("/")
    try:
        return f"/static/{path.lstrip('/')}?v={int(file.stat().st_mtime)}"
    except OSError:
        return f"/static/{path.lstrip('/')}"


templates.env.globals["asset"] = asset


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
        # Starlette only re-issues the cookie when the session dict changes, so
        # a fixed max_age would expire 60 days after sign-in however often you
        # visit. Touching it on each request makes the window rolling: you stay
        # signed in indefinitely while you keep using it, and only a real 60-day
        # absence (or clearing cookies) logs you out.
        if not auth.is_public(request.url.path):
            request.session["seen"] = int(datetime.now(timezone.utc).timestamp())
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
    # 60 days, refreshed on every request below, so regular use never expires.
    max_age=60 * 60 * 24 * 60,
    same_site="lax",
    # Marks the cookie Secure, so the browser will not send it over plain HTTP.
    # Off by default because that would break local development on http://; set
    # MIA_HTTPS_ONLY=1 wherever TLS actually terminates in front of the app.
    https_only=os.getenv("MIA_HTTPS_ONLY", "").lower() in ("1", "true", "yes"),
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
    from web import sharing

    share = sharing.for_analysis(d["id"]) if d else None
    return page(request, "digest.html", {
        "digest": d,
        "share": share,
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


@app.get("/digest/{digest_id}/research", response_class=HTMLResponse)
async def digest_research(request: Request, digest_id: int):
    """The brief's working papers: every pipeline stage, in full.

    What the Marshal prioritised, what each Scout found (with its quotes and
    sources), what each Analyst concluded, and what the Verifier checked —
    the evidence trail behind every sentence on the brief page.
    """
    d = db.one(
        "SELECT id,title,meta,created_at FROM analyses WHERE id=%s AND kind='digest'",
        (digest_id,),
    )
    if not d:
        return RedirectResponse("/")
    rows = db.query(
        """SELECT stage, beat, payload, usd, model, created_at
           FROM brief_runs WHERE analysis_id=%s ORDER BY id""",
        (digest_id,),
    )
    stages: dict = {"scout": {}, "analyst": {}}
    singles: dict = {}
    for r in rows:
        if r["stage"] in ("scout", "analyst"):
            stages[r["stage"]][r["beat"]] = r
        else:
            singles[r["stage"]] = r
    from brain.pipeline import beats as beats_mod

    # A beat that reported nothing is not a section — a heading over an empty
    # box reads as a broken page rather than as a quiet beat. The row appears
    # only when a scout found leads or an analyst produced findings.
    beat_rows = []
    for b in beats_mod.BEATS:
        s, a = stages["scout"].get(b["key"]), stages["analyst"].get(b["key"])
        leads = (s or {}).get("payload", {}).get("leads") or []
        findings = (a or {}).get("payload", {}).get("findings") or []
        if leads or findings:
            beat_rows.append({"beat": b, "scout": s if leads else None,
                              "analyst": a if findings else None})
    quiet = [b["section"] for b in beats_mod.BEATS
             if b["key"] not in {r["beat"]["key"] for r in beat_rows}
             and (stages["scout"].get(b["key"]) or stages["analyst"].get(b["key"]))]
    return page(request, "digest_research.html", {
        "digest": d,
        "beats": beat_rows,
        "marshal": singles.get("marshal"),
        "editor": singles.get("editor"),
        "verifier": singles.get("verifier"),
        "quiet": quiet,
        "active": "latest",
    })


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
        "conversations": conversation_list(),
        "charts_json": json.dumps(pack, default=str),
        "active": "ask",
    })


def conversation_list() -> list[dict]:
    """Every past exchange, quick and deep, newest first — the chat history."""
    rows = db.query(
        """SELECT id, coalesce(meta->>'question', title) AS question,
                  created_at, 'quick' AS depth
           FROM analyses WHERE kind='answer'
         UNION ALL
           SELECT id, question, created_at, 'deep' AS depth FROM research_notes
           ORDER BY created_at DESC LIMIT 60"""
    )
    for r in rows:
        r["url"] = (f"/research/{r['id']}" if r["depth"] == "deep"
                    else f"/answer/{r['id']}")
    return rows


@app.get("/ask", response_class=HTMLResponse)
async def ask_form(request: Request, q: str = "", depth: str = "quick"):
    return page(request, "ask.html", {
        "answer": None,
        "prefill": q[:400],
        "depth": depth if depth in ("quick", "deep") else "quick",
        "conversations": conversation_list(),
        "active": "ask",
    })


@app.post("/api/ask")
async def api_ask(request: Request):
    """Answer a question and return rendered HTML.

    The chat view posts here so it can show the question immediately and hold
    a thinking state while the agent works — a full page round-trip would
    leave the reader on a frozen form for a minute or more.
    """
    from brain import client

    payload = await request.json()
    q = (payload.get("question") or "").strip()
    if not q:
        return JSONResponse({"error": "empty question"}, 400)
    depth = "deep" if payload.get("depth") == "deep" else "quick"
    # The composer offers a tier, never a model id: no vendor name is ever
    # shipped to the browser, and the routing can change without the UI lying.
    model = {"extended": "gemini-pro-latest"}.get(payload.get("model") or "", "")
    client.AUTONOMOUS = False  # interactive: may use the reserved budget

    try:
        if depth == "deep":
            from brain import research as research_mod

            result = await asyncio.to_thread(research_mod.run, q, trigger="ask")
            if not result.get("note_id"):
                return JSONResponse({"error": "research produced no note"}, 500)
            return {"id": result["note_id"], "depth": "deep",
                    "url": f"/research/{result['note_id']}",
                    "html": render_markdown(result["body"], {})}

        from brain import ask as ask_mod

        result = await asyncio.to_thread(
            lambda: ask_mod.ask(q, model=model or None))
        if not result.get("analysis_id"):
            return JSONResponse(
                {"error": result.get("stopped") or "no answer produced"}, 500)
        return {"id": result["analysis_id"], "depth": "quick",
                "url": f"/answer/{result['analysis_id']}",
                "html": render_markdown(result.get("text") or "", {})}
    except Exception as exc:  # noqa: BLE001
        log.exception("ask failed")
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"[:300]}, 500)


@app.post("/ask")
async def ask_submit(question: str = Form(...), model: str = Form(""),
                     depth: str = Form("quick")):
    """One question surface, two depths. Quick is a single tool loop; deep is
    the same agent with the research machinery awake — parallel facet
    researchers and a synthesis — producing a stored note."""
    from brain import client

    client.AUTONOMOUS = False  # interactive: may use the reserved budget
    q = question.strip()

    if depth == "deep":
        from brain import research as research_mod

        result = await asyncio.to_thread(research_mod.run, q, trigger="ask")
        if result.get("note_id"):
            return RedirectResponse(f"/research/{result['note_id']}", status_code=303)
        return JSONResponse({"error": "deep research produced no note"}, 500)

    from brain import ask as ask_mod

    result = await asyncio.to_thread(
        lambda: ask_mod.ask(q, model=model or None))
    if result.get("analysis_id"):
        return RedirectResponse(f"/answer/{result['analysis_id']}", status_code=303)
    return JSONResponse({"error": result.get("stopped") or "no answer produced"}, 500)


@app.get("/research", response_class=HTMLResponse)
async def research_form(q: str = ""):
    """Ask owns the question surface now; deep research is its second depth.
    Old links (and the dig-deeper buttons) land on Ask with deep preselected."""
    from urllib.parse import quote

    return RedirectResponse(f"/ask?depth=deep&q={quote(q[:400])}", status_code=307)


@app.get("/research/{note_id}", response_class=HTMLResponse)
async def research_note(request: Request, note_id: int):
    note = db.one("SELECT * FROM research_notes WHERE id=%s", (note_id,))
    if not note:
        return RedirectResponse("/research")
    return page(request, "research.html", {
        "note": note,
        "body_html": render_markdown(note["body"], {}),
        "history": db.query(
            """SELECT id, question, created_at, usd FROM research_notes
               ORDER BY created_at DESC LIMIT 25"""),
        "active": "ask",
    })


# The charts page groups its figures under tabs; a 15-figure scroll is not a
# structure. Order within each tab is reading order.
CHART_TABS: list[tuple[str, list[str]]] = [
    ("Overview", ["regime_gauge", "returns_heatmap", "cross_asset_performance"]),
    ("Markets", ["normalised_performance", "global_equities", "drawdowns"]),
    ("Rates & credit", ["yield_curve", "net_liquidity", "real_yield_gold",
                        "credit_spreads", "macro_panel"]),
    ("FX", ["fx_performance"]),
    ("Correlations", ["rolling_correlations", "correlation_heatmap", "ratios"]),
]


@app.get("/markets", response_class=HTMLResponse)
async def markets_page(request: Request, ex: str = "US", sort: str = "change_pct",
                       days: int = 30, sector: str = ""):
    """The three covered exchanges, screenable. Live from the database on every
    request — this is a working surface, not a stored snapshot."""
    from brain import tools

    ex = ex.upper() if ex.upper() in ("US", "ASX", "NSE") else "US"
    snapshot = await asyncio.to_thread(
        tools.HANDLERS["market_snapshot"], exchange=ex, days=min(days, 90))
    screen = await asyncio.to_thread(
        tools.HANDLERS["screen_stocks"], exchange=ex, sector=sector or None,
        sort_by=sort, days=min(days, 365), limit=40)
    news = await asyncio.to_thread(
        tools.HANDLERS["stock_news"], exchange=ex, days=7, limit=14)
    sectors = db.query(
        """SELECT DISTINCT sector FROM securities
           WHERE exchange=%s AND sector IS NOT NULL ORDER BY sector""", (ex,))
    return page(request, "markets.html", {
        "ex": ex, "days": days, "sort": sort, "sector": sector,
        "snapshot": snapshot if "error" not in snapshot else None,
        "rows": screen.get("results", []),
        "news": news.get("items", []),
        "sectors": [r["sector"] for r in sectors],
        "markets": {"US": "United States", "ASX": "Australia", "NSE": "India"},
        "active": "markets",
    })


@app.get("/markets/{symbol}", response_class=HTMLResponse)
async def stock_page(request: Request, symbol: str):
    """One company: the measured profile, its price path, and its news file."""
    from brain import tools

    prof = await asyncio.to_thread(tools.HANDLERS["stock_profile"],
                                   symbol=symbol, days=365)
    if prof.get("error"):
        return RedirectResponse("/markets")
    sym = prof["security"]["symbol"]
    news = await asyncio.to_thread(tools.HANDLERS["stock_news"],
                                   symbol=sym, days=180, limit=30)
    return page(request, "stock.html", {
        "p": prof, "s": prof["security"], "news": news.get("items", []),
        "active": "markets",
    })


@app.get("/api/chart/stock/{symbol}")
async def api_stock_chart(symbol: str, days: int = 365):
    """Price path for one security, drawn by the same chart machinery."""
    from brain import tools

    prof = await asyncio.to_thread(tools.HANDLERS["stock_profile"],
                                   symbol=symbol, days=min(days, 2000))
    if prof.get("error") or not prof.get("series"):
        return JSONResponse({"error": "no data"}, 404)
    s = prof["security"]
    return {
        "key": f"stock:{s['symbol']}", "type": "line",
        "title": f"{s['name']} ({s['symbol']})",
        "subtitle": f"{s.get('currency') or ''} · last {prof['last']:,.2f} · "
                    f"{prof['change_pct']:+.1f}% over the window",
        "x": [p["d"] for p in prof["series"]],
        "yLabel": s.get("currency") or "",
        "series": [{"name": s["symbol"],
                    "data": [p["c"] for p in prof["series"]], "color": "#EA580C"}],
    }


@app.get("/charts", response_class=HTMLResponse)
async def charts_page(request: Request):
    """Every standing figure, tabbed, searchable, and live to the controls."""
    pack = chartdata.latest_pack()
    placed: set[str] = set()
    tabs = []
    for label, keys in CHART_TABS:
        specs = [{"key": k, "title": pack[k].get("title", k),
                  "subtitle": pack[k].get("subtitle", ""),
                  "rebuildable": k in chartdata._REBUILDABLE}
                 for k in keys if k in pack]
        placed.update(k for k in keys if k in pack)
        if specs:
            tabs.append({"label": label, "specs": specs})
    leftover = [{"key": k, "title": v.get("title", k), "subtitle": v.get("subtitle", ""),
                 "rebuildable": False}
                for k, v in pack.items() if k not in placed]
    if leftover:
        tabs.append({"label": "Other", "specs": leftover})
    return page(request, "charts.html", {
        "tabs": tabs,
        "priceable": chartdata.PRICEABLE,
        "currencies": list(chartdata.CURRENCIES),
        "periods": list(chartdata.PERIODS),
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
        "embedding": db.query(
            """SELECT coalesce(embed_model, '(not embedded)') AS model, count(*) AS docs
               FROM documents GROUP BY 1 ORDER BY 2 DESC"""
        ),
        "briefs": db.query(
            """SELECT id, created_at, meta->>'model' AS model,
                      (meta->>'words')::int AS words,
                      (meta->>'usd')::numeric AS usd,
                      jsonb_array_length(coalesce(meta->'citations','[]'::jsonb)) AS cites
               FROM analyses WHERE kind='digest'
               ORDER BY created_at DESC LIMIT 12"""
        ),
        "spend": db.query(
            """SELECT provider, model,
                      regexp_replace(purpose,':turn[0-9]+','') AS purpose,
                      count(*) AS calls, round(sum(usd)::numeric,4) AS usd
               FROM api_calls GROUP BY 1,2,3 ORDER BY sum(usd) DESC LIMIT 20"""
        ),
        # The army at a glance: each agent's most recent run and 24h record.
        "army": db.query(
            """SELECT agent,
                      max(started_at) AS last_run,
                      count(*) FILTER (WHERE started_at > now() - interval '24 hours') AS runs_24h,
                      count(*) FILTER (WHERE status='error'
                                       AND started_at > now() - interval '24 hours') AS errors_24h,
                      (array_agg(status ORDER BY started_at DESC))[1] AS last_status
               FROM agent_runs GROUP BY agent ORDER BY max(started_at) DESC"""
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


# ─────────────────────────── live chart + graph APIs ────────────────────────
@app.get("/api/chart/price")
async def api_price_chart(symbol: str, days: str = "1y", ccy: str = "USD",
                          compare: str = ""):
    """A single instrument, window and display currency, computed on request.

    The Charts tab is a live surface — unlike a brief's stored pack, these
    figures answer to the reader's controls, so they come fresh from Postgres
    every call. `compare` overlays up to three further symbols, rebased.
    """
    n = chartdata.PERIODS.get(days)
    if not n:
        return JSONResponse({"error": f"period must be one of {list(chartdata.PERIODS)}"}, 400)
    others = [s.strip().upper() for s in compare.split(",") if s.strip()]
    spec = await asyncio.to_thread(
        chartdata.price_history, symbol.upper(), n, ccy.upper(), others)
    if not spec:
        return JSONResponse({"error": "no data for that combination"}, 404)
    return spec


@app.get("/api/chart/{key}")
async def api_chart(key: str, days: str = ""):
    """A standing figure, optionally over a chosen window."""
    n = chartdata.PERIODS.get(days)
    if n and key in chartdata._REBUILDABLE:
        spec = await asyncio.to_thread(chartdata.rebuild, key, n)
        if spec:
            return spec
    spec = chartdata.latest_pack().get(key)
    return spec or JSONResponse({"error": "unknown chart"}, 404)


@app.get("/api/graph")
async def api_graph(days: int = 7, limit: int = 220,
                    urgency: str = "Medium", q: str = "", concept: str = ""):
    """The knowledge graph: documents, the concepts they mention, and the links."""
    from memory import graph as kg

    return await asyncio.to_thread(
        kg.build, min(days, 120), min(limit, 600), urgency, q.strip(),
        concept.strip())


@app.get("/api/graph/document/{doc_id}")
async def api_graph_document(doc_id: int):
    """One document with its nearest neighbours, for the detail panel."""
    from memory import graph as kg

    doc = db.one(
        """SELECT id, title, url, source, source_tier, urgency, summary,
                  left(body, 1200) AS excerpt, entities, themes, fetched_at
           FROM documents WHERE id = %s""",
        (doc_id,),
    )
    if not doc:
        return JSONResponse({"error": "not found"}, 404)
    doc["fetched_at"] = doc["fetched_at"].isoformat()
    neighbours = await asyncio.to_thread(kg.neighbours, doc_id)
    for n in neighbours:
        n["fetched_at"] = n["fetched_at"].isoformat()
        n["similarity"] = round(float(n["similarity"]), 3)
    return {"document": doc, "neighbours": neighbours}


@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request, concept: str = "", q: str = ""):
    """The map. `concept` deep-links from a brief straight into the
    neighbourhood of one idea."""
    counts = db.one(
        """SELECT (SELECT count(*) FROM documents) AS documents,
                  (SELECT count(*) FROM doc_links) AS links,
                  (SELECT count(*) FROM entities) AS entities,
                  (SELECT count(*) FROM securities) AS securities"""
    )
    return page(request, "graph.html", {
        "counts": counts, "concept": concept[:120], "q": q[:120],
        "active": "graph"})


# ───────────────────────────── knowledge library ────────────────────────────
@app.get("/add", response_class=HTMLResponse)
async def add_form(request: Request):
    from web import library

    return page(request, "add.html", {
        "result": None, "recent_docs": library.recent(), "active": "add",
    })


@app.post("/add", response_class=HTMLResponse)
async def add_submit(
    request: Request,
    content: str = Form(...),
    title: str = Form(""),
    note: str = Form(""),
):
    from web import library

    result = await asyncio.to_thread(library.add, content, title.strip(), note.strip())
    return page(request, "add.html", {
        "result": result, "recent_docs": library.recent(), "active": "add",
    })


# ─────────────────────────────── share links ───────────────────────────────
@app.post("/digest/{digest_id}/share")
async def create_share(request: Request, digest_id: int):
    from web import sharing

    exists = db.one("SELECT id FROM analyses WHERE id=%s AND kind='digest'", (digest_id,))
    if not exists:
        return JSONResponse({"error": "no such brief"}, 404)
    user = auth.current_user(request) or {}
    token = await asyncio.to_thread(sharing.create, digest_id, user.get("username"))
    return {"token": token, "url": f"{request.base_url}s/{token}".replace("http://", "https://", 1)
            if request.url.scheme == "https" else f"{request.base_url}s/{token}"}


@app.post("/digest/{digest_id}/unshare")
async def revoke_share(request: Request, digest_id: int):
    from web import sharing

    link = await asyncio.to_thread(sharing.for_analysis, digest_id)
    if link:
        await asyncio.to_thread(sharing.revoke, link["token"])
    return {"revoked": bool(link)}


@app.get("/s/{token}", response_class=HTMLResponse)
async def shared_brief(request: Request, token: str):
    """A single brief, to someone with no account.

    Deliberately its own template with no navigation: the recipient sees this
    brief and nothing else — no archive, no other briefs, no Ask box, no route
    back into the dashboard.
    """
    from web import sharing

    row = await asyncio.to_thread(sharing.resolve, token)
    if not row:
        return templates.TemplateResponse(
            request, "share_missing.html", {"charts_json": "{}"}, status_code=404)

    pack = chartdata.load_pack(row["id"]) or chartdata.latest_pack()
    return templates.TemplateResponse(request, "share.html", {
        "digest": row,
        "body_html": render_markdown(row["body"], pack),
        "charts_json": json.dumps(pack, default=str),
    })


@app.get("/healthz")
async def healthz():
    try:
        db.one("SELECT 1 AS ok")
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, 503)
