"""What the Today and Simulations pages show, gathered in one place.

Both pages read only what the scheduled jobs have already written — prices,
snapshots, ideas, evidence, replays — so they load in milliseconds and are
exactly as fresh as the last run, which each section states.
"""
from __future__ import annotations

import math

from markupsafe import Markup

import db
from advisor import REGIONS, settings

# Markets are named by where you would place the order, never by country.
REGION_NAME = {"IN": "NSE/BSE", "ASX": "ASX", "US": "NYSE/Nasdaq"}


# ─────────────────────────────── line charts ────────────────────────────────
def line_svg(dates: list, series: list[tuple], *, log: bool = False, zero: bool = False,
             mark: str | None = None, w: int = 640, h: int = 170) -> Markup:
    """A small multi-line chart, drawn server-side.

    series: (values, css colour, stroke width, dashed). `log` scales growth so
    a doubling reads the same at any height; `zero` draws the zero line;
    `mark` draws a vertical rule at a date (the start of the recent period).
    """
    vals = [v for vs, *_ in series for v in vs if v is not None and (not log or v > 0)]
    if not vals or len(dates) < 2:
        return Markup("")
    f = (lambda v: math.log(v)) if log else (lambda v: v)
    lo, hi = min(map(f, vals)), max(map(f, vals))
    if zero:
        lo, hi = min(lo, 0), max(hi, 0)
    span = (hi - lo) or 1
    pad = 6
    y = lambda v: pad + (h - 2 * pad) * (1 - (f(v) - lo) / span)  # noqa: E731
    x = lambda i: i / (len(dates) - 1) * w  # noqa: E731
    parts = []
    if zero:
        parts.append(f'<line x1="0" x2="{w}" y1="{y(0):.1f}" y2="{y(0):.1f}" '
                     f'stroke="var(--border)" stroke-width="1"/>')
    if mark and mark in dates:
        mx = x(dates.index(mark))
        parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="0" y2="{h}" stroke="var(--border)" '
                     f'stroke-width="1" stroke-dasharray="2 3"/>')
    for vs, color, width, dashed in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vs)
                       if v is not None and (not log or v > 0))
        if pts:
            dash = ' stroke-dasharray="3 3"' if dashed else ""
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}"'
                         f' stroke-linejoin="round"{dash} vector-effect="non-scaling-stroke" points="{pts}"/>')
    return Markup(f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
                  f'style="height:{h}px" role="img" aria-hidden="true">{"".join(parts)}</svg>')


def account_svg(curve: dict) -> Markup:
    return line_svg(curve.get("dates") or [], [
        (curve.get("survivors") or [], "var(--t3)", 1.2, True),
        (curve.get("random") or [], "var(--t3)", 1.2, False),
        (curve.get("index") or [], "var(--t1)", 1.4, False),
        (curve.get("strategy") or [], "var(--accent)", 2.2, False),
    ], log=True)


# ─────────────────────────────────── Today ──────────────────────────────────
def today() -> dict:
    from advisor import bars, scan
    from advisor.strategies import indicators
    from brokers import kite, portfolio as pf, webull

    accts = scan.accounts()
    books = []
    for name, mod in (("kite", kite), ("webull", webull)):
        data = pf.latest(name)
        tot = pf.totals(data["lines"]) if data else {}
        a = accts.get(name) or {}
        ccy = a.get("currency")
        books.append({
            "broker": name, "status": mod.status(), "value": a.get("value"),
            "cash": a.get("cash"), "currency": ccy,
            "pnl": sum(t["pnl"] for c, t in tot.items() if c == ccy) if tot else None,
            "as_of": data and data["snapshot"] and data["snapshot"]["taken_at"],
            "lines": len(data["lines"]) if data else 0,
        })

    # Holdings that have broken their trailing stop, read fresh from the bars.
    attention = []
    for sym, h in bars.holdings().items():
        df = bars.load(sym, years=2)
        if len(df) < 260:
            continue
        r = scan.holding_risk(sym, indicators(df), h)
        if r["below_stop"]:
            attention.append(r)
    attention.sort(key=lambda r: (not r["newly_below"], r["stop_gap_pct"]))

    ideas = db.query("""SELECT * FROM ideas WHERE status IN ('open', 'filled')
                        ORDER BY created_at DESC LIMIT 12""")
    closed = db.query("""SELECT symbol, strategy, r_multiple, exit_date FROM ideas
                         WHERE status = 'closed' ORDER BY exit_date DESC LIMIT 5""")
    regions = []
    for region in REGIONS:
        acct = db.one("SELECT stats, computed_at FROM portfolio_backtests WHERE region=%s", (region,))
        n = db.one("""SELECT count(*) FILTER (WHERE passed) AS p, count(*) AS n
                      FROM strategy_evidence WHERE market=%s AND symbol=''""", (region,))
        last = db.one("""SELECT started_at, ok, detail FROM job_runs WHERE job=%s
                         ORDER BY started_at DESC LIMIT 1""", (f"advise_{region.lower()}",))
        regions.append({"region": region, "name": REGION_NAME[region],
                        "sending": bool(acct and acct["stats"].get("passed")),
                        "strategies": n and n["p"], "of": n and n["n"], "last_scan": last})
    brief = db.one("""SELECT id, title, created_at, meta FROM analyses WHERE kind='digest'
                      ORDER BY created_at DESC LIMIT 1""")

    # Act now: exits due on what you hold, then tickets, then how much to fund.
    cands = db.query("""SELECT region, kind, symbol, strategy, as_of, detail, created_at
                        FROM scan_candidates ORDER BY region, kind""")
    exits = [c for c in cands if c["kind"] == "exit"]
    near = [c for c in cands if c["kind"] in ("near", "also")]
    near.sort(key=lambda c: (c["kind"] != "also", c["detail"].get("level") is None,
                             c["detail"].get("distance_pct") or 0))
    cfg = settings()
    funding = []
    for b in books:
        reserve = float((cfg.get("new_funds") or {}).get(b["broker"]) or 0)
        need = 0.0
        for i in ideas:
            if i["broker"] != b["broker"] or i["status"] != "open" or not i["quantity"]:
                continue
            cost = i["quantity"] * i["limit_price"]
            if i["currency"] != b["currency"]:
                rate = scan._in_currency(1.0, b["currency"], i["currency"]) or 1
                cost /= rate
            need += cost
        add = max(0.0, need - (b["cash"] or 0))
        funding.append({"broker": b["broker"], "currency": b["currency"], "reserve": reserve,
                        "tickets_cost": need, "add": add})

    # Portfolio at a glance: the largest holdings per broker.
    top = {}
    for name in ("kite", "webull"):
        data = pf.latest(name)
        if data and data["lines"]:
            top[name] = sorted((ln for ln in data["lines"] if ln["value"]),
                               key=lambda ln: -ln["value"])[:5]

    # Evidence at a glance: per market, what passes and how the account did.
    from advisor import ledger
    from advisor.strategies import STRATEGIES

    evidence = []
    for r in regions:
        acct = db.one("SELECT stats FROM portfolio_backtests WHERE region=%s", (r["region"],))
        st = acct and acct["stats"] or {}
        evidence.append({**r, "labels": [STRATEGIES[n].label for n in st.get("strategies", [])
                                         if n in STRATEGIES],
                         "me": st.get("strategy") or {}, "index": st.get("index") or {},
                         "index_symbol": st.get("index_symbol")})

    try:
        from brain import news_match

        news = news_match.recent(hours=36, per_name=3)
    except Exception:  # noqa: BLE001 - the dashboard must render without it
        news = []

    # ── One list of things to do, one row per instrument, most urgent first ──
    # Every row names the instrument, the verb, where to place it and the
    # numbers. Nothing is grouped by country: a country is not something you
    # can buy.
    def name_of(sym):
        return sym.split(".")[0]

    actions = []
    for e in exits:
        d = e["detail"]
        actions.append({"verb": "Sell", "tone": "bad", "symbol": name_of(e["symbol"]),
                        "where": d.get("broker"), "what": "at the next open",
                        "why": f"{d.get('reason')} ({e['strategy'].replace('_', ' ')})"})
    for i in ideas:
        if i["status"] != "open":
            continue
        actions.append({"verb": "Buy", "tone": "ok", "symbol": name_of(i["symbol"]), "where": i["broker"],
                        "what": f"limit {i['limit_price']:,.2f} · stop {i['stop_price']:,.2f}"
                                + (f" · qty {i['quantity']:g}" if i["quantity"] else ""),
                        "why": (i["reasons"] or [""])[0],
                        "by": i["valid_until"], "currency": i["currency"]})
    for f in funding:
        if f["add"] > 0:
            actions.append({"verb": "Fund", "tone": "warn", "symbol": "Kite" if f["broker"] == "kite" else "Webull",
                            "where": f["broker"], "currency": f["currency"], "amount": f["add"],
                            "what": "before placing the tickets above",
                            "why": f"open tickets need {f['tickets_cost']:,.0f} {f['currency']}; "
                                   f"you set aside {f['reserve']:,.0f}"})
    for h in attention:
        actions.append({"verb": "Review", "tone": "bad" if h["newly_below"] else "muted",
                        "symbol": h["display"], "where": h["broker"],
                        "what": f"closed {h['close']:,.2f}, under its stop {h['stop']:,.2f}",
                        "why": ("broke its trailing stop today" if h["newly_below"]
                                else "below its trailing stop") +
                               ("; under its 200-day average" if h["below_200d"] else ""),
                        "urgent": h["newly_below"]})

    # Watch: instruments a passing strategy is close to buying, from the
    # latest simulation run — the next tickets, if they fire.
    run = db.one("""SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL
                    ORDER BY started_at DESC LIMIT 1""")
    watch = []
    if run:
        seen = set()
        for c in db.query("""SELECT symbol, strategy, kind, score, detail FROM sim_candidates
                             WHERE run_id = %s AND passed_gate
                             ORDER BY (kind = 'signal') DESC, (detail->>'level') IS NULL,
                                      (detail->>'distance_pct')::float NULLS LAST, score DESC""",
                          (run["id"],)):
            d = c["detail"]
            key = d.get("display") or name_of(c["symbol"])
            if key in seen or key in {a["symbol"] for a in actions}:
                continue
            seen.add(key)
            watch.append({"symbol": key, "where": d.get("broker"), "strategy": d.get("label"),
                          "close": d.get("close"), "currency": d.get("currency"),
                          "waiting": ("fired at the last close — the ticket goes out with the next scan"
                                      if c["kind"] == "signal" else d.get("waiting")),
                          "held": d.get("held")})
            if len(watch) >= 10:
                break

    return {"news": news[:8], "actions": actions, "watch": watch,
            "books": books, "attention": attention, "ideas": ideas, "closed": closed,
            "regions": regions, "brief": brief, "exits": exits, "near": near[:14],
            "funding": funding, "top": top, "evidence": evidence,
            "board": ledger.scoreboard(),
            "scanned_at": max((c["created_at"] for c in cands), default=None)}


# ──────────────────────────────── Simulations ───────────────────────────────
def sim_index() -> dict:
    """Every stored run, newest first, and the latest one at a glance."""
    runs = db.query("""SELECT id, started_at, finished_at, trigger, bars_through, passing, tested,
                              candidates, seconds, error FROM sim_runs
                       ORDER BY started_at DESC LIMIT 84""")
    latest = next((r for r in runs if r["finished_at"] and not r["error"]), None)
    out = {"runs": runs, "latest": latest, "running": any(not r["finished_at"] for r in runs)}
    if latest:
        out.update(sim_run(latest["id"], top_only=True))
    return out


def sim_run(run_id: int, *, region: str = "", show: str = "", top_only: bool = False) -> dict:
    from advisor.strategies import STRATEGIES

    run = db.one("SELECT * FROM sim_runs WHERE id=%s", (run_id,))
    if not run:
        return {"run": None}
    prev = db.one("""SELECT id FROM sim_runs WHERE started_at < %s AND finished_at IS NOT NULL
                     AND error IS NULL ORDER BY started_at DESC LIMIT 1""", (run["started_at"],))
    results = db.query("SELECT strategy, region, passed, stats FROM sim_results WHERE run_id=%s",
                       (run_id,))
    before = {(r["strategy"], r["region"]): r["passed"] for r in db.query(
        "SELECT strategy, region, passed FROM sim_results WHERE run_id=%s", (prev["id"],))} if prev else {}

    # The gate as a matrix: strategies down, markets across.
    matrix: dict[str, dict] = {}
    changes = []
    from advisor import intraday

    for r in results:
        st = STRATEGIES.get(r["strategy"]) or intraday.STRATEGIES.get(r["strategy"])
        row = matrix.setdefault(r["strategy"], {
            "label": (st.label if st else r["stats"].get("label") or r["strategy"]),
            "horizon": getattr(st, "max_days", None),
            "intraday": r["strategy"] in intraday.STRATEGIES, "cells": {}})
        r["was"] = before.get((r["strategy"], r["region"]))
        row["cells"][r["region"]] = r
        if r["was"] is not None and r["was"] != r["passed"]:
            changes.append(r)
    order = sorted(matrix.items(), key=lambda kv: (kv[1]["intraday"],
                                                   -sum(c["passed"] for c in kv[1]["cells"].values()),
                                                   kv[1]["label"]))

    accounts = db.query("SELECT region, stats, curve FROM sim_accounts WHERE run_id=%s "
                        "ORDER BY array_position(ARRAY['IN','ASX','US'], region)", (run_id,))
    for a in accounts:
        a["svg"] = account_svg(a["curve"])
        base, _, kind = a["region"].partition(":")
        a["name"] = REGION_NAME.get(base, base) + (" · intraday" if kind else "")
        a["intraday"] = bool(kind)

    where, params = ["run_id = %s"], [run_id]
    if region in REGIONS:
        where.append("region = %s")
        params.append(region)
    groups = {
        "passed": ("Passed the gate — these become tickets", "kind='signal' AND passed_gate"),
        "missed": ("Fired, but did not pass the gate", "kind='signal' AND NOT passed_gate"),
        "near_passed": ("Close to firing — strategies that pass", "kind='near' AND passed_gate"),
        "near_other": ("Close to firing — strategies that do not pass", "kind='near' AND NOT passed_gate"),
    }
    limit = 8 if top_only else (500 if show else 25)
    cand = {}
    for key, (title, cond) in groups.items():
        if show and show != key:
            continue
        rows = db.query(f"""SELECT region, symbol, strategy, kind, passed_gate, score, detail
                            FROM sim_candidates WHERE {' AND '.join(where)} AND {cond}
                            ORDER BY score DESC LIMIT %s""", params + [limit])
        total = db.one(f"SELECT count(*) AS n FROM sim_candidates WHERE {' AND '.join(where)} AND {cond}",
                       params)["n"]
        cand[key] = {"title": title, "rows": rows, "total": total}
    return {"run": run, "prev": prev, "matrix": order, "changes": changes, "accounts": accounts,
            "candidates": cand, "region": region, "show": show,
            "regions": [(k, REGION_NAME[k]) for k in REGIONS]}


def method() -> dict:
    from advisor.strategies import STRATEGIES

    latest = db.one("SELECT id FROM sim_runs WHERE error IS NULL AND finished_at IS NOT NULL "
                    "ORDER BY started_at DESC LIMIT 1")
    passing = {}
    if latest:
        for r in db.query("SELECT strategy, region, passed FROM sim_results WHERE run_id=%s",
                          (latest["id"],)):
            if r["passed"]:
                passing.setdefault(r["strategy"], []).append(REGION_NAME.get(r["region"], r["region"]))
    data = db.one("SELECT count(DISTINCT symbol) AS names, count(*) AS bars, min(d) AS first FROM bars")
    from advisor import intraday

    rows = [{"name": k, "label": v.label, "summary": v.summary, "max_days": v.max_days,
             "passes": passing.get(k, [])} for k, v in STRATEGIES.items()]
    rows += [{"name": k, "label": v.label, "summary": v.summary, "max_days": "same day",
              "passes": passing.get(k, [])} for k, v in intraday.STRATEGIES.items()]
    hourly_rows = db.one("SELECT count(*) AS n, min(ts) AS first FROM bars_1h")
    return {"cfg": settings(), "data": data, "hourly": hourly_rows, "strategies": rows}


def _month_end(dates: list[str], start: str | None) -> str | None:
    """The first plotted month at or after the recent period's start."""
    if not start:
        return None
    return next((d for d in dates if d >= start), None)
