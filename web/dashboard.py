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

REGION_NAME = {"IN": "India", "ASX": "Australia", "US": "United States"}


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

    return {"books": books, "attention": attention, "ideas": ideas, "closed": closed,
            "regions": regions, "brief": brief, "exits": exits, "near": near[:14],
            "funding": funding, "top": top, "evidence": evidence,
            "board": ledger.scoreboard(),
            "scanned_at": max((c["created_at"] for c in cands), default=None)}


# ──────────────────────────────── Simulations ───────────────────────────────
def simulations() -> dict:
    from advisor.strategies import STRATEGIES

    cfg = settings()
    rows = db.query("""SELECT strategy, market, passed, stats, computed_at FROM strategy_evidence
                       WHERE symbol = '' ORDER BY market, strategy""")
    by_strategy: dict[str, dict] = {}
    for r in rows:
        s = r["stats"]
        curve = s.get("curve") or {}
        r["svg"] = line_svg(curve.get("dates") or [], [
            (curve.get("random") or [], "var(--t3)", 1.2, True),
            (curve.get("strategy") or [], "var(--accent)" if r["passed"] else "var(--t2)", 1.8, False),
        ], zero=True, mark=_month_end(curve.get("dates") or [], s.get("recent_from")), h=110)
        r["best"] = db.query(
            """SELECT symbol, stats FROM strategy_evidence
               WHERE strategy=%s AND market=%s AND symbol<>'' AND (stats->>'trades')::int >= %s
               ORDER BY (stats->>'expectancy_r')::float DESC LIMIT 5""",
            (r["strategy"], r["market"], cfg["gate"]["min_symbol_trades"]))
        r["region_name"] = REGION_NAME.get(r["market"], r["market"])
        by_strategy.setdefault(r["strategy"], {"rows": []})["rows"].append(r)
    for name, entry in by_strategy.items():
        st = STRATEGIES.get(name)
        entry.update(name=name, label=st.label if st else name, summary=st.summary if st else "",
                     max_days=st.max_days if st else None,
                     passes=sum(r["passed"] for r in entry["rows"]))
    order = sorted(by_strategy.values(), key=lambda e: (-e["passes"], e["label"]))

    accounts = db.query("""SELECT region, stats, curve, computed_at FROM portfolio_backtests
                           ORDER BY array_position(ARRAY['IN','ASX','US'], region)""")
    for a in accounts:
        a["svg"] = account_svg(a["curve"])
        a["name"] = REGION_NAME.get(a["region"], a["region"])

    data = db.one("SELECT count(DISTINCT symbol) AS names, count(*) AS bars, min(d) AS first FROM bars")
    when = rows[0]["computed_at"] if rows else None
    return {"strategies": order, "accounts": accounts, "cfg": cfg, "data": data,
            "computed_at": when, "passing": sum(r["passed"] for r in rows), "tested": len(rows)}


def _month_end(dates: list[str], start: str | None) -> str | None:
    """The first plotted month at or after the recent period's start."""
    if not start:
        return None
    return next((d for d in dates if d >= start), None)
