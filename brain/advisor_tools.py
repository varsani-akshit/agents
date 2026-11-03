"""The advisor's own knowledge, exposed to the chat agent.

Ask could reach the market — prices, statistics, news, the graph, the web —
but not the thing it is advising on. It could tell you what gold did and
nothing about your gold. "Should I add to BHP?" was unanswerable because the
agent could not see that you hold it, at what price, under which strategy, or
what the last run said about it.

These five close that gap. Four read what the advisor already decided, so an
answer agrees with the dashboard instead of re-deriving a second opinion from
scratch. The fifth lets the agent draw, because "silver is stretched against
gold" is a sentence, and the ratio over ten years is an argument.

Nothing here can place an order or change a position. They are reads.
"""
from __future__ import annotations

import contextvars

import db

# Charts the agent built during this answer. A contextvar rather than a
# return value because the chart has to reach the renderer, which is three
# layers above the tool call, and threading it through every frame would mean
# changing the signature of everything in between.
_TURN_CHARTS: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "alfred_turn_charts", default=None)


def collecting() -> dict:
    """Start collecting this turn's charts; returns the dict they land in."""
    pack: dict = {}
    _TURN_CHARTS.set(pack)
    return pack


DEFINITIONS = [
    {
        "name": "my_portfolio",
        "description": (
            "What the reader actually holds, right now, across Zerodha Kite "
            "(India), Webull (US and Australia) and Coinbase (crypto): each "
            "position's quantity, average cost, last price, value and profit, "
            "plus cash and total value per broker. Use this for any question "
            "about their own money — what they own, how much, how it is doing, "
            "whether they already hold something, how concentrated they are."),
        "input_schema": {
            "type": "object",
            "properties": {
                "broker": {"type": "string",
                           "description": "kite | webull | coinbase. Omit for all."},
            },
        },
    },
    {
        "name": "my_tickets",
        "description": (
            "Trade ideas Alfred has issued: the open ones the reader could act "
            "on now (with buy limit, stop, size and the written case for each), "
            "and the closed ones with what they actually returned. Use for "
            "'what should I buy', 'what is open', 'why did Alfred suggest X', "
            "'how have its calls done'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string",
                           "description": "open | filled | closed | withdrawn | all. "
                                          "Default open."},
                "symbol": {"type": "string", "description": "Limit to one name."},
                "limit": {"type": "integer", "description": "Max rows, default 20."},
            },
        },
    },
    {
        "name": "strategy_records",
        "description": (
            "Every strategy Alfred tests, its record, and whether it currently "
            "passes the gate in each market — expectancy in R, win rate, trade "
            "count, and the specific reasons any strategy fails. Use for 'what "
            "is working', 'why is nothing firing in India', 'which strategy is "
            "best', 'what does Alfred actually trade'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "IN | ASX | US | CRYPTO. "
                                                            "Omit for all."},
                "passing_only": {"type": "boolean",
                                 "description": "Only strategies that currently pass."},
            },
        },
    },
    {
        "name": "latest_simulation",
        "description": (
            "The most recent full simulation run: when it ran, how many "
            "strategy-market pairs passed, what changed since the run before, "
            "and the setups that cleared the gate. Use for 'what did the last "
            "run say', 'what changed', 'what is close to firing'."),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "IN | ASX | US | CRYPTO."},
            },
        },
    },
    {
        "name": "make_chart",
        "description": (
            "Draw a chart in your answer. Returns a key; embed it in your reply "
            "as ![title](charts/KEY.png) on its own line and it renders as an "
            "interactive figure. Use a chart whenever a claim is about a shape "
            "over time — a trend, a level being approached, two things diverging "
            "— rather than describing the shape in words. "
            "kind 'candle' draws one name's daily bars with its 50- and 200-day "
            "averages; 'line' draws one or more names' closes together, rebased "
            "to 100 so different price levels can be compared; 'ratio' draws the "
            "first symbol divided by the second."),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "candle | line | ratio"},
                "symbols": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Tickers as stored: RELIANCE.NS, BHP.AX, AAPL, "
                                   "BTC-USD. One for candle, one or more for line, "
                                   "exactly two for ratio."},
                "days": {"type": "integer", "description": "Sessions to show. Default 250."},
                "title": {"type": "string", "description": "What the chart shows."},
            },
            "required": ["kind", "symbols"],
        },
    },
]


# ──────────────────────────────── handlers ──────────────────────────────────
def my_portfolio(broker: str = "") -> dict:
    from advisor import scan
    from brokers import portfolio as pf

    want = (broker or "").strip().lower()
    # Account value and free cash come from scan.accounts(), which is where
    # that arithmetic already lives — the raw snapshot carries a `funds` blob
    # whose shape differs per broker, and a second reading of it here would
    # drift from the one the tickets are sized against.
    accts = scan.accounts()
    out, books = {}, ("kite", "webull", "coinbase")
    for name in books:
        if want and name != want:
            continue
        data = pf.latest(name)
        if not data or not data["lines"]:
            out[name] = {"positions": [], "note": "no snapshot stored"}
            continue
        snap = data["snapshot"] or {}
        acct = accts.get(name) or {}
        lines = sorted(data["lines"], key=lambda ln: -(ln["value"] or 0))
        out[name] = {
            "as_of": snap.get("taken_at"),
            "total_value": acct.get("value"),
            "cash": acct.get("cash"),
            "currency": acct.get("currency"),
            "positions": [{
                "symbol": ln["symbol"], "exchange": ln.get("exchange"),
                "quantity": _f(ln.get("quantity")), "avg_price": _f(ln.get("avg_price")),
                "last_price": _f(ln.get("last_price")), "value": _f(ln.get("value")),
                "profit": _f(ln.get("pnl")), "currency": ln.get("currency"),
            } for ln in lines],
        }
    if not out:
        return {"error": f"unknown broker '{broker}'"}
    return {"brokers": out,
            "note": "Values are from the last snapshot, not live quotes."}


def my_tickets(status: str = "open", symbol: str = "", limit: int = 20) -> dict:
    want = (status or "open").strip().lower()
    where, params = [], []
    if want != "all":
        where.append("status = %s")
        params.append(want)
    if symbol:
        where.append("(symbol = %s OR symbol LIKE %s)")
        params += [symbol.upper(), symbol.upper() + ".%"]
    sql = "SELECT * FROM ideas"
    if where:
        sql += " WHERE " + " AND ".join(where)
    rows = db.query(sql + " ORDER BY created_at DESC LIMIT %s",
                    params + [max(1, min(int(limit or 20), 60))])
    if not rows:
        return {"tickets": [], "note": f"no {want} tickets"}

    # The written case lives on `calls`, so an answer quotes the same
    # reasoning the dashboard shows. Keyed by name and strategy rather than
    # by signal date, because rebasing moves a ticket's signal_date onto the
    # latest close and a date-keyed join loses the case for every ticket that
    # has been brought forward.
    cases = {(r["symbol"], r["strategy"]): r["reasoning"] for r in db.query(
        """SELECT DISTINCT ON (symbol, strategy) symbol, strategy, reasoning
           FROM calls WHERE reasoning IS NOT NULL
           ORDER BY symbol, strategy, created_at DESC""")}
    out = []
    for r in rows:
        case = cases.get((r["symbol"], r["strategy"])) or {}
        out.append({
            "symbol": r["symbol"], "strategy": r["strategy"], "status": r["status"],
            "signal_date": r["signal_date"], "valid_until": r["valid_until"],
            "buy_limit": _f(r["limit_price"]), "stop": _f(r["stop_price"]),
            "quantity": _f(r["quantity"]), "currency": r["currency"],
            "where_to_place_it": r["broker"],
            "moved_onto_latest_close": r.get("rebase_note"),
            "withdrawn_because": r.get("withdrawn_reason"),
            "result_r": _f(r.get("r_multiple")), "exit_reason": r.get("exit_reason"),
            "why": (r["reasons"] or [])[:3],
            "case": {k: case.get(k) for k in
                     ("verdict", "headline", "why_now", "why_this_price",
                      "what_would_break_it", "against", "confidence")} if case else None,
        })
    return {"tickets": out,
            "note": "Advice only — Alfred never places orders. R is profit "
                    "divided by the risk taken."}


def strategy_records(region: str = "", passing_only: bool = False) -> dict:
    from advisor import REGIONS
    from advisor.strategies import STRATEGIES

    regs = [region.upper()] if region else list(REGIONS)
    rows = db.query(
        """SELECT strategy, market, passed, stats FROM strategy_evidence
           WHERE symbol = '' AND market = ANY(%s) ORDER BY market, strategy""", (regs,))
    out = []
    for r in rows:
        if passing_only and not r["passed"]:
            continue
        st = r["stats"] or {}
        recent = st.get("recent") or {}
        strat = STRATEGIES.get(r["strategy"])
        out.append({
            "strategy": r["strategy"],
            "what_it_does": strat.summary if strat else st.get("label"),
            "market": r["market"], "passes_the_gate": bool(r["passed"]),
            "expectancy_r_recent": _r(recent.get("expectancy_r"), 3),
            "win_rate_recent": _r(recent.get("win_rate"), 3),
            "trades_recent": recent.get("trades"),
            "why_it_fails": (st.get("fails") or [])[:4],
        })
    acct = {r["region"]: r["stats"] for r in
            db.query("SELECT region, stats FROM portfolio_backtests")}
    return {
        "strategies": out,
        "market_level_gate": {k: {"sending_tickets": bool((v or {}).get("passed")),
                                  "why_not": (v or {}).get("fails")}
                              for k, v in acct.items() if not region or k == region.upper()},
        "note": "A market sends tickets only if its one-account replay beats "
                "the index ETF on both Sharpe and worst fall. Expectancy is in "
                "R: the average result per trade as a multiple of what was risked.",
    }


def latest_simulation(region: str = "") -> dict:
    run = db.one("""SELECT * FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL
                    ORDER BY started_at DESC LIMIT 1""")
    if not run:
        return {"error": "no completed simulation run stored"}
    where, params = ["run_id = %s", "kind = 'signal'", "passed_gate"], [run["id"]]
    if region:
        where.append("region = %s")
        params.append(region.upper())
    passed = db.query(
        f"""SELECT region, symbol, strategy, score, detail FROM sim_candidates
            WHERE {' AND '.join(where)} ORDER BY score DESC LIMIT 25""", params)
    near = db.one(
        f"""SELECT count(*) AS n FROM sim_candidates WHERE run_id = %s AND kind = 'near'
            AND passed_gate {'AND region = %s' if region else ''}""",
        [run["id"]] + ([region.upper()] if region else []))
    return {
        "run_id": run["id"], "ran_at": run["started_at"],
        "prices_through": run["bars_through"],
        "strategy_market_pairs_passing": run["passing"], "of": run["tested"],
        "candidates_examined": run["candidates"],
        "cleared_the_gate": [{
            "symbol": c["symbol"], "market": c["region"], "strategy": c["strategy"],
            "buy_limit": (c["detail"] or {}).get("limit"),
            "stop": (c["detail"] or {}).get("stop"),
            "why": ((c["detail"] or {}).get("reasons") or [])[:2],
        } for c in passed],
        "close_to_firing": near["n"] if near else 0,
        "note": "Runs every two hours. A setup clearing the gate is a candidate, "
                "not an instruction: the news read can still hold it back.",
    }


def _closes(symbol: str, days: int):
    """Daily closes for a symbol from whichever table holds it.

    Alfred keeps two price universes: `bars` for tradeable listings under
    yfinance tickers (GC=F, BHP.AX), and `prices` for the macro series under
    plain names (GOLD, DXY, US10Y). The chat agent reaches for the macro
    names, because that is what query_prices and the statistics use — so a
    chart tool that only read `bars` failed silently on "GOLD" and the agent
    fell back to describing the shape in words.
    """
    import pandas as pd

    from advisor import bars

    df = bars.load(symbol, years=25)
    if not df.empty:
        return df["close"].tail(days)
    rows = db.query(
        """SELECT ts, price FROM prices WHERE symbol = %s AND grain = '1d'
           ORDER BY ts""", (symbol.upper(),))
    if not rows:
        return None
    ser = pd.Series([float(r["price"]) for r in rows],
                    index=pd.to_datetime([r["ts"] for r in rows]).tz_localize(None))
    return ser.tail(days)


def make_chart(kind: str, symbols: list[str], days: int = 250, title: str = "") -> dict:
    """Build a figure for this answer and hand back the key to embed."""
    from advisor import bare, bars
    from advisor.charts import setup as candle_setup

    pack = _TURN_CHARTS.get()
    if pack is None:
        return {"error": "charts are not available in this context"}
    kind = (kind or "line").strip().lower()
    syms = [s.strip() for s in (symbols or []) if s and s.strip()]
    if not syms:
        return {"error": "no symbols given"}
    days = max(20, min(int(days or 250), 2500))

    if kind == "candle":
        # Candles need open/high/low, which only `bars` carries; a macro
        # series has closes alone, so say so rather than drawing a flat box.
        spec = candle_setup(syms[0], "", levels={}, days=days)
        if not spec:
            return {"error": f"no daily bars for {syms[0]} — candles need "
                             f"open/high/low. Try kind 'line'."}
        spec["title"] = title or f"{bare(syms[0])} — daily bars"
    elif kind == "ratio":
        if len(syms) != 2:
            return {"error": "a ratio needs exactly two symbols"}
        a, b = (_closes(s, 10_000) for s in syms)
        if a is None or b is None:
            return {"error": f"no stored prices for {syms[0] if a is None else syms[1]}"}
        joined = (a / b.reindex(a.index)).dropna().tail(days)
        if joined.empty:
            return {"error": "those two have no overlapping history"}
        spec = {
            "type": "line", "key": "", "unit": "",
            "title": title or f"{bare(syms[0])} ÷ {bare(syms[1])}",
            "subtitle": "Ratio of daily closes. Rising means the first is "
                        "gaining on the second.",
            "x": [d.strftime("%Y-%m-%d") for d in joined.index],
            "series": [{"name": f"{bare(syms[0])}/{bare(syms[1])}",
                        "data": [round(float(v), 4) for v in joined]}],
        }
    else:
        series, index = [], None
        for s in syms[:6]:
            tail = _closes(s, days)
            if tail is None or tail.empty:
                continue
            index = tail.index if index is None or len(tail) > len(index) else index
            series.append((s, tail))
        if not series:
            return {"error": "no stored prices for any of those symbols"}
        # Rebased to 100 so names at 60 and 6,000 can share an axis; a chart
        # where one line is a flat worm against the bottom compares nothing.
        rebase = len(series) > 1
        spec = {
            "type": "line", "key": "", "unit": "",
            "title": title or (", ".join(bare(s) for s, _ in series)
                               + (" — rebased to 100" if rebase else "")),
            "subtitle": ("Daily closes, each starting at 100 so the shapes can be "
                         "compared." if rebase else "Daily closes."),
            "x": [d.strftime("%Y-%m-%d") for d in index],
            "series": [{
                "name": bare(s),
                "data": [None if v != v else round(float(v) / float(t.iloc[0]) * 100, 3)
                         if rebase else round(float(v), 4)
                         for v in t.reindex(index)],
            } for s, t in series],
        }

    key = f"ask{len(pack) + 1}_{kind}_{bare(syms[0]).lower()}"
    spec["key"] = key
    pack[key] = spec
    return {"key": key,
            "embed": f"![{spec['title']}](charts/{key}.png)",
            "note": "Put the embed line on its own line in your answer, where "
                    "the chart belongs in the argument."}


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _r(v, n):
    f = _f(v)
    return None if f is None else round(f, n)


HANDLERS = {
    "my_portfolio": my_portfolio,
    "my_tickets": my_tickets,
    "strategy_records": strategy_records,
    "latest_simulation": latest_simulation,
    "make_chart": make_chart,
}

SPAN_KIND = {
    "my_portfolio": "retrieval",
    "my_tickets": "retrieval",
    "strategy_records": "retrieval",
    "latest_simulation": "retrieval",
    "make_chart": "tool",
}
