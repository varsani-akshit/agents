"""The weekly review, emailed Sunday night (Monday morning in Melbourne).

One page: how the calls did, which strategies started or stopped passing,
which markets are sending tickets, how each market's conditions changed,
and what is scheduled in the week ahead for the names you hold. Built from
the database alone — no model writes any of it.
"""
from __future__ import annotations

from datetime import date, timedelta

import db
from advisor import REGIONS, bare, regime
from advisor.report import HEADER, REGION_NAME
from advisor.simulate import summarize


def _r(v) -> str:
    return "—" if v is None else f"{v:+.2f}R"


def scorecard(days: int = 7) -> list[str]:
    week = db.query("""SELECT passed_gate, story->>'direction' AS news, r FROM calls
                       WHERE status = 'closed' AND exit_date > current_date - %s""", (days,))
    total = db.query("SELECT passed_gate, story->>'direction' AS news, r FROM calls WHERE status = 'closed'")
    out = ["Every call, scored (taken or not)"]
    for label, rows in (("This week", week), ("Since the start", total)):
        a = summarize([float(r["r"]) for r in rows])
        if not a.get("trades"):
            out.append(f"  {label}: no call closed.")
            continue
        g = {k: summarize([float(r["r"]) for r in rows if (r["passed_gate"] if k == "passes" else not r["passed_gate"])])
             for k in ("passes", "fails")}
        n = {k: summarize([float(r["r"]) for r in rows if r["news"] == k])
             for k in ("supports", "neutral", "contradicts")}
        out.append(f"  {label}: {a['trades']} closed, {_r(a['expectancy_r'])} a call, "
                   f"{a['win_rate']:.0%} won. Gate: passes {_r(g['passes'].get('expectancy_r'))} "
                   f"({g['passes'].get('trades', 0)}), fails {_r(g['fails'].get('expectancy_r'))} "
                   f"({g['fails'].get('trades', 0)}). News: supports {_r(n['supports'].get('expectancy_r'))}, "
                   f"neutral {_r(n['neutral'].get('expectancy_r'))}, contradicts "
                   f"{_r(n['contradicts'].get('expectancy_r'))}.")
    return out


def tickets(days: int = 7) -> list[str]:
    sent = db.query("SELECT symbol, strategy, created_at FROM ideas WHERE created_at > now() - make_interval(days => %s)", (days,))
    closed = db.query("""SELECT symbol, strategy, exit_reason, r_multiple FROM ideas
                         WHERE status = 'closed' AND exit_date > current_date - %s""", (days,))
    live = db.query("SELECT symbol, status FROM ideas WHERE status IN ('open', 'filled')")
    out = [f"Tickets: {len(sent)} sent this week · {len(live)} live "
           f"({sum(r['status'] == 'filled' for r in live)} filled, "
           f"{sum(r['status'] == 'open' for r in live)} waiting for their limit)"]
    for r in closed:
        out.append(f"  closed {bare(r['symbol'])} ({r['strategy'].replace('_', ' ')}): "
                   f"{r['exit_reason']}, {_r(r['r_multiple'])}")
    return out


def gate_changes(days: int = 7) -> list[str]:
    last = db.one("""SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL
                     ORDER BY started_at DESC LIMIT 1""")
    first = db.one("""SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL
                      AND started_at < now() - make_interval(days => %s)
                      ORDER BY started_at DESC LIMIT 1""", (days,))
    if not last or not first:
        return ["Gate: not enough runs yet to compare a week."]
    was = {(r["strategy"], r["region"]): r["passed"] for r in db.query(
        "SELECT strategy, region, passed FROM sim_results WHERE run_id = %s", (first["id"],))}
    out = []
    for r in db.query("SELECT strategy, region, passed, stats->>'label' AS label FROM sim_results WHERE run_id = %s",
                      (last["id"],)):
        before = was.get((r["strategy"], r["region"]))
        if before is not None and before != r["passed"]:
            out.append(f"  {r['label'] or r['strategy']} on {REGION_NAME.get(r['region'], r['region'])}: "
                       f"{'now passes' if r['passed'] else 'stopped passing'}")
    return ["Gate changes this week:"] + (out or ["  none"])


def markets() -> list[str]:
    out = ["Markets"]
    for region in REGIONS:
        acct = db.one("SELECT stats FROM portfolio_backtests WHERE region = %s", (region,))
        st = (acct or {}).get("stats") or {}
        sending = "sends tickets" if st.get("passed") else "sends nothing"
        h = regime.history(region)
        changed = []
        if not h.empty and len(h) > 8:
            now, then = h.iloc[-1], h.iloc[-8]
            dims = regime.dimensions(region)
            for k in h.columns:
                if isinstance(now[k], str) and isinstance(then[k], str) and now[k] != then[k]:
                    changed.append(f"{dims[k].label.lower()} {then[k]} → {now[k]}")
        out.append(f"  {REGION_NAME.get(region, region)}: {sending}"
                   + (f"; conditions used" if st.get("conditions_used") else "")
                   + (f". This week: {'; '.join(changed)}." if changed else ". Conditions unchanged."))
    return out


def week_ahead(days: int = 7) -> list[str]:
    from advisor import bars

    names = {bare(s) for s in bars.holdings()} | {bare(r["symbol"]) for r in db.query(
        "SELECT symbol FROM ideas WHERE status IN ('open', 'filled')")}
    rows = db.query("""SELECT kind, key, event, at FROM calendar
                       WHERE at::date BETWEEN current_date AND current_date + %s
                         AND (importance = 'high' OR (kind = 'earnings' AND key = ANY(%s)))
                       ORDER BY at""", (days, sorted(names)))
    seen, out = set(), []
    for r in rows:
        what = f"{r['key']} earnings" if r["kind"] == "earnings" else f"{r['event']} ({r['key']})"
        if what in seen:
            continue
        seen.add(what)
        out.append(f"  {r['at']:%a %d %b}: {what}")
    return ["The week ahead:"] + (out or ["  nothing scheduled that touches your names"])


def compose() -> tuple[str, str]:
    today = date.today()
    parts = [HEADER, ""]
    for block in (scorecard(), tickets(), gate_changes(), markets(), week_ahead()):
        parts += block + [""]
    subject = f"Alfred · week to {today:%d %b}"
    return subject, "\n".join(parts)


def send() -> dict:
    from notify import email

    subject, text = compose()
    sent = email.send(subject, text, key=f"weekly-{date.today().isoformat()}")
    return {"emailed": bool(sent), "lines": text.count("\n")}
