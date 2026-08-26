"""MIA command line. `./mia <command>`"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import config
import db


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────── commands ───────────────────────────────
def cmd_init(args) -> int:
    from ingest import prices
    from memory import world_model

    db.apply_schema()
    n = prices.sync_instruments()
    world_model.ensure_seeded()
    print(f"schema applied; {n} instruments registered; world model seeded")
    return 0


def cmd_backfill(args) -> int:
    from ingest import fred, prices
    from notify import out

    with db.job("backfill") as detail:
        out.info("backfilling prices…")
        detail["prices"] = prices.backfill(args.period)
        out.info(f"  {detail['prices']}")
        out.info("backfilling FRED…")
        f = fred.sync(args.fred_days)
        detail["fred"] = {k: v for k, v in f.items() if k != "per_series"}
        out.info(f"  {detail['fred']}")
    return 0


def cmd_tick(args) -> int:
    """15-minute loop: ingest, compute, trigger, alert."""
    from brain import alert as alert_mod
    from brain import classify
    from ingest import feeds, prices
    from memory import store
    from notify import out
    from signals import stats, triggers

    with db.job("tick") as detail:
        detail["prices"] = prices.tick()
        harvest = feeds.harvest()
        detail["feeds"] = {k: v for k, v in harvest.items() if k != "new_ids"}
        detail["embedded"] = store.embed_documents(limit=120)
        detail["classified"] = classify.run(batch=20, max_batches=3, workers=3)

        pack = stats.build(persist=True)
        fired = triggers.evaluate(pack, doc_ids=harvest.get("new_ids"))
        detail["triggers_fired"] = len(fired)

        triggers.suppress_stale(hours=6)
        sent = []
        for ev in triggers.pending_critical():
            payload = alert_mod.write(ev)
            out.alert(payload)
            sent.append(ev["id"])
        triggers.mark_notified(sent)
        detail["alerts_sent"] = len(sent)

    out.info(
        f"tick: {detail['feeds'].get('inserted', 0)} new docs, "
        f"{detail['triggers_fired']} triggers, {detail['alerts_sent']} alerts"
    )
    return 0


def cmd_digest(args) -> int:
    from brain import client, digest
    from notify import out

    client.AUTONOMOUS = False   # manually invoked, not a scheduled cycle

    with db.job("digest") as detail:
        result = digest.run(hours=args.hours)
        detail.update({k: v for k, v in result.items() if k not in ("body",)})
        if not result.get("ok"):
            out.error(f"digest failed: {result.get('error')}")
            return 1
        out.digest(result)
    return 0


def cmd_ask(args) -> int:
    from brain import ask as ask_mod
    from brain import client
    from notify import out

    # A question typed by the user is interactive: allow it to draw on the
    # reserve that scheduled jobs cannot touch.
    client.AUTONOMOUS = False

    question = " ".join(args.question).strip()
    if not question:
        out.error("no question given")
        return 1
    result = ask_mod.ask(question, use_web_search=not args.no_web)
    if not result["answer"]:
        out.error(f"no answer produced ({result.get('stopped')})")
        return 1
    out.answer(result)
    return 0


def cmd_status(args) -> int:
    from brain import client, extract
    from ingest import feeds
    from rich.console import Console
    from rich.table import Table

    c = Console()

    counts = db.one(
        """SELECT (SELECT count(*) FROM documents) AS docs,
                  (SELECT count(*) FROM documents WHERE classified_at IS NOT NULL) AS classified,
                  (SELECT count(*) FROM documents WHERE embedding IS NOT NULL) AS embedded,
                  (SELECT count(*) FROM prices) AS prices,
                  (SELECT count(*) FROM analyses) AS analyses,
                  (SELECT count(*) FROM trigger_events) AS triggers,
                  (SELECT count(*) FROM entities) AS entities,
                  (SELECT count(*) FROM edges) AS edges"""
    )
    t = Table(title="MIA status", show_header=False, box=None)
    for k, v in counts.items():
        t.add_row(f"[cyan]{k}[/cyan]", str(v))
    c.print(t)

    b = client.budget_status()
    c.print(
        f"\n[bold]spend[/bold]  today ${b['spent_today_usd']:.4f}/"
        f"${b['daily_cap_usd']:.2f}   total ${b['spent_total_usd']:.4f}/"
        f"${b['total_cap_usd']:.2f}"
        f"\n[dim]scheduled jobs stop at ${b['autonomous_cap_usd']:.2f}; "
        f"${b['interactive_reserve_usd']:.4f} reserved for your own queries[/dim]"
    )

    jobs = db.query(
        """SELECT DISTINCT ON (job) job, started_at, finished_at, ok, error
           FROM job_runs ORDER BY job, started_at DESC"""
    )
    jt = Table(title="\nlast run per job")
    for col in ("job", "started", "ok", "error"):
        jt.add_column(col)
    for j in jobs:
        jt.add_row(
            j["job"],
            j["started_at"].strftime("%m-%d %H:%M"),
            "[green]yes[/green]" if j["ok"] else ("[red]no[/red]" if j["ok"] is False else "…"),
            (j["error"] or "")[:60],
        )
    c.print(jt)

    if args.sources:
        st = Table(title="\nsource health (7d)")
        for col in ("source", "tier", "docs", "last seen"):
            st.add_column(col)
        for s in feeds.source_health():
            st.add_row(s["source"][:40], str(s["source_tier"]), str(s["docs"]),
                       s["last_seen"].strftime("%m-%d %H:%M"))
        c.print(st)

    if args.graph:
        g = extract.graph_stats()
        c.print(f"\n[bold]graph[/bold] {g['entities']} entities / {g['edges']} edges")
        for e in g["strongest_edges"]:
            c.print(
                f"  {e['source_entity']} --{e['relation']}({e['direction']})--> "
                f"{e['target_entity']}  s={e['strength']} n={e['confirm_count']}"
            )
    return 0


def cmd_stats(args) -> int:
    from signals import stats

    pack = stats.build(persist=not args.no_save)
    if args.section:
        print(json.dumps(pack.get(args.section), indent=2, default=str))
    else:
        print(json.dumps(pack, indent=2, default=str))
    return 0


def cmd_backtest(args) -> int:
    from signals import backtest

    print(backtest.report())
    return 0


def cmd_worldmodel(args) -> int:
    from memory import world_model
    from rich.console import Console
    from rich.markdown import Markdown

    if args.history:
        for h in world_model.history():
            print(f"v{h['version']}  {h['created_at']}  {h['regime']}")
        return 0
    Console().print(Markdown(world_model.current_body()))
    return 0


def cmd_serve(args) -> int:
    import scheduler

    scheduler.main()
    return 0


# ─────────────────────────────────── parser ─────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mia", description="Macro Intelligence Agent")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="apply schema and register instruments").set_defaults(
        func=cmd_init
    )

    b = sub.add_parser("backfill", help="load price and macro history")
    b.add_argument("--period", default="2y")
    b.add_argument("--fred-days", type=int, default=900)
    b.set_defaults(func=cmd_backfill)

    sub.add_parser("tick", help="one 15-minute ingestion+trigger cycle").set_defaults(
        func=cmd_tick
    )

    d = sub.add_parser("digest", help="run a deep analysis cycle")
    d.add_argument("--hours", type=int, default=6)
    d.set_defaults(func=cmd_digest)

    a = sub.add_parser("ask", help="ask a question with full memory + web")
    a.add_argument("question", nargs="+")
    a.add_argument("--no-web", action="store_true", help="stored memory only")
    a.set_defaults(func=cmd_ask)

    s = sub.add_parser("status", help="system health, spend, job history")
    s.add_argument("--sources", action="store_true")
    s.add_argument("--graph", action="store_true")
    s.set_defaults(func=cmd_status)

    st = sub.add_parser("stats", help="print the computed stats pack")
    st.add_argument("--section")
    st.add_argument("--no-save", action="store_true")
    st.set_defaults(func=cmd_stats)

    sub.add_parser("backtest", help="calibrate alert thresholds on history").set_defaults(
        func=cmd_backtest
    )

    w = sub.add_parser("worldmodel", help="show the current world model")
    w.add_argument("--history", action="store_true")
    w.set_defaults(func=cmd_worldmodel)

    sub.add_parser("serve", help="run the scheduler in the foreground").set_defaults(
        func=cmd_serve
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
