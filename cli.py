"""Alfred command line. `./mia <command>`"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
        result = digest.run(hours=args.hours, model=args.model, effort=args.effort)
        detail.update({k: v for k, v in result.items() if k not in ("body",)})
        if not result.get("ok"):
            out.error(f"digest failed: {result.get('error')}")
            # Raise so job_runs records ok=false. Returning quietly logged a
            # successful run for a digest that produced nothing.
            raise RuntimeError(f"digest produced no output: {result.get('error')}")
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
    result = ask_mod.ask(question, use_web_search=not args.no_web,
                         model=args.model, max_turns=args.turns)
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
    t = Table(title="Alfred status", show_header=False, box=None)
    for k, v in counts.items():
        t.add_row(f"[cyan]{k}[/cyan]", str(v))
    c.print(t)

    b = client.budget_status()
    c.print(
        f"\n[bold]spend[/bold]  24h ${b['spent_today_usd']:.2f}/${b['daily_cap_usd']:.2f}   "
        f"30 days ${b['spent_month_usd']:.2f}/${b['monthly_cap_usd']:.2f}   "
        f"all time ${b['spent_total_usd']:.2f}"
        f"\n[dim]scheduled jobs stop at ${b['scheduled_daily_cap_usd']:.2f} a day; "
        f"${b['interactive_reserve_usd']:.2f} kept for your own questions[/dim]"
    )

    from brain import llm
    pt = Table(title="\nmodel routing", box=None)
    for col in ("task", "routed to", "spend"):
        pt.add_column(col)
    by_prov = {r["provider"]: r for r in client.spend_by_provider()}
    for r in llm.routing_table():
        prov = llm.parse_spec(r["spec"])[0]
        spent = by_prov.get(prov, {}).get("usd", 0)
        pt.add_row(r["task"], r["spec"], f"${float(spent):.4f} ({prov} total)")
    c.print(pt)

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


def cmd_user(args) -> int:
    """Create or reset a dashboard login, or list the existing ones."""
    import getpass

    from notify import out
    from web import auth

    if args.list:
        from brain import spend as spend_mod

        for u in auth.list_users():
            last = u["last_login"].strftime("%Y-%m-%d %H:%M") if u["last_login"] else "never"
            role = "admin " if u.get("is_admin") else "reader"
            if u.get("disabled_at"):
                role = "OFF   "
            name = u.get("display_name") or "—"
            st = spend_mod.status(u["username"])
            money = "unlimited" if st["unlimited"] else \
                f"${st['spent']:.2f} of ${st['limit']:.2f}"
            out.info(f"  {u['username']:<14} {name:<20} {role}  "
                     f"{money:<18} last login {last}")
        return 0

    if args.budget is not None:
        # Set on its own, like admin rights — never as a side effect of a
        # password reset.
        auth.set_budget(args.username, None if args.budget < 0 else args.budget)
        shown = "the default" if args.budget < 0 else f"${args.budget:.2f}"
        out.info(f"{args.username}'s monthly Ask budget is now {shown}")
        if not (args.password or args.create):
            return 0
    if not args.username:
        out.error("give a username, or --list")
        return 2
    if args.disable or args.enable:
        if args.enable and not args.disable:
            ok = auth.set_disabled(args.username, False)
        elif args.username.strip().lower() == config.OWNER:
            out.error("refusing to disable the owner account")
            return 2
        else:
            ok = auth.set_disabled(args.username, True)
        if not ok:
            out.error(f"no user '{args.username}'")
            return 2
        out.info(f"{args.username} is now {'enabled' if args.enable else 'disabled'}")
        return 0
    # Prompt rather than accept a password argument: anything on the command
    # line lands in shell history and in the process table.
    if args.admin or args.no_admin:
        # Rights are changed on their own, never as a side effect of a
        # password reset.
        auth.set_admin(args.username, bool(args.admin))
        out.info(f"{args.username} is now {'an admin' if args.admin else 'a reader'}")
        if not (args.password or args.create):
            return 0
    password = args.password or getpass.getpass("password: ")
    try:
        auth.create_user(args.username, password, is_admin=bool(args.admin),
                         display_name=args.name)
    except ValueError as exc:
        out.error(str(exc))
        return 2
    out.info(f"user '{args.username.lower()}' ready")
    return 0


def cmd_broker(args) -> int:
    """Broker connections: status, connect, pull, inspect. Read-only throughout."""
    from brokers import NotConnected, portfolio as pf
    from notify import out

    if args.action == "status":
        for name, mod in pf.ADAPTERS.items():
            st = mod.status()
            state = ("connected" if st["connected"] else
                     "not connected" if st["configured"] else "not configured")
            out.info(f"  {st['label']:<18} {state}"
                     + (f" · until {st['expires_at']}" if st.get("expires_at") else ""))
        return 0

    if args.action == "connect":
        if args.broker == "kite":
            from brokers import kite

            if not args.token:
                out.info("Open this, log in, and copy request_token from the URL you land on:")
                out.info(kite.login_url())
                out.info("then: mia broker connect kite --token REQUEST_TOKEN")
                return 0
            out.info(f"connected: {kite.exchange(args.token)}")
            return 0
        if args.broker == "webull":
            from brokers import webull

            out.info("Approve the request in the Webull app — waiting up to 5 minutes…")
            try:
                accounts = webull.connect()
            except NotConnected as exc:
                out.error(str(exc))
                return 2
            out.info(f"connected: {len(accounts)} account(s)")
            return 0
        out.error("connect kite or webull")
        return 2

    if args.action == "pull":
        for r in ([pf.pull(args.broker)] if args.broker else pf.pull_all()):
            out.info(f"  {r}")
        return 0

    if args.action == "show":
        if args.raw:
            if args.broker != "webull":
                out.error("--raw is for checking the Webull field mapping: show webull --raw")
                return 2
            from brokers import webull

            print(json.dumps(webull.raw(), indent=2, default=str))
            return 0
        print(pf.book_summary(max_lines=200) or "no snapshots yet")
        return 0
    return 2


def cmd_email(args) -> int:
    """Send a test email to the owner, to prove delivery end to end."""
    from notify import email, out

    if not email.configured():
        out.error("set RESEND_API_KEY and MIA_EMAIL_TO in .env first")
        return 2
    msg_id = email.send(
        "Alfred can reach you",
        "This is a test from Alfred. Trade ideas will arrive here — as advice "
        "only. Alfred never places orders; you do, in Kite or Webull.",
        key=f"test-{datetime.now(timezone.utc):%Y%m%d%H%M%S}")
    if not msg_id:
        out.error("Resend refused the message — see the log line above for why")
        return 1
    out.info(f"sent to {config.EMAIL_TO} (id {msg_id})")
    return 0


def cmd_advisor(args) -> int:
    """Trade ideas: backfill history, simulate, scan, review. Advice only."""
    from notify import out

    if args.action == "backfill":
        from advisor import bars

        syms = bars.universe() + bars.FX_PAIRS
        out.info(f"downloading {args.years} years of daily bars for {len(syms)} symbols…")
        out.info(f"{bars.fetch(syms, period=f'{args.years}y'):,} rows")
        return 0

    if args.action == "simulate":
        from advisor import simulate

        rep = simulate.run([args.region] if args.region else None)
        for (name, region), (ok, s) in rep.items():
            r, mc = s["recent"], s["monte_carlo"]
            out.info(f"  {name:<15} {region:<4} {r.get('trades', 0):>6} trades  "
                     f"{r.get('expectancy_r', 0):+.3f}R  PF {(r.get('profit_factor') or 0):.2f}  "
                     f"loss-chance {(mc.get('chance_of_loss') or 0) * 100:4.1f}%  "
                     + ("PASS" if ok else "fail: " + "; ".join(s["fails"])))
        return 0

    if args.action == "portfolio":
        from advisor import REGIONS, portfolio_sim

        for region in ([args.region] if args.region else list(REGIONS)):
            s = portfolio_sim.run(region)
            if not s:
                out.info(f"{region}: no strategy passes the gate; nothing to replay")
                continue
            out.info(f"{region} {s['from']} → {s['through']}  ({', '.join(s['strategies'])})")
            for label, key in (("Alfred", "strategy"), ("random picks", "random_picks"),
                               (f"index {s['index_symbol']}", "index"), ("survivors", "survivors")):
                m = s[key] or {}
                out.info(f"  {label:<20} {(m.get('cagr') or 0) * 100:+6.1f}%/yr  "
                         f"worst fall {(m.get('max_drawdown') or 0) * 100:6.1f}%  "
                         f"Sharpe {(m.get('sharpe') or 0):.2f}")
        return 0

    if args.action == "scan":
        from advisor import REGIONS, jobs, report, scan

        for region in ([args.region] if args.region else list(REGIONS)):
            if args.send:
                out.info(f"{region}: {jobs.cycle(region)}")
                continue
            res = scan.scan(region, preview=args.preview)
            composed = report.compose(res, scan.regime())
            print(composed[1] if composed else
                  f"{region}: nothing to act on (as of {res['as_of']})\n\n"
                  + report.holdings_text(res["holdings"]))
            print()
        return 0

    if args.action == "ideas":
        from advisor import ledger

        ledger.update()
        rows = db.query("SELECT * FROM ideas ORDER BY created_at DESC LIMIT 40")
        for r in rows:
            res = f"{r['r_multiple']:+.2f}R" if r["r_multiple"] is not None else ""
            out.info(f"  {r['signal_date']} {r['symbol']:<14} {r['strategy']:<15} "
                     f"{r['status']:<8} {res}")
        out.info(f"scoreboard: {ledger.scoreboard()['overall']}")
        return 0
    return 2


def cmd_data(args) -> int:
    """Run or inspect the non-broker data feeds."""
    import data
    from data import sync

    what = args.what
    if what == "status":
        rows = data.freshness()
        if not rows:
            print("no feed has run yet")
        for r in rows:
            state = "ok " if r["ok"] else "BAD"
            print(f"{state} {r['source']:<12} rows {r['rows']:>8,}  last good "
                  f"{(r['last_ok'] or '—')!s:.16}  {r['error'] or ''}")
        return 0
    if what in ("hourly", "daily", "backfill"):
        results = getattr(sync, what)()
    elif what in sync.SOURCES:
        results = [sync.one(what, backfill=args.backfill)]
    else:
        print(f"unknown feed {what!r}; one of: hourly, daily, backfill, "
              + ", ".join(sync.SOURCES))
        return 2
    for r in results:
        if r.get("ok"):
            extra = {k: v for k, v in r.items() if k not in ("source", "ok", "rows", "seconds")}
            print(f"ok  {r['source']:<12} {r['rows']:>8,} rows  {r.get('seconds', 0)}s  {extra}")
        else:
            print(f"BAD {r['source']:<12} {r.get('error')}")
    return 0 if all(r.get("ok") for r in results) else 1


def cmd_models(args) -> int:
    """Which model does each job, and does every one of them answer?"""
    from brain import llm, router
    from notify import out

    out.info(f"Azure configured: {'yes' if config.AZURE_READY else 'no — every role runs on Gemini'}")
    specs = []
    for role in ("bulk", "workhorse", "reason", "search", "deep", "premium"):
        chain = router._roles()[role]
        live = router.chain_for(role, premium_site="editor" if role == "premium" else None)
        out.info(f"  {role:<10} {' → '.join(chain)}"
                 + ("" if live == chain else f"   (usable now: {' → '.join(live)})"))
        specs += [c for c in live if c not in specs]
    if not args.probe:
        return 0
    bad = 0
    for spec in specs:
        try:
            llm.complete_text(spec, system="Reply with one word.", user="Say ok.",
                              purpose="probe.models", max_tokens=200, fallback=spec)
            out.info(f"  ok    {spec}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            out.error(f"{spec}: {str(exc)[:160]}")
    return 1 if bad else 0


def cmd_lockdown(args) -> int:
    """Back to one owner: disable every other account, revoke every share link.

    Reversible on purpose. Accounts are switched off, not deleted, and come
    back one at a time with `mia user NAME --enable`.
    """
    from notify import out
    from web import auth, sharing

    owner = (args.owner or config.OWNER).strip().lower()
    try:
        disabled = auth.lockdown(owner)
    except ValueError as exc:
        out.error(str(exc))
        return 2
    revoked = sharing.revoke_all()
    out.info(f"kept {owner}; disabled {', '.join(disabled) or 'nobody else'}")
    out.info(f"revoked {revoked} live share link{'' if revoked == 1 else 's'}")
    if config.SHARING_ENABLED:
        out.error("MIA_SHARING is on — new share links can still be created")
    return 0


def cmd_rewrite(args) -> int:
    """Re-render stored briefs in the current format.

    An important limitation, stated plainly: this reformats the analysis that
    was written at the time. It cannot re-research a past window, because the
    documents behind an old brief have often aged out under retention and the
    prices have moved on. What it does is restructure the existing findings into
    the topic sections, give the brief a headline of its own, and bring the
    register into line — so the archive reads consistently.

    For a window whose sources are still in the corpus, `digest --hours N` is
    the better tool: it genuinely re-analyses rather than re-presents.
    """
    import db
    from brain import rewrite
    from notify import out

    rows = db.query(
        """SELECT id, title, body, meta, created_at FROM analyses
           WHERE kind='digest' ORDER BY created_at DESC LIMIT %s""",
        (args.limit,),
    )
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]
    if not rows:
        out.error("no briefs matched")
        return 1

    out.info(f"rewriting {len(rows)} brief(s) with {config.DIGEST_MODEL}")
    total = 0.0
    for r in rows:
        try:
            res = rewrite.rewrite_brief(r, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            out.error(f"  #{r['id']}: {type(exc).__name__}: {exc}")
            continue
        total += res.get("usd", 0.0)
        flag = " (dry run)" if args.dry_run else ""
        out.info(f"  #{r['id']} {res['words']} words · ${res['usd']:.4f}{flag}")
        out.info(f"     {res['headline']}")
    out.info(f"total ${total:.4f}")
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
    d.add_argument("--hours", type=int, default=8)
    d.add_argument("--model", help="override the digest model")
    d.add_argument("--effort", help="low | medium | high | xhigh | max")
    d.set_defaults(func=cmd_digest)

    a = sub.add_parser("ask", help="ask a question with full memory + web")
    a.add_argument("question", nargs="+")
    a.add_argument("--no-web", action="store_true", help="stored memory only")
    a.add_argument("--model", help="override: claude-opus-5 | claude-sonnet-5 | claude-haiku-4-5")
    a.add_argument("--turns", type=int, default=8, help="max agent turns")
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

    u = sub.add_parser("user", help="create or reset a dashboard login")
    u.add_argument("username", nargs="?")
    u.add_argument("--password", help="skip the prompt (avoid: lands in shell history)")
    u.add_argument("--list", action="store_true", help="list existing users")
    u.add_argument("--admin", action="store_true",
                   help="grant administrator rights (Status, Add, sources)")
    u.add_argument("--no-admin", action="store_true", help="revoke administrator rights")
    u.add_argument("--name", help="the person's name, shown when greeting them")
    u.add_argument("--budget", type=float, metavar="USD",
                   help="monthly Ask budget for this reader; -1 restores the "
                        "default. Admins are unlimited regardless.")
    u.add_argument("--create", action="store_true",
                   help="with --admin, also set a password")
    u.add_argument("--disable", action="store_true",
                   help="switch the account off; nothing it owns is deleted")
    u.add_argument("--enable", action="store_true", help="switch the account back on")
    u.set_defaults(func=cmd_user)

    br = sub.add_parser("broker", help="broker connections (read-only)")
    br.add_argument("action", choices=["status", "connect", "pull", "show"])
    br.add_argument("broker", nargs="?", choices=["kite", "webull"])
    br.add_argument("--token", help="Kite request_token, for connecting without the web page")
    br.add_argument("--raw", action="store_true", help="with show webull: unparsed responses")
    br.set_defaults(func=cmd_broker)

    sub.add_parser("email-test", help="send a test email to MIA_EMAIL_TO").set_defaults(
        func=cmd_email)

    ad = sub.add_parser("advisor", help="trade ideas — advice only, never orders")
    ad.add_argument("action", choices=["backfill", "simulate", "portfolio", "scan", "ideas"])
    ad.add_argument("--region", choices=["IN", "ASX", "US"])
    ad.add_argument("--years", type=int, default=12, help="backfill depth")
    ad.add_argument("--preview", action="store_true",
                    help="scan ignoring the gate; shows tickets, stores and sends nothing")
    ad.add_argument("--send", action="store_true",
                    help="run the full cycle: refresh, settle, scan, store, email")
    ad.set_defaults(func=cmd_advisor)

    dt = sub.add_parser("data", help="the data feeds that are not brokers (conf/data.yaml)")
    dt.add_argument("what", nargs="?", default="status",
                    help="status | hourly | daily | backfill | one feed's name")
    dt.add_argument("--backfill", action="store_true", help="with a feed name: load its history")
    dt.set_defaults(func=cmd_data)

    md = sub.add_parser("models", help="which model does each job; --probe pings every one")
    md.add_argument("--probe", action="store_true", help="send each model a one-word test")
    md.set_defaults(func=cmd_models)

    lk = sub.add_parser("lockdown",
                        help="disable every account but the owner's and revoke share links")
    lk.add_argument("--owner", help=f"account to keep (default MIA_OWNER, '{config.OWNER}')")
    lk.set_defaults(func=cmd_lockdown)

    rw = sub.add_parser("rewrite", help="re-render stored briefs in the current format")
    rw.add_argument("--limit", type=int, default=5, help="how many recent briefs")
    rw.add_argument("--only", type=int, nargs="*", help="specific analysis ids")
    rw.add_argument("--dry-run", action="store_true", help="print without saving")
    rw.set_defaults(func=cmd_rewrite)

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
