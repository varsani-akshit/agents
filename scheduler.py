"""Scheduler: the loops that make Alfred continuous rather than on-demand.

  every 30 min   full tick: ingest -> embed -> classify -> stats -> triggers
  offset 15 min  prices-only tick (free), so triggers still see fresh prices
  twice daily    deep brief at 09:05 and 21:05 UTC (7pm and 7am Melbourne)
  daily 02:00    FRED refresh + daily price refresh
  daily 03:00    graph hygiene, retention, spend report

Each job records to `job_runs`, so `mia status` shows what ran and what broke.
Jobs never raise into the scheduler — a failing feed must not stop the loop.
"""
from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config
import db
from notify import out

log = logging.getLogger("mia.scheduler")


def _guard(name: str, fn) -> None:
    """Run a job, recording success/failure. Never propagates."""
    try:
        with db.job(name) as detail:
            result = fn()
            if isinstance(result, dict):
                detail.update(result)
    except Exception as exc:  # noqa: BLE001
        log.exception("job %s failed", name)
        out.error(f"{name}: {type(exc).__name__}: {exc}")


# ─────────────────────────────────── jobs ───────────────────────────────────
def job_tick(harvest_news: bool = True) -> dict:
    """One ingestion cycle.

    Prices and trigger evaluation are free — no model touches them — so they can
    run as often as we like. Harvest, embedding and classification are the only
    steps that spend AI credit, so the schedule runs them at half the cadence:
    the full tick every 30 minutes, a prices-only tick in between. News minutes
    old is indistinguishable from news half an hour old at a twice-daily brief.
    """
    from brain import alert as alert_mod
    from ingest import feeds, prices
    from signals import stats, triggers

    detail: dict = {}
    detail["prices"] = prices.tick()
    harvest = {}
    if harvest_news:
        from brain import classify
        from memory import store

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
        out.alert(alert_mod.write(ev))
        sent.append(ev["id"])
    triggers.mark_notified(sent)
    detail["alerts_sent"] = len(sent)

    out.info(
        f"tick · {detail.get('feeds', {}).get('inserted', 0)} new docs · "
        f"{detail['triggers_fired']} triggers · {detail['alerts_sent']} alerts"
    )
    return detail


def job_digest() -> dict:
    from brain import digest

    result = digest.run(hours=12)
    if not result.get("ok"):
        out.error(f"digest failed: {result.get('error')}")
        raise RuntimeError(f"digest produced no output: {result.get('error')}")
    out.digest(result)
    return {k: v for k, v in result.items() if k != "body"}


def job_daily_data() -> dict:
    from ingest import fred, prices

    detail = {"prices": prices.daily_refresh()}
    f = fred.sync(120)
    detail["fred"] = {k: v for k, v in f.items() if k != "per_series"}
    out.info(f"daily data refresh · {detail}")
    return detail


# ─────────────────────────────── retention ──────────────────────────────────
# What Alfred keeps, and for how long. Two principles decide these numbers.
#
# Storage is the reason to delete. Analysis is the reason to keep. Where they
# conflict, note that a stale document does not actually cause a wrong answer:
# search already decays relevance on a three-week half-life, every document the
# model reads carries its publication time, and a brief that cites an old item
# alongside a new one is showing a sequence. What deleting history does cost is
# the ability to recognise continuity — that this is the third time Treasury has
# floated the same idea — which is exactly the kind of observation the system
# exists to make.
#
# So: routine churn goes quickly, anything that mattered stays for years, and
# the series that feed quantitative work are never deleted at all.
# (label, table, timestamp column, window, extra predicate). The timestamp
# column is named rather than inferred: guessing "created_at" was wrong for
# job_runs, which uses started_at, and that rule would have failed silently at
# 03:00 every night while the rest of maintenance appeared to succeed.
RETENTION = [
    # -- news corpus, graded by how much each document turned out to matter ----
    ("docs_churn", "documents", "fetched_at", "21 days",
     "coalesce(urgency,'Low') = 'Low' AND source_tier >= 3"),
    ("docs_routine", "documents", "fetched_at", "90 days",
     "urgency = 'Medium' AND source_tier >= 3"),
    # Tier 1-2 are sources of record and low volume; High/Critical is whatever
    # the classifier thought mattered. Two years lets a brief compare a policy
    # move against the same month last year.
    ("docs_aged", "documents", "fetched_at", "730 days", "TRUE"),

    # -- operational history ---------------------------------------------------
    # Intraday bars exist to catch a same-session reversal. Weeks later the
    # daily bar carries everything they add, at a fraction of the rows.
    ("intraday_prices", "prices", "ts", "21 days", "grain = '15m'"),
    # Regenerable from the series in seconds, so there is no case for keeping them.
    ("stats_packs", "stats_packs", "created_at", "30 days", "TRUE"),
    # Figures for archived briefs. Older briefs keep their text and argument.
    ("chart_packs", "chart_packs", "created_at", "180 days", "TRUE"),
    # Long enough to backtest an alert rule against a full year of markets.
    ("trigger_events", "trigger_events", "created_at", "400 days", "TRUE"),
    # Spend analysis needs a couple of quarters, not forever.
    ("api_calls", "api_calls", "created_at", "180 days", "TRUE"),
    ("job_runs", "job_runs", "started_at", "90 days", "TRUE"),
]

# Never deleted, and listed explicitly so the omission reads as a decision
# rather than an oversight:
#   prices (grain='1d')  the analogue engine reads twelve years
#   fred_series          macro series are meaningless without decades
#   analyses             the briefs are the product
#   world_model          the standing view, and its history
#   users, instruments   configuration


def apply_retention() -> dict:
    """Age out data per the table above. Returns rows removed per rule."""
    out: dict[str, int] = {}
    for label, table, ts, window, predicate in RETENTION:
        # A document cited by a stored brief is never removed: a published
        # analysis must not end up pointing at a source that no longer exists.
        guard = ""
        if table == "documents":
            guard = """ AND NOT EXISTS (
                SELECT 1 FROM analyses a
                WHERE a.meta::text LIKE '%%' || d.url || '%%')"""
        alias = " d" if table == "documents" else ""
        out[label] = db.execute(
            f"""DELETE FROM {table}{alias}
                WHERE {ts} < now() - interval '{window}'
                  AND ({predicate}){guard}"""
        )
    return out


def job_maintenance() -> dict:
    from brain import client, extract

    detail = {"hygiene": extract.hygiene()}
    detail.update(apply_retention())
    # Refresh semantic links after pruning, so the graph reflects what remains.
    from memory import graph as kg

    detail["doc_links"] = kg.rebuild_links(days=7)
    detail["budget"] = client.budget_status()
    b = detail["budget"]
    out.info(
        f"maintenance · graph {detail['hygiene']} · "
        f"spend today ${b['spent_today_usd']:.3f} total ${b['spent_total_usd']:.3f}"
    )
    return detail


# ─────────────────────────────────── runner ─────────────────────────────────
def build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(
        timezone="UTC",
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
    )
    sched.add_job(
        lambda: _guard("tick", job_tick),
        IntervalTrigger(minutes=30),
        id="tick",
        next_run_time=datetime.now(timezone.utc),
    )
    # Offset by 15 minutes so the two never coincide.
    sched.add_job(
        lambda: _guard("tick", lambda: job_tick(harvest_news=False)),
        IntervalTrigger(minutes=30, start_date=datetime.now(timezone.utc) + timedelta(minutes=15)),
        id="tick_prices",
    )
    # 21:05 UTC / 09:05 UTC: morning and evening briefs in Melbourne, each
    # covering the 12 hours since the other.
    sched.add_job(
        lambda: _guard("digest", job_digest),
        CronTrigger(hour="9,21", minute=5),
        id="digest",
    )
    sched.add_job(
        lambda: _guard("daily_data", job_daily_data),
        CronTrigger(hour=2, minute=0),
        id="daily_data",
    )
    sched.add_job(
        lambda: _guard("maintenance", job_maintenance),
        CronTrigger(hour=3, minute=0),
        id="maintenance",
    )
    return sched


def build_background_scheduler():
    """The same jobs on a non-blocking scheduler, for running inside the web app.

    Two processes cost twice as much to host, and for a single-reader system the
    scheduler is idle almost all the time. Setting MIA_EMBEDDED_SCHEDULER=1 runs
    it in a thread beside the web server so one service does both.

    This is only safe at exactly one instance. With two, both would tick, both
    would write a brief, and the duplicates would look like real data. The web
    app checks that before starting it.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(
        timezone="UTC",
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
    )
    for job_id, fn, trigger in (
        ("tick", job_tick, IntervalTrigger(minutes=30)),
        ("tick_prices", lambda: job_tick(harvest_news=False),
         IntervalTrigger(minutes=30, start_date=datetime.now(timezone.utc) + timedelta(minutes=15))),
        ("digest", job_digest, CronTrigger(hour="9,21", minute=5)),
        ("daily_data", job_daily_data, CronTrigger(hour=2, minute=0)),
        ("maintenance", job_maintenance, CronTrigger(hour=3, minute=0)),
    ):
        sched.add_job(lambda f=fn, i=job_id: _guard(i, f), trigger, id=job_id)
    return sched


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(config.LOGS / "scheduler.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    for noisy in ("httpx", "httpcore", "apscheduler.executors", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sched = build_scheduler()

    def shutdown(signum, frame):  # noqa: ARG001
        out.info("shutting down scheduler…")
        sched.shutdown(wait=False)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    out.info("MIA scheduler starting — tick/15m, digest/6h, data/daily, maintenance/daily")
    sched.start()


if __name__ == "__main__":
    main()
