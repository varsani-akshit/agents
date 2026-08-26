"""Scheduler: the loops that make MIA continuous rather than on-demand.

  every 15 min   ingest -> embed -> classify -> stats -> triggers -> alerts
  every 8 hours  deep analysis digest (00:05, 08:05, 16:05 UTC)
  daily 02:00    FRED refresh + daily price refresh
  daily 03:00    graph hygiene, stats vacuum, spend report

Each job records to `job_runs`, so `mia status` shows what ran and what broke.
Jobs never raise into the scheduler — a failing feed must not stop the loop.
"""
from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime, timezone

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
def job_tick() -> dict:
    from brain import alert as alert_mod
    from brain import classify
    from ingest import feeds, prices
    from memory import store
    from signals import stats, triggers

    detail: dict = {}
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
        out.alert(alert_mod.write(ev))
        sent.append(ev["id"])
    triggers.mark_notified(sent)
    detail["alerts_sent"] = len(sent)

    out.info(
        f"tick · {detail['feeds'].get('inserted', 0)} new docs · "
        f"{detail['triggers_fired']} triggers · {detail['alerts_sent']} alerts"
    )
    return detail


def job_digest() -> dict:
    from brain import digest

    result = digest.run(hours=8)
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


def prune_documents() -> dict:
    """Age out the corpus by how much each document turned out to matter.

    Ingestion runs about a thousand documents a day, each carrying a 1024-dim
    embedding plus its index entry, so the corpus grows a few hundred megabytes a
    month with nothing to stop it. That is fine on a laptop disk and fatal on any
    hosted plan with a storage cap.

    The windows are deliberately generous and graded by signal: routine churn
    goes first, anything a source of record published or the classifier flagged
    is kept for a year. Documents referenced by a stored analysis are never
    deleted, so no brief ends up citing a source that no longer exists.
    """
    keep = """
      AND NOT EXISTS (
        SELECT 1 FROM analyses a
        WHERE a.meta::text LIKE '%%' || d.url || '%%')
    """
    windows = [
        ("low", "d.urgency = 'Low' AND d.source_tier >= 3", "30 days"),
        ("medium", "d.urgency = 'Medium' AND d.source_tier >= 3", "120 days"),
        ("aged", "TRUE", "365 days"),
    ]
    out_counts: dict[str, int] = {}
    for name, predicate, window in windows:
        out_counts[f"docs_pruned_{name}"] = db.execute(
            f"""DELETE FROM documents d
                WHERE d.fetched_at < now() - interval '{window}'
                  AND ({predicate}){keep}"""
        )
    # Intraday bars exist to catch a same-session reversal; a fortnight later
    # they carry nothing the daily bar does not.
    out_counts["intraday_pruned"] = db.execute(
        "DELETE FROM prices WHERE grain='15m' AND ts < now() - interval '21 days'"
    )
    return out_counts


def job_maintenance() -> dict:
    from brain import client, extract

    detail = {"hygiene": extract.hygiene()}
    # Keep the stats_packs table from growing without bound.
    detail["packs_pruned"] = db.execute(
        "DELETE FROM stats_packs WHERE created_at < now() - interval '30 days'"
    )
    detail.update(prune_documents())
    # Chart packs are ~100KB each and three briefs a day is ~9MB a month. Old
    # briefs keep their text and their argument; only the figures expire.
    detail["chart_packs_pruned"] = db.execute(
        "DELETE FROM chart_packs WHERE created_at < now() - interval '120 days'"
    )
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
        IntervalTrigger(minutes=15),
        id="tick",
        next_run_time=datetime.now(timezone.utc),
    )
    sched.add_job(
        lambda: _guard("digest", job_digest),
        CronTrigger(hour="0,8,16", minute=5),
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
        ("tick", job_tick, IntervalTrigger(minutes=15)),
        ("digest", job_digest, CronTrigger(hour="0,8,16", minute=5)),
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
