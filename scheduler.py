"""Scheduler: the loops that make MIA continuous rather than on-demand.

  every 15 min   ingest -> embed -> classify -> stats -> triggers -> alerts
  every 6 hours  deep analysis digest (00:05, 06:05, 12:05, 18:05 UTC)
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

    result = digest.run(hours=6)
    if result.get("ok"):
        out.digest(result)
        return {k: v for k, v in result.items() if k != "body"}
    out.error(f"digest failed: {result.get('error')}")
    return {"ok": False, "error": str(result.get("error"))}


def job_daily_data() -> dict:
    from ingest import fred, prices

    detail = {"prices": prices.daily_refresh()}
    f = fred.sync(120)
    detail["fred"] = {k: v for k, v in f.items() if k != "per_series"}
    out.info(f"daily data refresh · {detail}")
    return detail


def job_maintenance() -> dict:
    from brain import client, extract

    detail = {"hygiene": extract.hygiene()}
    # Keep the stats_packs table from growing without bound.
    detail["packs_pruned"] = db.execute(
        "DELETE FROM stats_packs WHERE created_at < now() - interval '30 days'"
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
        CronTrigger(hour="0,6,12,18", minute=5),
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
