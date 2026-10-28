"""A simulation run, end to end, kept so it can be opened later.

Every two hours: top up prices, re-run every strategy on every name, replay
each market as one account, and record every candidate. The latest results
still feed the gate and the tickets through strategy_evidence and
portfolio_backtests; the run tables are the history the Simulations page
lists.

Daily bars, not intraday: a session in progress is excluded on purpose, so a
run between closes repeats the last close's verdict. Running often means a
market's close reaches the evidence within two hours rather than overnight.
"""
from __future__ import annotations

import json
import logging
import time

import db
from advisor import REGIONS, bars, candidates, portfolio_sim, simulate

log = logging.getLogger("alfred.runs")

KEEP_DAYS = 14


# A run takes 2-5 minutes. One with no finish after this long was killed
# mid-way (a scheduler restart does that) and will never finish.
STALE_MINUTES = 20


def close_interrupted() -> int:
    """Mark runs that were killed part-way as failed, so nothing waits on them."""
    return db.execute(
        """UPDATE sim_runs SET finished_at = now(),
                 error = 'interrupted: the process stopped before the run finished'
           WHERE finished_at IS NULL
             AND started_at < now() - make_interval(mins => %s)""", (STALE_MINUTES,))


def run(trigger: str = "schedule", refresh: bool = True) -> dict:
    close_interrupted()
    t0 = time.time()
    row = db.one("INSERT INTO sim_runs (trigger) VALUES (%s) RETURNING id", (trigger,))
    run_id = row["id"]
    try:
        if refresh:
            bars.refresh(period="5d")
        report = simulate.run()
        db.executemany(
            "INSERT INTO sim_results (run_id, strategy, region, passed, stats) VALUES (%s,%s,%s,%s,%s)",
            [(run_id, name, region, ok,
              json.dumps(candidates._clean({k: v for k, v in stats.items() if k != "curve"}),
                         default=str))
             for (name, region), (ok, stats) in report.items()])

        for region in REGIONS:
            portfolio_sim.run(region)
            acct = db.one("SELECT stats, curve FROM portfolio_backtests WHERE region=%s", (region,))
            if acct:
                db.execute("INSERT INTO sim_accounts (run_id, region, stats, curve) VALUES (%s,%s,%s,%s)",
                           (run_id, region, json.dumps(acct["stats"]), json.dumps(acct["curve"])))

        # Intraday: hourly prices topped up, then the same-day strategies.
        try:
            from advisor import hourly, intraday

            if refresh:
                hourly.fetch(bars.universe(), period="5d")
            ireport = intraday.run()
        except Exception as exc:  # noqa: BLE001 - daily results still stand
            log.warning("intraday simulation failed: %s", exc)
            ireport = {}
        for key, val in ireport.items():
            if key[0] == "account":
                curve = val.pop("curve", {})
                db.execute("INSERT INTO sim_accounts (run_id, region, stats, curve) VALUES (%s,%s,%s,%s)",
                           (run_id, f"{key[1]}:intraday",
                            json.dumps(candidates._clean(val), default=str), json.dumps(curve)))
            else:
                ok, stats = val
                report[key] = (ok, stats)
                db.execute("INSERT INTO sim_results (run_id, strategy, region, passed, stats) "
                           "VALUES (%s,%s,%s,%s,%s)",
                           (run_id, key[0], key[1], ok,
                            json.dumps(candidates._clean(stats), default=str)))

        through, cands = {}, []
        for region in REGIONS:
            found = candidates.collect(region)
            cands += found
            dates = [c["as_of"] for c in found]
            through[region] = max(dates) if dates else None
        db.executemany(
            """INSERT INTO sim_candidates (run_id, region, symbol, strategy, kind, passed_gate, score, detail)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(run_id, c["region"], c["symbol"], c["strategy"], c["kind"], c["passed_gate"], c["score"],
              json.dumps(candidates._clean(c), default=str)) for c in cands])

        passing = sum(ok for ok, _ in report.values())
        db.execute(
            """UPDATE sim_runs SET finished_at=now(), bars_through=%s, passing=%s, tested=%s,
                 candidates=%s, seconds=%s WHERE id=%s""",
            (json.dumps(through), passing, len(report), len(cands), round(time.time() - t0, 1), run_id))
        db.execute("DELETE FROM sim_runs WHERE started_at < now() - make_interval(days => %s)",
                   (KEEP_DAYS,))
        return {"run_id": run_id, "passing": passing, "tested": len(report),
                "candidates": len(cands), "seconds": round(time.time() - t0, 1)}
    except Exception as exc:
        db.execute("UPDATE sim_runs SET finished_at=now(), error=%s, seconds=%s WHERE id=%s",
                   (f"{type(exc).__name__}: {exc}"[:1000], round(time.time() - t0, 1), run_id))
        raise
