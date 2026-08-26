"""Deterministic alert rules. Runs every tick; no LLM in the detection path.

The model never decides *whether* something is a big move — code does, against
explicit thresholds. The model's only job downstream is explaining a move that
code already flagged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone

import yaml

import config
import db
from signals import stats

log = logging.getLogger("mia.triggers")
_CONF = config.CONFIG_DIR / "triggers.yaml"

SEV_ORDER = {"High": 1, "Critical": 2}


def load_rules() -> dict:
    return yaml.safe_load(_CONF.read_text()) or {}


def _dedupe_key(rule: str, subject: str, bucket: str) -> str:
    return hashlib.sha256(f"{rule}|{subject}|{bucket}".encode()).hexdigest()[:32]


def _day_bucket(hours: int) -> str:
    """Coarse time bucket so the same condition doesn't re-alert every tick."""
    now = datetime.now(timezone.utc)
    return f"{now.date()}-{now.hour // max(hours, 1)}"


# ───────────────────────────────── detectors ────────────────────────────────
def check_prices(pack: dict, rules: dict) -> list[dict]:
    events = []
    perf = {p["symbol"]: p for p in pack.get("performance", [])}
    window = int(rules.get("dedupe_window_hours", 12))

    for spec in rules.get("price_moves", []):
        row = perf.get(spec["symbol"])
        if not row:
            continue
        move = row.get("chg_1d_pct") or 0
        mag = abs(move)
        sev = None
        if mag >= spec.get("critical_pct", 1e9):
            sev = "Critical"
        elif mag >= spec.get("high_pct", 1e9):
            sev = "High"
        if sev:
            events.append(
                {
                    "rule": "price_move",
                    "severity": sev,
                    "symbol": spec["symbol"],
                    "detail": {
                        "move_pct": move,
                        "threshold_pct": spec.get(f"{sev.lower()}_pct"),
                        "last": row.get("last"),
                        "zscore": row.get("z_1d_move"),
                    },
                    "dedupe_key": _dedupe_key("price_move", spec["symbol"], _day_bucket(window)),
                }
            )

    for spec in rules.get("yield_moves", []):
        row = perf.get(spec["symbol"])
        if not row or row.get("chg_1d_bp") is None:
            continue
        bp = row["chg_1d_bp"]
        mag = abs(bp)
        sev = None
        if mag >= spec.get("critical_bp", 1e9):
            sev = "Critical"
        elif mag >= spec.get("high_bp", 1e9):
            sev = "High"
        if sev:
            events.append(
                {
                    "rule": "yield_move",
                    "severity": sev,
                    "symbol": spec["symbol"],
                    "detail": {
                        "move_bp": bp,
                        "threshold_bp": spec.get(f"{sev.lower()}_bp"),
                        "last_yield": row.get("last"),
                    },
                    "dedupe_key": _dedupe_key("yield_move", spec["symbol"], _day_bucket(window)),
                }
            )

    zconf = rules.get("zscore", {})
    for sym in zconf.get("symbols", []):
        row = perf.get(sym)
        if not row:
            continue
        z = row.get("z_1d_move") or 0
        sev = None
        if abs(z) >= zconf.get("critical", 1e9):
            sev = "Critical"
        elif abs(z) >= zconf.get("high", 1e9):
            sev = "High"
        if sev:
            events.append(
                {
                    "rule": "zscore_outlier",
                    "severity": sev,
                    "symbol": sym,
                    "detail": {"zscore": z, "move_pct": row.get("chg_1d_pct")},
                    "dedupe_key": _dedupe_key("zscore_outlier", sym, _day_bucket(window)),
                }
            )
    return events


def check_structural(pack: dict, rules: dict) -> list[dict]:
    events = []
    conf = rules.get("structural", {})
    window = int(rules.get("dedupe_window_hours", 12))

    gs = next((r for r in pack.get("ratios", []) if r["ratio"] == "gold_silver"), None)
    thresh = conf.get("gold_silver_ratio_z")
    if gs and thresh and gs.get("z_vs_1y") is not None and abs(gs["z_vs_1y"]) >= thresh:
        events.append(
            {
                "rule": "gold_silver_stretch",
                "severity": "High",
                "symbol": "GOLD/SILVER",
                "detail": gs,
                "dedupe_key": _dedupe_key("gold_silver_stretch", "ratio", _day_bucket(24)),
            }
        )

    curve = pack.get("yield_curve", {})
    if conf.get("curve_flip") and curve.get("2s10s_bp") is not None:
        prior = db.one(
            """SELECT payload->'yield_curve'->>'2s10s_bp' AS bp FROM stats_packs
               WHERE created_at < now() - interval '12 hours'
               ORDER BY created_at DESC LIMIT 1"""
        )
        if prior and prior["bp"] is not None:
            try:
                was, now_bp = float(prior["bp"]), float(curve["2s10s_bp"])
                if (was < 0) != (now_bp < 0):
                    events.append(
                        {
                            "rule": "curve_flip",
                            "severity": "Critical",
                            "symbol": "2s10s",
                            "detail": {"prior_bp": was, "now_bp": now_bp},
                            "dedupe_key": _dedupe_key("curve_flip", "2s10s", _day_bucket(48)),
                        }
                    )
            except (TypeError, ValueError):
                pass

    delta_thresh = conf.get("correlation_flip_delta")
    if delta_thresh:
        for flip in pack.get("correlation_flips", []):
            if abs(flip.get("delta", 0)) >= delta_thresh:
                events.append(
                    {
                        "rule": "correlation_break",
                        "severity": "High",
                        "symbol": flip["pair"],
                        "detail": flip,
                        "dedupe_key": _dedupe_key(
                            "correlation_break", flip["pair"], _day_bucket(24)
                        ),
                    }
                )
    return events


def check_news(rules: dict, doc_ids: list[int] | None = None) -> list[dict]:
    """Escalate official-source items and credible keyword hits."""
    conf = rules.get("news", {})
    keywords = [k.lower() for k in conf.get("critical_keywords", [])]
    max_tier = int(conf.get("keyword_max_tier", 2))
    window = int(rules.get("dedupe_window_hours", 12))

    # Two populations: documents ingested this tick, and documents the classifier
    # only just reached. Classification lags ingestion when a harvest is large,
    # so without the second query a doc later judged Critical would never be
    # re-examined and would silently never alert.
    docs = db.query(
        """SELECT id,title,body,source,source_tier,url,urgency FROM documents
           WHERE id = ANY(%s)
              OR (classified_at > now() - interval '90 minutes'
                  AND urgency IN ('Critical','High'))""",
        (doc_ids or [],),
    )

    events = []
    for d in docs:
        hay = f"{d['title']} {d.get('body') or ''}".lower()
        hits = [k for k in keywords if k in hay]
        urgency = d.get("urgency")

        # Keyword presence alone is far too loose to mean Critical — phrases like
        # "rate cut" and "default" appear in routine market copy every day. A
        # Critical news alert requires either the classifier independently
        # judging it Critical, or an official (tier-1) source carrying a trigger
        # keyword. Everything else that matches is High and rides the digest.
        sev = None
        if hits and d["source_tier"] <= max_tier:
            sev = "High"
        if d["source_tier"] == 1:
            sev = "High"
        if urgency == "Critical":
            sev = "Critical"
        elif d["source_tier"] == 1 and hits:
            sev = "Critical"
        if not sev:
            continue
        events.append(
            {
                "rule": "news_signal",
                "severity": sev,
                "symbol": None,
                "doc_id": d["id"],
                "detail": {
                    "title": d["title"],
                    "source": d["source"],
                    "tier": d["source_tier"],
                    "keywords": hits,
                    "url": d.get("url"),
                },
                "dedupe_key": _dedupe_key("news_signal", str(d["id"]), _day_bucket(window * 4)),
            }
        )
    return events


# ───────────────────────────────── persistence ──────────────────────────────
def record(events: list[dict]) -> list[dict]:
    """Insert events, skipping ones already seen. Returns only genuinely new rows."""
    fresh = []
    with db.conn() as c, c.cursor() as cur:
        for e in events:
            cur.execute(
                """INSERT INTO trigger_events (rule,severity,symbol,doc_id,detail,dedupe_key)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (dedupe_key) DO NOTHING RETURNING id""",
                (
                    e["rule"],
                    e["severity"],
                    e.get("symbol"),
                    e.get("doc_id"),
                    json.dumps(e.get("detail", {}), default=str),
                    e["dedupe_key"],
                ),
            )
            row = cur.fetchone()
            if row:
                e["id"] = row[0]
                fresh.append(e)
    return fresh


def evaluate(pack: dict | None = None, doc_ids: list[int] | None = None) -> list[dict]:
    """Full trigger sweep. Returns newly-fired (deduped) events."""
    rules = load_rules()
    pack = pack or stats.latest_pack() or stats.build()
    events = check_prices(pack, rules) + check_structural(pack, rules) + check_news(rules, doc_ids)
    return record(events)


def pending_critical(max_per_run: int = 3, max_per_hour: int = 5) -> list[dict]:
    """Critical events awaiting delivery, rate-limited.

    A burst of alerts trains the reader to ignore alerts, which defeats the
    purpose of having them. Anything suppressed here is not lost — it is already
    recorded and will be folded into the next digest.
    """
    recent = db.one(
        """SELECT count(*) n FROM trigger_events
           WHERE notified_at > now() - interval '1 hour'"""
    )
    budget = max(0, max_per_hour - int(recent["n"] if recent else 0))
    if budget == 0:
        return []
    return db.query(
        """SELECT * FROM trigger_events
           WHERE severity='Critical' AND notified_at IS NULL
           ORDER BY created_at DESC LIMIT %s""",
        (min(max_per_run, budget),),
    )


def suppress_stale(hours: int = 6) -> int:
    """Mark old un-notified Criticals as handled so they never alert late."""
    return db.execute(
        """UPDATE trigger_events SET notified_at=now()
           WHERE severity='Critical' AND notified_at IS NULL
             AND created_at < now() - make_interval(hours => %s)""",
        (hours,),
    )


def mark_notified(ids: list[int]) -> int:
    if not ids:
        return 0
    return db.execute(
        "UPDATE trigger_events SET notified_at=now() WHERE id = ANY(%s)", (ids,)
    )
