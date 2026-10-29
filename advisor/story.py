"""The Story read, and the record of every call.

**Calls.** Every signal a simulation run surfaces becomes a call — one per
name, strategy and signal day, taken or not, passing the gate or not — and
is followed to its exit at the prices Alfred named, by the same play() the
backtest and the ledger use. That is the scorecard: not what the backtest
promised, but what each call did afterwards.

**The Story read.** For each new call, code assembles a dossier — the
company's last three days of news (fetched fresh if Alfred has little), its
filings, the earnings and central-bank dates inside the holding window, the
market's conditions, and the latest brief's top line — and a model reads it
and says whether the story *supports*, is *neutral* to, or *contradicts* the
setup, citing the articles it used and naming the one thing that would
change its mind. Decision-1 adds a probability of a rise or a fall.

The Story read cannot be backtested honestly: a model asked about 2019
already knows 2020. So it is judged the only fair way — forward. The
scorecard compares the calls it supported with the calls it contradicted;
until that comparison has enough closed calls behind it, the read is shown
beside a ticket but does not decide one.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import db
from advisor import CURRENCY, bare, bars, market_of, settings

log = logging.getLogger("alfred.story")

READS_PER_RUN = 20          # model reads per cycle; the budget caps still apply
NEWS_HOURS = 72
MAX_DOCS = 12
MIN_DOCS = 3                # fewer than this, and the read fetches fresh news first
_STATUS = {"pending": "pending", "expired": "expired", "open": "filled", "closed": "closed"}
_SUFFIX = re.compile(r"[,.]?\s+(limited|ltd|inc|incorporated|corporation|corp|plc|holdings?|"
                     r"group|company|co|n\.?v|s\.?a|ag|se|trust|fund|etf)\.?$", re.I)
REGION_WORDS = {"IN": "NSE/BSE (India)", "ASX": "ASX (Australia)", "US": "NYSE/Nasdaq (US)"}


# ───────────────────────────────── recording ────────────────────────────────
def record(run_id: int | None = None) -> int:
    """Every signal from a run, as a call. Returns how many were new."""
    run = db.one("""SELECT id FROM sim_runs WHERE finished_at IS NOT NULL AND error IS NULL
                    AND (%s::bigint IS NULL OR id = %s) ORDER BY started_at DESC LIMIT 1""",
                 (run_id, run_id))
    if not run:
        return 0
    rows = db.query("""SELECT region, symbol, strategy, passed_gate, score, detail
                       FROM sim_candidates WHERE run_id = %s AND kind = 'signal'""", (run["id"],))
    new = 0
    for c in rows:
        d = c["detail"] or {}
        if d.get("limit") is None or d.get("stop") is None or not d.get("as_of"):
            continue
        keep = {k: d.get(k) for k in ("display", "label", "broker", "currency", "close",
                                      "stop_pct", "quantity", "reasons", "regime_why",
                                      "strategy_passed", "market_sending", "held")}
        keep["median_days"] = ((d.get("evidence") or {}).get("median_days"))
        keep["expectancy_r"] = ((d.get("evidence") or {}).get("expectancy_r"))
        new += db.execute(
            """INSERT INTO calls (run_id, region, symbol, strategy, signal_date, limit_price,
                                  stop_price, target_price, passed_gate, regime_on, score, detail)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (symbol, strategy, signal_date) DO NOTHING""",
            (run["id"], c["region"], c["symbol"], c["strategy"], d["as_of"], d["limit"], d["stop"],
             d.get("target"), c["passed_gate"], d.get("regime_on"), c["score"],
             json.dumps(keep, default=str)))
    return new


# ───────────────────────────────── settling ─────────────────────────────────
def settle() -> dict:
    """Replay every open call to today. Returns counts by new status."""
    from advisor import ledger
    from advisor.strategies import STRATEGIES, add_momentum_rank, indicators

    open_calls = db.query("SELECT * FROM calls WHERE status IN ('pending', 'filled')")
    ranked: dict[str, dict] = {}       # momentum rank needs the whole region, once
    moved: dict[str, int] = {}
    for call in open_calls:
        strat = STRATEGIES.get(call["strategy"])
        x = bars.load(call["symbol"], years=2)
        if strat is None or len(x) < 220:
            continue
        x = indicators(x)
        if strat.name == "momentum_rotation":
            region = call["region"]
            if region not in ranked:
                frames = {}
                for s in bars.universe():
                    if market_of(s) in _markets(region):
                        f = bars.load(s, years=2)
                        if len(f) > 260:
                            frames[s] = indicators(f)
                add_momentum_rank(frames)
                ranked[region] = frames
            x = ranked[region].get(call["symbol"], x)
        out = ledger.replay_one(call["symbol"], x, strat, call["signal_date"], call["limit_price"],
                                call["stop_price"], call["target_price"])
        if out is None:
            continue
        status = _STATUS[out.status]
        d = lambda k: x.index[k].date() if k is not None else None  # noqa: E731
        db.execute(
            """UPDATE calls SET status=%s, fill_date=%s, fill_price=%s, exit_date=%s,
                 exit_price=%s, exit_reason=%s, r=%s, settled_at=now() WHERE id=%s""",
            (status, d(out.fill_i), out.fill, d(out.exit_i), out.exit, out.reason, out.r, call["id"]))
        if status != call["status"]:
            moved[status] = moved.get(status, 0) + 1
    return {"open": len(open_calls), "moved": moved}


def _markets(region: str) -> tuple:
    from advisor import REGIONS

    return REGIONS[region]


# ───────────────────────────────── the dossier ──────────────────────────────
def short_name(name: str) -> str:
    """"Darden Restaurants, Inc." → "Darden Restaurants": what headlines say."""
    s = name.strip()
    for _ in range(3):
        s2 = _SUFFIX.sub("", s).strip(" ,.")
        if s2 == s:
            break
        s = s2
    return s or name


def _company(symbol: str) -> str:
    from brain.news_match import _name

    return _name(symbol) or bare(symbol)


def _docs(symbol: str, name: str, hours: int = NEWS_HOURS) -> list[dict]:
    """Recent articles about the company: confirmed by the news matcher, or
    naming the company or its ticker in the headline or opening lines.

    A Finnhub tag alone is not enough: Finnhub files anything that mentions
    the S&P indexes under S&P Global, and Uber under half the market."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    short = short_name(name)
    ticker = bare(symbol)
    pats = [re.escape(short)]
    if len(ticker) >= 3 and ticker.isalpha():
        pats.append(r"\m" + re.escape(ticker) + r"\M")
    pat = "(" + "|".join(pats) + ")"
    rows = db.query(
        """SELECT DISTINCT ON (d.id) d.id, d.title, d.source, d.source_tier, d.published_at,
                  left(coalesce(d.summary, d.body, ''), 400) AS gist
           FROM documents d
           LEFT JOIN news_matches m ON m.doc_id = d.id AND m.symbol = %s
           WHERE coalesce(d.published_at, d.fetched_at) > %s
             AND (m.symbol IS NOT NULL OR d.title ~* %s
                  OR left(coalesce(d.summary, d.body, ''), 400) ~* %s)
           ORDER BY d.id""",
        (symbol, since, pat, pat))
    rows.sort(key=lambda r: (r["source_tier"], -(r["published_at"] or since).timestamp()))
    return rows[:MAX_DOCS]


def fetch_news(symbol: str, name: str) -> int:
    """Fresh news for one name, into `documents`: Google News in the name's
    own market, and Finnhub company news for US names."""
    from urllib.parse import quote_plus

    from ingest import feeds

    m = market_of(symbol)
    hl, gl = {"NSE": ("en-IN", "IN"), "BSE": ("en-IN", "IN"),
              "ASX": ("en-AU", "AU")}.get(m, ("en-US", "US"))
    src = {"name": f"Google News · {short_name(name)} (proxy)", "tier": 3, "proxy": True,
           "url": (f"https://news.google.com/rss/search?q={quote_plus(chr(34) + short_name(name) + chr(34))}"
                   f"+when:3d&hl={hl}&gl={gl}&ceid={gl}:en")}
    docs = feeds.fetch_source(src, max_age_hours=NEWS_HOURS)
    n = feeds.store(docs)["inserted"]
    if m == "US":
        try:
            from data import finnhub

            to = date.today()
            items = finnhub._get("/company-news", {"symbol": bare(symbol),
                                                   "from": (to - timedelta(days=3)).isoformat(),
                                                   "to": to.isoformat()})
            n += finnhub._store_docs(finnhub.news_docs(bare(symbol), items or [],
                                                       f"Finnhub · {bare(symbol)}"))
        except Exception as exc:  # noqa: BLE001 - Google News alone is still a read
            log.info("finnhub news for %s: %s", symbol, exc)
    return n


def _events(symbol: str, region: str, start: date, days: int) -> list[str]:
    end = start + timedelta(days=max(days, 5))
    out, seen = [], set()
    for r in db.query("""SELECT kind, key, event, at, importance FROM calendar
                         WHERE at::date BETWEEN %s AND %s
                           AND ((kind = 'earnings' AND key = %s)
                                OR (kind = 'meeting' AND key = %s)
                                OR (kind = 'release' AND importance = 'high'))
                         ORDER BY at""",
                      (start, end, bare(symbol), {"IN": "IN", "ASX": "AU", "US": "US"}[region])):
        what = "earnings report" if r["kind"] == "earnings" else r["event"]
        if what in seen:            # a release listed every day is one event
            continue
        seen.add(what)
        out.append(f"{r['at']:%a %d %b}: {what}")
    return out[:10]


def _filings(symbol: str, days: int = 14) -> list[str]:
    rows = db.query("""SELECT source, kind, title, filed_at, sensitive FROM filings
                       WHERE symbol = %s AND filed_at > now() - make_interval(days => %s)
                       ORDER BY filed_at DESC LIMIT 8""", (symbol, days))
    return [f"{r['filed_at']:%d %b} · {r['source'].upper()} {r['kind']}: {r['title'][:160]}"
            + (" (price-sensitive)" if r["sensitive"] else "") for r in rows]


def _moves(symbol: str) -> dict:
    x = bars.load(symbol, years=2)
    if x.empty:
        return {}
    c = x["close"]
    pct = lambda n: float((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100) if len(c) > n else None  # noqa: E731
    hi = float(c.tail(252).max())
    return {"last": float(c.iloc[-1]), "as_of": str(x.index[-1].date()),
            "d1": pct(1), "d5": pct(5), "d20": pct(20), "d60": pct(60),
            "from_52w_high": float((c.iloc[-1] / hi - 1) * 100)}


def _brief_line() -> str:
    r = db.one("SELECT body FROM analyses WHERE kind = 'digest' AND created_at > now() - interval '2 days' "
               "ORDER BY created_at DESC LIMIT 1")
    if not r:
        return ""
    body = re.sub(r"\s+", " ", r["body"] or "")
    return body[:900]


def dossier(call: dict) -> dict:
    """Everything the Story read may use, built in code."""
    from advisor import regime

    sym, region = call["symbol"], call["region"]
    name = _company(sym)
    docs = _docs(sym, name)
    if len(docs) < MIN_DOCS:
        try:
            fetch_news(sym, name)
            docs = _docs(sym, name)
        except Exception as exc:  # noqa: BLE001 - a read with less news is still a read
            log.info("fresh news for %s failed: %s", sym, exc)
    d = call.get("detail") or {}
    hold = int(d.get("median_days") or 10)
    try:
        cond = [f"{s['label']}: {s['words']}" for s in regime.now(region)["states"]]
    except Exception:  # noqa: BLE001
        cond = []
    return {
        "name": name, "ticker": bare(sym), "exchange": REGION_WORDS.get(region, region),
        "currency": CURRENCY[market_of(sym)], "moves": _moves(sym),
        "setup": {"strategy": d.get("label") or call["strategy"], "limit": call["limit_price"],
                  "stop": call["stop_price"], "target": call["target_price"],
                  "stop_pct": d.get("stop_pct"), "typical_hold_days": hold,
                  "why_it_fired": d.get("reasons") or [],
                  "strategy_record_r": d.get("expectancy_r"),
                  "passes_the_gate": bool(call["passed_gate"])},
        "events_in_window": _events(sym, region, call["signal_date"], int(hold * 1.5)),
        "filings": _filings(sym),
        "news": [{"id": r["id"], "when": f"{r['published_at']:%d %b %H:%M}" if r["published_at"] else "",
                  "source": r["source"], "title": r["title"], "gist": r["gist"]} for r in docs],
        "conditions": cond,
        "world": _brief_line(),
    }


# ───────────────────────────────── the read ─────────────────────────────────
SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["supports", "neutral", "contradicts"]},
        "strength": {"type": "number"},
        "thesis": {"type": "string"},
        "risk": {"type": "string"},
        "event_risk": {"type": "boolean"},
        "cites": {"type": "array", "items": {"type": "integer"}},
        "flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["direction", "strength", "thesis", "risk", "event_risk", "cites", "flags"],
    "additionalProperties": False,
}

SYSTEM = """You read the news around one trade setup and say whether the story supports it.

The setup is a mechanical signal from a tested strategy: buy at the limit, exit at the stop or by the strategy's rule, usually within the typical hold. Your job is not to re-judge the chart. It is to say whether what is happening to this company, its sector, its country and the world makes the setup more or less likely to work over that hold.

Use only the dossier. Do not use anything you remember about this company or about events after the dossier's dates. Cite the ids of the articles you rely on.

- supports: concrete recent news that should help the price over the hold — an earnings beat or upgrade, a contract, a policy or sector tailwind that names this business, money flowing into the theme.
- contradicts: concrete recent news that should hurt it — a profit warning, downgrade, investigation, fraud or governance issue, a lost contract, a policy or sector headwind that names this business, or a scheduled event inside the hold (an earnings report, a central-bank decision) big enough to gap the price through the stop.
- neutral: nothing material, or the good and bad cancel. "No news" is neutral, not supportive.

strength: 0 to 1, how much the story should move the odds. thesis: at most two plain sentences on why. risk: the one development that would change your read. event_risk: true if a scheduled event inside the hold could gap the price. flags: short labels such as "earnings in 4 days", "regulatory probe", "no company news"."""


def read(call: dict) -> dict:
    """One Story read: the model's judgement and Decision-1's probability."""
    from brain import decide, router

    dos = dossier(call)
    user = json.dumps(dos, default=str, ensure_ascii=False)
    payload, spec = router.complete_json("reason", system=SYSTEM, user=user, schema=SCHEMA,
                                         purpose="story.read", max_tokens=700)
    known = {n["id"] for n in dos["news"]}
    payload["cites"] = [i for i in payload.get("cites") or [] if i in known]
    payload["strength"] = float(np.clip(payload.get("strength") or 0, 0, 1))
    payload["model"] = spec
    payload["news_count"] = len(dos["news"])
    payload["events"] = dos["events_in_window"]
    try:
        hold = dos["setup"]["typical_hold_days"]
        q = {"move": {"type": "choice",
                      "instructions": f"Given this dossier, over the next {hold} trading days is "
                                      f"{dos['name']}'s share price more likely to rise, fall, or "
                                      "end roughly where it is?",
                      "criteria": {"rise": "likely higher", "fall": "likely lower",
                                   "flat": "no clear direction"}}}
        a = decide.ask(user[:20000], q, purpose="story.decide").get("move") or {}
        payload["decision"] = {"choice": a.get("choice"), "confidence": a.get("confidence"),
                               "probabilities": a.get("probabilities")}
    except Exception as exc:  # noqa: BLE001 - the model's read stands on its own
        payload["decision"] = {"error": str(exc)[:120]}
    return payload


def read_new(limit: int = READS_PER_RUN) -> dict:
    """Story reads for the newest unread calls: those that pass first, then
    names you hold, then by score."""
    from brain.client import BudgetExceeded

    rows = db.query("""SELECT * FROM calls WHERE story IS NULL
                         AND signal_date > current_date - 7
                       ORDER BY passed_gate DESC, (detail->>'held')::boolean DESC NULLS LAST,
                                score DESC NULLS LAST
                       LIMIT %s""", (limit,))
    done, failed = 0, 0
    for call in rows:
        try:
            s = read(call)
        except BudgetExceeded as exc:
            log.warning("story reads stopped: %s", exc)
            break
        except Exception as exc:  # noqa: BLE001 - one bad read does not stop the rest
            log.warning("story read %s %s failed: %s", call["symbol"], call["strategy"], exc)
            failed += 1
            continue
        db.execute("UPDATE calls SET story = %s, story_at = now() WHERE id = %s",
                   (json.dumps(s, default=str), call["id"]))
        done += 1
    return {"read": done, "failed": failed, "waiting": len(rows) - done - failed}


def cycle() -> dict:
    """Record the latest run's signals, settle every open call, read the new ones."""
    return {"recorded": record(), "settled": settle(), "story": read_new()}


# ───────────────────────────────── scorecard ────────────────────────────────
def scorecard() -> dict:
    """What the calls did, by gate and by Story read."""
    from advisor.simulate import summarize

    rows = db.query("""SELECT strategy, region, passed_gate, regime_on, r,
                              story->>'direction' AS direction
                       FROM calls WHERE status = 'closed' AND r IS NOT NULL""")
    counts = {r["status"]: r["n"] for r in db.query(
        "SELECT status, count(*) AS n FROM calls GROUP BY status")}

    def group(key) -> dict:
        out: dict = {}
        for r in rows:
            out.setdefault(key(r), []).append(float(r["r"]))
        return {k: summarize(v) for k, v in out.items()}

    return {
        "counts": counts,
        "all": summarize([float(r["r"]) for r in rows]),
        "by_gate": group(lambda r: "passes" if r["passed_gate"] else "fails"),
        "by_story": group(lambda r: r["direction"] or "not read"),
        "by_strategy": group(lambda r: (r["strategy"], r["region"])),
    }


# ───────────────────────────── tickets and pages ────────────────────────────
def for_ticket(t: dict, region: str) -> dict | None:
    """The Story read for a ticket about to be emailed: recorded as a call
    (if the run has not already), read now if it has not been."""
    from brain.client import BudgetExceeded

    ev = t.get("evidence") or {}
    keep = {"display": t.get("display"), "label": t.get("label"), "broker": t.get("broker"),
            "currency": t.get("currency"), "stop_pct": t.get("stop_pct"),
            "quantity": t.get("quantity"), "reasons": t.get("reasons"),
            "median_days": ev.get("median_days"), "expectancy_r": ev.get("expectancy_r"),
            "strategy_passed": True, "market_sending": True}
    db.execute(
        """INSERT INTO calls (region, symbol, strategy, signal_date, limit_price, stop_price,
                              target_price, passed_gate, regime_on, score, detail)
           VALUES (%s,%s,%s,%s,%s,%s,%s,true,true,%s,%s)
           ON CONFLICT (symbol, strategy, signal_date) DO NOTHING""",
        (region, t["symbol"], t["strategy"], t["signal_date"], t["limit"], t["stop"],
         t.get("target"), t.get("mom_rank"), json.dumps(keep, default=str)))
    call = db.one("SELECT * FROM calls WHERE symbol=%s AND strategy=%s AND signal_date=%s",
                  (t["symbol"], t["strategy"], t["signal_date"]))
    if call["story"]:
        return call["story"]
    try:
        s = read(call)
    except BudgetExceeded:
        return None
    db.execute("UPDATE calls SET story = %s, story_at = now() WHERE id = %s",
               (json.dumps(s, default=str), call["id"]))
    return s


def attach(rows: list[dict]) -> list[dict]:
    """Put each candidate's Story read (if any) on it, for the pages."""
    keys = [(r["symbol"], r["strategy"], (r.get("detail") or {}).get("as_of")) for r in rows]
    keys = [k for k in keys if k[2]]
    if not keys:
        return rows
    found = {(c["symbol"], c["strategy"], str(c["signal_date"])): c["story"] for c in db.query(
        """SELECT symbol, strategy, signal_date, story FROM calls
           WHERE story IS NOT NULL AND (symbol, strategy, signal_date::text) IN
                 (SELECT * FROM unnest(%s::text[], %s::text[], %s::text[]))""",
        ([k[0] for k in keys], [k[1] for k in keys], [str(k[2]) for k in keys]))}
    for r in rows:
        r["story"] = found.get((r["symbol"], r["strategy"], str((r.get("detail") or {}).get("as_of"))))
    return rows
