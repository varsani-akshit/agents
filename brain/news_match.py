"""Which articles are about the reader's own names — every article, every name.

After each harvest, every new article is put to Decision-1 once, with one
yes/no question per name the reader holds or has an open ticket on: "is this
article materially about this company?". Seventeen names cost one call and
about a second. Where a name matches, a second call asks whether the news is
good or bad for a shareholder. The brief's "Your money" and the Today page
read the matches; nothing here writes prose or touches prices.

A judgement, not a fact. Matching ("is this about the company?") is
reliable at the high threshold used here, so a passing mention of Coal India
in an oil story does not count. The good/bad read is a hint: it is shown with
the name's measured price move that session, and the brief treats matches as
leads to verify.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

import db
from brain import decide

log = logging.getLogger("mia.news_match")

THRESHOLD = 0.85        # p(article is materially about the name)
MAX_PER_RUN = 400       # articles per harvest; a backlog drains over a few runs
WINDOW_HOURS = 48       # only judge recent news — older is history, not news
WORKERS = 6

# Quote pages and data widgets that the personal news searches also return:
# about the name, but not news. Marked judged without a call.
_NOT_NEWS = re.compile(
    r"share price|stock price|price today|live (?:nse|bse|asx)|stock quote|quotes \d|"
    r"shares issued under|price chart|market cap(?:italisation)? today|technical analysis",
    re.I)


# ───────────────────────────────── the names ────────────────────────────────
def names() -> dict[str, str]:
    """{yfinance ticker: company name} for holdings and open tickets."""
    from advisor import bare
    from advisor import bars as adv_bars

    tickers = set(adv_bars.holdings())
    tickers |= {r["symbol"] for r in db.query(
        "SELECT symbol FROM ideas WHERE status IN ('open', 'filled')")}
    out = {}
    for t in sorted(tickers):
        out[t] = _name(t) or bare(t)
    return out


def _name(ticker: str) -> str | None:
    """The company's name: the index universe first, then a one-time lookup."""
    alt = ticker.replace(".BO", ".NS")
    row = db.one("SELECT name FROM securities WHERE symbol IN (%s, %s) AND name IS NOT NULL LIMIT 1",
                 (ticker, alt)) or db.one("SELECT name FROM name_cache WHERE symbol=%s", (ticker,))
    if row:
        return row["name"]
    try:
        import yfinance as yf

        info = yf.Ticker(alt if alt.endswith(".NS") else ticker).info or {}
        name = info.get("longName") or info.get("shortName")
    except Exception:  # noqa: BLE001 - a missing name falls back to the ticker
        name = None
    if name:
        db.execute("INSERT INTO name_cache (symbol, name) VALUES (%s, %s) "
                   "ON CONFLICT (symbol) DO UPDATE SET name = EXCLUDED.name", (ticker, name))
    return name


# ──────────────────────────────── the judging ───────────────────────────────
def _question(name: str, ticker: str) -> dict:
    from advisor import bare

    return {"type": "noul",
            "instructions": f"Is this article materially about {name} ({bare(ticker)}) — the "
                            "company itself, its shares, its results or its business? A passing "
                            "mention in a story about something else does not count."}


def judge(doc: dict, roster: dict[str, str]) -> list[dict]:
    """Matches for one article: [{symbol, name, p_about, impact, impact_p}]."""
    state = f"{doc['title']}\n\n{(doc.get('summary') or doc.get('body') or '')[:2500]}"
    keys = {f"n{i}": t for i, t in enumerate(roster)}
    answers = decide.ask(state, {k: _question(roster[t], t) for k, t in keys.items()},
                         purpose="news_match.about")
    hits = []
    for k, t in keys.items():
        p = float((answers.get(k) or {}).get("noul") or 0)
        if p >= THRESHOLD:
            hits.append({"symbol": t, "name": roster[t], "p_about": p})
    if hits:
        qs = {f"i{i}": {"type": "choice",
                        "instructions": f"For a shareholder of {h['name']}, is this news good, "
                                        "bad, or neither?",
                        "criteria": {"positive": "likely good for the shareholder",
                                     "negative": "likely bad for the shareholder",
                                     "neutral": "no clear effect either way"}}
              for i, h in enumerate(hits)}
        impact = decide.ask(state, qs, purpose="news_match.impact")
        for i, h in enumerate(hits):
            a = impact.get(f"i{i}") or {}
            h["impact"] = a.get("choice")
            h["impact_p"] = a.get("confidence")
    return hits


def run(limit: int = MAX_PER_RUN) -> dict:
    """Judge every unjudged recent article against the reader's names."""
    if not decide.available():
        return {"skipped": "Azure not configured"}
    roster = names()
    if not roster:
        return {"skipped": "no holdings or tickets"}
    docs = db.query(
        """SELECT id, title, summary, body FROM documents
           WHERE matched_at IS NULL
             AND coalesce(published_at, fetched_at) > now() - make_interval(hours => %s)
           ORDER BY coalesce(published_at, fetched_at) DESC LIMIT %s""",
        (WINDOW_HOURS, limit))

    def one(doc):
        if _NOT_NEWS.search(doc["title"] or ""):
            return doc["id"], [], None
        try:
            return doc["id"], judge(doc, roster), None
        except Exception as exc:  # noqa: BLE001 - one article must not stop the run
            return doc["id"], [], exc

    judged = matched = failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for doc_id, hits, err in ex.map(one, docs):
            if err:
                failed += 1
                continue
            judged += 1
            for h in hits:
                matched += 1
                db.execute(
                    """INSERT INTO news_matches (doc_id, symbol, name, p_about, impact, impact_p)
                       VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (doc_id, symbol) DO NOTHING""",
                    (doc_id, h["symbol"], h["name"], h["p_about"], h.get("impact"), h.get("impact_p")))
            db.execute("UPDATE documents SET matched_at = now() WHERE id = %s", (doc_id,))
    if failed:
        log.warning("news_match: %d of %d articles failed", failed, len(docs))
    return {"names": len(roster), "judged": judged, "matched": matched, "failed": failed}


def recent(hours: int = 48, per_name: int = 4) -> list[dict]:
    """The matches the brief and Today read: newest first, a few per name."""
    rows = db.query(
        """SELECT m.symbol, m.name, m.p_about, m.impact, m.impact_p, d.id AS doc_id, d.title,
                  d.url, d.source, d.source_tier, coalesce(d.published_at, d.fetched_at) AS at
           FROM news_matches m JOIN documents d ON d.id = m.doc_id
           WHERE coalesce(d.published_at, d.fetched_at) > now() - make_interval(hours => %s)
           ORDER BY at DESC""", (hours,))
    out, seen = [], {}
    titles = set()
    moves: dict[str, object] = {}
    for r in rows:
        # The same story from two outlets: news searches append " - Publisher".
        key = (r["symbol"], re.sub(r"\s+-\s+[^-]{2,60}$", "", r["title"]).lower()[:80])
        if key in titles or seen.get(r["symbol"], 0) >= per_name or _NOT_NEWS.search(r["title"]):
            continue
        titles.add(key)
        seen[r["symbol"]] = seen.get(r["symbol"], 0) + 1
        r["move_pct"] = session_move(r["symbol"], r["at"], moves)
        out.append(r)
    return out


def session_move(symbol: str, at, cache: dict) -> float | None:
    """The name's measured % move on the session the article landed in.

    Decision-1's good/bad read is a model's judgement and occasionally wrong
    ("Nifty IT slips 4%; TCS tanks" came back as good news). The price is not
    a judgement, so it is shown beside the read and the reader sees both.
    """
    import pandas as pd

    from advisor import bars

    if symbol not in cache:
        try:
            cache[symbol] = bars.load(symbol, years=1, final_only=False)["close"]
        except Exception:  # noqa: BLE001
            cache[symbol] = None
    close = cache[symbol]
    if close is None or len(close) < 2:
        return None
    day = pd.Timestamp(at).tz_convert(None).normalize() if pd.Timestamp(at).tzinfo else pd.Timestamp(at).normalize()
    after = close.index[close.index >= day]
    i = close.index.get_loc(after[0]) if len(after) else len(close) - 1
    if i == 0:
        return None
    return round((close.iloc[i] / close.iloc[i - 1] - 1) * 100, 2)
