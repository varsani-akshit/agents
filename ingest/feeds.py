"""RSS/Atom ingestion with per-source credibility tiers and dedup."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import feedparser
import httpx
import yaml
from bs4 import BeautifulSoup

import config
import db
from ingest.dedupe import canonical_url, content_hash, title_fingerprint

log = logging.getLogger("mia.feeds")
_CONF = config.CONFIG_DIR / "sources.yaml"

# Several official sites (federalreserve.gov, ecb.europa.eu) reject feedparser's
# default fetch — either on user-agent or on macOS's missing cert chain. Fetching
# bytes ourselves with httpx (certifi-backed, browser UA) and handing those to
# feedparser fixes both, and lets feedparser use its lenient parser on the way in.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HEADERS = {"user-agent": _UA, "accept": "application/rss+xml, application/xml, text/xml, */*"}


def load_sources() -> list[dict]:
    return yaml.safe_load(_CONF.read_text()) or []


def _clean_html(raw: str) -> str:
    if not raw:
        return ""
    text = BeautifulSoup(raw, "lxml").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _entry_time(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        val = getattr(entry, key, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def _entry_body(entry) -> str:
    parts = []
    if getattr(entry, "content", None):
        parts.extend(c.get("value", "") for c in entry.content)
    for key in ("summary", "description"):
        val = getattr(entry, key, None)
        if val:
            parts.append(val)
    return _clean_html(" ".join(parts))[:6000]


# Official bodies publish on a days-to-weeks cadence; news proxies publish
# continuously. One global cutoff would silently drop every central-bank release,
# so the retention window scales with tier. Dedup makes the long tier-1 window
# free after the first harvest.
# Hours of history to accept, per source tier. Tier 1 is generous because
# central banks publish on a days-to-weeks cadence and a 72-hour cutoff silently
# discarded every one of them. Tiers 2-4 hold a week so the corpus always covers
# the window a brief is summarising — and so a gap in ingestion is filled in
# rather than becoming a permanent hole. This has to be at least as wide as the
# `when:` window in the news-proxy queries in conf/sources.yaml, or the fetch
# pulls a week and the filter throws most of it away.
_AGE_BY_TIER = {1: 30 * 24, 2: 7 * 24, 3: 7 * 24, 4: 7 * 24}


def age_window(src: dict, override: int | None = None) -> int:
    if override is not None:
        return override
    return _AGE_BY_TIER.get(int(src.get("tier", 3)), 72)


def fetch_source(src: dict, max_age_hours: int | None = None) -> list[dict]:
    """Parse one feed into normalised document dicts. Never raises."""
    try:
        with httpx.Client(timeout=25, headers=_HEADERS, follow_redirects=True) as client:
            resp = client.get(src["url"])
        if resp.status_code != 200:
            log.warning("feed %s HTTP %s", src["name"], resp.status_code)
            return []
        parsed = feedparser.parse(resp.content)
    except Exception as exc:  # noqa: BLE001
        log.warning("feed %s failed: %s", src["name"], exc)
        return []
    if not parsed.entries:
        log.warning("feed %s returned 0 entries", src["name"])
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=age_window(src, max_age_hours))
    docs = []
    for entry in parsed.entries[:60]:
        title = _clean_html(getattr(entry, "title", ""))
        if not title:
            continue
        published = _entry_time(entry)
        if published and published < cutoff:
            continue
        url = canonical_url(getattr(entry, "link", ""))
        body = _entry_body(entry)
        docs.append(
            {
                "url": url or None,
                "title": title[:500],
                "body": body,
                "source": src["name"],
                "source_tier": int(src.get("tier", 3)),
                "official": bool(src.get("official", False)),
                "published_at": published,
                "content_hash": content_hash(title, body, url),
            }
        )
    return docs


def fetch_all(max_age_hours: int | None = None, workers: int = 8) -> list[dict]:
    sources = load_sources()
    docs: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda s: fetch_source(s, max_age_hours), sources):
            docs.extend(result)
    return docs


def store(docs: list[dict]) -> dict:
    """Insert new documents. Returns counts and the ids of genuinely new rows."""
    if not docs:
        return {"seen": 0, "inserted": 0, "new_ids": []}

    # In-batch dedup by loose title fingerprint, preferring the most credible source.
    best: dict[str, dict] = {}
    for d in docs:
        fp = title_fingerprint(d["title"])
        cur = best.get(fp)
        if cur is None or d["source_tier"] < cur["source_tier"]:
            best[fp] = d

    new_ids: list[int] = []
    with db.conn() as c, c.cursor() as cur:
        for d in best.values():
            cur.execute(
                # Untargeted DO NOTHING: `documents` is unique on both
                # content_hash and url, and a targeted clause can only cover one.
                # A re-published story keeps its URL while its hash changes, so
                # targeting content_hash alone raises on the url constraint.
                """INSERT INTO documents
                     (url,content_hash,title,body,source,source_tier,published_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING
                   RETURNING id""",
                (
                    d["url"],
                    d["content_hash"],
                    d["title"],
                    d["body"],
                    d["source"],
                    d["source_tier"],
                    d["published_at"],
                ),
            )
            row = cur.fetchone()
            if row:
                new_ids.append(row[0])
    return {"seen": len(docs), "deduped": len(best), "inserted": len(new_ids),
            "new_ids": new_ids}


def harvest(max_age_hours: int | None = None) -> dict:
    docs = fetch_all(max_age_hours)
    result = store(docs)
    log.info("feeds: %s seen, %s inserted", result["seen"], result["inserted"])
    return result


def source_health() -> list[dict]:
    """Per-source item counts over the last 7 days — used by `mia status`."""
    return db.query(
        """SELECT source, source_tier, count(*) AS docs, max(fetched_at) AS last_seen
           FROM documents WHERE fetched_at > now() - interval '7 days'
           GROUP BY source, source_tier ORDER BY source_tier, docs DESC"""
    )
