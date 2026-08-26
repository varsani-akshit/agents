"""User-contributed knowledge: paste a URL or raw text, get a corpus document.

The RSS sources are broad but not complete — a paywalled FT piece, a research
note, a thread worth keeping will never arrive by feed. This is the side door:
whatever the reader pastes is stored as a tier-1 document (user-curated is the
strongest credibility signal we have), embedded and classified immediately, and
from there it flows into search, the graph, and the next brief exactly like any
other document.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone

import httpx

import db

log = logging.getLogger("alfred.library")

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/127.0 Safari/537.36")


def _extract_readable(html_text: str) -> tuple[str, str]:
    """Title and body text from a page, article-first."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", root.get_text("\n", strip=True))
    return title[:300], text


def add(content: str, title: str = "", note: str = "") -> dict:
    """Ingest pasted content. Returns {'ok', 'doc_id'|'error', 'title'}.

    A URL is fetched and its article text extracted; anything else is stored as
    the text itself. The reader's note rides along at the top of the body so the
    analyst sees *why* this was worth saving, not just what it says.
    """
    content = (content or "").strip()
    if len(content) < 8:
        return {"ok": False, "error": "Nothing to add."}

    url = None
    if re.match(r"^https?://\S+$", content):
        url = content
        try:
            resp = httpx.get(url, headers={"User-Agent": _UA},
                             follow_redirects=True, timeout=30)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Could not fetch that URL: {exc}"}
        page_title, body = _extract_readable(resp.text)
        title = title or page_title or url
        if len(body) < 200:
            return {"ok": False,
                    "error": "That page yielded almost no text — likely a paywall. "
                             "Paste the article text instead."}
    else:
        body = content
        title = title or (content.split("\n", 1)[0][:120])

    if note:
        body = f"[Reader's note: {note}]\n\n{body}"
    body = body[:60000]

    content_hash = hashlib.sha256(body.encode()).hexdigest()
    row = db.one(
        """INSERT INTO documents
             (url, content_hash, title, body, source, source_tier, published_at)
           VALUES (%s, %s, %s, %s, 'user-added', 1, %s)
           ON CONFLICT DO NOTHING
           RETURNING id""",
        (url, content_hash, title, body, datetime.now(timezone.utc)),
    )
    if not row:
        return {"ok": False, "error": "Already in the knowledge base."}

    # Embed and classify now rather than waiting for the next tick: the reader
    # is watching, and "added" should mean usable.
    embedded = classified = False
    try:
        from memory import store

        store.embed_documents(limit=10)
        embedded = True
    except Exception as exc:  # noqa: BLE001
        log.warning("immediate embed failed (next tick will retry): %s", exc)
    try:
        from brain import classify

        classify.run(batch=5, max_batches=1, workers=1)
        classified = True
    except Exception as exc:  # noqa: BLE001
        log.warning("immediate classify failed (next tick will retry): %s", exc)

    return {"ok": True, "doc_id": row["id"], "title": title,
            "embedded": embedded, "classified": classified,
            "chars": len(body)}


def recent(limit: int = 20) -> list[dict]:
    return db.query(
        """SELECT id, title, url, urgency, summary, fetched_at
           FROM documents WHERE source='user-added'
           ORDER BY fetched_at DESC LIMIT %s""",
        (limit,),
    )
