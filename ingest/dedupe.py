"""URL canonicalisation and content hashing. Pure functions, no I/O."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlparse, urlunparse

_TRACKING = re.compile(
    r"^(utm_|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|guccounter|__twitter|igshid|s_cid)",
    re.I,
)
_WS = re.compile(r"\s+")


def canonical_url(url: str) -> str:
    """Strip tracking params, unwrap Google News redirects, normalise casing."""
    if not url:
        return ""
    url = url.strip()
    parsed = urlparse(url)

    # Google News wraps the real link in ?url= on some feed shapes.
    if "news.google.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "url" in qs and qs["url"]:
            return canonical_url(qs["url"][0])

    keep = [
        (k, v)
        for k, v in parse_qs(parsed.query, keep_blank_values=False).items()
        if not _TRACKING.match(k)
    ]
    query = "&".join(f"{k}={v[0]}" for k, v in sorted(keep))
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    url = urlunparse((parsed.scheme.lower() or "https", netloc, path, "", query, ""))

    # Postgres cannot index a btree row past ~2704 bytes, and some feeds emit
    # URLs carrying an entire encoded payload in the query string. One such
    # link aborted the whole ingestion transaction — and with it the tick that
    # also refreshes prices and evaluates triggers — 48 times in a day. A link
    # this long has a unusable query anyway, so it is cut back to its path.
    if len(url.encode("utf-8")) > 1800:
        url = urlunparse((parsed.scheme.lower() or "https", netloc, path, "", "", ""))
        if len(url.encode("utf-8")) > 1800:
            url = url.encode("utf-8")[:1800].decode("utf-8", "ignore")
    return url


def normalise_text(text: str) -> str:
    return _WS.sub(" ", (text or "")).strip().lower()


def content_hash(title: str, body: str = "", url: str = "") -> str:
    """Stable identity for a document.

    Title+body is the primary signal so the same story syndicated to two URLs
    collapses to one row. URL is a fallback when body is empty.
    """
    basis = normalise_text(title) + "|" + normalise_text(body)[:2000]
    if not basis.strip("|"):
        basis = canonical_url(url)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def title_fingerprint(title: str) -> str:
    """Loose fingerprint: alphanumeric words only, sorted. Catches reworded headlines."""
    words = re.findall(r"[a-z0-9]+", normalise_text(title))
    stop = {"the", "a", "an", "of", "to", "in", "on", "for", "as", "at", "by", "is", "and"}
    keep = sorted(w for w in words if w not in stop and len(w) > 2)
    return hashlib.sha256(" ".join(keep).encode()).hexdigest()[:16]
