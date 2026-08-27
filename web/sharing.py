"""Share links: one brief, one unguessable URL, no account required.

A recipient of a share link sees exactly that brief and nothing else — no
navigation, no archive, no Ask box, no other briefs. The token is the only
credential, so it is generated from a cryptographic source and is long enough
that guessing is not a realistic attack on a personal dashboard.

Links are revocable and counted. Counting matters more than it sounds: it is the
only way to know whether a link you sent was ever opened, and the only way to
notice a link circulating more widely than intended.
"""
from __future__ import annotations

import logging
import secrets

import db

log = logging.getLogger("alfred.sharing")

# 32 hex characters, 128 bits. Long enough that enumeration is not a threat,
# short enough to paste into a message without wrapping.
_TOKEN_BYTES = 16


def create(analysis_id: int, created_by: str | None = None) -> str:
    """Mint a share link, reusing any live one for the same brief.

    Reuse is deliberate: pressing Share twice should not quietly leave two valid
    links in circulation, only one of which you remember to revoke.
    """
    existing = db.one(
        """SELECT token FROM brief_shares
           WHERE analysis_id = %s AND revoked_at IS NULL
           ORDER BY created_at DESC LIMIT 1""",
        (analysis_id,),
    )
    if existing:
        return existing["token"]

    token = secrets.token_hex(_TOKEN_BYTES)
    db.execute(
        "INSERT INTO brief_shares (token, analysis_id, created_by) VALUES (%s,%s,%s)",
        (token, analysis_id, created_by),
    )
    log.info("share link created for analysis %s", analysis_id)
    return token


def resolve(token: str) -> dict | None:
    """The brief behind a token, or None if unknown or revoked."""
    if not token or len(token) != _TOKEN_BYTES * 2:
        return None
    row = db.one(
        """SELECT a.id, a.title, a.body, a.meta, a.created_at, s.token
           FROM brief_shares s JOIN analyses a ON a.id = s.analysis_id
           WHERE s.token = %s AND s.revoked_at IS NULL""",
        (token,),
    )
    if row:
        db.execute(
            """UPDATE brief_shares
               SET view_count = view_count + 1, last_viewed = now()
               WHERE token = %s""",
            (token,),
        )
    return row


def revoke(token: str) -> bool:
    return bool(db.execute(
        "UPDATE brief_shares SET revoked_at = now() WHERE token = %s AND revoked_at IS NULL",
        (token,),
    ))


def for_analysis(analysis_id: int) -> dict | None:
    return db.one(
        """SELECT token, created_at, view_count, last_viewed FROM brief_shares
           WHERE analysis_id = %s AND revoked_at IS NULL
           ORDER BY created_at DESC LIMIT 1""",
        (analysis_id,),
    )
