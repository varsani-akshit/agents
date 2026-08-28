"""Re-render a stored brief in the current format.

This restructures analysis that already exists. It does not re-research: the
documents behind an older brief have often aged out under retention, and the
prices have moved on, so asking the model to redo the work would produce a brief
about today wearing yesterday's timestamp. Reformatting is the honest operation
— same findings, current structure.

Everything the original established is preserved. Nothing is added, because
anything added here would be invention rather than analysis.
"""
from __future__ import annotations

import logging

from brain import digest, router

log = logging.getLogger("alfred.rewrite")

SYSTEM = """You restructure an existing macro brief into a newer format.

This is a formatting and editing task, not an analytical one. Every finding,
number, source and judgement in the original must survive into the rewrite.

Strict rules:
- Invent nothing. No new figures, no new claims, no new sources. If the original
  does not establish something, it does not appear.
- Change no number, and no direction of any claim.
- Keep every URL exactly as it appears; never construct one.
- Where the original lacks material for a section, omit that section rather than
  filling it.

What you may do: reorganise into the required topic sections, write the headline
and standfirst, give each development a title, add cross-references between
sections where the original's own content supports them, and bring the register
into line — formal international English, no American colloquialism, British
spelling except for fixed proper nouns."""


def rewrite_brief(row: dict, dry_run: bool = False) -> dict:
    """Rewrite one stored brief. Returns headline, word count and cost."""
    original = row["body"]
    meta = row["meta"] or {}
    when = row["created_at"].strftime("%d %B %Y, %H:%M UTC")

    user = f"""Restructure the brief below into the required format.

It was written at {when}. Its regime label was: {meta.get('regime') or 'unrecorded'}

# Required format
{digest.FORMAT}

# The original brief
{original}"""

    # The same single writer that produces new briefs re-presents old ones, so
    # the archive reads in one voice. The Verifier deliberately does NOT run
    # here: it audits against today's database, and "correcting" last month's
    # true prices to today's would falsify the record, not fix it.
    text, spec, usd = router.complete_text(
        "premium", premium_site="editor",
        system=SYSTEM,
        user=user,
        purpose="rewrite",
        max_tokens=24000,
    )

    body, wm, regime = digest._split_world_model(text)
    body, headline, standfirst = digest._split_headline(body)

    result = {
        "headline": headline or row["title"],
        "standfirst": standfirst,
        "words": len(body.split()),
        "usd": usd,
        "body": body,
    }
    if dry_run:
        return result

    import db

    new_meta = dict(meta)
    new_meta |= {
        "headline": headline,
        "standfirst": standfirst,
        "words": len(body.split()),
        # Recorded so it is always clear which briefs were re-presented rather
        # than originally written in this format.
        "rewritten": True,
        "rewrite_model": spec,
        "regime": regime or meta.get("regime"),
    }
    import json as _json

    db.execute(
        "UPDATE analyses SET title=%s, body=%s, meta=%s WHERE id=%s",
        (headline or row["title"], body, _json.dumps(new_meta, default=str), row["id"]),
    )
    log.info("rewrote analysis %s: %s", row["id"], headline)
    return result
