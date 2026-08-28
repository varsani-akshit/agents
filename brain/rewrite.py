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
    """Re-present one stored brief. Returns headline, word count and cost.

    This is not a separate agent: it is the brief pipeline's editor stage,
    re-triggered on stored content — so it traces as a `brief` run in
    re-present mode, with only the compose sub-step underneath. The research
    stages are absent because their inputs no longer exist, not because a
    different agent ran.
    """
    from brain import observe

    with observe.run("brief", trigger="manual",
                     meta={"mode": "re-present", "analysis_id": row["id"]}) as rec:
        rec.set_input({"analysis_id": row["id"], "old_title": row["title"]})
        result = _rewrite(row, dry_run=dry_run)
        rec.set_output({k: v for k, v in result.items() if k != "body"})
        return result


def _rewrite(row: dict, dry_run: bool = False) -> dict:
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
    from brain import observe

    with observe.stage("compose", kind="generic",
                       input={"analysis_id": row["id"], "written": when,
                              "original_words": len(original.split())}) as sp:
        text, spec, usd = router.complete_text(
            "premium", premium_site="editor",
            system=SYSTEM,
            user=user,
            purpose="rewrite",
            max_tokens=24000,
        )
        sp.set_attribute("model", spec)
        sp.set_output({"chars": len(text)})

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
