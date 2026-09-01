"""The five stages of a brief, each doing one job with full attention.

The architecture follows the pattern proven by deep-research systems: parallel,
context-isolated research; compression at every hand-off; exactly one writer.
A single model call doing gathering, measuring, interpreting and writing does
each at half depth — these stages exist so nothing is done at half depth.

  Marshal   writes the run brief: what this window demands attention on.
  Scout     per beat: grounded search + full-article reads → compressed leads.
  Analyst   per beat: leads + measured evidence + graph → titled findings.
  Editor    the single writer. gpt-5.4, once, sees everything.
  Verifier  every number in the draft checked against the database.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import db
from brain import agent_gemini, observe, router

log = logging.getLogger("mia.pipeline")


def _spend(purpose_prefix: str, since: datetime) -> float:
    row = db.one(
        "SELECT coalesce(sum(usd),0) s FROM api_calls WHERE purpose LIKE %s AND created_at >= %s",
        (purpose_prefix + "%", since),
    )
    return float(row["s"]) if row else 0.0


def _loose_json(text: str) -> dict | None:
    """Parse JSON out of agent prose: fenced, bare, or embedded."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    if text.startswith("{"):
        candidates.append(text)
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:text.rfind("}") + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:  # noqa: BLE001
            continue
    return None


# ─────────────────────────────────── marshal ────────────────────────────────
_MARSHAL_SCHEMA = {
    "type": "object",
    "properties": {
        "window_summary": {"type": "string"},
        "priorities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string"},
                    "why": {"type": "string"},
                    "beats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["focus", "why"],
            },
        },
        "beat_guidance": {
            "type": "object",
            "properties": {},
            "description": "beat key -> one sentence of specific direction",
        },
        "carry_forward": {"type": "string",
                          "description": "threads from the previous brief that need resolution"},
    },
    "required": ["window_summary", "priorities"],
}


def recent_coverage(limit: int = 3) -> str:
    """What the last briefs already told the reader.

    Without this the desk re-reports standing stories — the same central bank,
    the same war — because they are permanently in the news. A brief is a
    despatch, not a summary of the world: it earns its place by what has
    changed since the last one.
    """
    rows = db.query(
        """SELECT title, body, created_at FROM analyses WHERE kind='digest'
           ORDER BY created_at DESC LIMIT %s""", (limit,))
    out = []
    for r in rows:
        titles = re.findall(r"^\*\*(.+?)\*\*\s*$", r["body"] or "", re.M)[:14]
        out.append(f"— {r['created_at'].strftime('%d %b %H:%M')} · {r['title']}\n  "
                   + "\n  ".join("· " + t for t in titles))
    return "\n".join(out) or "(no previous briefs)"


def marshal(*, hours: int, slim: dict, dlines: list[str], prior_wm: str,
            triggers: list[dict], last_headline: str | None) -> tuple[dict, str]:
    """The run brief: the north star every downstream agent is steered by."""
    user = f"""You are the Marshal of a macro research team. A brief covering the last
{hours} hours is about to be produced by beat Scouts and Analysts. Write their
marching orders.

# Already reported in recent briefs — do NOT commission these again
{recent_coverage()}

A story listed above returns only if it MOVED in this window: a decision taken,
a number printed, a threshold crossed. "Still tense", "continues to weigh",
"remains a risk" is not movement — it is the same despatch with a new date, and
the reader has read it. Prefer a development nobody has filed yet over a fuller
account of one they have. If a standing story genuinely advanced, the priority
must name the advance, not the story.

Previous brief headline: {last_headline or '(none)'}

# Code-detected trigger events this window
{json.dumps([{k: str(v) for k, v in t.items()} for t in triggers], default=str)[:3000]}

# Measured state (compact)
{json.dumps({k: slim.get(k) for k in ('regime_score', 'intraday_moves', 'correlation_flips', 'anomalies', 'net_liquidity')}, default=str)[:8000]}

# Headlines this window ({len(dlines)})
{chr(10).join(dlines[:80])}

# Prior world model
{prior_wm[:4000]}

Set 3-6 priorities: the developments this window that most deserve deep
investigation, each with why and which beats it touches (keys: monetary,
fiscal, geopolitics, energy, equities, digital, currencies, real_assets).
Add one sentence of beat_guidance for any beat needing specific direction —
including what the previous brief left unresolved (carry_forward)."""
    payload, spec = router.complete_json(
        "deep", system="You direct macro research coverage. Be specific; name the "
        "development, not the topic.", user=user, schema=_MARSHAL_SCHEMA,
        purpose="brief.marshal", max_tokens=6000,
    )
    return payload, spec


def north_star_text(marshal_out: dict) -> str:
    lines = [f"Window summary: {marshal_out.get('window_summary', '')}", "Priorities:"]
    for p in marshal_out.get("priorities", []):
        lines.append(f"- {p.get('focus')} — {p.get('why')} (beats: {', '.join(p.get('beats') or [])})")
    if marshal_out.get("carry_forward"):
        lines.append(f"Carry-forward: {marshal_out['carry_forward']}")
    return "\n".join(lines)


# ─────────────────────────────────── scout ──────────────────────────────────
_SCOUT_SYSTEM = """You are a beat Scout on a macro research desk. Your beat is
defined below. You gather and compress; you do not interpret.

Method, in order:
1. Read the desk's marching orders and your beat charter.
2. Search the live web for what happened on your beat this window — at least
   two distinct searches, phrased for facts (decisions, data, statements), not
   commentary. Prefer primary and tier-1 sources.
3. For the two or three most load-bearing stories, fetch the article itself
   (fetch_url) and take real quotes and real numbers from it — the snippet is
   not evidence.
4. Check search_memory for whether a story is genuinely new or a continuation.

Novelty is the bar. The desk has already reported the stories listed in your
orders; a lead that restates one of them is worthless however well sourced.
Return a lead only if something happened in this window: a decision, a print, a
filing, a strike, a level broken. Ongoing situations qualify ONLY through their
new development, and the lead must lead with that development. If the beat
produced nothing new, return fewer leads — an empty beat is an honest report and
costs the desk nothing.

Then return ONLY a JSON object, no prose around it:
{"leads": [{
  "title": "finding stated as a sentence with its key number",
  "what_happened": "2-4 sentences of dated fact, no interpretation",
  "numbers": ["each figure you found, with source"],
  "quotes": [{"text": "verbatim quote", "who": "speaker/institution", "source_url": "..."}],
  "sources": [{"url": "...", "title": "...", "tier": "primary|wire|commentary"}],
  "novelty": "new|develops_prior|background",
  "significance": 1-5
}]}
3 to 8 leads. A quiet beat returns fewer leads, never padding. Only include a
URL you actually saw in a tool or search result."""


def scout(*, beat: dict, north_star: str, hours: int, dlines: list[str]) -> tuple[dict, dict]:
    """One beat's gathering pass. Returns (leads_payload, agent_result)."""
    user = f"""Marching orders from the Marshal:
{north_star}

# Already reported — a lead restating any of these is rejected
{recent_coverage(2)}

Your beat: {beat['section']}
Charter: {beat['charter']}
Window: the last {hours} hours, now {datetime.now(timezone.utc).isoformat()}.

Feed headlines already ingested (check these before searching — your search
should ADD to them, not repeat them):
{chr(10).join(dlines[:60]) or '(none)'}

Gather, verify, compress, return the JSON."""
    result = agent_gemini.run_agent(
        system=_SCOUT_SYSTEM,
        user_message=user,
        model="gemini-flash-latest",
        purpose=f"brief.scout.{beat['key']}",
        max_turns=6,
        max_tokens=10000,
        effort="low",
        use_web_search=True,
    )
    text = result.get("text") or ""
    payload = _loose_json(text)
    if payload is None and len(text) > 200:
        # The scout did the work but wrapped it badly. A reformat costs a
        # fraction of a cent; discarding a beat's research costs the beat.
        try:
            payload, _ = router.complete_json(
                "workhorse",
                system="Convert this research report into the requested JSON. "
                "Invent nothing; drop nothing that is there.",
                user=text[:20000] + '\n\nSchema: {"leads": [{"title", "what_happened", '
                '"numbers": [], "quotes": [{"text","who","source_url"}], '
                '"sources": [{"url","title","tier"}], "novelty", "significance"}]}',
                schema={"type": "object", "properties": {"leads": {"type": "array"}},
                        "required": ["leads"]},
                purpose=f"brief.scout.{beat['key']}.rescue", max_tokens=8000,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("scout %s rescue failed: %s", beat["key"], exc)
    payload = payload or {"leads": [], "raw_text": text[:4000]}
    leads = payload.get("leads") or []
    # Attach grounding citations the scout saw but may not have inlined.
    payload["citations"] = result.get("citations", [])
    payload["searched"] = True
    log.info("scout %s: %d leads, $%.4f", beat["key"], len(leads), result.get("usd", 0))
    return payload, result


# ─────────────────────────────────── analyst ────────────────────────────────
_ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string",
                              "description": "the finding as a sentence, number first where one exists"},
                    "analysis": {"type": "string",
                                 "description": "2-5 sentences: mechanism, evidence, the non-obvious implication"},
                    "numbers_used": {"type": "array", "items": {"type": "string"}},
                    "chart_keys": {"type": "array", "items": {"type": "string"}},
                    "related_beats": {"type": "array", "items": {"type": "string"},
                                      "description": "other beat keys this connects to, with the connection stated in analysis"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "rank": {"type": "integer", "description": "1 = most consequential on this beat"},
                },
                "required": ["title", "analysis", "rank"],
            },
        },
        "beat_summary": {"type": "string", "description": "one sentence: what this window meant on this beat"},
    },
    "required": ["findings", "beat_summary"],
}


def analyst(*, beat: dict, leads: dict, north_star: str, slim: dict,
            graph_ctx: dict, prior_wm: str, chart_keys: list[str]) -> tuple[dict, str]:
    """One beat's interpretation pass: leads + measured evidence → findings."""
    beat_stats = {
        "performance": [r for r in slim.get("performance", [])
                        if r.get("symbol") in set(beat.get("series", []))],
        "intraday": {k: v for k, v in (slim.get("intraday_moves") or {}).items()
                     if k in set(beat.get("series", []))} if isinstance(slim.get("intraday_moves"), dict)
                    else slim.get("intraday_moves"),
        "regime_score": slim.get("regime_score"),
        "correlation_flips": slim.get("correlation_flips"),
        "anomalies": slim.get("anomalies"),
    }
    user = f"""Marching orders:
{north_star}

Your beat: {beat['section']}

# Scout's leads (gathered and sourced this window)
{json.dumps(leads.get('leads', []), default=str)[:14000]}

# Measured evidence for your beat (authoritative for every number)
{json.dumps(beat_stats, default=str)[:9000]}

# Knowledge-graph neighbourhood (relationships already established)
{json.dumps(graph_ctx, default=str)[:4000]}

# Prior world model (reconcile against, don't repeat)
{prior_wm[:3500]}

# Chart keys you may cite in chart_keys
{', '.join(chart_keys)}

Turn the leads into findings. Each title states the conclusion with its
number. Analysis gives the mechanism and the implication that is not obvious
from the headline — and where the connection runs through ANOTHER beat, name
that beat in related_beats and state the connection. Discard leads that turned
out to be noise. British English, no American colloquialism."""
    payload, spec = router.complete_json(
        "deep",
        system="You are a beat Analyst on a macro desk. Numbers come only from "
        "the measured evidence or the sourced leads — never from memory. "
        "Interpretation is your job; invention is failure.",
        user=user, schema=_ANALYST_SCHEMA,
        purpose=f"brief.analyst.{beat['key']}", max_tokens=8000,
    )
    return payload, spec


# ─────────────────────────────────── editor ─────────────────────────────────
def editor(*, findings_by_beat: dict[str, dict], north_star: str, slim: dict,
           chart_lines: str, prior_wm: str, hours: int,
           citation_urls: list[dict], role: str, fmt: str) -> tuple[str, str, float]:
    """The single writer. Sees every beat's findings; writes the whole brief."""
    sections = []
    for key, payload in findings_by_beat.items():
        sections.append(f"## beat:{key} — {payload.get('beat_summary', '')}\n"
                        + json.dumps(payload.get("findings", []), default=str)[:9000])
    user = f"""Write the {hours}-hour brief from your desk's completed research.
Timestamp: {datetime.now(timezone.utc).isoformat()}

# Marching orders this window
{north_star}

# Findings, by beat (already verified against measured data; ranked)
{chr(10).join(sections)[:60000]}

# Compact measured state (for the Signals section and any number you add)
{json.dumps({k: slim.get(k) for k in ('performance', 'intraday_moves', 'yield_curve', 'regime_score', 'net_liquidity', 'correlation_flips', 'ratios', 'historical_analogues')}, default=str)[:20000]}

# Charts available — link in prose with the exact form shown
{chart_lines}

# Source URLs your researchers actually saw (cite only from this list)
{json.dumps(citation_urls, default=str)[:6000]}

# Prior world model — reconcile and then replace via the world_model block
{prior_wm[:5000]}

You are writing from completed research, not researching. Weave the findings
into the required structure: developments keep their titled-finding form, and
every cross-beat connection your Analysts flagged becomes an explicit
cross-reference between sections. Cut anything that does not earn its place.

Every development you file must rest on something that happened INSIDE this
window and be datable to it. A paragraph whose substance would have been equally
true last week does not belong in this brief, no matter how important the
subject: that is background, and the reader has it. Where a standing story
returns, the title states what advanced ("Bank of Japan lifts the policy rate to
0.75%"), never the standing condition ("Bank of Japan remains under pressure").

The word budget is a hard ceiling, not a target: with seven beats reporting you
will have more findings than fit, and the discipline is to DROP the weakest
developments entirely — never to compress everything until it all just fits.
A brief of 2,400 sharp words beats 3,200 thorough ones."""
    text, spec, usd = router.complete_text(
        "premium", premium_site="editor",
        system=role + "\n\n---\n\n" + fmt,
        user=user, purpose="brief.editor", max_tokens=30000,
    )
    return text, spec, usd


# ─────────────────────────────────── verifier ───────────────────────────────
_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string", "description": "exact text from the draft containing the wrong figure"},
                    "corrected_quote": {"type": "string", "description": "the same text with the figure corrected"},
                    "why": {"type": "string"},
                    "severity": {"type": "string", "enum": ["wrong", "imprecise", "unverifiable"]},
                },
                "required": ["quote", "why", "severity"],
            },
        },
        "checked_claims": {"type": "integer"},
    },
    "required": ["issues", "checked_claims"],
}


def verifier(*, body: str, slim: dict) -> tuple[dict, int]:
    """Audit every numeric claim in the draft against the measured data.

    Corrections apply only as exact string replacements — the verifier proposes,
    code disposes, and anything that does not match verbatim is flagged rather
    than rewritten. A model must never redraft another model's prose here.
    """
    payload, _ = router.complete_json(
        "reason",
        system="You audit numeric claims in a draft against measured data. "
        "Only figures the measured data can actually check: prices, changes, "
        "yields, correlations. Report ONLY discrepancies: a claim that matches "
        "the measured data is counted in checked_claims and NOT listed as an "
        "issue; a figure only an external source could confirm is severity "
        "'unverifiable' and listed only if it looks implausible. Copy quotes "
        "EXACTLY as they appear.",
        user=f"""# Draft
{body[:40000]}

# Measured data (authoritative)
{json.dumps({k: slim.get(k) for k in ('performance', 'intraday_moves', 'yield_curve', 'regime_score', 'net_liquidity', 'ratios')}, default=str)[:25000]}

List every numeric claim that contradicts the measured data. Tolerance: 2%
relative or one basis point on yields. checked_claims = how many you audited.""",
        schema=_VERIFY_SCHEMA, purpose="brief.verifier", max_tokens=6000,
    )
    fixed = 0
    for issue in payload.get("issues", []):
        q, c = issue.get("quote"), issue.get("corrected_quote")
        if issue.get("severity") == "wrong" and q and c and q != c:
            body, applied = _apply_correction(body, q, c)
            if applied:
                issue["applied"] = True
                fixed += 1
    payload["fixed"] = fixed
    return {"body": body, "audit": payload}, fixed


def _apply_correction(body: str, quote: str, corrected: str) -> tuple[str, bool]:
    """Replace one flagged passage, tolerating markdown drift in the quote.

    Models copy the words faithfully but shed the `**` and backticks around
    them, so a byte-exact match fails on precisely the passages most likely to
    be flagged — the bold numbers. The match therefore treats emphasis marks
    and whitespace as elastic, but still requires every word in order, and
    applies only when the passage occurs exactly once: an ambiguous match is
    left flagged rather than half-fixed.
    """
    if body.count(quote) == 1:
        return body.replace(quote, corrected, 1), True
    words = [w for w in re.split(r"\s+", re.sub(r"[*_`]", " ", quote)) if w]
    if len(words) < 3:  # too short to match safely without exact bytes
        return body, False
    pattern = r"[*_`]*" + r"[\s*_`]+".join(re.escape(w) for w in words) + r"[*_`]*"
    matches = list(re.finditer(pattern, body))
    if len(matches) != 1:
        return body, False
    m = matches[0]
    return body[: m.start()] + corrected + body[m.end():], True
