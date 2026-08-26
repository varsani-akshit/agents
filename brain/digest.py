"""The 6-hour deep analysis cycle — the system's primary output.

Inputs: the computed stats pack, new documents since the last cycle, the prior
world model, and the principles corpus. The agent may also search memory and the
live web mid-analysis. Side effects: writes the digest, updates the world model,
and extracts relationship edges.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import config
import db
from brain import agent, client, extract
from memory import store, world_model
from signals import stats

log = logging.getLogger("mia.digest")

ROLE = """You are MIA, a senior macro strategist running continuously for one
sophisticated individual investor. Your domain is the intersection of precious
metals, fiat currencies, sovereign debt, central bank policy, and crypto.

You are not a news summariser. Your value is in relationships the reader would
not spot alone: a correlation that broke, a move inconsistent with its usual
driver, a policy action whose second-order effect lands three assets away. Lead
with those.

The principles below are your standing analytical frame. The stats pack is your
only source of numbers. Apply the analysis discipline strictly — especially the
rule that every quantitative claim comes from a tool result, never from memory."""

FORMAT = """Write the digest in this exact structure, in markdown.

Length discipline: this is a briefing for someone who reads four of these a day.
Aim for 700-1000 words total. Every sentence must carry information the reader
does not already have. Cut throat-clearing, cut restatement, cut any section that
has nothing in it this cycle down to a single line. A short digest on a quiet day
is correct output, not laziness.


## Bottom Line
Two or three sentences. What changed, what it means, what to watch. If nothing
material changed, say so plainly — that is a valid and useful finding.

## Prices
The moves that matter, with measured numbers. Skip instruments that did nothing.
For yields use basis points, not percent.

## Key Developments
New information since the last cycle that has macro consequence. Cite sources by
name and note tier where credibility matters. Omit routine churn entirely.

## Correlations & Anomalies
The statistical heart of the digest. Which relationships are holding, which
broke, and what a break implies. Quote measured correlation values. If the pack
shows no anomalies, say the relationships are behaving normally and name the one
worth watching.

## Framework View
Map the current picture onto the debt-cycle / debasement / repression frames.
Which of the four levers is being pulled? What regime does the evidence support?

## Tensions
Where credible sources or the data disagree. State both sides, say which the
data favours and by how much, and name the observation that would resolve it.
Omit this section only if there is genuinely no tension.

## Scenarios
Two or three forward paths with rough likelihoods and the leading indicator that
would confirm each. These are scenarios, not predictions.

## Confidence
What you are confident about, what is uncertain, and what would change your mind.

---
After the digest, output a fenced block exactly like this:

```world_model
## Regime label
<a short label, e.g. "fiscal-dominance / debasement-led">

## What we believe about the current regime
<the standing view, updated with this cycle's evidence>

## Key relationships currently holding
<with measured values>

## Relationships currently broken or unusual
<with measured values, and what it would take to restore them>

## Open questions
<what the next cycle should resolve>

## Changed since last cycle
<explicitly state what you revised and why — or "no material revision">

## Confidence
<by area>
```
The world_model block replaces the previous one wholesale, so carry forward
anything still true. It is read by the next cycle as its starting point."""


def _build_prompt(hours: int) -> tuple[str, dict]:
    pack = stats.build(persist=True)
    prior = world_model.current_body()
    docs = store.recent_documents(hours=hours, limit=70)
    triggers = db.query(
        """SELECT rule, severity, symbol, detail, created_at FROM trigger_events
           WHERE created_at > now() - make_interval(hours => %s)
           ORDER BY created_at DESC LIMIT 25""",
        (hours,),
    )

    # Send a compact brief, not the whole pack. Everything omitted here is one
    # get_stats_pack call away, and the full pack would otherwise be re-billed on
    # every turn of the tool loop.
    slim = {
        "generated_at": pack.get("generated_at"),
        "performance": [
            {k: v for k, v in row.items()
             if k in ("symbol", "last", "chg_1d_pct", "chg_1d_bp", "chg_1w_pct",
                      "chg_1m_pct", "z_1d_move")}
            for row in pack.get("performance", [])
        ],
        "yield_curve": pack.get("yield_curve"),
        "ratios": pack.get("ratios"),
        "gold_in_currencies": pack.get("gold_in_currencies"),
        "correlation_flips": pack.get("correlation_flips"),
        "anomalies": pack.get("anomalies"),
        "lead_lag": pack.get("lead_lag"),
        "correlations_30d": pack.get("correlations", {}).get("30d", {}),
        "macro_keys": sorted((pack.get("macro") or {}).keys()),
        "_note": (
            "Compact brief. Full sections (macro levels, 90d/180d correlation "
            "matrices, intraday) available via get_stats_pack."
        ),
    }

    doc_lines = []
    for d in docs:
        if d.get("urgency") == "Low":
            continue
        doc_lines.append(
            f"- [{d.get('urgency') or '?'}|tier{d['source_tier']}] {d['title']}"
            f" — {d.get('summary') or ''} ({d['source']})"
        )

    user = f"""Run the {hours}-hour deep analysis cycle. Timestamp: {datetime.now(timezone.utc).isoformat()}

# Computed statistics (authoritative for all numbers)
```json
{json.dumps(slim, default=str)[:9000]}
```

# Code-detected trigger events this window
```json
{json.dumps([{k: v for k, v in t.items() if k != 'created_at'} for t in triggers], default=str)[:3000]}
```

# New documents since last cycle ({len(doc_lines)} non-Low of {len(docs)} total)
{chr(10).join(doc_lines[:60]) or "(nothing above Low urgency)"}

# Prior world model — reconcile your analysis against this
{prior}

---
Investigate before concluding. Use search_memory to check whether a development
is genuinely new or a continuation of something already tracked, and use
query_prices or get_stats_pack for any number you are unsure of.

Use web_search at least once every cycle, and specifically when:
- the stats pack shows a move the stored documents do not explain
- a tier-2 source makes a factual claim worth verifying against a primary one
- a scheduled event (data print, speech, auction, decision) fell inside this
  window and you need its actual outcome rather than the preview coverage
- the feeds' newest item on a live topic is more than a few hours old

The RSS corpus lags and is incomplete by construction; treating it as the whole
world is how this system would miss the thing that mattered. Cite what you find.

Then write the digest in the required format."""
    return user, pack


def _split_world_model(text: str) -> tuple[str, str | None, str | None]:
    """Separate the digest body from the trailing world_model block."""
    marker = "```world_model"
    if marker not in text:
        return text.strip(), None, None
    body, _, rest = text.partition(marker)
    wm = rest.split("```", 1)[0].strip()
    regime = None
    for line in wm.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            regime = line.strip()[:120]
            break
    return body.strip(), wm, regime


def run(hours: int = 8, extract_edges: bool = True, model: str | None = None,
        effort: str | None = None) -> dict:
    """Execute one deep-analysis cycle end to end."""
    world_model.ensure_seeded()
    user, pack = _build_prompt(hours)

    system = [
        {"type": "text", "text": ROLE},
        # Principles are stable across every cycle, so they sit behind a cache
        # breakpoint — repeated cycles read them at ~10% of input cost.
        {
            "type": "text",
            "text": agent.load_principles(),
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": FORMAT},
    ]

    provider = "openai" if (model or "").startswith(("gpt-", "o3", "o4")) else "anthropic"
    chosen_effort = effort or os.getenv("MIA_DIGEST_EFFORT", "low")

    if provider == "openai":
        from brain import agent_openai

        # The OpenAI loop takes a single system string, not Anthropic's block list.
        flat_system = "\n\n---\n\n".join(
            b["text"] if isinstance(b, dict) else str(b) for b in system
        )
        result = agent_openai.run_agent(
            system=flat_system,
            user_message=user,
            model=model,
            purpose="digest",
            max_turns=6,
            max_tokens=10000,
            effort=chosen_effort,
            use_web_search=True,
        )
    else:
        result = agent.run_agent(
            system=system,
            user_message=user,
            model=model or config.DIGEST_MODEL,
            purpose="digest",
            max_turns=6,
            max_tokens=10000,
            use_web_search=True,
            # Adaptive thinking bills as output tokens, so effort is the dominant
            # cost lever. Measured per digest: high ~$0.40, medium ~$0.33, low ~$0.26.
            effort=chosen_effort,
        )

    text = result["text"]
    if not text:
        return {"ok": False, "error": result.get("stopped"), "usd": result["usd"]}

    body, wm_block, regime = _split_world_model(text)
    title = f"6h Digest — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

    analysis_id = store.save_analysis(
        "digest",
        title,
        body,
        meta={
            "hours": hours,
            "tool_calls": result["tool_calls"],
            "turns": result["turns"],
            "usd": result["usd"],
            "stopped": result.get("stopped"),
            "regime": regime,
            "citations": result.get("citations", []),
            "model": model or config.DIGEST_MODEL,
            "provider": provider,
            "effort": chosen_effort,
            "provider": provider,
        },
    )

    wm_version = None
    if wm_block:
        wm_version = world_model.save(wm_block, regime=regime, analysis_id=analysis_id)

    edges = 0
    if extract_edges:
        try:
            edges = extract.run(hours=hours)
        except Exception as exc:  # noqa: BLE001
            log.warning("edge extraction failed: %s", exc)

    return {
        "ok": True,
        "analysis_id": analysis_id,
        "title": title,
        "body": body,
        "world_model_version": wm_version,
        "regime": regime,
        "edges_extracted": edges,
        "turns": result["turns"],
        "tool_calls": result["tool_calls"],
        "usd": result["usd"],
        "stopped": result.get("stopped"),
    }


def latest() -> dict | None:
    return db.one("SELECT * FROM analyses WHERE kind='digest' ORDER BY created_at DESC LIMIT 1")
