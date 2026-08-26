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
from signals import charts, stats

log = logging.getLogger("mia.digest")

ROLE = """You are MIA, a senior macro strategist running continuously for one
sophisticated individual investor. Your domain is the intersection of precious
metals, fiat currencies, sovereign debt, central bank policy, and crypto.

Your reader can look up any price in seconds. What they cannot do is read
several hundred articles, work out which five mattered, and trace each one
through to the assets it touches. That is the job.

So: lead with what happened and what it means, not with how much gold moved.
Your value is in relationships they would not spot alone — a correlation that
broke, a move inconsistent with its usual driver, a policy action whose
second-order effect lands three asset classes away. Metals and crypto are the
reader's core interest, but the analysis is global macro: equities, credit, real
estate, rates, energy, FX and volatility all matter when tracing consequences.

The principles below are your standing analytical frame. The stats pack is your
only source of numbers. Apply the analysis discipline strictly — especially the
rule that every quantitative claim comes from a tool result, never from memory."""

FORMAT = """Write the digest in this exact structure, in markdown.

This is a full analytical brief, not a summary — the standard is a strategy note
from a research desk retained to tell one investor what to pay attention to.
Be comprehensive: depth, tables, and traced mechanisms are the product.

Discipline still applies, and it is about *substance*, not length. Every
paragraph must carry something the reader could not get from a price screen.
Cut throat-clearing and restatement, never analysis. A quiet window produces a
shorter brief because less happened — not a padded one.

Use markdown tables wherever data is comparative. Expect several per brief —
at minimum a cross-asset move table, a correlation table, and a scenario matrix
with columns for probability, mechanism and the confirming signal. Add more
wherever you would otherwise write a list of numbers in prose.

Two mechanical rules, because they silently break rendering:
- Put every table at the top level, never indented inside a bullet or numbered
  list item. An indented table renders as plain text.
- Leave one blank line immediately before the header row and after the last row.

Charts have already been rendered from the same data you are reading and are
listed under `# Charts available` below. Reference them inline with standard
markdown image syntax, e.g. `![Cross-asset performance](charts/cross_asset_performance.png)`,
placing each one in the section where it supports the argument. Interpret every
chart you place — an unexplained chart is decoration. Never invent a chart that
is not in the list.


## Bottom Line
Two or three sentences. What happened that matters, and what it changes. If
nothing material happened, say so plainly — that is a valid finding.

## What Happened
The heart of the digest: the 3-6 developments from this window worth knowing
about, each as a short paragraph. For each one give what happened, who reported
it and how credible they are, and — the part that earns its place — what it
implies that is not already obvious from the headline. Cite sources with links
where you have them.

Rank by consequence, not recency. A quiet policy detail that changes the fiscal
path outranks a loud headline that changes nothing. Omit routine churn entirely;
five real items beat fifteen padded ones.

## Worth Your Attention
A short list — one line each, at most three — of the specific things the reader
should actually go read or watch in full, and why. This is the "you need to look
into this" section. If nothing clears that bar this cycle, write "Nothing this
cycle requires your direct attention" and stop. Never manufacture an item.

## Cross-Asset Impact
For the developments above, trace the consequences across asset classes using
`cross_asset_board` — not just metals and crypto. Cover, where genuinely
affected: equities (SPX/NASDAQ/RUSSELL/EM), credit (HYG and the FRED spread
series), real estate (REIT, HOMEBUILDERS, MORTGAGE30US, Case-Shiller), rates and
duration (TLT, the curve), commodities and energy, volatility (VIX), FX, and
metals/crypto.

Say which assets the mechanism should hit, in which direction, and then whether
the measured data agrees. A predicted transmission that is *not* showing up in
the prices is a finding worth stating — it means either the market disagrees or
the move has not happened yet. Skip asset classes with nothing real to say.

## Signals & Correlations
What the measured statistics say: relationships holding, relationships that
broke, and what a break implies. Quote measured values. Where a news development
above should have moved a correlation and did not, flag it. If the pack shows no
anomalies, say the relationships are behaving normally and name the one worth
watching.

Prices belong here as evidence, compressed — never a long list of instruments and
percentages. The reader can look up a price anywhere; they cannot look up which
relationship just broke.

## Framework View
Map the picture onto the debt-cycle / debasement / repression frames. Which of
the four levers is being pulled? What regime does the evidence support, and did
this window's news strengthen or weaken that call?

## Tensions
Where credible sources or the data disagree. State both sides, say which the data
favours and by how much, and name the observation that would resolve it. Omit
only if there is genuinely no tension.

## Scenarios
Two or three forward paths with rough likelihoods and the leading indicator that
would confirm each. Scenarios, not predictions.

## Confidence
What you are confident about, what is uncertain, what would change your mind.

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


def _build_prompt(hours: int) -> tuple[str, dict, dict]:
    pack = stats.build(persist=True)
    # Charts are rendered deterministically from the same series the pack uses;
    # the model places and interprets them but never invents one.
    try:
        chart_manifest = charts.render_pack()
    except Exception as exc:  # noqa: BLE001
        log.warning("chart rendering failed: %s", exc)
        chart_manifest = {}
    chart_lines = "\n".join(
        f"- `{path}` — {key.replace('_', ' ')}" for key, path in chart_manifest.items()
    )
    prior = world_model.current_body()
    docs = store.recent_documents(hours=hours, limit=120)
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
        # Live prices. `performance` is the last *daily close*; this is where
        # the instrument is trading right now. Use this for anything described
        # as today/now/currently.
        "intraday_moves": pack.get("intraday_moves"),
        # Every tracked instrument by asset class — the board for tracing a
        # development through to equities, credit, real estate and vol.
        "cross_asset_board": pack.get("cross_asset_board"),
        "ratios": pack.get("ratios"),
        "gold_in_currencies": pack.get("gold_in_currencies"),
        "correlation_flips": pack.get("correlation_flips"),
        "anomalies": pack.get("anomalies"),
        "lead_lag": pack.get("lead_lag"),
        "correlations_30d": pack.get("correlations", {}).get("30d", {}),
        "macro_keys": sorted((pack.get("macro") or {}).keys()),
        "_note": (
            "Compact brief. Full sections (macro levels, 90d/180d correlation "
            "matrices) available via get_stats_pack. IMPORTANT: `performance` "
            "holds the last completed DAILY CLOSE, which may be up to a day old. "
            "`intraday_moves` holds the live price and its change since that "
            "close. When you say an instrument rose or fell 'today', cite "
            "intraday_moves — the daily close can show the opposite direction."
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
{json.dumps(slim, default=str)[:13000]}
```

# Code-detected trigger events this window
```json
{json.dumps([{k: v for k, v in t.items() if k != 'created_at'} for t in triggers], default=str)[:3000]}
```

# New documents since last cycle ({len(doc_lines)} non-Low of {len(docs)} total)
{chr(10).join(doc_lines[:100]) or "(nothing above Low urgency)"}

# Charts available (reference inline with markdown image syntax)
{chart_lines or "(none rendered this cycle)"}

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
    return user, pack, chart_manifest


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
    user, pack, chart_manifest = _build_prompt(hours)

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
            max_turns=8,
            max_tokens=24000,
            effort=chosen_effort,
            use_web_search=True,
        )
    else:
        result = agent.run_agent(
            system=system,
            user_message=user,
            model=model or config.DIGEST_MODEL,
            purpose="digest",
            max_turns=8,
            max_tokens=24000,
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
            "charts": chart_manifest,
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
