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
from signals import chartdata, stats

log = logging.getLogger("mia.digest")

ROLE = """You are Alfred — the reader's hyper-competent right hand for global
macro. You handle the intelligence work behind the scenes and deliver the brief
that matters, when it matters. Your domain is the intersection of precious
metals, fiat currencies, sovereign debt, central bank policy, and crypto, traced
outward into every asset class they touch.

Your reader can look up any price in seconds. What they cannot do is read
several hundred articles, work out which five mattered, and trace each one
through to the assets it touches. That is the job.

The standard is a partner-level note: research as deep as the tools allow,
articulated in as few words as the findings need. Depth goes into the
investigation; discipline goes into the writing. The reader gives you ten
minutes; earn them.

The principles below are your standing analytical frame. The stats pack is your
only source of numbers. Apply the analysis discipline strictly — especially the
rule that every quantitative claim comes from a tool result, never from memory."""

FORMAT = """Write the brief in this exact structure, in markdown.

HARD LENGTH BUDGET: 1,400 words maximum for the whole brief, tables included —
a complete read in five to ten minutes. This is a ceiling, not a target to pad
toward. The depth lives in the investigation (use the tools as much as the
window needs); the writing is the condensed result. If you are over, cut the
weakest item entirely rather than compressing everything into mush. Never cut
Bottom Line, Scenarios or Positioning.

Per-section ceilings, which together leave room to spare:
  Bottom Line 70 · What Happened 450 · Worth Your Attention 100
  Signals 250 · Scenarios 200 · Positioning 300 · Confidence 90

What condensation means in practice:
- One insight per paragraph, stated once. No restating a table in prose.
- Numbers appear where they carry an argument, not as inventory. Three numbers
  that prove the point beat twelve that describe the day.
- The reader can interrogate anything further in Ask — write the finding, not
  the full workings behind it.

Unit and sign conventions in the stats pack. Each of these has produced a wrong
statement in a previous brief, so read them as hard rules:
- Yields carry basis points only, in `chg_*_bp`. A 1% change in a 4.7% yield is
  4.7bp, not 100bp.
- `fx_board` resolves quote direction for you. Use `dollar_1d` and
  `currency_1m_vs_usd_pct` rather than reasoning from the raw pair change.
- `intraday_moves` is the live price against the prior close; the daily bar is
  history. Say which one you are quoting.
- `net_liquidity` is weekly. Correlations are computed on returns, never levels.

Charts: the dashboard has a Charts tab where every standing figure (prices,
correlations, liquidity, drawdowns, the regime gauge) is always available and
interactive — do NOT embed those here. Embed a chart ONLY when this cycle
produced a specific insight that the figure demonstrates — a correlation that
just flipped, a divergence that just opened. Zero inline charts is the normal
case.

When a standing figure supports a point, link it in prose using the exact link
given for it in `# Charts available` below, e.g.
`[the gold/TLT correlation](/charts#rolling_correlations)`. Do this whenever you
reference something a figure shows — it is how the reader gets from a claim to
the interactive chart behind it. To embed the rare insight chart instead, use
image syntax: `![title](charts/<key>.png)`.

Tables render properly only at the top level with a blank line before and after
— never indented inside a list item.

House style: no horizontal rules; em dashes sparingly; plain declaratives; bold
only where a number or claim is genuinely load-bearing.


## Bottom Line
Two or three sentences. What changed this window and what it means. If nothing
material happened, say so — that is a valid finding and earns a short brief.

## What Happened
The 3-5 developments worth knowing, ranked by consequence. Each: 2-4 sentences —
what happened, the one implication that is not obvious from the headline, source
linked inline. Omit routine churn entirely.

## Worth Your Attention
At most three one-liners: the specific things to go read in full, and why. If
nothing clears the bar, write "Nothing this cycle requires your direct
attention."

## Signals
250 words. The two or three relationships that CHANGED this cycle, and what each
change means. Not a tour of the stats pack: no sub-headed survey of metals, then
crypto, then FX, then the curve. If a relationship is behaving as it did last
cycle, it does not appear here at all.

Pick the changes with the largest measured drift, give each two or three
sentences with the values that carry the argument, and link the standing chart
rather than narrating what it shows. Everything you leave out remains one
question away in Ask.

## Scenarios
A table: scenario, rough probability, mechanism, confirming signal. Two or three
rows, one tight sentence per cell.

## Positioning Implications
A table with one row per asset class where this cycle actually changed the
argument: direction, mechanism, key evidence, what would invalidate it. At most
five rows, one tight sentence per cell. Classes where nothing changed are
omitted — not filled with "unchanged". Analysis of
alignment, never trade advice: "the argument for duration weakened", never
"reduce duration".

## Confidence
Three lines: most confident of, least, and the single observation next cycle
that would most change the picture.

After the brief, output the fenced world_model block exactly as before:

```world_model
## Regime label
<short label>

## What we believe about the current regime
<the standing view, updated with this cycle's evidence>

## Key relationships currently holding
<with measured values>

## Relationships currently broken or unusual
<with measured values, and what would restore them>

## Open questions
<what the next cycle should resolve>

## Changed since last cycle
<what you revised and why — or "no material revision">

## Confidence
<by area>
```
The world_model block replaces the previous one wholesale and is NOT part of the
length budget — keep it as complete as the evidence requires. It is read by the
next cycle as its starting point."""


def _build_prompt(hours: int) -> tuple[str, dict, dict]:
    pack = stats.build(persist=True)
    # Charts are rendered deterministically from the same series the pack uses;
    # the model places and interprets them but never invents one.
    try:
        chart_manifest = chartdata.build_pack()
    except Exception as exc:  # noqa: BLE001
        log.warning("chart rendering failed: %s", exc)
        chart_manifest = {}
    # The model sees the catalogue, not the series: it places and interprets a
    # figure by key, and the browser draws it from the same data the pack used.
    chart_lines = "\n".join(
        f"- {spec.get('title', key)} — link as `[text](/charts#{key})`, "
        f"embed as `![{spec.get('title', key)}](charts/{key}.png)`. "
        f"{spec.get('subtitle', '')}"
        for key, spec in chart_manifest.items()
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
        # Currencies beyond DXY, with quote direction already resolved.
        "fx_board": pack.get("fx_board"),
        "ratios": pack.get("ratios"),
        "gold_in_currencies": pack.get("gold_in_currencies"),
        "correlation_flips": pack.get("correlation_flips"),
        "themed_correlations": pack.get("themed_correlations"),
        "anomalies": pack.get("anomalies"),
        "lead_lag": pack.get("lead_lag"),
        "correlations_30d": pack.get("correlations", {}).get("30d", {}),
        # Position within the range, and whether vol is expanding — momentum
        # alone reads the same on a breakout and on a dead-cat bounce.
        "drawdowns": pack.get("drawdowns"),
        "volatility": (pack.get("volatility") or [])[:12],
        "breadth": pack.get("breadth"),
        # Derived macro state: the scored regime, system liquidity, credit, and
        # the nearest historical analogues.
        "regime_score": pack.get("regime_score"),
        "net_liquidity": pack.get("net_liquidity"),
        "credit_conditions": pack.get("credit_conditions"),
        "historical_analogues": pack.get("historical_analogues"),
        "macro_keys": sorted((pack.get("macro") or {}).keys()),
        "_note": (
            "Compact brief. Full sections (macro levels, 90d correlation "
            "matrix) available via get_stats_pack. IMPORTANT: `performance` "
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
{json.dumps(slim, default=str)[:30000]}
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
    title = f"Brief — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

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
            # Keys only. The series themselves go to chart_packs — meta is read
            # by every list query, and a pack is ~100KB.
            "charts": sorted(chart_manifest.keys()),
            "model": model or config.DIGEST_MODEL,
            "effort": chosen_effort,
            "provider": provider,
        },
    )

    if chart_manifest:
        # Stored per brief so an archived page redraws exactly the figures it was
        # written against, rather than today's data under yesterday's argument.
        try:
            chartdata.save_pack(analysis_id, chart_manifest)
        except Exception as exc:  # noqa: BLE001
            log.warning("chart pack not stored: %s", exc)

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
