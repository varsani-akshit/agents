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

Unit and sign conventions in the stats pack. Each of these has produced a wrong
statement in a previous digest, so read them as hard rules:
- Yields carry basis points only, in `chg_*_bp`. There is no percent change of a
  yield in the pack because it is a misleading quantity: a 1% change in a 4.7%
  yield is 4.7bp, not 100bp.
- `fx_board` resolves quote direction for you. Use the `dollar_1d` field and
  `currency_1m_vs_usd_pct` rather than reasoning from the raw pair change —
  USDJPY rising means the dollar strengthened and the yen weakened.
- `intraday_moves` is the live price against the prior close. When it disagrees
  with the daily bar, the intraday figure is what is happening now, and the daily
  bar is history. Say which one you are quoting.
- `net_liquidity` is weekly. `regime_score` components are votes in [-1, +1],
  not returns or probabilities.
- Correlations are computed on returns, never on levels.

Two mechanical rules, because they silently break rendering:
- Put every table at the top level, never indented inside a bullet or numbered
  list item. An indented table renders as plain text.
- Leave one blank line immediately before the header row and after the last row.

House style. The brief should read as though a person wrote it:
- No horizontal rules (`---`) between sections. The headings already separate
  them; adding rules on top gives the page a generated, templated look.
- Use em dashes sparingly — at most one or two in a section. Where one is doing
  the work of a comma, a colon or a full stop, use that instead. Strings of
  dashes are the clearest tell of machine-written prose.
- Prefer plain declaratives to the "not just X, but Y" and "it isn't A — it's B"
  constructions. State the finding.
- Bold carries emphasis only where a number or claim is genuinely load-bearing.
  A paragraph with six bold phrases has none.

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

Three further inputs belong here, because a return without context misleads:
- `drawdowns` — how far each asset sits from its own 52-week high. "Gold +14% in
  a month" and "gold 12% below its high" are both true and point opposite ways.
- `volatility` — whether a move came with expanding or contracting vol, and where
  30-day vol sits in its own one-year range.
- `fx_board` — currencies beyond the dollar index. DXY is 58% euro and cannot
  speak for the yen, the yuan or the Australian dollar.

## Signals & Correlations
What the measured statistics say: relationships holding, relationships that
broke, and what a break implies. Quote measured values. Where a news development
above should have moved a correlation and did not, flag it. If the pack shows no
anomalies, say the relationships are behaving normally and name the one worth
watching.

`themed_correlations` carries the pairs that answer a standing question — each
one ships with the question it answers, plus its 30d value, its 90d value and
the drift between them. Lead with the pairs whose drift is largest. Prices belong
here as evidence, compressed — never a long list of instruments and percentages.

## Regime & Liquidity
Open with the scored regime reading from `regime_score`: the number, the label,
and — the part that matters — which components dissent. A score of +0.2 built
from six bullish votes and two strongly bearish ones is a different world from
+0.2 built from eight lukewarm ones. Name the dissenters and say what would flip
them.

Then `net_liquidity`: the level, the direction over one and three months, and
whether the hard-asset complex is behaving consistently with it. Note that this
is a weekly series — do not describe it as a daily move. Bring in
`credit_conditions` as the counterweight: spread widening is a deflationary
impulse that cuts against the debasement trade even though both read as "bad
macro". Use `breadth` to say whether a move is broad or narrow: one asset at a
52-week high is a story about that asset, eleven across four asset classes is a
story about money.

## Historical Analogue
`historical_analogues` returns past dates whose macro fingerprint most resembles
today, with what actually followed. Treat this with visible discipline:

- Say how the matches were found and over how much history — the pack tells you.
- Report the median forward returns, but lead with the *dispersion*. If gold
  ranged from -5% to +28% across four matches, the median is nearly meaningless
  and you should say so.
- Ask what is genuinely different now versus each analogue. This is the section's
  real value: the ways today does *not* rhyme.
- Never phrase any of it as a forecast or a base rate. Four matches is four
  observations. If the engine reports `available: false`, say the sample was
  insufficient and move on — do not substitute recalled history.

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
would confirm each. Scenarios, not predictions. Present these as a table with
columns for scenario, rough probability, mechanism, and the confirming signal.

## Positioning Implications
The section the reader acts on. For each asset class where this cycle actually
changed something, state the implication and the reasoning behind it. Use a table
with columns: asset class, direction of the argument, the specific mechanism, the
measured evidence, and what would invalidate it.

Cover the full board where there is something real to say — precious metals,
crypto, equities by region and sector, credit, rates and duration, real estate,
energy and commodities, FX. Metals and crypto are the reader's core interest but
they are not the whole portfolio.

Three hard rules:
- This is analysis of how assets are positioned against the macro, not advice to
  buy or sell. Never state or imply a recommended trade, size, entry or exit.
  Write "the argument for duration weakened" — never "reduce duration".
- An asset class where nothing changed gets one line saying so, or is omitted.
  Do not manufacture a view to fill the table.
- Every row must trace to measured evidence in the pack. A row you cannot
  evidence does not belong in the table.

## Confidence
What you are confident about, what is uncertain, what would change your mind.
Close with the single observation over the next cycle that would most change the
picture — one line, specific enough to check.

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
        chart_manifest = chartdata.build_pack()
    except Exception as exc:  # noqa: BLE001
        log.warning("chart rendering failed: %s", exc)
        chart_manifest = {}
    # The model sees the catalogue, not the series: it places and interprets a
    # figure by key, and the browser draws it from the same data the pack used.
    chart_lines = "\n".join(
        f"- `charts/{key}.png` — {spec.get('title', key)}. {spec.get('subtitle', '')}"
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
