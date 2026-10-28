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

ROLE = """You are Alfred — the reader's hyper-competent right hand for markets and
the economy. You handle the intelligence work behind the scenes and deliver the
brief that matters, when it matters. Your domain is whatever moved the world's
money in this window — central banks, governments, companies, currencies,
commodities, credit, geopolitics — read for what it means for the reader's own
holdings in India, Australia and the US.

Your reader can look up any price in seconds. What they cannot do is read
several hundred articles, work out which five mattered, and trace each one
through to the assets it touches. That is the job.

The standard is a partner-level note: research as deep as the tools allow,
articulated in as few words as the findings need. Depth goes into the
investigation; discipline goes into the writing. The reader gives you ten
minutes; earn them.

There is no house worldview: the lenses below are tools, chosen per story. The
stats pack and your tools are your only source of numbers. Apply the analysis discipline strictly — especially the
rule that every quantitative claim comes from a tool result, never from memory."""

FORMAT = """Write the brief in this structure, in markdown.

LENGTH: 1,800-3,000 words. The news decides the length: a heavy window gets the
upper end, a quiet one the lower. Depth comes from following each story to the
reader's money, not from more words on the same subject.

REGISTER: Formal international English. Use the technical vocabulary of markets
and policy freely — term premium, carry, bear steepening, fiscal dominance —
because the reader knows it and precision matters. Avoid American colloquialism
and newsroom slang entirely: no "hawkish jitters", no "risk-off mood", no
"walking a tightrope", no "eye-watering". State what happened and what follows
from it. Prefer British spelling (labour, centre, programme, recognise,
normalise) except where a proper noun or market term fixes it (Treasury, Federal
Reserve, Treasuries, S&P 500).

Begin with exactly these two lines, before any heading:

    # <headline>
    *<standfirst>*

The headline is specific to THIS window and its content — the single most
consequential thing that happened, stated as a finding, in eight to fourteen
words. "Treasury buybacks meet a rising term premium as Tokyo turns hawkish",
not "Macro brief" and not the regime label. Two consecutive briefs should never
carry the same headline; if the same theme persists, the headline says what
advanced in it.

The standfirst is one sentence in italics summarising the window for someone
deciding whether to read on.

## Top line

Three bullets, one sentence each: the three things from this window the reader
most needs to know, most important first. Where several share one cause, say so.

## Your money

What this window meant for the reader's own book (given to you below as
"The reader's book"). One bullet per holding, ticket or market that a story
actually touched — name it, say what happened that reaches it, and whether
the news strengthens or weakens the case for holding it, with the measured
evidence. Then one bullet on the largest risk to the book as it stands, if a
story raised one. Holdings untouched by the news are not listed. If nothing in
the window reached the book, say so in one line and name what came closest.

Analysis, never instruction: "the case for holding Coal India weakened as the
coal tax took effect", not "sell Coal India". Alfred's advisor sends the
actionable tickets; this section is the context around them.

## The stories

The body of the brief: the stories your desk chose, most important first, each
under its own `###` heading. The heading is the finding, stated specifically —
"RBI holds at 6.25% but cuts the CRR, freeing ₹1.2 trillion for lending", not
"India update" and never a fixed category name. Five to eight stories; fewer in
a quiet window. Do NOT force stories into fixed topics, and do not default to
gold, the dollar or US Treasuries unless something genuinely happened to them.

Each story, in prose (paragraphs, not bullets):
- What happened: dated fact, the numbers, the source — linked inline where you
  have a real URL — and how credible that source is.
- Why it matters: the mechanism, through the lens that fits (central banks,
  fiscal, earnings, currencies, commodities, credit, India, Australia...).
- Who and what it touches: markets, sectors and named listed companies with
  their measured figures. Name the reader's holdings explicitly when reached.
- What to watch: the next observable that confirms or reverses it.

Embed the chart built for a story (from `# Charts available`, listed first and
marked with the story it was made for) directly beneath that story's first
paragraph. Most stories carry one chart; a story with no useful chart carries
none. Never embed a chart that does not evidence the story it sits in.

Cover the world through the reader's markets: India, Australia and the US
first, then what Europe, China, Japan, the Gulf and the commodity exporters
transmitted to them. A brief entirely about one country has failed.

CROSS-LINKING, where a genuine connection exists:
- One story explaining another: name it — "the same rupee weakness in the
  oil story above".
- A story continuing one from a previous brief: say what advanced.
- A listed company the argument rests on: link its page on first mention —
  `[BHP](/markets/BHP.AX)`, `[Coal India](/markets/COALINDIA.NS)`.

## Markets at a glance

One markdown table: the instruments that actually moved this window across
India, Australia, the US, currencies, commodities and rates — instrument,
level, change over the window, and a one-line reason (a story above, or
"no clear catalyst" — never an invented one). Eight to twelve rows, sorted by
absolute move. Numbers only from the measured data below.

## What's next

A short table — date, event, why it matters to the reader — of scheduled
events in the next week or two that the sources or data actually mention:
central bank meetings, data releases, earnings for the reader's names,
auctions, elections. Never invent a date; if the sources give none, leave the
event out. Three to eight rows.

## Opportunities

Two to five bullets connecting this window's news to Alfred's advisor (its
open tickets and the names near a trigger are given below): which of them
today's stories support or undercut, and why. Name the ticket or the trigger
it is waiting for. If nothing connects, one line saying so. This is context
for the tickets, not new advice — the advisor's gate decides what is sent.

## Confidence

Two short paragraphs: what you are most and least sure of, and the single
observation next window that would most change the picture.

Unit and sign conventions in the statistics pack. Each has produced a wrong
statement in a previous brief, so read them as hard rules:
- Yields carry basis points only, in `chg_*_bp`. A 1% change in a 4.7% yield is
  4.7bp, not 100bp.
- `fx_board` resolves quote direction for you. Use `dollar_1d` and
  `currency_1m_vs_usd_pct` rather than the raw pair change.
- `intraday_moves` is the live price against the prior close; the daily bar is
  history. Say which you are quoting.
- `net_liquidity` is weekly. Correlations are computed on returns, never levels.
- Every document below carries its publication time. The corpus spans about a
  week, so check dates before drawing a line between two items: an expectation
  published on Monday and the outcome on Thursday are a sequence, not a
  contradiction, and the older is superseded rather than competing.

Citations: link only to a URL that appeared verbatim in a tool or search result,
including its full path. Never assemble one from a publisher's name — a bare
homepage looks like a citation and is not one. With no exact URL, name the source
in plain text. Every source found by search is listed under the brief
automatically.

CHARTS are made for the stories. `# Charts available` lists first the charts
built for this window's stories, each marked with the story it was made for,
then a few standing boards. Embed with `![title](charts/KEY.png)` beneath the
story each one evidences. Do not embed a standing board unless a story is
about exactly what it shows.

Tables render only at the top level, with a blank line before and after, never
indented inside a list item.

House style: no horizontal rules; em dashes sparingly; bold for genuinely load-bearing
numbers, nowhere else.

After the brief, output the fenced world_model block:

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
length budget. It is read by the next cycle as its starting point."""


def chart_link_lines(chart_manifest: dict) -> str:
    """The chart catalogue as link instructions — shared with the pipeline."""
    return "\n".join(
        f"- {spec.get('title', key)} — embed as `![{spec.get('title', key)}](charts/{key}.png)`. "
        f"{spec.get('subtitle', '')}"
        + (f" [made for the story: {spec['for']}]" if spec.get("for") else "")
        for key, spec in chart_manifest.items()
    )


def slim_stats(pack: dict) -> dict:
    """The compact statistics brief sent in prompts.

    Send a compact brief, not the whole pack. Everything omitted here is one
    get_stats_pack call away, and the full pack would otherwise be re-billed on
    every turn of the tool loop.
    """
    return {
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


def doc_lines(docs: list[dict], include_low: bool = False) -> list[str]:
    """Document lines for prompts, dated — shared with the pipeline.

    The date is not decoration. Reading a week of coverage undated, the model
    cannot tell Monday's expectation from Friday's outcome, and will cheerfully
    present a superseded claim alongside the thing that superseded it.
    """
    out = []
    for d in docs:
        if not include_low and d.get("urgency") == "Low":
            continue
        when = d.get("published_at") or d.get("fetched_at")
        stamp = when.strftime("%d %b %H:%M") if when else "undated"
        out.append(
            f"- [{stamp}] [{d.get('urgency') or '?'}|tier{d['source_tier']}] "
            f"{d['title']} — {d.get('summary') or ''} ({d['source']})"
        )
    return out


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
    chart_lines = chart_link_lines(chart_manifest)
    prior = world_model.current_body()
    docs = store.recent_documents(hours=hours, limit=120)
    triggers = db.query(
        """SELECT rule, severity, symbol, detail, created_at FROM trigger_events
           WHERE created_at > now() - make_interval(hours => %s)
           ORDER BY created_at DESC LIMIT 25""",
        (hours,),
    )
    slim = slim_stats(pack)
    dlines = doc_lines(docs)

    user = f"""Run the {hours}-hour deep analysis cycle. Timestamp: {datetime.now(timezone.utc).isoformat()}

# Computed statistics (authoritative for all numbers)
```json
{json.dumps(slim, default=str)[:30000]}
```

# Code-detected trigger events this window
```json
{json.dumps([{k: v for k, v in t.items() if k != 'created_at'} for t in triggers], default=str)[:3000]}
```

# New documents since last cycle ({len(dlines)} non-Low of {len(docs)} total)
{chr(10).join(dlines[:100]) or "(nothing above Low urgency)"}

# Charts available — LINK these in prose using the link form shown.
# Embedding is the rare exception, only for a figure that demonstrates a
# specific finding from THIS cycle.
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


def _split_headline(body: str) -> tuple[str, str | None, str | None]:
    """Lift the leading `# headline` and italic standfirst out of the body.

    Every brief previously stored the same generated title and the page showed
    the regime label as its heading — and the regime label barely changes, so an
    archive of a dozen briefs read as a dozen copies of one entry. The model now
    writes a headline specific to the window, and it becomes the title
    everywhere: page, archive, sidebar, share link.

    Both lines are removed from the body, since the page renders them as the
    header rather than as prose.
    """
    lines = body.lstrip().splitlines()
    headline = standfirst = None
    consumed = 0

    if lines and lines[0].startswith("# "):
        headline = lines[0][2:].strip()
        consumed = 1
        # The standfirst is the next non-empty line, italicised.
        for i in range(1, min(4, len(lines))):
            candidate = lines[i].strip()
            if not candidate:
                continue
            if candidate.startswith("*") and candidate.endswith("*"):
                standfirst = candidate.strip("*").strip()
                consumed = i + 1
            break

    return "\n".join(lines[consumed:]).lstrip(), headline, standfirst


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

    name = model or config.DIGEST_MODEL
    if name.startswith(("gpt-", "o3", "o4")):
        provider = "openai"
    elif name.startswith("gemini"):
        provider = "gemini"
    else:
        provider = "anthropic"
    chosen_effort = effort or os.getenv("MIA_DIGEST_EFFORT", "low")

    # Only the Anthropic path takes a block list with cache breakpoints; the
    # others take one system string.
    flat_system = "\n\n---\n\n".join(
        b["text"] if isinstance(b, dict) else str(b) for b in system
    )

    if provider == "openai":
        from brain import agent_openai

        result = agent_openai.run_agent(
            system=flat_system,
            user_message=user,
            model=name,
            purpose="digest",
            max_turns=8,
            max_tokens=24000,
            effort=chosen_effort,
            use_web_search=True,
        )
    elif provider == "gemini":
        from brain import agent_gemini

        result = agent_gemini.run_agent(
            system=flat_system,
            user_message=user,
            model=name,
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
            model=name,
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
    body, headline, standfirst = _split_headline(body)
    # Fall back to a timestamp only if the model omitted the headline entirely;
    # a brief with no distinguishing title is worse than an ugly one.
    title = headline or f"Brief — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"

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
            # Recorded so length drift is visible on Status rather than
            # something noticed only when a brief feels long.
            "words": len(body.split()),
            "headline": headline,
            "standfirst": standfirst,
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
