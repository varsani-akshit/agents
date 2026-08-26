# MIA — Morning Report

Built overnight, 2026-08-25 → 2026-08-26. **The system is running now** as a
launchd service and has produced real digests from real data.

---

## Start here

```bash
cd ~/Documents/agents/mia
./mia status --sources --graph      # health, spend, feeds, relationship graph
./mia worldmodel                    # MIA's current view of the macro regime
cat outbox/2026-08-26/*digest.md    # the digests it wrote overnight
./mia ask "what changed in the metals complex overnight?"
```

## What's live

| | |
|---|---|
| Scheduler | running under launchd, restarts on crash and at login |
| Instruments | 15 tracked, 2 years of daily history (10,983 price rows) |
| Macro series | 14 FRED series, 4,679 observations |
| News corpus | 1,116 documents from 25 live sources, all embedded |
| Analyses | 4 digests, 4 world-model versions |
| Tests | 28 passing |
| Spend | $1.92 of the $3.00 cap I set (your key had $5) |

## Does it actually find non-obvious things?

From the digests it wrote last night, unprompted:

- **BTC leads gold by one day** (correlation 0.471) while the *same-day*
  correlation is only 0.104. A predictive relationship exists where the obvious
  contemporaneous one doesn't.
- **The dollar is currently a stronger driver of gold than yields are**:
  DXY→GOLD −0.543 versus US10Y→GOLD −0.244. The textbook real-rate channel is
  the weaker one right now.
- **BTC is in its debasement regime, not its risk-on regime** — established from
  a measured BTC/SPX correlation of −0.157, not assumed.
- **Gold is outrunning industrial metals broadly**, not just silver: copper/gold
  −12.5% on the month, so the gold/silver stretch isn't a silver-specific story.
- It identified the Treasury-buyback → yield-suppression → dollar → metals chain
  as the dominant channel, and flagged the "repression fingerprint" — heavy
  issuance with the long end *falling*.

When I asked it why silver lagged gold, it caught a **factual error in a news
article**: the piece claimed the gold/silver ratio was "near 60" against a
measured 69.16, and it deferred to the computed value while flagging the
discrepancy. That is the grounding discipline working as intended.

## Design decisions worth knowing

**Numbers from code, meaning from the model.** Correlations, z-scores, ratios,
lead/lag, and every alert threshold are computed in Python. The model is handed
those numbers and forbidden from inventing any. This is the main defence against
hallucinated correlations.

**No Neo4j, as discussed.** The graph is a Postgres `edges` table traversed with
a recursive CTE. It already does multi-hop: querying "US Treasury" at depth 3
returns `US Treasury → US Dollar → Bitcoin`. The schema is graph-shaped, so
moving to Neo4j later is a script rather than a redesign.

**Embeddings run on OpenAI, not Voyage.** Voyage's finance-tuned model was the
better fit on paper, but its free tier without a payment method is capped at
**3 requests/minute and 10K tokens/minute** — it cannot embed a news corpus or
serve interactive queries. OpenAI's `text-embedding-3-small` shortens natively to
1024 dimensions, so the schema was unchanged; the whole corpus embeds in ~14
seconds for about a cent. Your Gemini key is wired as a second fallback. Each
row records which model produced its vector, because vectors from different
models are not comparable and must never be mixed.

**Alert thresholds are backtested.** Replayed over 624 trading days: 0.27
Critical alerts/day, 85.7% of days completely silent. The busiest day was
2025-04-03 with 9 — the tariff selloff. It stays quiet and fires on real events.

## Bugs I hit and fixed

1. **Every tier-1 official feed was silently dropping.** Two causes: feedparser's
   default fetch was rejected on user-agent/certs (fixed by fetching with httpx),
   and my 72-hour age cutoff discarded them all — central banks publish on a
   days-to-weeks cadence while news proxies publish constantly. Retention is now
   per-tier: 30 days for official sources, 72 hours for press.
2. **Yields were being described in the wrong unit.** A −1.02% change in a 4.7%
   yield is −4.8bp; an alert rendered it as "fell 102bp". The stats table emitted
   both units, inviting the misread. Rate instruments now emit basis points
   *only*. Regression test added.
3. **Sign was double-encoded in the relationship graph.** Verbs like `supports`
   carried sign *and* so did the `direction` field, producing contradictions like
   `supports(negative)` and an inverted `US Dollar --supports(positive)--> Gold`
   edge (the measured correlation is −0.54). "Dollar *weakness* lifts gold" lost
   its negation when collapsed to the entity "US Dollar". Verbs are now
   sign-neutral; direction alone carries sign.
4. **Duplicate-URL crash.** `documents` is unique on both content hash and URL;
   a targeted `ON CONFLICT` covered only the first, so a re-published story with
   a tweaked headline killed the tick.
5. **Alert fatigue on first run — 9 alerts in one tick.** Keyword matching alone
   escalated to Critical, but "rate cut" and "default" appear in routine copy
   daily. Critical now requires either the classifier independently judging it
   Critical, or an official source carrying a trigger keyword — plus a rate limit
   of 3 per tick / 5 per hour.
6. **Cost blowout from re-billing history.** In a tool loop the whole message
   history is resent each turn, so a 22K-character stats blob was charged on
   every turn. Added incremental prompt caching (hit rate went 0–28% → 70–97%)
   and slimmed the opening prompt, since the agent can fetch any section on
   demand.

## The one thing that needs your decision: budget

Measured costs on your workload:

| | |
|---|---|
| Digest (low effort) | **$0.26** |
| Digest (medium / high) | $0.33 / $0.40 |
| Interactive `ask` | ~$0.17 |
| Tick (classification) | ~$0.01 |
| Data sources | **$0** — all free tiers |

At 4 digests/day that's roughly **$1.05/day ≈ $31/month**, above the $10–30 you
had in mind. Levers, in order of bluntness: drop to 2–3 digests/day, keep effort
at `low` (already the default), or accept ~$31.

**Right now $1.08 remains under my $3.00 cap**, and about $1.90 of your $5 is
spent — most of it one-time build and testing that won't recur. To keep it
running past today, raise the caps in `.env`:

```bash
MIA_TOTAL_USD_CAP=4.50        # lifetime guard
MIA_AUTONOMOUS_USD_CAP=4.00   # scheduled jobs stop here
MIA_DAILY_USD_CAP=1.20        # steady state: 4 digests + ticks
```

Scheduled jobs stop at the autonomous cap so they can never consume the budget
you need for your own questions — currently $0.45 is reserved for `ask`.

## Known gaps

- **Relationship graph is thin** — 6 entities, 3 edges. It only grows from
  documents rated Medium or above, and it's had one night. Expect it to become
  useful over about a week.
- **Slack is wired but off**, per your instruction. Output goes to console and
  `outbox/`. Flip `MIA_NOTIFY_SLACK=true`, or add a sink in `notify/out.py` for
  whatever platform you choose.
- **Silver's industrial-demand read is weak.** Copper is a proxy; real signal
  would need solar/industrial data I didn't add.
- **No portfolio overlay** — out of scope for Phase 1.
- **Runs on this Mac.** If the laptop sleeps, ticks pause and resume on wake.
  True 24/7 with <5-minute alert latency needs a small always-on host; the code
  is deployment-agnostic and the only stateful dependency is Postgres.
- **`ask` costs real money per question.** No caching of repeated questions yet.

## Where things live

```
mia/
├── mia                     launcher — ./mia <command>
├── conf/                   instruments, sources (tiered), trigger thresholds
├── ingest/                 prices, FRED, RSS feeds, dedup
├── signals/                stats pack, triggers, backtest      ← no LLM in here
├── memory/                 embeddings, semantic search, world model
├── brain/                  classify, tools, agent loop, digest, alert, ask
│   └── principles/         Dalio frameworks + analysis discipline
├── notify/                 console + file outbox (+ Slack, off)
├── scheduler.py            the loops
└── tests/                  28 tests
```

Editing `brain/principles/*.md` changes how MIA reasons — that's the intended
steering mechanism, and worth a read since it encodes the frameworks we discussed.
