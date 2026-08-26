# Alfred — Macro Intelligence Agent

Continuous research on the gold / silver / fiat / sovereign-debt / crypto complex.
Ingests prices, official releases, and news; computes statistics deterministically;
uses Claude to interpret them against Dalio debt-cycle and debasement frameworks;
alerts on material events and produces deep digests every 6 hours.

**Core design rule: numbers come from code, meaning comes from the model.**
Every correlation, z-score, ratio, and threshold is computed in Python. The model
never asserts a number it wasn't handed. This is what separates real correlation
discovery from a plausible-sounding narrative.

## Quick start

```bash
./mia status              # health, spend, last run of each job
./mia ask "why is silver lagging gold?"
./mia digest              # run a deep analysis cycle now
./mia worldmodel          # current standing view of the macro regime
```

## Architecture

```
free data sources                    deterministic                 model
─────────────────                    ─────────────                 ─────
yfinance   ─┐                                                    
CoinGecko  ─┼─> prices ──────> signals/stats.py ──> stats pack ──┐
FRED       ─┘                  (correlations, z-scores,          │
                                ratios, lead/lag, flips)         │
                                        │                        ▼
25 RSS feeds ─> documents ──────────────┼──────────────> brain/digest.py
(tier 1 official                        │                 (Sonnet 5 + tools)
 → tier 4 noise)                        ▼                        │
       │                        signals/triggers.py              │
       │                        (thresholds → alerts)            │
       ▼                                │                        ▼
  pgvector embeddings                   ▼                  digest + world model
  (OpenAI, 1024-dim)            brain/alert.py             + relationship edges
       │                        (Haiku, no tools)                │
       └────────────> memory/store.py <────────────────────────-─┘
                      (semantic recall)
```

### Why these choices

- **Postgres + pgvector, no Neo4j.** The relationship graph is an `edges` table
  traversed with a recursive CTE. At a few thousand entities that answers the
  same questions Cypher would, in milliseconds, with one less service to run.
  The schema is graph-shaped, so migrating later is a script, not a redesign.
- **No TimescaleDB.** A dozen instruments at daily/15-minute grain is trivial
  volume for an indexed table.
- **Manual agent loop, not the SDK tool runner.** Every iteration checks the
  spend ledger, so an unattended loop can halt itself mid-flight.
- **Triggers are code, not model calls.** Alert latency is bounded by the
  15-minute tick, not by inference, and costs nothing.

## Commands

| Command | Purpose |
|---|---|
| `./mia init` | Apply schema, register instruments, seed world model |
| `./mia backfill` | Load 2y of prices + 900d of FRED history |
| `./mia tick` | One ingestion → classify → trigger → alert cycle |
| `./mia digest [--hours N]` | Deep analysis cycle |
| `./mia ask "<question>"` | Research question over memory + web (`--no-web` for stored only) |
| `./mia status [--sources] [--graph]` | Health, spend, job history, feed health, graph |
| `./mia stats [--section X]` | Print the computed stats pack |
| `./mia backtest` | Replay thresholds over history to check alert frequency |
| `./mia worldmodel [--history]` | Current or historical regime view |
| `./mia serve` | Run the scheduler in the foreground |

## Scheduler

Installed as a launchd agent (`com.mia.scheduler`), restarts on crash and at login.

| Job | Cadence |
|---|---|
| tick | every 15 min |
| digest | 00:05, 08:05, 16:05 UTC (3/day) |
| daily data refresh | 02:00 UTC |
| maintenance (graph hygiene, pruning) | 03:00 UTC |

```bash
launchctl list | grep mia                                     # is it running
tail -f logs/scheduler.log                                    # what it's doing
launchctl unload ~/Library/LaunchAgents/com.mia.scheduler.plist   # stop
launchctl load   ~/Library/LaunchAgents/com.mia.scheduler.plist   # start
```

## Cost control

Every model call is priced and written to `api_calls` before the next one is
allowed. Two hard caps in `.env`; a call that would breach either raises
`BudgetExceeded` instead of running, and jobs degrade gracefully rather than
failing.

```
MIA_DAILY_USD_CAP=2.50        # rolling 24-hour rate limit
MIA_TOTAL_USD_CAP=4.50        # lifetime guard on the prepaid balance
MIA_AUTONOMOUS_USD_CAP=4.00   # scheduled jobs stop here; rest reserved for `ask`
MIA_DIGEST_EFFORT=low         # low ~$0.26 | medium ~$0.33 | high ~$0.40 per digest
```

**Caps guard Anthropic spend only.** Routed Gemini/Groq/OpenAI spend is tracked
separately — counting it against these caps would defeat the point of routing
work off Anthropic. Data sources are all free tiers, so the only running cost is
inference.

## Model routing

Not every call needs Claude. Tasks are routed by requirement, not preference:

| Task | Model | Why |
|---|---|---|
| classify | `gemini:gemini-flash-latest` | high volume, narrow schema |
| extract_edges | `gemini:gemini-flash-latest` | high volume, narrow schema |
| alert | `anthropic:claude-haiku-4-5` | user-facing prose, low volume |
| digest | `anthropic:claude-sonnet-5` | deep reasoning + tool use + caching |
| ask | `anthropic:claude-sonnet-5` | deep reasoning + tool use |

Override any of these with `MIA_CLASSIFY_SPEC`, `MIA_EXTRACT_SPEC`,
`MIA_ALERT_SPEC`. Format is `provider:model`; a bare name means Anthropic.
Providers are `anthropic`, `gemini`, `groq`, `openai`. Any provider failure falls
back to Anthropic automatically, so a routing choice cannot break the pipeline.
`./mia status` shows the live routing table and per-provider spend.

On a benchmark headline ("Fed holds rates steady, signals no cuts"), Gemini Flash
rated it High correctly while gpt-4o-mini and Gemini Flash-Lite both under-rated
it Medium — which is why classification routes to Flash rather than the cheapest
option available.

## Output

Console (rich formatting) plus a durable markdown outbox at
`outbox/YYYY-MM-DD/HHMMSS-{alert,digest,answer}.md`. Slack webhooks are wired
but off — set `MIA_NOTIFY_SLACK=true` to enable, or add a sink in
`notify/out.py` for whatever platform you settle on.

## Configuration

| File | Contents |
|---|---|
| `conf/instruments.yaml` | Tracked instruments and FRED series |
| `conf/sources.yaml` | RSS feeds with credibility tiers (1 official → 4 noise) |
| `conf/triggers.yaml` | Alert thresholds — run `./mia backtest` after editing |
| `brain/principles/*.md` | Dalio frameworks and analysis discipline, injected into every cycle |

Editing `brain/principles/` changes how MIA reasons. That is the intended way to
steer it.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers dedup, statistics (including that correlations use returns not levels),
trigger thresholds, entity canonicalisation, unit handling for yields, and
cost accounting.

## Dashboard

Runs as a launchd service (`com.mia.web`) on port 8100, bound to `0.0.0.0` so
it is reachable from your phone on the same network.

| | |
|---|---|
| On this Mac | http://localhost:8100 |
| On your LAN | `http://<mac-ip>:8100` — find it with `ipconfig getifaddr en0` |

| Page | Purpose |
|---|---|
| Latest | Most recent digest, charts and tables rendered inline |
| Archive | Every past digest and question |
| Ask | Ask a question, pick the model, read the answer |
| Alerts | Fired trigger events with the written alert |
| World model | Current regime view plus version history |
| Status | Model routing, spend by task, job history, feed health, graph |

```bash
launchctl unload ~/Library/LaunchAgents/com.mia.web.plist   # stop
launchctl load   ~/Library/LaunchAgents/com.mia.web.plist   # start
./mia web --reload                                          # dev mode
```

`GET /api/latest` returns the newest digest as JSON — that is what a Telegram or
Slack notifier would call to build its "new digest" ping.

### Reaching it from outside the house

The service binds to the LAN only. For access from anywhere, put it behind
Tailscale (simplest, private, no ports opened) or a Cloudflare Tunnel. Do not
port-forward it — there is no authentication on the dashboard yet.
