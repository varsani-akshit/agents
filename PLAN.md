# MIA — Overnight Build Plan (v1)
Date: 2026-08-25 → 2026-08-26 morning
Target: fully working local system on this Mac — ingestion, signals, memory w/ embeddings, digest agent, ask mode, Slack alerts — tested end-to-end with real data.

## Architecture recap (as agreed)
- Numbers from code, meaning from the model. Stats/correlations/triggers are deterministic Python; Claude does semantic linking, digests, Q&A.
- One Postgres (local, Homebrew) + pgvector. Plain tables for prices (no Timescale). Plain `edges` table for the relationship graph (no Neo4j).
- Claude Agent SDK (Python) for the brain. Haiku 4.5 for classification/extraction; Sonnet 5 (upgradeable to Opus 5) for 6-hour digests and ask mode.
- Embeddings: Voyage `voyage-finance-2` if VOYAGE_API_KEY provided; else local `sentence-transformers` fallback (all-MiniLM, zero cost). Stored in pgvector; used for memory search + news dedup.
- Slack via incoming webhooks: #macro-critical (alerts), #macro-digests (6h summaries).

## Stage 0 — Environment (no keys needed)
- [ ] brew install pgvector; ensure postgresql@17 service running; create db `mia`
- [ ] uv-managed venv on Python 3.12; project skeleton, pyproject.toml
- [ ] Repo layout as agreed: ingest/ signals/ brain/ memory/ notify/ scheduler.py tests/
- [ ] git init + initial commit (local only)

## Stage 1 — Schema (no keys needed)
Tables: instruments, prices (ts, instrument, ohlc/last, source), documents (url, title, body, source, credibility, published_at, content_hash, embedding vector, urgency, entities jsonb), entities (canonical_id, name, type, aliases), edges (source, target, relation, direction, strength, evidence_doc_ids, first_seen, last_confirmed), analyses (kind: digest|alert|answer, body, embedding, created_at), world_model (versioned doc), fred_series, job_runs (observability).
- [ ] schema.sql + migration runner + smoke test

## Stage 2 — Ingestion (no keys needed except FRED)
- [ ] prices.py: yfinance — GC=F, SI=F, DX-Y.NYB, ^TNX/^TYX/2y proxy, ^GSPC, CL=F, EURUSD; CoinGecko — BTC, ETH. 15-min bars, upsert, gap-tolerant.
- [ ] Backfill: 2 years daily + recent intraday for all instruments.
- [ ] feeds.py: curated RSS — Fed press releases, US Treasury press, ECB, BoE, BoJ (en), IMF, FT/Reuters/Bloomberg via public RSS mirrors where free, Kitco/metals feeds, ZeroHedge (tagged low-credibility), FRED releases. Source credibility tags in a sources.yaml.
- [ ] fred.py: DGS2/DGS10/DGS30, WALCL (Fed balance sheet), M2SL, GFDEBTN, RRPONTSYD, T10YIE (breakevens), DTWEXBGS (broad dollar).
- [ ] dedupe.py: url canonicalization + content hash + (later) embedding near-dup check.
- [ ] Unit tests with recorded fixtures; live smoke test hitting real endpoints.

## Stage 3 — Signals & triggers (no keys needed)
- [ ] stats.py: rolling correlations (30/90/180d) across all pairs; z-scores of daily & intraday moves; ratios (gold/silver, gold/SPX, BTC/gold); correlation-flip detection; divergence-from-history flags; realized vol; lead/lag hints. Output = one JSON "stats pack" per run, persisted.
- [ ] triggers.py: rule engine (yaml-configurable thresholds): |move| z>2.5 intraday, gold/silver ±2% day, BTC ±5%, 10y ±10bp day, DXY ±0.8%, new doc from Fed/Treasury official feeds = auto-High. Emits events → alert path.
- [ ] Backtest triggers against backfilled history to sanity-check alert frequency (tune so ~0–3 critical/day, not 30).

## Stage 4 — Memory & embeddings (VOYAGE key or fallback)
- [ ] store.py: embed(text)->vector, upsert docs/analyses, semantic search with recency + credibility weighting, near-dup suppression.
- [ ] principles/: Dalio debt-cycle summary, debasement/financial-repression framework, monetary system map — written as markdown, injected into digest/ask prompts. (I draft these; Akshit reviews later.)
- [ ] world_model.py: read latest, write new version each digest cycle.

## Stage 5 — Brain (needs ANTHROPIC_API_KEY)
- [ ] classify.py (Haiku 4.5): batch-classify new docs → urgency (Critical/High/Med/Low), entities, one-line summary, relations mentioned. Structured output, strict JSON schema.
- [ ] tools.py: query_prices, get_stats_pack, search_memory, query_relationships (recursive CTE), get_world_model, web_search (SDK-native), fetch_url.
- [ ] digest.py (Sonnet 5): 6-hour cycle — inputs: stats pack, new docs since last cycle, prior world model, principles. Output sections: Prices | Key Developments | Correlations & Anomalies | Framework View | Tensions/Conflicting Narratives | Bottom Line | Confidence notes. Side effects: update world_model, upsert edges, embed analysis into memory.
- [ ] alert.py (Haiku): trigger event → concise alert w/ data + 2-sentence why + links.
- [ ] ask.py: CLI `mia ask "..."` — same tools, conversational; searches stored memory AND fresh web.
- [ ] extract.py (Haiku): entity canonicalization pass + edge upsert with evidence.

## Stage 6 — Notify (needs Slack webhooks)
- [ ] slack.py: block-kit formatting for alerts + digests; thread-friendly; retry w/ backoff; dry-run mode that logs instead of posting (used until webhooks provided).

## Stage 7 — Scheduler & ops
- [ ] scheduler.py (APScheduler): 15-min ingest+trigger tick; 6-hour digest cycle; daily FRED pull; nightly edge-hygiene/distillation pass.
- [ ] launchd plist → runs on login, restarts on crash; logs to mia/logs/.
- [ ] job_runs table + `mia status` CLI (last runs, errors, counts, est. API spend).

## Stage 8 — Test / correct / harden loop (rest of night)
- [ ] Unit tests green (mocked). Live smoke tests per source.
- [ ] Full simulated day: replay backfilled data through triggers + run 2 real digest cycles + several ask queries; read outputs critically; fix and rerun.
- [ ] Failure injection: dead feed, API timeout, malformed doc, DB restart.
- [ ] Cost check: estimate $/month from actual token counts; verify within budget.
- [ ] README.md + MORNING_REPORT.md: what was built, what ran, sample outputs, known gaps, how to operate.

## Definition of "done by morning"
1. `mia status` shows healthy scheduled runs over several hours.
2. Real 6-hour digest generated from real data (posted to Slack if webhooks provided, else saved + shown in morning report).
3. At least one end-to-end trigger→alert demonstrated (replayed or live).
4. `mia ask` answers a question using stored memory + fresh web search.
5. Tests green; morning report written.
