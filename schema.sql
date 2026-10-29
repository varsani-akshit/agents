-- MIA schema. Idempotent: safe to re-run.
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────── instruments & prices ───────────────────────────
CREATE TABLE IF NOT EXISTS instruments (
    symbol       TEXT PRIMARY KEY,      -- canonical: GOLD, SILVER, BTC, DXY, US10Y ...
    name         TEXT NOT NULL,
    asset_class  TEXT NOT NULL,         -- metal | fx | rate | crypto | equity | energy
    source       TEXT NOT NULL,         -- yfinance | coingecko | fred
    source_id    TEXT NOT NULL,         -- GC=F, bitcoin, DGS10 ...
    is_rate      BOOLEAN NOT NULL DEFAULT FALSE  -- true => "price" is a yield in %
);

CREATE TABLE IF NOT EXISTS prices (
    symbol   TEXT NOT NULL REFERENCES instruments(symbol) ON DELETE CASCADE,
    ts       TIMESTAMPTZ NOT NULL,
    price    DOUBLE PRECISION NOT NULL,
    open     DOUBLE PRECISION,
    high     DOUBLE PRECISION,
    low      DOUBLE PRECISION,
    volume   DOUBLE PRECISION,
    grain    TEXT NOT NULL,             -- 1d | 15m
    source   TEXT NOT NULL,
    PRIMARY KEY (symbol, ts, grain)
);
CREATE INDEX IF NOT EXISTS prices_symbol_ts_idx ON prices (symbol, grain, ts DESC);

-- FRED macro series (non-price economic data: M2, WALCL, debt, RRP ...)
CREATE TABLE IF NOT EXISTS fred_series (
    series_id TEXT NOT NULL,
    ts        DATE NOT NULL,
    value     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (series_id, ts)
);

-- ─────────────────────────── documents (news/official) ──────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT UNIQUE,
    content_hash  TEXT UNIQUE NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL,
    source_tier   INT  NOT NULL DEFAULT 3,   -- 1 official, 2 top-tier press, 3 general, 4 low-credibility
    published_at  TIMESTAMPTZ,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    urgency       TEXT,                       -- Critical | High | Medium | Low (set by classifier)
    urgency_score INT,
    summary       TEXT,
    entities      JSONB NOT NULL DEFAULT '[]',
    themes        JSONB NOT NULL DEFAULT '[]',
    classified_at TIMESTAMPTZ,
    embedding     vector(1024)
);
CREATE INDEX IF NOT EXISTS documents_published_idx ON documents (published_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS documents_unclassified_idx ON documents (classified_at) WHERE classified_at IS NULL;
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents
    USING hnsw (embedding vector_cosine_ops);

-- ─────────────────────────── entity / relationship graph ────────────────────
CREATE TABLE IF NOT EXISTS entities (
    id           BIGSERIAL PRIMARY KEY,
    canonical    TEXT UNIQUE NOT NULL,       -- "Federal Reserve"
    kind         TEXT NOT NULL,              -- institution | person | asset | policy | event | country
    aliases      JSONB NOT NULL DEFAULT '[]',
    first_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    mention_count INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id             BIGSERIAL PRIMARY KEY,
    source_entity  TEXT NOT NULL,
    target_entity  TEXT NOT NULL,
    relation       TEXT NOT NULL,            -- suppresses | supports | pressures | funds | correlates_with ...
    direction      TEXT NOT NULL DEFAULT 'positive',  -- positive | negative | ambiguous
    strength       DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    evidence_doc_ids BIGINT[] NOT NULL DEFAULT '{}',
    rationale      TEXT,
    first_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_confirmed TIMESTAMPTZ NOT NULL DEFAULT now(),
    confirm_count  INT NOT NULL DEFAULT 1,
    UNIQUE (source_entity, target_entity, relation)
);
CREATE INDEX IF NOT EXISTS edges_source_idx ON edges (source_entity);
CREATE INDEX IF NOT EXISTS edges_target_idx ON edges (target_entity);

-- ─────────────────────────── analyses & memory ──────────────────────────────
CREATE TABLE IF NOT EXISTS analyses (
    id          BIGSERIAL PRIMARY KEY,
    kind        TEXT NOT NULL,              -- digest | alert | answer
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    meta        JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector(1024)
);
CREATE INDEX IF NOT EXISTS analyses_created_idx ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS analyses_embedding_idx ON analyses
    USING hnsw (embedding vector_cosine_ops);

-- The living "World Model" — versioned macro regime document.
CREATE TABLE IF NOT EXISTS world_model (
    version     BIGSERIAL PRIMARY KEY,
    body        TEXT NOT NULL,              -- markdown
    regime      TEXT,                       -- short label
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_analysis_id BIGINT REFERENCES analyses(id) ON DELETE SET NULL
);

-- Deterministic statistics pack, one row per signals run.
CREATE TABLE IF NOT EXISTS stats_packs (
    id         BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload    JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS stats_packs_created_idx ON stats_packs (created_at DESC);

-- Fired trigger events (code-detected, pre-LLM).
CREATE TABLE IF NOT EXISTS trigger_events (
    id          BIGSERIAL PRIMARY KEY,
    rule        TEXT NOT NULL,
    severity    TEXT NOT NULL,              -- Critical | High
    symbol      TEXT,
    doc_id      BIGINT REFERENCES documents(id) ON DELETE SET NULL,
    detail      JSONB NOT NULL DEFAULT '{}',
    dedupe_key  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    notified_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS trigger_events_dedupe_idx ON trigger_events (dedupe_key);

-- ─────────────────────────── observability & cost ───────────────────────────
CREATE TABLE IF NOT EXISTS job_runs (
    id          BIGSERIAL PRIMARY KEY,
    job         TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    ok          BOOLEAN,
    detail      JSONB NOT NULL DEFAULT '{}',
    error       TEXT
);
CREATE INDEX IF NOT EXISTS job_runs_job_idx ON job_runs (job, started_at DESC);

CREATE TABLE IF NOT EXISTS api_calls (
    id             BIGSERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider       TEXT NOT NULL DEFAULT 'anthropic',
    model          TEXT NOT NULL,
    purpose        TEXT NOT NULL,
    input_tokens   INT NOT NULL DEFAULT 0,
    output_tokens  INT NOT NULL DEFAULT 0,
    cache_read     INT NOT NULL DEFAULT 0,
    cache_write    INT NOT NULL DEFAULT 0,
    web_searches   INT NOT NULL DEFAULT 0,
    usd            DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS api_calls_created_idx ON api_calls (created_at DESC);

-- Embeddings from different models occupy different vector spaces and must not
-- be compared. Track which model produced each vector so search can filter.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embed_model TEXT;
ALTER TABLE analyses  ADD COLUMN IF NOT EXISTS embed_model TEXT;
CREATE INDEX IF NOT EXISTS documents_embed_model_idx ON documents (embed_model);

-- Chart data, stored per digest so the page can redraw an old brief exactly as
-- it was written. Kept out of analyses.meta because a pack is ~100KB of series
-- and meta is read on every list query.
CREATE TABLE IF NOT EXISTS chart_packs (
  id          BIGSERIAL PRIMARY KEY,
  analysis_id BIGINT UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
  payload     JSONB NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chart_packs_created_idx ON chart_packs (created_at DESC);

-- Dashboard login. Scrypt with a per-user salt; no third-party dependency.
CREATE TABLE IF NOT EXISTS users (
  id            BIGSERIAL PRIMARY KEY,
  username      TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  salt          TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_login    TIMESTAMPTZ
);

-- Extraction bookkeeping: without it the same top-urgency documents are
-- re-extracted every cycle, inflating confirm counts while fresh documents
-- never get read.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS extracted_at TIMESTAMPTZ;

-- Semantic links between documents: the edges of the knowledge graph. Computed
-- from the embeddings already stored on each document, so this adds no model
-- cost — it is an index over work already done. src_id < dst_id keeps each
-- undirected pair once.
CREATE TABLE IF NOT EXISTS doc_links (
  src_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  dst_id     BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  similarity REAL   NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (src_id, dst_id),
  CHECK (src_id < dst_id)
);
CREATE INDEX IF NOT EXISTS doc_links_src_idx ON doc_links (src_id);
CREATE INDEX IF NOT EXISTS doc_links_dst_idx ON doc_links (dst_id);
CREATE INDEX IF NOT EXISTS doc_links_sim_idx ON doc_links (similarity DESC);

-- Share links: one unguessable token per brief, so a single brief can be sent to
-- someone without giving them an account or sight of anything else. Revocable,
-- and counted so you can see whether a link was ever opened.
CREATE TABLE IF NOT EXISTS brief_shares (
  token       TEXT PRIMARY KEY,
  analysis_id BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  created_by  TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  revoked_at  TIMESTAMPTZ,
  view_count  INTEGER NOT NULL DEFAULT 0,
  last_viewed TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS brief_shares_analysis_idx ON brief_shares (analysis_id);

-- The agent army's trace mirror. Trodo (app.trodo.ai) is the operations view;
-- this table is the product record that drill-down pages render from, so it
-- must survive independently of any external service. One row per agent run.
CREATE TABLE IF NOT EXISTS agent_runs (
  id         BIGSERIAL PRIMARY KEY,
  agent      TEXT NOT NULL,
  trigger    TEXT NOT NULL DEFAULT 'schedule',
  status     TEXT NOT NULL DEFAULT 'running',   -- running | ok | error
  input      JSONB,
  output     JSONB,
  meta       JSONB,
  error      TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS agent_runs_agent_idx ON agent_runs (agent, started_at DESC);

-- Every stage of every brief, in full: the Marshal's north star, each Scout's
-- sourced leads, each Analyst's findings, the Editor's draft, the Verifier's
-- audit. This is what makes a sentence in a brief clickable down to its
-- evidence. beat is NULL for whole-brief stages (marshal, editor, verifier).
CREATE TABLE IF NOT EXISTS brief_runs (
  id          BIGSERIAL PRIMARY KEY,
  analysis_id BIGINT REFERENCES analyses(id) ON DELETE CASCADE,
  stage       TEXT NOT NULL,                    -- marshal|scout|analyst|editor|verifier
  beat        TEXT,                             -- monetary|fiscal|geopolitics|...
  payload     JSONB NOT NULL,
  usd         NUMERIC(10,5) NOT NULL DEFAULT 0,
  model       TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS brief_runs_analysis_idx ON brief_runs (analysis_id, stage);

-- Deep-research notes: standalone investigations triggered from the dashboard
-- or chat, stored like analyses but with their sub-researcher material.
CREATE TABLE IF NOT EXISTS research_notes (
  id         BIGSERIAL PRIMARY KEY,
  question   TEXT NOT NULL,
  body       TEXT NOT NULL,
  facets     JSONB,          -- per-sub-researcher findings
  usd        NUMERIC(10,5) NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Public equities: US, Australia, India ─────────────────────────────────
-- Kept apart from `instruments`/`prices` on purpose. Those drive the macro
-- statistics — correlations, the analogue engine, the regime score — and
-- folding 750 single names into that frame would turn a cross-asset matrix
-- into noise and slow every pack build. Securities are a parallel universe
-- with their own cadence, read by the market beats and the chatbot's screens.
CREATE TABLE IF NOT EXISTS securities (
  symbol           TEXT PRIMARY KEY,        -- yfinance ticker: BHP.AX, RELIANCE.NS
  name             TEXT,
  exchange         TEXT NOT NULL,           -- US | ASX | NSE
  index_member     TEXT,                    -- SP500 | ASX200 | NIFTY50
  currency         TEXT,
  sector           TEXT,
  industry         TEXT,
  market_cap       NUMERIC,
  trailing_pe      NUMERIC,
  forward_pe       NUMERIC,
  dividend_yield   NUMERIC,
  week52_high      NUMERIC,
  week52_low       NUMERIC,
  beta             NUMERIC,
  fundamentals_at  TIMESTAMPTZ,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS securities_exchange_idx ON securities (exchange);
CREATE INDEX IF NOT EXISTS securities_sector_idx ON securities (sector);

CREATE TABLE IF NOT EXISTS security_prices (
  symbol TEXT NOT NULL REFERENCES securities(symbol) ON DELETE CASCADE,
  d      DATE NOT NULL,
  close  NUMERIC NOT NULL,
  volume NUMERIC,
  PRIMARY KEY (symbol, d)
);
CREATE INDEX IF NOT EXISTS security_prices_d_idx ON security_prices (d DESC);

-- The per-stock context layer: what has been said about each name, kept so a
-- question about a stock reaches its history rather than only today's tape.
CREATE TABLE IF NOT EXISTS security_news (
  id           BIGSERIAL PRIMARY KEY,
  symbol       TEXT NOT NULL REFERENCES securities(symbol) ON DELETE CASCADE,
  title        TEXT NOT NULL,
  publisher    TEXT,
  url          TEXT,
  summary      TEXT,
  published_at TIMESTAMPTZ,
  fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (symbol, title)
);
CREATE INDEX IF NOT EXISTS security_news_sym_idx ON security_news (symbol, published_at DESC);

-- ── Multi-user ────────────────────────────────────────────────────────────
-- Alfred is becoming multi-tenant: several people read the same briefs and
-- the same market data, because those are the product, but a question is
-- private to whoever asked it. Ownership is recorded on the two surfaces a
-- user creates: answers and research notes.
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS owner TEXT;
ALTER TABLE research_notes ADD COLUMN IF NOT EXISTS owner TEXT;
CREATE INDEX IF NOT EXISTS analyses_owner_idx ON analyses (kind, owner, created_at DESC);
CREATE INDEX IF NOT EXISTS research_notes_owner_idx ON research_notes (owner, created_at DESC);

-- A person's name, for greeting them. The username is a credential; it is not
-- what you call someone at the top of their morning brief.
ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name TEXT;

-- Per-reader spending. Briefs, prices and the graph are shared infrastructure
-- and cost the same whoever reads them; Ask is the one surface where a reader
-- spends money on demand, so it is the one surface with a cap. Attribution
-- lives on api_calls rather than on the saved answer, because a question that
-- errors halfway still burned tokens and must still count against the month.
ALTER TABLE api_calls ADD COLUMN IF NOT EXISTS owner TEXT;
CREATE INDEX IF NOT EXISTS api_calls_owner_month_idx
  ON api_calls (owner, created_at DESC) WHERE owner IS NOT NULL;

-- NULL means "the default for a reader", not "unlimited" — an unset budget
-- must never be the permissive case. Admins are unlimited in code, not by a
-- row here, so revoking admin restores the cap without a second edit.
ALTER TABLE users ADD COLUMN IF NOT EXISTS monthly_usd NUMERIC(10,2);

-- ── Private mode ──────────────────────────────────────────────────────────
-- Alfred is going back to one owner while it learns to trade. Accounts are
-- disabled, not deleted: their answers and notes stay attributed, and turning
-- someone back on is one command rather than a rebuild.
ALTER TABLE users ADD COLUMN IF NOT EXISTS disabled_at TIMESTAMPTZ;

-- ── Brokers (read-only) ───────────────────────────────────────────────────
-- What the owner actually holds, captured on a schedule. A snapshot is one
-- pull from one broker; a failed pull is recorded too, so the page can say
-- "Kite has not answered since 09:15" instead of showing stale numbers as live.
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  id         BIGSERIAL PRIMARY KEY,
  broker     TEXT NOT NULL,
  taken_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  ok         BOOLEAN NOT NULL,
  error      TEXT,
  funds      JSONB
);
CREATE INDEX IF NOT EXISTS portfolio_snapshots_broker_idx
  ON portfolio_snapshots (broker, taken_at DESC);

-- `kind` separates long-term holdings from open positions: on Kite these are
-- different endpoints and different risks (delivery vs intraday/F&O).
CREATE TABLE IF NOT EXISTS portfolio_lines (
  snapshot_id    BIGINT NOT NULL REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
  kind           TEXT NOT NULL,
  account        TEXT,
  symbol         TEXT NOT NULL,
  exchange       TEXT,
  quantity       NUMERIC NOT NULL,
  avg_price      NUMERIC,
  last_price     NUMERIC,
  pnl            NUMERIC,
  day_change_pct NUMERIC,
  currency       TEXT,
  product        TEXT,
  raw            JSONB
);
CREATE INDEX IF NOT EXISTS portfolio_lines_snapshot_idx ON portfolio_lines (snapshot_id);

-- ── Advisor ───────────────────────────────────────────────────────────────
-- Daily OHLC, split- and dividend-adjusted, for everything the advisor scans.
-- Separate from security_prices (closes only, one year, read by the macro
-- beats): simulating a stop or a limit fill needs the day's high and low, and
-- judging a strategy needs a decade, not a year.
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL,
  d      DATE NOT NULL,
  open   DOUBLE PRECISION NOT NULL,
  high   DOUBLE PRECISION NOT NULL,
  low    DOUBLE PRECISION NOT NULL,
  close  DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION,
  PRIMARY KEY (symbol, d)
);

-- What each strategy has proven, per market (symbol NULL: pooled across the
-- market) and per name. Rewritten whole on each simulation run, so a strategy
-- that stops clearing the gate stops sending ideas the same day.
CREATE TABLE IF NOT EXISTS strategy_evidence (
  strategy    TEXT NOT NULL,
  market      TEXT NOT NULL,
  symbol      TEXT NOT NULL DEFAULT '',
  passed      BOOLEAN NOT NULL,
  stats       JSONB NOT NULL,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (strategy, market, symbol)
);

-- Every recommendation Alfred sends, and what then happened to it. The outcome
-- is computed with the same fill/stop/exit rules as the simulation, so the
-- scoreboard is the forward test of the evidence that justified the idea.
CREATE TABLE IF NOT EXISTS ideas (
  id           BIGSERIAL PRIMARY KEY,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  signal_date  DATE NOT NULL,
  strategy     TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  market       TEXT NOT NULL,
  broker       TEXT NOT NULL,
  action       TEXT NOT NULL,
  currency     TEXT,
  limit_price  DOUBLE PRECISION NOT NULL,
  stop_price   DOUBLE PRECISION NOT NULL,
  target_price DOUBLE PRECISION,
  quantity     DOUBLE PRECISION,
  risk_amount  DOUBLE PRECISION,
  valid_until  DATE NOT NULL,
  reasons      JSONB NOT NULL DEFAULT '[]',
  evidence     JSONB NOT NULL DEFAULT '{}',
  status       TEXT NOT NULL DEFAULT 'open',  -- open | filled | expired | closed
  fill_date    DATE,
  fill_price   DOUBLE PRECISION,
  exit_date    DATE,
  exit_price   DOUBLE PRECISION,
  exit_reason  TEXT,
  r_multiple   DOUBLE PRECISION,
  emailed_at   TIMESTAMPTZ,
  UNIQUE (strategy, symbol, signal_date)
);
CREATE INDEX IF NOT EXISTS ideas_status_idx ON ideas (status, created_at DESC);

-- The system replayed as one account: every passing strategy, the per-scan
-- cap, the open-position limit and cash that can only be spent once. One row
-- per region, rewritten on each run. `curve` holds weekly points for the
-- strategy and its benchmarks, normalised to 1.0 at the start.
CREATE TABLE IF NOT EXISTS portfolio_backtests (
  region      TEXT PRIMARY KEY,
  stats       JSONB NOT NULL,
  curve       JSONB NOT NULL,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- What the last scan of each region saw beyond the tickets it sent: signals
-- the per-scan cap left out, names close to triggering a passing strategy,
-- and exits due on ideas you hold. Rewritten per region each scan; read by
-- the Today page so it never has to re-scan 780 names to render.
CREATE TABLE IF NOT EXISTS scan_candidates (
  region     TEXT NOT NULL,
  kind       TEXT NOT NULL,          -- also | near | exit
  symbol     TEXT NOT NULL,
  strategy   TEXT NOT NULL,
  as_of      DATE,
  detail     JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS scan_candidates_region_idx ON scan_candidates (region, kind);

-- Which articles are about the reader's own names, judged by Decision-1.
-- One row per (article, name) that matched; `impact` is the model's call on
-- whether the news is good or bad for a shareholder. documents.matched_at
-- marks an article as judged, matched or not, so nothing is asked twice.
CREATE TABLE IF NOT EXISTS news_matches (
  doc_id      BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  symbol      TEXT NOT NULL,
  name        TEXT NOT NULL,
  p_about     DOUBLE PRECISION NOT NULL,
  impact      TEXT,
  impact_p    DOUBLE PRECISION,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (doc_id, symbol)
);
CREATE INDEX IF NOT EXISTS news_matches_symbol_idx ON news_matches (symbol, created_at DESC);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS matched_at TIMESTAMPTZ;

-- Company names for tickers outside the index universe (Kite holdings like
-- JINDALSAW), looked up once: a name finds the story, a ticker rarely does.
CREATE TABLE IF NOT EXISTS name_cache (
  symbol     TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Simulation runs ─────────────────────────────────────────────────────────
-- Every run is kept, so each can be opened later: what the gate decided, how
-- each market replayed as one account, and every candidate the run surfaced —
-- including signals from strategies that did not pass, labelled as such.
-- Pruned after 14 days; strategy_evidence and portfolio_backtests still hold
-- the latest run for the rest of the app.
CREATE TABLE IF NOT EXISTS sim_runs (
  id           BIGSERIAL PRIMARY KEY,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at  TIMESTAMPTZ,
  trigger      TEXT NOT NULL DEFAULT 'schedule',
  bars_through JSONB NOT NULL DEFAULT '{}',   -- region -> last final session
  passing      INT,
  tested       INT,
  candidates   INT,
  seconds      REAL,
  error        TEXT
);
CREATE INDEX IF NOT EXISTS sim_runs_started_idx ON sim_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS sim_results (
  run_id   BIGINT NOT NULL REFERENCES sim_runs(id) ON DELETE CASCADE,
  strategy TEXT NOT NULL,
  region   TEXT NOT NULL,
  passed   BOOLEAN NOT NULL,
  stats    JSONB NOT NULL,
  PRIMARY KEY (run_id, strategy, region)
);

CREATE TABLE IF NOT EXISTS sim_accounts (
  run_id BIGINT NOT NULL REFERENCES sim_runs(id) ON DELETE CASCADE,
  region TEXT NOT NULL,
  stats  JSONB NOT NULL,
  curve  JSONB NOT NULL,
  PRIMARY KEY (run_id, region)
);

-- kind: signal (fired on the last close) | near (close to firing)
CREATE TABLE IF NOT EXISTS sim_candidates (
  run_id      BIGINT NOT NULL REFERENCES sim_runs(id) ON DELETE CASCADE,
  region      TEXT NOT NULL,
  symbol      TEXT NOT NULL,
  strategy    TEXT NOT NULL,
  kind        TEXT NOT NULL,
  passed_gate BOOLEAN NOT NULL,
  score       DOUBLE PRECISION,
  detail      JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS sim_candidates_run_idx ON sim_candidates (run_id, passed_gate DESC, score DESC);

-- Hourly bars for the intraday strategies: about three years is what Yahoo
-- serves at this interval. Stored in UTC; sessions are grouped in each
-- exchange's own time zone when loaded.
CREATE TABLE IF NOT EXISTS bars_1h (
  symbol TEXT NOT NULL,
  ts     TIMESTAMPTZ NOT NULL,
  open   DOUBLE PRECISION NOT NULL,
  high   DOUBLE PRECISION NOT NULL,
  low    DOUBLE PRECISION NOT NULL,
  close  DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION,
  PRIMARY KEY (symbol, ts)
);

-- ───────────────────────────── data feeds (not brokers) ─────────────────────
-- One row per feed: when it last ran, when it last succeeded, and why not.
-- The Status page reads this; a feed that goes quiet shows as stale.
CREATE TABLE IF NOT EXISTS data_sources (
  source        TEXT PRIMARY KEY,
  last_run      TIMESTAMPTZ,
  last_ok       TIMESTAMPTZ,
  ok            BOOLEAN,
  rows          INT NOT NULL DEFAULT 0,
  seconds       DOUBLE PRECISION,
  error         TEXT,
  detail        JSONB NOT NULL DEFAULT '{}'
);

-- Perp funding and open interest: crypto positioning read as sentiment.
CREATE TABLE IF NOT EXISTS derivatives (
  venue         TEXT NOT NULL,            -- hyperliquid | binance
  symbol        TEXT NOT NULL,            -- BTC, ETH …
  ts            TIMESTAMPTZ NOT NULL,
  funding       DOUBLE PRECISION,         -- rate per funding period, as a fraction
  open_interest DOUBLE PRECISION,         -- in coins
  mark          DOUBLE PRECISION,
  volume_24h    DOUBLE PRECISION,         -- USD notional
  premium       DOUBLE PRECISION,
  PRIMARY KEY (venue, symbol, ts)
);

-- GDELT average news tone per query per day, since 2017.
CREATE TABLE IF NOT EXISTS tone (
  query  TEXT NOT NULL,
  d      DATE NOT NULL,
  tone   DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (query, d)
);

-- Scalar daily series that are not FRED and not prices (crypto market cap,
-- NSE FII/DII flows …). FRED stays in fred_series.
CREATE TABLE IF NOT EXISTS macro_series (
  series TEXT NOT NULL,
  d      DATE NOT NULL,
  value  DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (series, d)
);

-- CFTC speculative positioning, weekly.
CREATE TABLE IF NOT EXISTS positioning (
  market        TEXT NOT NULL,            -- our short name: us_equity, gold …
  report_date   DATE NOT NULL,
  longs         DOUBLE PRECISION NOT NULL,
  shorts        DOUBLE PRECISION NOT NULL,
  open_interest DOUBLE PRECISION,
  PRIMARY KEY (market, report_date)
);

-- Exchange and regulator filings for names Alfred cares about.
CREATE TABLE IF NOT EXISTS filings (
  id         BIGSERIAL PRIMARY KEY,
  source     TEXT NOT NULL,               -- sec | nse | asx
  symbol     TEXT NOT NULL,               -- yfinance ticker
  kind       TEXT NOT NULL,               -- 8-K, Form 4, "Board meeting" …
  title      TEXT NOT NULL,
  url        TEXT NOT NULL,
  filed_at   TIMESTAMPTZ,
  sensitive  BOOLEAN NOT NULL DEFAULT false,
  raw        JSONB NOT NULL DEFAULT '{}',
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (source, url)
);
CREATE INDEX IF NOT EXISTS filings_symbol_idx ON filings (symbol, filed_at DESC);

-- What is scheduled: earnings dates, data releases, central-bank meetings.
CREATE TABLE IF NOT EXISTS calendar (
  id         BIGSERIAL PRIMARY KEY,
  kind       TEXT NOT NULL,               -- earnings | release | meeting
  key        TEXT NOT NULL,               -- symbol, or country code
  event      TEXT NOT NULL,
  at         TIMESTAMPTZ NOT NULL,
  estimate   DOUBLE PRECISION,
  actual     DOUBLE PRECISION,
  prior      DOUBLE PRECISION,
  importance TEXT,                        -- high | medium | low
  raw        JSONB NOT NULL DEFAULT '{}',
  UNIQUE (kind, key, event, at)
);
CREATE INDEX IF NOT EXISTS calendar_at_idx ON calendar (at);

-- Second opinion on closes: Yahoo vs Twelve Data for the names in play.
CREATE TABLE IF NOT EXISTS price_checks (
  symbol   TEXT NOT NULL,
  d        DATE NOT NULL,
  yahoo    DOUBLE PRECISION,
  twelve   DOUBLE PRECISION,
  diff_pct DOUBLE PRECISION,
  PRIMARY KEY (symbol, d)
);

-- ───────────────────────────── every call, scored ──────────────────────────
-- Every signal a simulation run surfaced — one per name, strategy and signal
-- day, taken or not, passing the gate or not — followed to its exit at the
-- prices Alfred named, by the same play() the backtest and the ledger use.
-- `story` is the Story read: what the news said about the setup at the time.
CREATE TABLE IF NOT EXISTS calls (
  id           BIGSERIAL PRIMARY KEY,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  run_id       BIGINT,
  region       TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  strategy     TEXT NOT NULL,
  signal_date  DATE NOT NULL,
  limit_price  DOUBLE PRECISION NOT NULL,
  stop_price   DOUBLE PRECISION NOT NULL,
  target_price DOUBLE PRECISION,
  passed_gate  BOOLEAN NOT NULL,
  regime_on    BOOLEAN,
  score        DOUBLE PRECISION,
  detail       JSONB NOT NULL DEFAULT '{}',
  story        JSONB,
  story_at     TIMESTAMPTZ,
  status       TEXT NOT NULL DEFAULT 'pending',   -- pending | filled | closed | expired
  fill_date    DATE,
  fill_price   DOUBLE PRECISION,
  exit_date    DATE,
  exit_price   DOUBLE PRECISION,
  exit_reason  TEXT,
  r            DOUBLE PRECISION,
  settled_at   TIMESTAMPTZ,
  UNIQUE (symbol, strategy, signal_date)
);
CREATE INDEX IF NOT EXISTS calls_open_idx ON calls (status) WHERE status IN ('pending', 'filled');
CREATE INDEX IF NOT EXISTS calls_story_idx ON calls (story_at) WHERE story IS NULL;

-- Which exchange a US name actually trades on (NYSE, Nasdaq …), from Yahoo,
-- so a ticket can say where to find it.
ALTER TABLE name_cache ADD COLUMN IF NOT EXISTS exchange TEXT;
