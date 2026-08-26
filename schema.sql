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
