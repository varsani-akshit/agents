# Deploying MIA

MIA is two processes against one Postgres database:

| Process | Command | What it does |
|---|---|---|
| **web** | `uvicorn web.app:app --host 0.0.0.0 --port $PORT` | The dashboard you read |
| **scheduler** | `python cli.py serve` | Ingests every 15 min, writes a brief at 00:05 / 08:05 / 16:05 UTC |

The web process is stateless. The scheduler must be a **single** long-running
instance — two copies would double-ingest and write duplicate briefs.

**One-service option.** Set `MIA_EMBEDDED_SCHEDULER=1` and the web process runs
the schedule in a background thread, so you pay for one service instead of two.
Only do this at exactly one instance: if the host ever scales to two, both would
tick and both would write a brief, and the duplicates would look like real data.

---

## 1. Database

Postgres 15+ with the `pgvector` extension. Create it, then load the schema:

```bash
psql "$DATABASE_URL" -c 'CREATE EXTENSION IF NOT EXISTS vector;'
psql "$DATABASE_URL" -f schema.sql
```

`schema.sql` is idempotent — safe to re-run on every deploy.

Then seed history and create your login:

```bash
python cli.py backfill --period 12y --fred-days 6000
python cli.py user akshit
```

The backfill pulls roughly 150k price rows and takes about two minutes.

---

## 2. Environment variables

**Required — the app will not start or will fail its first cycle without these.**

| Variable | Notes |
|---|---|
| `DATABASE_URL` | `postgresql://user:pass@host:5432/mia` — must have pgvector |
| `MIA_SESSION_SECRET` | Signs the login cookie. `python -c "import secrets;print(secrets.token_hex(32))"`. **If unset, a random key is generated per boot and every deploy logs you out.** |
| `OPENAI_API_KEY` | Writes the briefs (gpt-5.1) and computes embeddings |
| `ANTHROPIC_API_KEY` | Powers `Ask` and the alert writer |

**Recommended**

| Variable | Notes |
|---|---|
| `GEMINI_API_KEY` | Classification and entity extraction route here — far cheaper than Claude for those. Without it they fall back to Anthropic and cost more. |
| `FRED_API_KEY` | Free from the St. Louis Fed. Without it the 37 macro series stop updating and the liquidity, term-premium and credit analysis go stale. |

**Optional**

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | Another cheap routing target |
| `VOYAGE_API_KEY` | — | Alternative embeddings. Do not switch providers on a populated database without re-embedding: vectors from different models are not comparable, and the code scopes queries by `embed_model` precisely to stop them being mixed. |
| `MIA_DIGEST_MODEL` | `claude-sonnet-5` | Set to `gpt-5.1` for the current setup — roughly a third the cost per brief |
| `MIA_DIGEST_EFFORT` | `medium` | `high` produces the deepest briefs at about $0.34 each |
| `MIA_ASK_MODEL` | `claude-sonnet-5` | Kept on Anthropic deliberately: server-side web search only exists on that path |
| `MIA_CLASSIFY_SPEC` | `gemini:gemini-flash-latest` | `provider:model` |
| `MIA_EXTRACT_SPEC` | `gemini:gemini-flash-latest` | |
| `MIA_DAILY_USD_CAP` | `0.60` | Rolling 24-hour Anthropic spend ceiling |
| `MIA_TOTAL_USD_CAP` | `3.00` | Lifetime Anthropic ceiling. **Raise both, or scheduled cycles will stop.** |
| `MIA_AUTONOMOUS_USD_CAP` | 75% of total | Scheduled jobs stop here, reserving the rest for your interactive questions |
| `MIA_EMBEDDED_SCHEDULER` | unset | `1` runs the schedule inside the web process. Single instance only. |

The caps count **Anthropic spend only** — they exist to protect one prepaid
balance, and counting routed Gemini or OpenAI spend against them would defeat the
point of routing work off Anthropic.

---

## 3. Host notes

Any host that runs a long-lived process works: Render, Railway, Fly.io, or a
plain VPS. Two things to check before you pick one.

**The scheduler needs a persistent process.** On a serverless platform (Vercel,
Netlify, Lambda) the web app can be adapted, but APScheduler cannot — nothing
stays running between requests. There you would drop `cli.py serve` and drive the
work from platform cron instead:

```
*/15 * * * *   python cli.py tick
5 0,8,16 * * * python cli.py digest --hours 8
```

**Free tiers that sleep will break ingestion.** If the process is suspended when
idle, the 15-minute tick simply does not happen, and a brief written after a gap
analyses a corpus with a hole in it. Use a plan that stays awake.

### Render

- **Web Service** — build `pip install -r requirements.txt`, start
  `uvicorn web.app:app --host 0.0.0.0 --port $PORT`
- **Background Worker** — same build, start `python cli.py serve`
- **Postgres** — add-on, then enable pgvector as above

Both services need the same environment group.

### Vercel

Suitable for the dashboard only, and it needs a serverless entrypoint plus an
external Postgres (Neon and Supabase both ship pgvector). Run the scheduler
elsewhere — Vercel Cron can call `tick` and `digest` endpoints, but you would
need to expose them as routes first. Given the app is already two clean
processes, a host that just runs them is less work.

---

## 4. After deploying

```bash
curl https://your-host/healthz     # {"ok": true}
```

Then sign in and check **Status** — it shows source health, model routing, spend
by task, and the last 30 job runs. If briefs are not appearing, that page tells
you why before you go reading logs.

### Security

Login is a signed session cookie over scrypt-hashed passwords, which is
appropriate for a private dashboard with one or two users. It is not hardened
for public exposure: there is no rate limiting on the login form, no CSRF token,
and no 2FA. Keep the host's TLS on, and set `https_only=True` on the session
cookie in `web/app.py` once you are behind a domain you control.
