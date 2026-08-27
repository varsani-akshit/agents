# Deploying Alfred to Oracle Cloud (Always Free)

Oracle's Ampere A1 free tier gives 4 ARM cores, 24 GB RAM and 200 GB of storage,
free indefinitely. That runs Postgres and both Alfred processes with room to
spare, and removes the storage ceiling that every managed free tier eventually
hits.

Everything below assumes Ubuntu 24.04. Total time is about 25 minutes, most of
it the historical backfill.

---

## 1. Create the VM  *(your part)*

In the Oracle Cloud console: **Compute → Instances → Create instance**

| Setting | Value |
|---|---|
| Image | Canonical Ubuntu 24.04 |
| Shape | `VM.Standard.A1.Flex` — **Ampere**, 4 OCPU, 24 GB |
| SSH keys | Save the private key it generates; you cannot download it later |
| Boot volume | 100 GB is plenty (default 47 GB also works) |

**Ampere capacity is often exhausted in a region.** If creation fails with
"out of host capacity", either retry over a few hours or pick a different
availability domain — this is normal for the free tier and not a
misconfiguration.

### Open the ports

Two separate firewalls have to allow traffic, and missing the second is the
most common reason a fresh Oracle box appears unreachable.

**a. The VCN security list** — Networking → Virtual Cloud Networks → your VCN →
Security Lists → Default. Add ingress rules:

| Source CIDR | Protocol | Port |
|---|---|---|
| `0.0.0.0/0` | TCP | 443 |
| `0.0.0.0/0` | TCP | 80 |

Port 22 is already open by default.

**b. The instance's own iptables** — Oracle's Ubuntu images ship with rules that
drop everything except SSH. The bootstrap script below handles this.

---

## 2. Point a hostname at it  *(your part, optional but recommended)*

Add an **A record** for something like `alfred.akshit.fyi` pointing at the
instance's public IP. With a hostname, Caddy gets a real certificate
automatically; without one you would be on plain HTTP, and the login cookie
would travel in clear text.

---

## 3. Bootstrap  *(one command)*

```bash
ssh -i ~/path/to/key ubuntu@YOUR_PUBLIC_IP
git clone --depth 1 https://github.com/varsani-akshit/agents.git /tmp/alfred-src
bash /tmp/alfred-src/deploy/bootstrap.sh
```

That installs Python and PostgreSQL, enables pgvector, creates the database and
a random password, clones the app to `/opt/alfred`, installs dependencies,
loads the schema, opens the local firewall, and installs both systemd services.

It is idempotent — re-running it upgrades the checkout and leaves your `.env`
alone.

---

## 4. Add your key and load the data

```bash
nano /opt/alfred/.env          # add GEMINI_API_KEY, and FRED_API_KEY if you have one
sudo systemctl restart alfred-web alfred-scheduler

cd /opt/alfred
./.venv/bin/python cli.py backfill --period 12y --fred-days 6000   # ~3 min
./.venv/bin/python cli.py user akshit                              # sets your password
```

`GEMINI_API_KEY` is the only model key required — briefs, Ask, alerts,
classification, extraction and embeddings all run on Gemini. `OPENAI_API_KEY`
and `ANTHROPIC_API_KEY` are optional fallbacks.

Check it is alive:

```bash
curl -s localhost:8100/healthz     # {"ok":true}
systemctl status alfred-scheduler --no-pager
```

---

## 5. HTTPS

```bash
sudo apt-get install -y caddy
sudo cp /opt/alfred/deploy/Caddyfile.template /etc/caddy/Caddyfile
sudo sed -i "s|__DOMAIN__|alfred.akshit.fyi|" /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

Caddy obtains and renews the certificate on its own. The app listens only on
`127.0.0.1:8100`, so it is never reachable except through Caddy.

Then tighten the session cookie, now that TLS is real — in `web/app.py` set
`https_only=True` on `SessionMiddleware` and restart the web service.

---

## 6. Backups

The database is the whole system: the corpus, its embeddings, every brief.

```bash
sudo -u postgres sh -c 'mkdir -p /var/backups/alfred'
echo '0 4 * * * postgres pg_dump alfred | gzip > /var/backups/alfred/alfred-$(date +\%F).sql.gz && find /var/backups/alfred -mtime +14 -delete' \
  | sudo tee /etc/cron.d/alfred-backup
```

That keeps a fortnight of nightly dumps on the box. Copying them off it
periodically is what makes them a real backup — a snapshot on the same machine
does not survive losing the machine.

---

## 7. Updating

```bash
cd /opt/alfred && git pull
./.venv/bin/pip install -q -r requirements.txt
psql "$DATABASE_URL" -q -f schema.sql        # idempotent
sudo systemctl restart alfred-web alfred-scheduler
```

Both processes read `.env` once at startup, so **any** configuration change
needs that restart. This is worth taking seriously: after the embedding provider
changed locally, the still-running scheduler kept writing vectors from the old
model for an hour, and because search filters by embedding model those documents
silently stopped being findable. The **Status** page has an *Embedding coverage*
table — more than one row means the corpus is split, and the next few ticks
repair it automatically.

---

## What it costs

Nothing for the VM. Model usage is Gemini only, measured at **$0.06 per brief**
at high effort, two briefs a day — roughly **$4 a month**, against your existing
Gemini credits. Ingestion, statistics, charts and the knowledge graph cost
nothing per cycle: they are arithmetic over data already stored.
