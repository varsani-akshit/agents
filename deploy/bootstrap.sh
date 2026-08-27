#!/usr/bin/env bash
# Alfred bootstrap for Ubuntu 24.04 on Oracle Ampere A1 (arm64).
# Idempotent - safe to re-run. Run as the default `ubuntu` user.
set -euo pipefail

# iptables-persistent opens a dialog asking whether to save current rules, which
# hangs any unattended run. Answer it in advance.
export DEBIAN_FRONTEND=noninteractive
echo 'iptables-persistent iptables-persistent/autosave_v4 boolean true' | sudo debconf-set-selections
echo 'iptables-persistent iptables-persistent/autosave_v6 boolean true' | sudo debconf-set-selections

REPO="${REPO:-https://github.com/varsani-akshit/agents.git}"
APP_DIR="${APP_DIR:-/opt/alfred}"
DB_NAME=alfred
DB_USER=alfred

echo "==> System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip git curl \
     postgresql postgresql-contrib build-essential libpq-dev iptables-persistent

PGMAJ="$(psql --version | grep -oE '[0-9]+' | head -1)"
echo "==> pgvector for PostgreSQL ${PGMAJ}"
sudo apt-get install -y -qq "postgresql-${PGMAJ}-pgvector" || {
  echo "   package unavailable, building from source"
  sudo apt-get install -y -qq "postgresql-server-dev-${PGMAJ}"
  tmp="$(mktemp -d)"; git clone --depth 1 https://github.com/pgvector/pgvector.git "$tmp"
  make -C "$tmp" && sudo make -C "$tmp" install
}

echo "==> Database"
DB_PASS="$(openssl rand -hex 24)"
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1; then
  sudo -u postgres psql -qc "CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}'"
  NEW_DB=1
fi
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 \
  || sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
sudo -u postgres psql -d "${DB_NAME}" -qc 'CREATE EXTENSION IF NOT EXISTS vector'

echo "==> Application"
sudo mkdir -p "$APP_DIR"; sudo chown "$USER:$USER" "$APP_DIR"
if [ -d "$APP_DIR/.git" ]; then git -C "$APP_DIR" pull --ff-only; else git clone --depth 1 "$REPO" "$APP_DIR"; fi
cd "$APP_DIR"
python3.12 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

if [ ! -f .env ]; then
  cp .env.template .env
  {
    echo "DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"
    echo "MIA_SESSION_SECRET=$(openssl rand -hex 32)"
    echo "MIA_EMBED_PROVIDER=gemini"
    echo "MIA_DIGEST_MODEL=gemini-flash-latest"
    echo "MIA_ASK_MODEL=gemini-flash-latest"
    echo "MIA_DIGEST_EFFORT=high"
    echo "MIA_DAILY_USD_CAP=999"
    echo "MIA_TOTAL_USD_CAP=999"
  } >> .env
  echo "   .env created - add GEMINI_API_KEY and FRED_API_KEY before starting"
elif [ "${NEW_DB:-0}" = "1" ]; then
  echo "   NOTE: a new database role was created but .env already exists;"
  echo "   check DATABASE_URL points at it."
fi

echo "==> Schema"
set -a; . ./.env; set +a
psql "$DATABASE_URL" -q -f schema.sql

echo "==> Firewall (Oracle images drop everything but SSH)"
sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 5 -p tcp --dport 80 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save >/dev/null

echo "==> systemd"
sudo cp deploy/alfred-web.service deploy/alfred-scheduler.service /etc/systemd/system/
sudo sed -i "s|__APP_DIR__|${APP_DIR}|g; s|__USER__|${USER}|g" \
  /etc/systemd/system/alfred-web.service /etc/systemd/system/alfred-scheduler.service
sudo systemctl daemon-reload
sudo systemctl enable --now alfred-web alfred-scheduler

echo
echo "Bootstrap complete. Next:"
echo "  1. nano ${APP_DIR}/.env          # add GEMINI_API_KEY, FRED_API_KEY"
echo "  2. sudo systemctl restart alfred-web alfred-scheduler"
echo "  3. cd ${APP_DIR} && ./.venv/bin/python cli.py backfill --period 12y --fred-days 6000"
echo "  4. ./.venv/bin/python cli.py user akshit"
echo "  5. curl -s localhost:8100/healthz"
