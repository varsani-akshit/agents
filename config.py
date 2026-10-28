"""Central configuration. Reads .env once, exposes typed settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/mia")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

DIGEST_MODEL = os.getenv("MIA_DIGEST_MODEL", os.getenv("DIGEST_MODEL", "gemini-flash-latest"))
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gemini-flash-latest")
# `ask` stays on Claude independently of the digest model: server-side web
# search exists only on the Anthropic path, and interactive questions are where
# fresh-source lookup matters most. Decoupled deliberately — inheriting from
# DIGEST_MODEL would silently hand a GPT id to the Anthropic loop.
ASK_MODEL = os.getenv("MIA_ASK_MODEL", "gemini-flash-latest")

# Task routing. "provider:model"; a bare name means Anthropic. High-volume
# schema-constrained work goes to a fast cheap model; deep reasoning stays on
# Claude. See brain/llm.py.
CLASSIFY_SPEC = os.getenv("MIA_CLASSIFY_SPEC", "gemini:gemini-flash-latest")
EXTRACT_SPEC = os.getenv("MIA_EXTRACT_SPEC", "gemini:gemini-flash-latest")
ALERT_SPEC = os.getenv("MIA_ALERT_SPEC", "gemini:gemini-flash-latest")

# Where a routed task goes when its own provider errors. Must name a real model
# on a real provider: this used to be built as f"anthropic:{CLASSIFY_MODEL}",
# which silently became "anthropic:gemini-flash-latest" the moment
# classification moved to Gemini — a fallback that 404s is worse than none,
# because it only fires when something is already wrong.
FALLBACK_SPEC = os.getenv("MIA_FALLBACK_SPEC", "gemini:gemini-flash-latest")

# Azure AI Foundry: one endpoint, one key, a deployment per model. When it is
# configured, the high-volume tasks run on Llama 4 Maverick there; otherwise
# on Gemini Flash. brain/router.py maps the rest of the roles.
AZURE_READY = bool(os.getenv("AZURE_FOUNDRY_ENDPOINT") and os.getenv("AZURE_FOUNDRY_API_KEY"))
_MAVERICK = "azure:" + os.getenv("AZURE_DEPLOY_MAVERICK", "Llama-4-Maverick-17B-128E-Instruct-FP8")
if AZURE_READY:
    CLASSIFY_SPEC = os.getenv("MIA_CLASSIFY_SPEC", _MAVERICK)
    EXTRACT_SPEC = os.getenv("MIA_EXTRACT_SPEC", _MAVERICK)

# Hard spend guards (USD). The ledger refuses calls once these are reached.
DAILY_USD_CAP = _f("MIA_DAILY_USD_CAP", 0.60)
TOTAL_USD_CAP = _f("MIA_TOTAL_USD_CAP", 3.00)

# Scheduled jobs stop at this, leaving the remainder of TOTAL_USD_CAP available
# for interactive `ask`. Without it a busy night of autonomous cycles can consume
# the whole balance and leave the user unable to ask their own questions —
# the one use that actually needs the budget to be there.
AUTONOMOUS_USD_CAP = _f("MIA_AUTONOMOUS_USD_CAP", TOTAL_USD_CAP * 0.75)

# What one reader may spend on Ask in a calendar month. Admins are exempt in
# code. This is the default for anyone without a budget of their own; a
# per-person figure lives on users.monthly_usd.
READER_MONTHLY_USD = _f("MIA_READER_MONTHLY_USD", 5.00)


# Private mode. The owner is the one account `mia lockdown` leaves standing.
# Share links are off unless switched back on: a link is a credential, and a
# dashboard that is about to show holdings should not be handing them out.
OWNER = os.getenv("MIA_OWNER", "akshit").strip().lower()
SHARING_ENABLED = os.getenv("MIA_SHARING", "false").lower() in {"1", "true", "yes"}

# Brokers, read-only for now. Tokens live on disk under SECRETS, mode 0600,
# never in the database or the logs.
SECRETS = ROOT / "secrets"
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
WEBULL_APP_KEY = os.getenv("WEBULL_APP_KEY", "")
WEBULL_APP_SECRET = os.getenv("WEBULL_APP_SECRET", "")
# "au" for Webull Australia, "us" for Webull US. The SDK picks the host from it.
WEBULL_REGION = os.getenv("WEBULL_REGION", "au").strip().lower()

# Email, send-only via Resend. The key should be created with "Sending access",
# not "Full access". Until a domain is verified in Resend, mail can only go
# from onboarding@resend.dev to the address the Resend account was opened with.
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
EMAIL_TO = os.getenv("MIA_EMAIL_TO", "").strip()
EMAIL_FROM = os.getenv("MIA_EMAIL_FROM", "Alfred <onboarding@resend.dev>").strip()

EMBED_MODEL = "voyage-finance-2"
EMBED_DIM = 1024

OUTBOX = ROOT / "outbox"
LOGS = ROOT / "logs"
PRINCIPLES_DIR = ROOT / "brain" / "principles"
CONFIG_DIR = ROOT / "conf"

for _d in (OUTBOX, LOGS):
    _d.mkdir(exist_ok=True)

# Anthropic pricing, USD per million tokens. Conservative (non-promotional) rates
# so the budget guard never under-estimates.
PRICING = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
WEB_SEARCH_USD = 10.0 / 1000  # $10 per 1,000 searches


def price_for(model: str) -> tuple[float, float]:
    for key, val in PRICING.items():
        if model.startswith(key):
            return val
    return (5.0, 25.0)  # unknown model: assume expensive
