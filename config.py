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

# Hard spend guards (USD). The ledger refuses calls once these are reached.
DAILY_USD_CAP = _f("MIA_DAILY_USD_CAP", 0.60)
TOTAL_USD_CAP = _f("MIA_TOTAL_USD_CAP", 3.00)

# Scheduled jobs stop at this, leaving the remainder of TOTAL_USD_CAP available
# for interactive `ask`. Without it a busy night of autonomous cycles can consume
# the whole balance and leave the user unable to ask their own questions —
# the one use that actually needs the budget to be there.
AUTONOMOUS_USD_CAP = _f("MIA_AUTONOMOUS_USD_CAP", TOTAL_USD_CAP * 0.75)

# Output routing. Slack stays off until a platform is chosen.
NOTIFY_SLACK = os.getenv("MIA_NOTIFY_SLACK", "false").lower() in {"1", "true", "yes"}
SLACK_WEBHOOK_CRITICAL = os.getenv("SLACK_WEBHOOK_CRITICAL", "")
SLACK_WEBHOOK_DIGESTS = os.getenv("SLACK_WEBHOOK_DIGESTS", "")

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
