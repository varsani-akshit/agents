"""Delivery. Console + durable file outbox now; Slack behind a flag for later.

Platform choice is deferred, so nothing above this module knows or cares where
output lands. Adding a destination means adding a sink here, not touching the
brain.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

import config

log = logging.getLogger("mia.notify")
console = Console()

SEV_STYLE = {"Critical": "bold red", "High": "bold yellow", "Medium": "cyan", "Low": "dim"}


def _outbox_path(kind: str, when: datetime | None = None) -> Path:
    when = when or datetime.now(timezone.utc)
    day = config.OUTBOX / when.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    return day / f"{when.strftime('%H%M%S')}-{kind}.md"


def _write_file(kind: str, title: str, body: str, meta: dict | None = None) -> Path:
    path = _outbox_path(kind)
    front = [
        "---",
        f"kind: {kind}",
        f"title: {title}",
        f"generated_at: {datetime.now(timezone.utc).isoformat()}",
    ]
    if meta:
        front.append(f"meta: {json.dumps(meta, default=str)}")
    front.append("---\n")
    path.write_text("\n".join(front) + body + "\n")
    return path


def _slack(webhook: str, text: str) -> bool:
    if not (config.NOTIFY_SLACK and webhook):
        return False
    try:
        r = httpx.post(webhook, json={"text": text[:38000]}, timeout=15)
        return r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        log.warning("slack post failed: %s", exc)
        return False


# ─────────────────────────────────── public ─────────────────────────────────
def alert(payload: dict) -> Path:
    """Deliver a critical alert."""
    sev = payload.get("severity", "High")
    style = SEV_STYLE.get(sev, "white")
    detail = payload.get("detail") or {}
    detail_line = "  ".join(f"{k}={v}" for k, v in detail.items()
                            if isinstance(v, (int, float, str)))

    console.print()
    console.print(
        Panel(
            f"{payload['text']}\n\n[dim]{detail_line}[/dim]",
            title=f"[{style}]⚠  {payload['title']}[/{style}]",
            subtitle=f"[dim]{payload.get('rule')} · {payload.get('created_at','')[:19]}[/dim]",
            border_style=style.split()[-1],
        )
    )

    body = f"**{payload['title']}**\n\n{payload['text']}\n\n`{detail_line}`"
    path = _write_file("alert", payload["title"], body, {"rule": payload.get("rule")})
    _slack(config.SLACK_WEBHOOK_CRITICAL, f":rotating_light: *{payload['title']}*\n{payload['text']}")
    return path


def digest(result: dict) -> Path:
    """Deliver a 6-hour digest."""
    console.print()
    console.rule(f"[bold cyan]{result['title']}[/bold cyan]")
    console.print(Markdown(result["body"]))
    meta = (
        f"regime={result.get('regime')}  turns={result.get('turns')}  "
        f"tools={len(result.get('tool_calls', []))}  cost=${result.get('usd', 0):.4f}"
    )
    console.print(f"[dim]{meta}[/dim]")

    path = _write_file(
        "digest",
        result["title"],
        result["body"],
        {
            "regime": result.get("regime"),
            "usd": result.get("usd"),
            "analysis_id": result.get("analysis_id"),
            "world_model_version": result.get("world_model_version"),
        },
    )
    _slack(config.SLACK_WEBHOOK_DIGESTS, f"*{result['title']}*\n\n{result['body'][:3500]}")
    return path


def answer(result: dict) -> Path:
    console.print()
    console.print(Panel(f"[bold]{result['question']}[/bold]", border_style="cyan"))
    console.print(Markdown(result["answer"]))
    console.print(
        f"[dim]tools={', '.join(result.get('tools_used') or []) or 'none'}  "
        f"turns={result.get('turns')}  cost=${result.get('usd', 0):.4f}[/dim]"
    )
    return _write_file(
        "answer",
        result["question"][:80],
        f"**Q: {result['question']}**\n\n{result['answer']}",
        {"usd": result.get("usd"), "tools": result.get("tools_used")},
    )


def info(msg: str) -> None:
    console.print(f"[dim]{datetime.now(timezone.utc).strftime('%H:%M:%S')}[/dim] {msg}")


def error(msg: str) -> None:
    console.print(f"[bold red]error[/bold red] {msg}")
