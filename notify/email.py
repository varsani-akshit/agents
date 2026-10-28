"""Email delivery through Resend — send-only, never a mailbox login.

The design choice is what Alfred holds, not how mail travels. Sending from
Gmail would mean storing an app password, and a Gmail app password opens the
whole mailbox: read, search, delete, send as you. A Resend API key scoped to
"sending access" can do exactly one thing — send mail — and is revoked in one
click without touching the inbox. The owner's Gmail is only ever a recipient.

Plain REST over httpx, like the model clients: one POST does not need an SDK.
"""
from __future__ import annotations

import hashlib
import logging

import httpx

import config

log = logging.getLogger("alfred.email")

API = "https://api.resend.com/emails"


def configured() -> bool:
    return bool(config.RESEND_API_KEY and config.EMAIL_TO)


def send(subject: str, text: str, html: str | None = None,
         key: str | None = None) -> str | None:
    """Send one email to the owner. Returns Resend's message id, or None.

    `key` makes the send idempotent: a scheduler retry or a double tick with
    the same key delivers once, not twice. Defaults to a hash of the content,
    so an identical alert cannot arrive as two emails a minute apart.

    Never raises: a mail outage must not take down the job that wanted to
    tell you something. The failure is logged and the caller gets None.
    """
    if not configured():
        log.info("email not configured; skipped %r", subject)
        return None
    idem = key or hashlib.sha256(f"{subject}\n{text}".encode()).hexdigest()
    payload = {
        "from": config.EMAIL_FROM,
        "to": [config.EMAIL_TO],
        "subject": subject[:200],
        "text": text,
    }
    if html:
        payload["html"] = html
    try:
        r = httpx.post(
            API, json=payload, timeout=20,
            headers={"Authorization": f"Bearer {config.RESEND_API_KEY}",
                     "Idempotency-Key": idem[:256]},
        )
    except httpx.HTTPError as exc:
        log.warning("email send failed: %s", exc)
        return None
    if r.status_code >= 300:
        # Resend explains itself in `message` — e.g. an unverified sender, or
        # a recipient other than the account owner while on the test domain.
        try:
            reason = r.json().get("message")
        except ValueError:
            reason = r.text[:200]
        log.warning("email refused (%s): %s", r.status_code, reason)
        return None
    return r.json().get("id")
