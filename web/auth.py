"""Dashboard login.

Deliberately small: a signed session cookie and scrypt-hashed passwords in
Postgres, with no dependency beyond what the app already carries. This gates a
private research dashboard for one or two people, not a public service — but the
parts that are cheap to get right (per-user salt, a real KDF, constant-time
comparison, no password ever written to a log) are done properly, because those
are the parts that are expensive to retrofit.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone

import db

log = logging.getLogger("mia.auth")

# scrypt parameters. n=2^15 with r=8 costs roughly 100ms per verification here,
# which is imperceptible on a login form and expensive in bulk.
_N, _R, _P, _DKLEN = 2 ** 15, 8, 1, 64
# These parameters need 128 * r * n = 33MB, just over OpenSSL's 32MB default,
# which fails with a bare "memory limit exceeded" that reads like a system fault
# rather than a parameter choice. State the ceiling instead of weakening the KDF
# to fit an implementation default.
_MAXMEM = 128 * _R * _N * 2

# /s/ is the share-link surface: a recipient has no account, and the token in
# the URL is the credential. Everything else stays behind the session gate.
PUBLIC_PATHS = ("/login", "/logout", "/healthz", "/static", "/s")


def _derive(password: str, salt: str) -> str:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=bytes.fromhex(salt),
        n=_N, r=_R, p=_P, dklen=_DKLEN, maxmem=_MAXMEM,
    ).hex()


def create_user(username: str, password: str) -> int:
    """Create or reset a user. Returns the user id."""
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    salt = secrets.token_hex(16)
    row = db.one(
        """INSERT INTO users (username, password_hash, salt) VALUES (%s, %s, %s)
           ON CONFLICT (username) DO UPDATE
             SET password_hash = EXCLUDED.password_hash, salt = EXCLUDED.salt
           RETURNING id""",
        (username.strip().lower(), _derive(password, salt), salt),
    )
    log.info("user %s created or reset", username)
    return row["id"]


def verify(username: str, password: str) -> dict | None:
    """Check a credential pair. Returns the user row, or None."""
    user = db.one(
        "SELECT id, username, password_hash, salt FROM users WHERE username=%s",
        ((username or "").strip().lower(),),
    )
    if not user:
        # Derive anyway against a throwaway salt. Returning early on an unknown
        # username makes the response measurably faster than a wrong password,
        # which tells an attacker which usernames exist.
        _derive(password or "", secrets.token_hex(16))
        return None
    if not hmac.compare_digest(_derive(password or "", user["salt"]), user["password_hash"]):
        return None
    db.execute("UPDATE users SET last_login = now() WHERE id = %s", (user["id"],))
    return {"id": user["id"], "username": user["username"]}


def list_users() -> list[dict]:
    return db.query(
        "SELECT username, created_at, last_login FROM users ORDER BY created_at")


def session_secret() -> str:
    """Cookie signing key.

    A generated fallback keeps local development working, but it changes on every
    restart, so sessions do not survive a redeploy. In production set the env var
    or everyone is logged out on each deploy.
    """
    secret = os.getenv("MIA_SESSION_SECRET") or os.getenv("SESSION_SECRET")
    if not secret:
        log.warning(
            "MIA_SESSION_SECRET is not set — using an ephemeral key, so sessions "
            "will not survive a restart"
        )
        return secrets.token_hex(32)
    return secret


def current_user(request) -> dict | None:
    user = request.session.get("user")
    return user if isinstance(user, dict) and user.get("username") else None


def login_session(request, user: dict) -> None:
    request.session["user"] = {"id": user["id"], "username": user["username"]}
    request.session["at"] = datetime.now(timezone.utc).isoformat()


def is_public(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PATHS)
