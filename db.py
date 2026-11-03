"""Postgres access layer. Thin wrappers over psycopg3 — no ORM."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any, Iterable, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

import config

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=6, open=True)
    return _pool


@contextmanager
def conn() -> Iterator[psycopg.Connection]:
    with pool().connection() as c:
        yield c


def _scrub(v: Any) -> Any:
    """Strip NUL from strings on their way into Postgres.

    Postgres text cannot hold U+0000 in any form — it rejects the whole
    statement with "unsupported Unicode escape sequence", and inside a JSON
    value it takes the surrounding row with it. A single NUL in one scraped
    article killed an entire brief this way on 1 November.

    Nothing legitimately carries a NUL: it arrives from a truncated download,
    a mis-decoded byte, or a model emitting \u0000 inside JSON. Dropping it at
    the boundary is right because every writer is equally exposed, and the
    alternative is remembering to sanitise at each of a few hundred call
    sites. Scrubbing here cannot corrupt real data — there is no valid value
    this changes.
    """
    if not isinstance(v, str):
        return v
    if "\x00" in v:
        v = v.replace("\x00", "")
    # json.dumps does not emit a raw NUL — it emits the six-character escape
    # \u0000, which jsonb rejects on its own terms. Only scrub that inside a
    # value that is actually JSON: in a plain text column the same six
    # characters are storable and might be someone's code sample, and this is
    # the one case where scrubbing could alter real data. A JSON document
    # carrying the escape cannot be stored at all, so removing it there loses
    # nothing that was ever going to survive.
    if "\\u0000" in v and v[:1] in "{[":
        v = v.replace("\\u0000", "")
    return v


def _bind(params: Any) -> Any:
    """Pass mappings through; make everything else a tuple.

    psycopg picks its placeholder style from the parameter type, so wrapping a
    dict in tuple() turns %(name)s into "named placeholders require a mapping".
    """
    if isinstance(params, Mapping):
        return {k: _scrub(v) for k, v in params.items()}
    return tuple(_scrub(v) for v in params)


def query(sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> list[dict]:
    with conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, _bind(params))
        return cur.fetchall() if cur.description else []


def one(sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, _bind(params))
        return cur.rowcount


def executemany(sql: str, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn() as c, c.cursor() as cur:
        # Bulk inserts carry scraped text too, and one bad byte in one row
        # would reject the whole batch.
        cur.executemany(sql, [_bind(r) for r in rows])
        return cur.rowcount


def apply_schema() -> None:
    sql = (config.ROOT / "schema.sql").read_text()
    with conn() as c, c.cursor() as cur:
        cur.execute(sql)


def now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────── job observability ──────────────────────────
@contextmanager
def job(name: str) -> Iterator[dict]:
    """Record a job run. Mutate the yielded dict to attach detail."""
    detail: dict = {}
    row = one(
        "INSERT INTO job_runs (job) VALUES (%s) RETURNING id", (name,)
    )
    jid = row["id"] if row else None
    try:
        yield detail
    except Exception as exc:  # noqa: BLE001 - we re-raise after recording
        execute(
            "UPDATE job_runs SET finished_at=now(), ok=false, error=%s, detail=%s WHERE id=%s",
            (f"{type(exc).__name__}: {exc}"[:2000], json.dumps(detail, default=str), jid),
        )
        raise
    else:
        execute(
            "UPDATE job_runs SET finished_at=now(), ok=true, detail=%s WHERE id=%s",
            (json.dumps(detail, default=str), jid),
        )
