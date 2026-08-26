"""Postgres access layer. Thin wrappers over psycopg3 — no ORM."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
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


def query(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    with conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall() if cur.description else []


def one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, tuple(params))
        return cur.rowcount


def executemany(sql: str, rows: list[tuple]) -> int:
    if not rows:
        return 0
    with conn() as c, c.cursor() as cur:
        cur.executemany(sql, rows)
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
