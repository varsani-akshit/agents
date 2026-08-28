"""Tracing for the agent army: Trodo runs/spans plus a local mirror.

Every agent run is written twice, deliberately. Trodo is the operations view —
latency, cost, failures, the trace tree at app.trodo.ai. The local `agent_runs`
row is the product: drill-down pages render an agent's reasoning from Postgres,
which must work even if the tracing backend is down, rate-limited, or has aged
the run out. Neither write may break the agent itself, so everything here fails
into a log line, never a raise.

Run boundaries follow one rule: an agent that is independently triggered is its
own run (watchman, librarian, quant, sentinel, deep-research, ask); a joint
pipeline is one run whose stages and sub-agents are spans (the brief). Nested
`wrap_agent` would create sibling runs, not children — so pipeline stages use
`stage()` / `tool_span()` / `llm_call()` inside the single enclosing `run()`.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import db

log = logging.getLogger("mia.observe")

DISTINCT_ID = "akshit"  # single-reader system; matches the dashboard login

_initialised = False


def _trodo():
    """The trodo module if tracing is configured, else None."""
    if not os.getenv("TRODO_SITE_ID"):
        return None
    try:
        import trodo

        return trodo
    except ImportError:
        return None


def init(service: str) -> None:
    """Initialise tracing once per process. Safe to call when unconfigured.

    httpx/requests instrumentation is disabled: the harvester makes hundreds of
    feed fetches per tick, and our LLM calls are raw httpx recorded explicitly
    via `llm_call`/`record_llm` — auto HTTP spans would bury the signal and
    double-count the model traffic. The anthropic SDK instrumentor stays on;
    it prices its own spans and `record_llm` skips that provider to match.
    """
    global _initialised
    t = _trodo()
    if _initialised or t is None:
        return
    try:
        t.init(
            site_id=os.environ["TRODO_SITE_ID"],
            disable_instrumentations=["httpx", "requests"],
        )
        _initialised = True
        log.info("trodo tracing initialised (service=%s)", service)
    except Exception as exc:  # noqa: BLE001
        log.warning("trodo init failed, tracing disabled: %s", exc)


class RunRecorder:
    """Handle yielded by `run()`: mirrors input/output to Trodo and Postgres."""

    def __init__(self, agent: str, trigger: str, trodo_run):
        self.agent = agent
        self.trigger = trigger
        self._trodo_run = trodo_run
        self.input: Any = None
        self.output: Any = None
        self.meta: dict = {}
        self.id: int | None = None

    def set_input(self, value: Any) -> None:
        self.input = value
        if self._trodo_run:
            try:
                self._trodo_run.set_input(value)
            except Exception:  # noqa: BLE001
                pass

    def set_output(self, value: Any) -> None:
        self.output = value
        if self._trodo_run:
            try:
                self._trodo_run.set_output(value)
            except Exception:  # noqa: BLE001
                pass

    def set_metadata(self, **kwargs: Any) -> None:
        self.meta.update(kwargs)
        if self._trodo_run:
            try:
                self._trodo_run.set_metadata(kwargs)
            except Exception:  # noqa: BLE001
                pass


def _js(value: Any) -> str:
    return json.dumps(value, default=str)[:65536]


@contextmanager
def run(agent: str, *, trigger: str = "schedule", meta: dict | None = None) -> Iterator[RunRecorder]:
    """One agent run: a Trodo run plus an `agent_runs` row.

    The Postgres row is written on entry and completed on exit, so a crashed
    agent still leaves a record with status='error' rather than vanishing.
    """
    t = _trodo()
    row = db.one(
        """INSERT INTO agent_runs (agent, trigger, status, input, meta)
           VALUES (%s, %s, 'running', NULL, %s) RETURNING id""",
        (agent, trigger, _js(meta or {})),
    )
    rid = row["id"] if row else None

    def _finish(rec: RunRecorder, status: str, error: str | None) -> None:
        try:
            db.execute(
                """UPDATE agent_runs
                   SET status=%s, error=%s, input=%s, output=%s, meta=%s,
                       ended_at=now()
                   WHERE id=%s""",
                (status, error, _js(rec.input), _js(rec.output),
                 _js(rec.meta), rid),
            )
        except Exception:  # noqa: BLE001
            log.exception("agent_runs update failed for %s", agent)

    if t and _initialised:
        try:
            with t.wrap_agent(agent, distinct_id=DISTINCT_ID,
                              metadata={"trigger": trigger, **(meta or {})}) as trun:
                rec = RunRecorder(agent, trigger, trun)
                rec.id = rid
                try:
                    yield rec
                except Exception as exc:
                    _finish(rec, "error", f"{type(exc).__name__}: {exc}"[:2000])
                    raise
                _finish(rec, "ok", None)
            return
        except Exception:
            raise
    # Tracing off: local record only.
    rec = RunRecorder(agent, trigger, None)
    rec.id = rid
    try:
        yield rec
    except Exception as exc:
        _finish(rec, "error", f"{type(exc).__name__}: {exc}"[:2000])
        raise
    _finish(rec, "ok", None)


@contextmanager
def stage(name: str, *, kind: str = "generic", input: Any = None) -> Iterator[Any]:
    """A named span inside the current run (pipeline stage, sub-agent).

    Yields the Trodo SpanHandle, or a no-op stand-in when tracing is off, so
    callers can unconditionally set_output/set_attribute.
    """
    t = _trodo()
    if t and _initialised:
        try:
            with t.span(name, kind=kind, input=input) as s:
                yield s
            return
        except Exception:
            raise
    yield _NoopSpan()


class _NoopSpan:
    def set_input(self, *a, **k):  # noqa: D102
        pass

    def set_output(self, *a, **k):
        pass

    def set_attribute(self, *a, **k):
        pass

    def set_llm(self, *a, **k):
        pass

    def set_error(self, *a, **k):
        pass


def record_llm(*, spec: str, purpose: str, input_tokens: int, output_tokens: int,
               usd: float, prompt: Any = None, completion: Any = None) -> None:
    """Record one raw-HTTP model call as an llm span under the active run.

    No-op outside a run, when tracing is off, and for the anthropic provider
    (its SDK auto-instruments; recording again would double-count tokens).
    """
    t = _trodo()
    if not (t and _initialised) or spec.startswith("anthropic"):
        return
    provider, _, model = spec.partition(":")
    try:
        t.track_llm_call(
            model=model, provider=provider, name=purpose,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost=usd, prompt=prompt, completion=completion,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("track_llm_call failed: %s", exc)


def flush() -> None:
    t = _trodo()
    if t and _initialised:
        try:
            t.flush()
        except Exception:  # noqa: BLE001
            pass
