"""Restricted Python sandbox: how agents get numbers without inventing them.

The core rule is "numbers from code, meaning from the model". Precomputed
statistics cover the common questions; this sandbox covers the long tail — an
Analyst who wants gold's correlation to real yields over a window nobody
precomputed writes three lines of pandas instead of guessing.

Each execution is a fresh subprocess: no network (all sockets are severed
before user code runs), a hard wall-clock timeout, a memory cap, and no
argument surface back into the parent. Statelessness is the safety model —
nothing an agent does in one call can leak into the next, and every result is
reproducible from the code that produced it.

The child is offered `series(symbol_or_fred_id)` which returns a pandas Series
of daily closes (or FRED values) — data flows in read-only through a temp file,
never through a live database handle.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import db

log = logging.getLogger("mia.sandbox")

TIMEOUT_S = 12
MEMORY_MB = 512
MAX_OUTPUT = 20_000

_PRELUDE = """
import resource, socket, builtins, json, sys
# Memory cap. Enforced on Linux (the production VM); macOS refuses to lower
# RLIMIT_AS below its own notion of the hard limit, so development boxes fall
# back to the wall-clock timeout as the only backstop.
try:
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
except ValueError:
    pass

# Sever the network before user code runs. A subclass rather than a bare
# function: ssl subclasses socket.socket at import time, and replacing the
# class with a function turns "network is disabled" into an inscrutable
# TypeError from inside ssl's class definition.
class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise RuntimeError("network access is disabled in the sandbox")
def _no_net(*a, **k):
    raise RuntimeError("network access is disabled in the sandbox")
socket.socket = _NoNet
socket.create_connection = _no_net

import pandas as pd
import numpy as np

_DATA = json.load(open({data_path!r}))

def series(name):
    \"\"\"Daily close (instruments) or value (FRED) as a pandas Series.\"\"\"
    if name not in _DATA:
        raise KeyError(f"unknown series {{name!r}}; available: {{sorted(_DATA)}}")
    d = _DATA[name]
    s = pd.Series(d["values"], index=pd.to_datetime(d["dates"]), name=name)
    return s.sort_index()

AVAILABLE = sorted(_DATA)
"""


def _export_series(names: list[str]) -> dict:
    """Pull the requested series out of Postgres for the child process."""
    out: dict[str, dict] = {}
    for name in names[:12]:
        rows = db.query(
            """SELECT ts::date AS d, price AS v FROM prices
               WHERE symbol = %s AND grain = '1d' ORDER BY ts""",
            (name,),
        )
        if not rows:
            rows = db.query(
                "SELECT ts::date AS d, value AS v FROM fred_series WHERE series_id = %s ORDER BY ts",
                (name,),
            )
        if rows:
            out[name] = {
                "dates": [str(r["d"]) for r in rows],
                "values": [float(r["v"]) for r in rows if r["v"] is not None],
            }
    return out


def run(code: str, series_names: list[str] | None = None) -> dict:
    """Execute agent-written analysis code. Returns {ok, stdout, error}.

    The child prints its result; whatever lands on stdout is the answer. The
    parent never evals anything that comes back.
    """
    with tempfile.TemporaryDirectory(prefix="alfred-sbx-") as tmp:
        data_path = Path(tmp) / "data.json"
        data_path.write_text(json.dumps(_export_series(series_names or [])))
        script = Path(tmp) / "job.py"
        script.write_text(
            _PRELUDE.format(mem=MEMORY_MB * 1024 * 1024, data_path=str(data_path))
            + "\n"
            + textwrap.dedent(code)
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=TIMEOUT_S,
                cwd=tmp, env={"PYTHONPATH": _site_packages()},
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "error": f"timed out after {TIMEOUT_S}s"}
    stdout = proc.stdout[:MAX_OUTPUT]
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"ok": False, "stdout": stdout, "error": "\n".join(tail)[:2000]}
    return {"ok": True, "stdout": stdout, "error": None}


def _site_packages() -> str:
    """The venv's site-packages, so -I (isolated) still finds pandas."""
    for p in sys.path:
        if p.endswith("site-packages"):
            return p
    return ""
