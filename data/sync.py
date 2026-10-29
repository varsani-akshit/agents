"""Which feeds run when. Each is run through data.run(), so one failing feed
is recorded and the rest still run."""
from __future__ import annotations

from data import run


def hourly() -> list[dict]:
    """While markets move: crypto bars and funding, company news, filings."""
    from data import crypto, filings, finnhub

    return [run("crypto", crypto.sync),
            run("finnhub", finnhub.sync, calendar=False),
            run("filings", filings.sync)]


def daily() -> list[dict]:
    """Once a day: macro series and proxies (with the news-based uncertainty
    and geopolitical-risk indices), calendars, positioning, the second
    opinion on closes. GDELT tone is not scheduled — see conf/data.yaml."""
    from data import finnhub, macro, positioning, twelve

    return [run("macro", macro.sync),
            run("positioning", positioning.sync),
            run("earnings", finnhub.earnings),
            run("twelve", twelve.sync)]


def backfill() -> list[dict]:
    """History for the feeds that have it."""
    from data import crypto, macro, positioning

    return [run("crypto", crypto.sync, backfill=True),
            run("macro", macro.sync, backfill=True),
            run("positioning", positioning.sync, backfill=True)]


SOURCES = {
    "crypto": lambda **kw: __import__("data.crypto", fromlist=["sync"]).sync(**kw),
    "finnhub": lambda **kw: __import__("data.finnhub", fromlist=["sync"]).sync(**kw),
    "earnings": lambda **kw: __import__("data.finnhub", fromlist=["earnings"]).earnings(),
    "filings": lambda **kw: __import__("data.filings", fromlist=["sync"]).sync(),
    "tone": lambda **kw: __import__("data.tone", fromlist=["sync"]).sync(**kw),
    "macro": lambda **kw: __import__("data.macro", fromlist=["sync"]).sync(**kw),
    "positioning": lambda **kw: __import__("data.positioning", fromlist=["sync"]).sync(**kw),
    "twelve": lambda **kw: __import__("data.twelve", fromlist=["sync"]).sync(),
}


def one(source: str, backfill: bool = False) -> dict:
    fn = SOURCES[source]
    kw = {"backfill": True} if backfill and source in ("crypto", "tone", "macro", "positioning") else {}
    return run(source, fn, **kw)
