"""Fetch a page at a time, not everything.

Several pages grew unbounded. The simulations index was a flat `LIMIT 84`,
which at twelve runs a day silently began dropping runs after a week — the
worst failure mode, because the page still looks complete. One run's candidate
list was capped at 500 rows, and a run holds over a thousand; expanding a
group rendered a quarter-megabyte of HTML in one response.

A cap that truncates without saying so is worse than a pager: you cannot tell
a short list from a trimmed one. So every list that can grow says how many
there are, shows one page, and lets you walk.

Page numbers rather than cursors on purpose: these are stable, ordered,
owner-only lists, and being able to link someone to page 3 is worth more here
than the cursor's correctness under concurrent insertion. A row arriving
mid-walk shifts the boundary by one, which for a reading surface is a
non-event.
"""
from __future__ import annotations

from dataclasses import dataclass

PER_PAGE = 50
MAX_PER_PAGE = 200


@dataclass(frozen=True)
class Page:
    """One slice of a list, and everything a pager needs to render itself."""

    rows: list
    total: int
    page: int
    per_page: int

    @property
    def pages(self) -> int:
        return max(1, -(-self.total // self.per_page))  # ceil

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    @property
    def has_next(self) -> bool:
        return self.page < self.pages

    @property
    def first(self) -> int:
        """1-based index of the first row shown, for "51–100 of 1,121"."""
        return 0 if not self.total else (self.page - 1) * self.per_page + 1

    @property
    def last(self) -> int:
        return min(self.page * self.per_page, self.total)

    @property
    def window(self) -> list[int]:
        """Page numbers worth offering: the ends, and a few either side.

        0 marks an elision, so the template can draw a gap instead of 23
        numbers nobody will click.
        """
        if self.pages <= 9:
            return list(range(1, self.pages + 1))
        near = {1, self.pages, *range(max(1, self.page - 2), min(self.pages, self.page + 2) + 1)}
        out: list[int] = []
        for n in sorted(near):
            if out and n != out[-1] + 1:
                out.append(0)
            out.append(n)
        return out


def params(page: int | str | None, per_page: int | str | None = None) -> tuple[int, int, int]:
    """Clean (page, per_page, offset) from whatever arrived in the query string.

    Anything unparseable becomes page 1 rather than an error: a mistyped URL
    should show the list, not a stack trace. per_page is capped because the
    query string is not a place to ask for forty thousand rows.
    """
    try:
        p = max(1, int(page or 1))
    except (TypeError, ValueError):
        p = 1
    try:
        n = int(per_page or PER_PAGE)
    except (TypeError, ValueError):
        n = PER_PAGE
    n = min(max(1, n), MAX_PER_PAGE)
    return p, n, (p - 1) * n


def slice_query(sql: str, count_sql: str, args: list | tuple = (),
                *, page: int = 1, per_page: int = PER_PAGE) -> Page:
    """Run the count and the page in one place.

    `sql` must already carry its ORDER BY and end where LIMIT/OFFSET belong;
    an unordered page is a different set of rows each time you look at it.
    """
    import db

    page, per_page, _ = params(page, per_page)
    total = db.one(count_sql, list(args))["n"]
    # A page past the end is a stale link or a typo, not an error. Land on the
    # last page rather than showing an empty table under a count that reads
    # "49,901-805 of 805".
    page = min(page, max(1, -(-total // per_page)))
    rows = db.query(f"{sql} LIMIT %s OFFSET %s", [*args, per_page, (page - 1) * per_page])
    return Page(rows=rows, total=total, page=page, per_page=per_page)
