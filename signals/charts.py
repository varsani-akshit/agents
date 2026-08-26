"""Chart rendering for the digest. Deterministic — no model involved.

Charts are data visualisation, so they are generated in code from the same
series the stats pack uses. The model references them and interprets them; it
never invents one. Each digest gets a standard pack plus any extras its own
content calls for.

Rendered as PNG into outbox/<date>/charts/ and referenced from the markdown.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a scheduled run
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import db  # noqa: E402
from signals import stats  # noqa: E402

log = logging.getLogger("mia.charts")

# Dark, legible, consistent across the pack.
INK = "#e8e6e3"
GRID = "#2a2d33"
BG = "#16181d"
ACCENT = {
    "gold": "#e2b13c", "silver": "#b9c0c8", "btc": "#f5922f", "eth": "#7b8cde",
    "up": "#3fb27f", "down": "#e0575b", "neutral": "#6c7480", "line": "#5aa9e6",
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.edgecolor": GRID, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "600",
    "figure.dpi": 130,
})


def _outdir(when: datetime | None = None) -> Path:
    when = when or datetime.now(timezone.utc)
    d = config.OUTBOX / when.strftime("%Y-%m-%d") / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _finish(fig, ax_or_axes, path: Path) -> str:
    for ax in np.atleast_1d(ax_or_axes).ravel():
        ax.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


# ─────────────────────────────────── charts ─────────────────────────────────
def cross_asset_performance(wide: pd.DataFrame, outdir: Path) -> str | None:
    """Horizontal bars: 1-month % change by asset, coloured by direction."""
    rows = []
    classes = {r["symbol"]: r["asset_class"]
               for r in db.query("SELECT symbol, asset_class FROM instruments")}
    for sym in wide.columns:
        if sym.startswith("US") and sym.endswith("Y"):
            continue  # yields belong on the curve chart, not a % bar
        v = stats.pct_change(wide[sym].dropna(), 21)
        if v is not None:
            rows.append((sym, v, classes.get(sym, "other")))
    if not rows:
        return None
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.28 * len(rows))))
    names = [f"{s}  ({c})" for s, _, c in rows]
    vals = [v for _, v, _ in rows]
    colors = [ACCENT["up"] if v >= 0 else ACCENT["down"] for v in vals]
    ax.barh(names, vals, color=colors, height=0.62)
    ax.axvline(0, color=INK, linewidth=0.8, alpha=0.6)
    ax.set_title("Cross-asset performance — 1 month (%)")
    ax.set_xlabel("% change")
    for i, v in enumerate(vals):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    return _finish(fig, ax, outdir / "cross_asset_performance.png")


def normalised_performance(wide: pd.DataFrame, outdir: Path, days: int = 90) -> str | None:
    """Rebased lines: the debasement complex against equities and the dollar."""
    picks = [("GOLD", ACCENT["gold"]), ("SILVER", ACCENT["silver"]),
             ("BTC", ACCENT["btc"]), ("SPX", ACCENT["line"]), ("DXY", ACCENT["neutral"])]
    have = [(s, c) for s, c in picks if s in wide.columns]
    if not have:
        return None
    sub = wide[[s for s, _ in have]].dropna().iloc[-days:]
    if len(sub) < 20:
        return None
    rebased = sub / sub.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    for sym, color in have:
        ax.plot(rebased.index, rebased[sym], label=sym, color=color, linewidth=1.7)
    ax.axhline(100, color=INK, linewidth=0.7, alpha=0.4, linestyle="--")
    ax.set_title(f"Rebased to 100 — last {len(sub)} sessions")
    ax.set_ylabel("index (start = 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, ncols=len(have), fontsize=8, loc="upper left")
    return _finish(fig, ax, outdir / "normalised_performance.png")


def correlation_heatmap(wide: pd.DataFrame, outdir: Path, window: int = 30) -> str | None:
    """30-day return-correlation matrix across the core complex."""
    cols = [c for c in stats.CORE if c in wide.columns]
    if len(cols) < 4:
        return None
    rets = wide[cols].pct_change().dropna().iloc[-window:]
    if len(rets) < window // 2:
        return None
    m = rets.corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(m.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#101216" if abs(v) > 0.45 else INK)
    ax.set_title(f"{window}-day return correlations")
    fig.colorbar(im, ax=ax, shrink=0.75, label="correlation")
    ax.grid(False)
    fig.tight_layout()
    path = outdir / "correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def yield_curve_chart(outdir: Path) -> str | None:
    """Current curve against one month ago — shape change, not just level."""
    tenors = [("US2Y", 2), ("US5Y", 5), ("US10Y", 10), ("US30Y", 30)]
    now_pts, then_pts = [], []
    for sym, yrs in tenors:
        rows = db.query(
            "SELECT ts, price FROM prices WHERE symbol=%s AND grain='1d' "
            "ORDER BY ts DESC LIMIT 25", (sym,))
        if not rows:
            continue
        now_pts.append((yrs, float(rows[0]["price"])))
        if len(rows) >= 21:
            then_pts.append((yrs, float(rows[20]["price"])))
    if len(now_pts) < 3:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(*zip(*now_pts), marker="o", color=ACCENT["line"], linewidth=2, label="today")
    if len(then_pts) >= 3:
        ax.plot(*zip(*then_pts), marker="o", color=ACCENT["neutral"],
                linewidth=1.4, linestyle="--", label="1 month ago")
    ax.set_title("US Treasury curve")
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("yield %")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, ax, outdir / "yield_curve.png")


def ratio_history(wide: pd.DataFrame, outdir: Path, days: int = 252) -> str | None:
    """Gold/silver and copper/gold with ±1σ bands — regime context, not a level."""
    if "GOLD" not in wide.columns:
        return None
    panels = []
    if "SILVER" in wide.columns:
        panels.append(("Gold / Silver", (wide["GOLD"] / wide["SILVER"]).dropna()))
    if "COPPER" in wide.columns:
        panels.append(("Copper / Gold ×1000",
                       (wide["COPPER"] / wide["GOLD"] * 1000).dropna()))
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.6 * len(panels)), sharex=True)
    for ax, (title, series) in zip(np.atleast_1d(axes), panels):
        s = series.iloc[-days:]
        mean, sd = s.mean(), s.std()
        ax.plot(s.index, s, color=ACCENT["gold"], linewidth=1.6)
        ax.axhline(mean, color=INK, alpha=0.45, linewidth=0.9, linestyle="--")
        ax.fill_between(s.index, mean - sd, mean + sd, color=ACCENT["neutral"], alpha=0.16)
        ax.set_title(f"{title}   last {s.iloc[-1]:.2f}  (mean {mean:.2f}, ±1σ shaded)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return _finish(fig, axes, outdir / "ratios.png")


def macro_panel(outdir: Path) -> str | None:
    """Fiscal/monetary plumbing: term premium, real yield, TGA, HY spread."""
    series = [
        ("THREEFYTP10", "10y term premium (ACM)", ACCENT["line"]),
        ("DFII10", "10y real yield (TIPS) %", ACCENT["gold"]),
        ("WTREGEN", "Treasury General Account ($mn)", ACCENT["neutral"]),
        ("BAMLH0A0HYM2", "US high-yield spread (%)", ACCENT["down"]),
    ]
    data = []
    for sid, label, color in series:
        rows = db.query(
            "SELECT ts, value FROM fred_series WHERE series_id=%s "
            "AND ts > now() - interval '400 days' ORDER BY ts", (sid,))
        if len(rows) > 10:
            data.append((label, color,
                         pd.Series([r["value"] for r in rows],
                                   index=pd.to_datetime([r["ts"] for r in rows]))))
    if not data:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2))
    for ax, (label, color, s) in zip(axes.ravel(), data):
        ax.plot(s.index, s.values, color=color, linewidth=1.5)
        ax.set_title(f"{label}   last {s.iloc[-1]:,.2f}", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(data):]:
        ax.set_visible(False)
    return _finish(fig, axes, outdir / "macro_panel.png")


# ─────────────────────────────────── the pack ───────────────────────────────
def render_pack(when: datetime | None = None) -> dict[str, str]:
    """Render every chart. Returns {key: filename}; failures are skipped, not fatal."""
    outdir = _outdir(when)
    wide = stats.load_daily()
    if wide.empty:
        return {}

    jobs = {
        "cross_asset_performance": lambda: cross_asset_performance(wide, outdir),
        "normalised_performance": lambda: normalised_performance(wide, outdir),
        "correlation_heatmap": lambda: correlation_heatmap(wide, outdir),
        "yield_curve": lambda: yield_curve_chart(outdir),
        "ratios": lambda: ratio_history(wide, outdir),
        "macro_panel": lambda: macro_panel(outdir),
    }
    out: dict[str, str] = {}
    for key, fn in jobs.items():
        try:
            name = fn()
            if name:
                out[key] = f"charts/{name}"
        except Exception as exc:  # noqa: BLE001
            log.warning("chart %s failed: %s", key, exc)
    log.info("rendered %d charts into %s", len(out), outdir)
    return out
