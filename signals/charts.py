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
                    color=INK if abs(v) > 0.6 else "#101216")
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


def regime_gauge(outdir: Path) -> str | None:
    """The regime score and every component vote that produced it.

    A regime label in prose is unfalsifiable. Shown as eight weighted votes, the
    reader can see immediately which components disagree — usually the most
    interesting thing on the page.
    """
    from signals import regime as regime_mod

    rs = regime_mod.regime_score()
    if not rs or not rs.get("components"):
        return None
    comps = rs["components"]
    names = list(comps.keys())[::-1]
    vals = [comps[n]["vote"] for n in names]
    labels = [n.replace("_", " ") for n in names]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.8, 0.42 * len(names) + 2.2),
        gridspec_kw={"height_ratios": [1, max(2.2, 0.34 * len(names))]})

    score = rs["score"]
    ax1.barh([""], [score], color=ACCENT["up"] if score >= 0 else ACCENT["down"], height=0.5)
    ax1.set_xlim(-1, 1)
    ax1.axvline(0, color=INK, linewidth=0.9, alpha=0.7)
    for x in (-0.45, -0.15, 0.15, 0.45):
        ax1.axvline(x, color=GRID, linewidth=0.7, linestyle=":")
    ax1.set_title(f"Regime score {score:+.2f}  —  {rs['label']}", fontsize=10)
    ax1.set_yticks([])
    ax1.set_xlabel("tight money  ←                    → debasement", fontsize=8)

    colors = [ACCENT["up"] if v >= 0 else ACCENT["down"] for v in vals]
    ax2.barh(labels, vals, color=colors, height=0.6)
    ax2.axvline(0, color=INK, linewidth=0.8, alpha=0.6)
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_title("Component votes (weighted)", fontsize=9)
    for i, n in enumerate(names):
        ax2.text(1.12, i, f"w={rs['weights'].get(n, 0):.2f}",
                 va="center", ha="right", fontsize=7, color=ACCENT["neutral"])
    return _finish(fig, [ax1, ax2], outdir / "regime_gauge.png")


def net_liquidity_chart(outdir: Path) -> str | None:
    """Net liquidity against the hard-asset complex — the clearest driver pairing."""
    from signals import regime as regime_mod

    walcl = regime_mod.fred_frame("WALCL", 1500)
    tga = regime_mod.fred_frame("WTREGEN", 1500)
    rrp = regime_mod.fred_frame("RRPONTSYD", 1500)
    if walcl.empty:
        return None
    frame = pd.concat({"w": walcl, "t": tga, "r": rrp}, axis=1, sort=True).sort_index().ffill()
    net = (frame["w"] - frame["t"] - frame["r"].fillna(0) * 1000).dropna() / 1e6
    # Weekly, matching the H.4.1 grid — see net_liquidity() for why daily is fake.
    net = net.resample("W-WED").last().dropna()
    if len(net) < 12:
        return None

    wide = stats.load_daily(lookback_days=1500)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    # Raw weekly plus a quarterly trend. The week-to-week swings are real (tax
    # dates and settlement move the TGA by hundreds of billions), but the trend
    # is what the hard-asset complex actually tracks.
    ax.plot(net.index, net.values, color=ACCENT["line"], linewidth=0.9, alpha=0.45,
            label="Net liquidity ($tn, weekly)")
    ax.plot(net.index, net.rolling(13, min_periods=4).mean().values,
            color=ACCENT["line"], linewidth=2.2, label="13-week average")
    ax.set_ylabel("net liquidity ($tn)", color=ACCENT["line"])
    ax.set_title("Net liquidity (Fed assets − TGA − RRP) vs hard assets")

    ax2 = ax.twinx()
    for sym, color in (("GOLD", ACCENT["gold"]), ("BTC", ACCENT["btc"])):
        if sym in wide.columns:
            s = wide[sym].dropna()
            s = s[s.index >= net.index[0]]
            if len(s) > 30:
                ax2.plot(s.index, s / s.iloc[0] * 100, color=color, linewidth=1.4,
                         alpha=0.85, label=f"{sym} (rebased)")
    ax2.set_ylabel("rebased to 100", color=ACCENT["gold"])
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return _finish(fig, ax, outdir / "net_liquidity.png")


def rolling_correlations(wide: pd.DataFrame, outdir: Path, days: int = 400) -> str | None:
    """Key correlations through time, not as a single snapshot.

    A heatmap says gold and TLT correlate +0.23 today. It cannot say that the
    figure was −0.36 six weeks ago, which is the actual news.
    """
    pairs = [("GOLD", "TLT", ACCENT["gold"]), ("GOLD", "DXY", ACCENT["silver"]),
             ("BTC", "SPX", ACCENT["btc"]), ("GOLD", "US10Y", ACCENT["line"])]
    have = [(a, b, c) for a, b, c in pairs if a in wide.columns and b in wide.columns]
    if not have:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 4))
    plotted = 0
    for a, b, color in have:
        cs = stats.corr_series(wide[a], wide[b], 30)
        if len(cs) < 60:
            continue
        cs = cs.iloc[-days:]
        ax.plot(cs.index, cs.values, color=color, linewidth=1.6, label=f"{a}/{b}")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    ax.axhline(0, color=INK, linewidth=0.9, alpha=0.6)
    ax.set_ylim(-1, 1)
    ax.set_title("30-day rolling return correlations")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(frameon=False, ncols=len(have), fontsize=8, loc="upper left")
    return _finish(fig, ax, outdir / "rolling_correlations.png")


def real_yield_vs_gold(outdir: Path) -> str | None:
    """The relationship the whole gold framework rests on, plotted so it can break."""
    from signals import regime as regime_mod

    real = regime_mod.fred_frame("DFII10", 1500)
    if real.empty:
        return None
    wide = stats.load_daily(lookback_days=1500)
    if "GOLD" not in wide.columns:
        return None
    gold = wide["GOLD"].dropna()
    gold = gold[gold.index >= real.index[0]]

    fig, ax = plt.subplots(figsize=(8.4, 4))
    # Inverted, because the textbook claim is that gold rises as real yields fall.
    ax.plot(real.index, real.values, color=ACCENT["line"], linewidth=1.6)
    ax.invert_yaxis()
    ax.set_ylabel("10y TIPS real yield % (inverted)", color=ACCENT["line"])
    ax.axhline(0, color=ACCENT["down"], linewidth=0.9, linestyle="--", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(gold.index, gold.values, color=ACCENT["gold"], linewidth=1.7)
    ax2.set_ylabel("gold $/oz", color=ACCENT["gold"])
    ax2.grid(False)
    ax.set_title("Gold vs 10y real yield — the rule, and whether it still holds")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return _finish(fig, ax, outdir / "real_yield_gold.png")


def fx_performance(wide: pd.DataFrame, outdir: Path) -> str | None:
    """Every tracked currency against the dollar, sign-corrected for quote convention."""
    board = stats.fx_board(wide)
    if not board:
        return None
    board = sorted(board, key=lambda r: r["currency_1m_vs_usd_pct"])
    names = [r["currency"] for r in board]
    vals = [r["currency_1m_vs_usd_pct"] for r in board]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(names, vals, color=[ACCENT["up"] if v >= 0 else ACCENT["down"] for v in vals],
            height=0.6)
    ax.axvline(0, color=INK, linewidth=0.8, alpha=0.6)
    ax.set_title("Currencies vs USD — 1 month (positive = stronger than the dollar)")
    ax.set_xlabel("% vs USD")
    for i, v in enumerate(vals):
        ax.text(v + (0.06 if v >= 0 else -0.06), i, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    return _finish(fig, ax, outdir / "fx_performance.png")


def drawdown_chart(wide: pd.DataFrame, outdir: Path) -> str | None:
    """Distance from the 52-week high — momentum's missing half."""
    rows = [r for r in stats.drawdown_table(wide) if r["pct_from_high"] is not None]
    if len(rows) < 5:
        return None
    rows = sorted(rows, key=lambda r: r["pct_from_high"])
    names = [r["symbol"] for r in rows]
    vals = [r["pct_from_high"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, max(3.5, 0.24 * len(rows))))
    colors = [ACCENT["up"] if v > -2 else (ACCENT["gold"] if v > -12 else ACCENT["down"])
              for v in vals]
    ax.barh(names, vals, color=colors, height=0.66)
    ax.set_title("Distance from 52-week high (%)")
    ax.set_xlabel("% below high")
    ax.tick_params(labelsize=7)
    return _finish(fig, ax, outdir / "drawdowns.png")


def credit_spreads(outdir: Path) -> str | None:
    """High-yield, investment-grade and EM spreads on one axis."""
    from signals import regime as regime_mod

    series = [("BAMLH0A0HYM2", "US high yield", ACCENT["down"]),
              ("BAMLC0A0CM", "US investment grade", ACCENT["line"]),
              ("BAMLEMCBPIOAS", "EM corporate", ACCENT["gold"])]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    plotted = 0
    for sid, label, color in series:
        s = regime_mod.fred_frame(sid, 1500)
        if len(s) < 30:
            continue
        ax.plot(s.index, s.values, color=color, linewidth=1.5,
                label=f"{label} ({s.iloc[-1]:.2f}%)")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title("Credit spreads — the deflationary counterweight to the debasement trade")
    ax.set_ylabel("option-adjusted spread %")
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return _finish(fig, ax, outdir / "credit_spreads.png")


def returns_heatmap(wide: pd.DataFrame, outdir: Path) -> str | None:
    """Every asset by every horizon — where the whole board is, at a glance."""
    horizons = [("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)]
    syms, rows = [], []
    classes = {r["symbol"]: r["asset_class"]
               for r in db.query("SELECT symbol, asset_class FROM instruments")}
    for sym in sorted(wide.columns, key=lambda s: (classes.get(s, "zz"), s)):
        if sym.startswith("US") and sym.endswith("Y"):
            continue
        s = wide[sym].dropna()
        if len(s) < 260:
            continue
        vals = [stats.pct_change(s, n) for _, n in horizons]
        if any(v is None for v in vals):
            continue
        syms.append(f"{sym}")
        rows.append(vals)
    if len(rows) < 5:
        return None

    arr = np.array(rows)
    lim = float(np.nanpercentile(np.abs(arr), 92)) or 10.0
    fig, ax = plt.subplots(figsize=(6.4, max(4.5, 0.235 * len(syms))))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(horizons)), [h for h, _ in horizons], fontsize=8)
    ax.set_yticks(range(len(syms)), syms, fontsize=7)
    for i in range(len(syms)):
        for j in range(len(horizons)):
            v = arr[i, j]
            # RdYlGn is pale at the midpoint and dark at both ends, so the text
            # contrast flips with magnitude, not with sign.
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6,
                    color=INK if abs(v) > lim * 0.6 else "#101216")
    ax.set_title("Total return by horizon (%)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.5, label="%")
    ax.grid(False)
    fig.tight_layout()
    path = outdir / "returns_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def global_equities(wide: pd.DataFrame, outdir: Path, days: int = 252) -> str | None:
    """Regional equity blocks rebased — is this a US story or a global one?"""
    picks = [("SPX", ACCENT["line"]), ("EUROPE", ACCENT["gold"]), ("JAPAN", ACCENT["down"]),
             ("CHINA", ACCENT["up"]), ("INDIA", ACCENT["btc"]), ("EM", ACCENT["neutral"])]
    have = [(s, c) for s, c in picks if s in wide.columns]
    if len(have) < 3:
        return None
    sub = wide[[s for s, _ in have]].dropna().iloc[-days:]
    if len(sub) < 40:
        return None
    rebased = sub / sub.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(8.4, 4))
    for sym, color in have:
        ax.plot(rebased.index, rebased[sym], color=color, linewidth=1.6,
                label=f"{sym} {rebased[sym].iloc[-1] - 100:+.0f}%")
    ax.axhline(100, color=INK, linewidth=0.7, alpha=0.4, linestyle="--")
    ax.set_title(f"Regional equities rebased — last {len(sub)} sessions")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="upper left")
    return _finish(fig, ax, outdir / "global_equities.png")


# ─────────────────────────────────── the pack ───────────────────────────────
def render_pack(when: datetime | None = None) -> dict[str, str]:
    """Render every chart. Returns {key: filename}; failures are skipped, not fatal."""
    outdir = _outdir(when)
    wide = stats.load_daily()
    if wide.empty:
        return {}

    jobs = {
        "regime_gauge": lambda: regime_gauge(outdir),
        "returns_heatmap": lambda: returns_heatmap(wide, outdir),
        "cross_asset_performance": lambda: cross_asset_performance(wide, outdir),
        "normalised_performance": lambda: normalised_performance(wide, outdir),
        "net_liquidity": lambda: net_liquidity_chart(outdir),
        "real_yield_gold": lambda: real_yield_vs_gold(outdir),
        "rolling_correlations": lambda: rolling_correlations(wide, outdir),
        "correlation_heatmap": lambda: correlation_heatmap(wide, outdir),
        "fx_performance": lambda: fx_performance(wide, outdir),
        "global_equities": lambda: global_equities(wide, outdir),
        "credit_spreads": lambda: credit_spreads(outdir),
        "yield_curve": lambda: yield_curve_chart(outdir),
        "ratios": lambda: ratio_history(wide, outdir),
        "drawdowns": lambda: drawdown_chart(wide, outdir),
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
