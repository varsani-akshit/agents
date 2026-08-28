"""The beats: one Scout and one Analyst per beat, every brief.

A beat is a patch of the world with its own sources, its own rhythm, and its
own failure mode when neglected. One agent covering everything produces the
context-clash brief — American policy in every paragraph because that is where
the loudest coverage is. Beats are how the brief covers the world.

`section` matches the `###` headings in the brief format, so an Analyst's
findings flow into the section the Editor will file them under.
"""
from __future__ import annotations

BEATS: list[dict] = [
    {
        "key": "monetary",
        "section": "Monetary Policy and Central Banks",
        "charter": (
            "Central bank decisions, speeches, minutes and shifts in market-implied "
            "policy paths. The Federal Reserve matters most but is one of many: "
            "ECB, Bank of Japan, Bank of England, RBA, RBI, PBoC and the major EM "
            "central banks all move capital. Hunt for: decisions and the votes "
            "behind them, guidance changes, implied-path repricing, and any gap "
            "between what a central bank said and what its market now prices."
        ),
        "series": ["US2Y", "US10Y", "DXY", "GOLD"],
    },
    {
        "key": "fiscal",
        "section": "Fiscal Policy and Sovereign Debt",
        "charter": (
            "Sovereign issuance, auctions, buybacks, deficits, downgrades, term "
            "premium and debt-sustainability arguments — any government, not only "
            "the US Treasury. Hunt for: auction results (tails, bid-to-cover), "
            "issuance calendar changes, official-sector buying or selling, "
            "ratings actions, and fiscal packages with a financing consequence."
        ),
        "series": ["US10Y", "US30Y", "GOLD"],
    },
    {
        "key": "geopolitics",
        "section": "Geopolitics, Conflict and Sanctions",
        "charter": (
            "War, sanctions, elections, trade measures and strategic resources. "
            "Hunt for: battlefield or negotiation developments with market "
            "consequence, new sanctions or export controls, chokepoint risk "
            "(Hormuz, Suez, Taiwan Strait, Black Sea), critical-minerals policy, "
            "and election outcomes that change fiscal or trade direction. State "
            "the market transmission, not just the event."
        ),
        "series": ["OIL", "GOLD", "DEFENSE"],
    },
    {
        "key": "energy",
        "section": "Energy and Commodities",
        "charter": (
            "Oil, gas, power, and the metals complex — precious and industrial. "
            "Hunt for: OPEC+ decisions, inventory surprises, refinery and "
            "pipeline outages, LNG flows, and for metals the physical side — "
            "central bank gold buying, ETF flows, COMEX/LBMA stress, mine "
            "supply. Distinguish spot moves from curve moves where the sources "
            "allow."
        ),
        "series": ["OIL", "NATGAS", "GOLD", "SILVER", "COPPER"],
    },
    {
        "key": "equities",
        "section": "Equities and Credit",
        "charter": (
            "Equity indices, sector rotation, single names only when they carry a "
            "macro signal (mega-cap earnings, systemic banks), credit spreads and "
            "issuance. Hunt for: earnings with read-through beyond the company, "
            "breadth divergences, credit-market stress or ease, IPO/buyback "
            "signals, and regional divergence — Europe, Japan, China, India "
            "against the US."
        ),
        "series": ["SPX", "FINANCIALS", "HYG", "IG"],
    },
    {
        "key": "digital",
        "section": "Digital Assets",
        "charter": (
            "Bitcoin, ether and the market structure around them: ETF flows, "
            "stablecoin supply, regulation, institutional adoption, and the "
            "on-chain or derivatives positioning behind a move. Hunt for the "
            "mechanism, not the price alone — the price is already measured. "
            "Treat crypto as a macro asset: what is it expressing about "
            "liquidity, debasement, or risk appetite this window?"
        ),
        "series": ["BTC", "ETH"],
    },
    {
        "key": "currencies",
        "section": "Currencies",
        "charter": (
            "The dollar against the world, and the crosses that carry a story: "
            "yen (policy divergence), euro, yuan (fixings, outflows), and the "
            "reader's own exposures — Australian dollar, Indian rupee, Kenyan "
            "shilling. Hunt for: intervention or its threat, carry unwind risk, "
            "EM stress, and official commentary that moves a currency. RBA and "
            "RBI policy noted here when the currency is the story."
        ),
        "series": ["DXY", "USDJPY", "AUDUSD", "USDINR", "USDKES"],
    },
    {
        "key": "real_assets",
        "section": "Real Assets and Property",
        "charter": (
            "Property, REITs, infrastructure and hard-asset investment flows. "
            "Hunt for: commercial real estate stress (refinancing walls, office "
            "values, regional-bank exposure), housing-market policy and rate "
            "transmission, and institutional allocation shifts into or out of "
            "real assets. This beat is often quiet; an empty report is a valid "
            "report."
        ),
        "series": ["REIT", "US10Y"],
    },
]


def by_key(key: str) -> dict:
    for b in BEATS:
        if b["key"] == key:
            return b
    raise KeyError(key)
