# Alfred v2 — what it is for and how it works

Date: 2026-10-29. Draft for alignment, not yet built.

## The objective

Alfred helps Akshit decide, quickly and continuously, what to buy and sell
across public markets and crypto, and when. It does the work a person would
need hours or days for — reading the news, the data, the positioning, the
macro picture, and the price history — for hundreds of instruments at once,
every hour of every day, and it says plainly: this instrument, on this
exchange, through this broker, at this price, with this stop and this exit,
funded from here, because of these three things.

Alfred never places an order. Kite and Webull are where the orders go, not
where Alfred thinks. Data comes from wherever it is best and cheapest.

The measure of Alfred is whether it is right. Every call it makes is
recorded and scored, including the ones Akshit does not take, and the
kinds of call that turn out wrong are retired on their own.

## What was wrong with v1

MSCI and DRI were "the 20-day average crossed the 100-day", with no idea
why MSCI had fallen 2.6% the day before, what the company does, what the
market was pricing, or what the world was worried about that week. That
is a pattern, not a decision. v1 only ever looked at price.

v1 also treated every holding the same (gold would have been flagged for
review below a stop), scanned three times a day, knew nothing about
crypto, and never said which exchange or currency the money had to be in.

## How a decision is made

Every idea is a case with three reads. Each read is produced on its own,
with its evidence attached, and the verdict needs them to agree.

1. **Price** — what the chart and the tape say. The simulated strategies
   from v1 (kept: they are honest), plus crypto-specific ones (funding
   extremes, liquidation cascades, range breaks) and intraday ones.
   Each has a tested record. Output: a setup with entry, stop, target,
   expectancy and how long it usually takes.

2. **Conditions** — what kind of world this is, and whether this setup
   works in it. A regime built every hour from rates, the yield curve, the
   dollar, oil, credit spreads, VIX, each market's own trend, crypto
   funding and open interest, and a news-tone index per country and
   industry going back to 2017 (GDELT). The simulator tests every setup
   *conditional on regime*, so Alfred can say "this works when the dollar
   is falling and fear is high, which is now" — or "it does not, skip it".
   This is the part of "macro behaviour" that can be tested honestly.

3. **Story** — what is happening and what the world thinks. Models read
   the last 48 hours of news for the instrument, its sector, its country
   and the global backdrop; earnings dates and surprises; central-bank and
   government moves; exchange filings (SEC, NSE/BSE, ASX); positioning
   (CFTC, short interest, crypto funding); and what prediction markets
   price (Polymarket) for the events that matter. Output: a read with
   cited stories, a direction, a strength, and the one thing that would
   change it. Decision-1 gives a probability; the LLMs give the argument.

A **verdict** is written only when Price has a setup *and* Conditions say
it works now *and* Story does not contradict it. Story can also *start* a
case — a policy change, a disaster, a surprise print — and then Price and
Conditions decide whether there is a tradeable setup behind it, on which
instrument, on which exchange.

Every ticket shows all three reads in one line each, so Akshit can see
why, and disagree.

The Story read cannot be backtested honestly (a model asked about 2019
already knows 2020), so it earns its weight the only fair way: its calls
are scored from the first day, and the scorecard decides how much it is
trusted per family of idea.

## What Alfred looks for

| Family | Horizon | Where |
|---|---|---|
| News-driven moves: earnings, policy, filings, surprises | hours to days | all |
| Trend and momentum (v1's proven strategies) | weeks | all |
| Cross-market lead/lag: US close → India open, China → ASX miners, oil → energy names | one session | NSE, ASX, US |
| Crypto range breaks, funding and open-interest extremes | hours to days | crypto, 24/7 |
| Macro rotations: which sectors and countries the regime favours | weeks to months | ETFs on ASX and US |
| Gold trim and re-buy on a spike (PMGOLD stays core) | days | ASX |
| Prediction-market vs price divergence around dated events | to the event | all |

Horizons are not tiers. The same pipeline runs at three clocks (intraday,
days, weeks); each family has its own record and its own loss limit.

## Where the money is, and where it can go

- **Kite** (NRI non-PIS): NSE and BSE stocks, delivery *and* intraday.
- **Webull AU**: ASX, NYSE/Nasdaq, Hong Kong, China A-shares, US options.
  US and HK stocks are $0 brokerage; ASX A$1 or 0.03%.
- **Coinbase**: crypto, spot. Public candles for 845 pairs need no key
  (verified); balances need a Coinbase API key made with **View**
  permission only — never Trade — pasted into `.env`.
- **Indian mutual funds**: Akshit buys at the bank. Alfred can still say
  which.
- **No direct Japan, Europe or UK**: reached through ETFs on ASX or US.

Every ticket carries: broker, exchange, currency, quantity, and a funding
line ("move US$1,200 into your Webull USD balance", "transfer ₹40,000 to
Kite"). Alfred knows the free cash per currency from the broker pulls and
does not suggest more than can be funded.

## Data: free, and not from the brokers

Verified from this Mac today, no key needed:

| Source | What | Notes |
|---|---|---|
| Hyperliquid info API | 235 perps: funding, open interest, candles | one POST, no key |
| Binance spot + futures | candles, funding history | no key |
| CoinGecko | prices in USD/INR/AUD, market data | demo key raises limits |
| Deribit | crypto options, index prices, implied vol | no key |
| Polymarket Gamma | event odds | no key |
| GDELT | global news tone by query, 2017→ | 1 request / 5 s |
| DBnomics | RBI, RBA, ECB, IMF, BIS, OECD series | one API for all |
| SEC EDGAR | US fundamentals (XBRL), filings, insider Form 4 | needs a user-agent |
| CFTC | futures positioning (COT) | weekly |
| NSE site JSON | announcements, bulk deals, FII/DII flows | works, fragile |
| ASX announcements | daily listing + PDFs | page/CSV, 20-min delay |
| FRED, yfinance, Google News | already in use | |

Free with a key (Akshit signs up, pastes into `.env`) — agreed 2026-10-29:

| Source | What | Why it helps |
|---|---|---|
| Finnhub ✓ | US/global company news, earnings calendar (1,500 reports next week), economic calendar, basic fundamentals | the Story read's calendar |
| Twelve Data ✓ | second price feed for US/ASX/FX/crypto (NSE needs a paid plan; Yahoo covers it) | when Yahoo is wrong or down |
| CoinGecko demo ✓ | higher limits, prices in AUD/INR/USD | |
| Coinbase (View-only key) | balances; candles are public anyway | to be created |

✓ = key tested on the Mac and on the server, 2026-10-29.

**Indian intraday data**: every free Indian API (Dhan, Upstox, Groww,
Fyers, Angel One) needs an Indian account with an active Indian mobile,
which Akshit does not have. Verified today that Yahoo gives NSE minute
bars free: 1-minute for the last 7 days, 5-minute for 60 days, hourly for
2 years. That is enough to run Indian intraday live and to test it over
the last two months. Kite's historical API (₹2,000/month, deep minute
history) is bought only once Indian intraday has shown a record worth
the fee.

**TradingView**: not needed. It is a chart and alert tool, not a data
source — it has no API to pull data out, and its NSE/ASX real-time feeds
cost extra on top of the plan (Essential US$14.95/month, Plus US$34.95,
Premium US$69.95; webhooks need a paid plan). Alfred draws its own charts
and sends its own alerts. If Akshit likes TradingView for looking at
charts, the free plan is fine alongside.

**Perp exchanges** (Hyperliquid, Binance futures) are read as sentiment —
funding, open interest, liquidations — not traded.

Not worth paying for yet: Kite historical, real-time ASX feed, Bloomberg
anything.

## Cadence

- **Heartbeat, every 5 minutes**: prices and news for everything held or
  under a live case; crypto always. A stop hit, a target hit, or a story
  that changes a case triggers an email at once.
- **Scan, every hour while a market is open**: new cases for that market.
  Crypto scans around the clock.
- **Regime, every hour**: the Conditions read refreshed.
- **Simulations, every 2 hours**: as now, plus the conditional tests.
- **Weekly review, Sunday**: the scorecard by family and horizon, what
  was retired, what was promoted, and the plan for the week.

Emails stay sparse: one per action, one daily plan, one weekly review.

## Being right: the scorecard

Every case, taken or not, is followed to its exit at the prices Alfred
said. Per family × horizon × market: hit rate, expectancy, worst run,
and the Story read's own accuracy. A family that is below water after
enough calls is retired automatically and says so on the Simulations
page. New families start with small size until they have a record.

Each horizon has a loss limit on the new money; when hit, that horizon
pauses and Alfred emails why. Akshit restarts it.

## What does not change

- No execution, ever. The repo test enforces it.
- PMGOLD is core: never a Sell or Review, only "trim N, re-buy at Y".
- The strict price gate still decides what counts as a proven setup; the
  other two reads are added, not substituted.
- Secrets in `.env` only.

## Build order

1. **Data layer** (crypto, GDELT, DBnomics, EDGAR, CFTC, NSE, ASX,
   Finnhub, Dhan) with one `sources` table saying what is fresh.
2. **Conditions**: the regime index, the news-tone history, and the
   conditional simulator.
3. **Story**: the case reader with citations and Decision-1 probability;
   the earnings/economic/policy calendar; the scorecard from day one.
4. **Cases and verdicts**: the three reads joined, ticket format with
   exchange and funding line, gold-core rule, crypto tickets.
5. **Cadence**: heartbeat, hourly scans, 24/7 crypto, weekly review.
6. **Today page**: cases with their three reads; the scorecard.

Each step ships on its own and is visible on the site when it lands.

## Progress

- **Step 1, data layer** — shipped 2026-10-29 (dce7a3d). Binance futures
  refuses US addresses, so on the server Hyperliquid alone supplies funding
  and open interest. GDELT throttles far below its documented rate; the tone
  feed waits, skips and resumes.
- **Step 2, conditions** — shipped 2026-10-29. advisor/regime.py: seven to
  nine conditions per market; rules learnt on the older years, tested on the
  last four, kept only if the market's account replay is no worse with them.
  First real run: rules held out of sample for momentum rotation on the ASX
  and the US, and switched the 55-day breakout on for India in fearful,
  falling-dollar conditions — but India's account did worse with it (Sharpe
  0.74 vs 0.92), so India stays on the plain gate. Only the ASX uses its
  rules today. Crypto becomes a market in step 4, with its own costs
  (Coinbase ≈ 1.2% a round trip at this volume).
- **News conditions, revised** — 2026-10-29. GDELT's API refuses the
  server (and any address that backfills), so the regime's news conditions
  now use newspaper-based indices with decades of history: geopolitical risk
  (Caldara & Iacoviello, daily since 1985), stock-market and US policy
  uncertainty (daily), India and trade-policy uncertainty (monthly, lagged
  40 days). Of 71 news rules the older years suggested, almost none held on
  the recent four; none made an account better. News enters decisions
  through the Story read, judged forward, not through backtested rules.
- **Step 3, Story read and scorecard** — shipped 2026-10-29. advisor/story.py:
  every signal from every run is a call, settled by the same play() as the
  ledger; each new call gets a Story read (GPT-5.4 mini over a code-built
  dossier: the company's news — matched, or naming it, never a bare Finnhub
  tag — filings, earnings and central-bank dates inside the hold, the
  market's conditions, the brief's top line) plus a Decision-1 probability.
  About $0.002 a read. Tickets email the read; the Ideas page scores calls
  by gate and by read. The read informs, it does not yet decide: that waits
  for the scorecard to show supported calls beating contradicted ones.
  First reads flagged CoStar (fell 14% on earnings), Zoetis and AJG
  (earnings inside the hold) — three of the tickets v1 sent blind.

- **Step 4, verdicts and crypto** — 2026-10-29. Every ticket names its
  exchange (NYSE, Nasdaq, NYSE Arca, NSE, BSE, ASX, Coinbase) and where the
  money must be ("US$1,300 in Webull's USD balance — about A$2,000,
  converted in Webull"). A ticket whose Story read contradicts it with
  strength ≥ 0.5 is held back and listed in the email with the reason; the
  next one takes its place; the call is still scored. PMGOLD is a core
  holding: no stop, no review — only a trim-and-buy-back note when it is
  7 ATRs above its 50-day average (the top 5% of days). Crypto is a fourth
  market on Coinbase (14 pairs, 1.2% round trip, A$500 set aside, funded in
  AUD, daily cycle at 00:45 UTC). First run: the 55-day breakout passes
  (+0.90R over 121 recent trades), but the account replay (+12%/yr, worst
  −23%, Sharpe 0.62) does not beat holding Bitcoin (+41%/yr, worst −84%,
  Sharpe 0.71) on Sharpe, so crypto sends no tickets yet; it is simulated
  and scored every two hours.
- **Step 5, cadence** — 2026-10-29. advisor/watch.py, every five minutes:
  a ticket's limit reached, a filled ticket's stop or target touched, a
  holding moving 2.5 daily ranges from yesterday's close (core holdings get
  no action suggested), and news the matcher reads as bad for a name you
  hold or have a ticket on — stored once, emailed together, closed markets
  skipped. advisor/review.py, Sunday 21:00 UTC: the scorecard by gate and
  by news read, tickets, gate changes over the week, which markets send,
  how their conditions moved, and the week ahead for your names. Hourly
  *trading* scans were not added: the hourly strategies still lose after
  costs, and the daily ones decide on finished bars.
- **Step 6, Today** — 2026-10-29. Today shows each case that passed the gate
  with its three reads (price setup and record, conditions, news), the ones
  the news held back, every Buy with its exchange, funding line and news
  read, the heartbeat's alerts since yesterday, each market's conditions in
  one line, and the scorecard. Coinbase joins the account tiles. The plan's
  six steps are built; what remains is the scorecard filling up, which
  decides how much weight the news read earns.