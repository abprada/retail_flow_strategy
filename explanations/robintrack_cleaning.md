# Robintrack cleaning

How `scripts/01_clean_explore_robintrack.py` turns raw Robintrack and CRSP files into the clean panels, and why each step is there. Two rules run through all of it:

- **Point in time.** Anything used to predict at 15:50 ET on day t must have been known by then.
- **Choices made by looking at values are made on the training period only.** Fixing how data was recorded is done on the full sample.

## 1. Raw data (`scripts/00_data_*.py`, stored unchanged in `data/raw/`)

| Source | Content |
|---|---|
| Robintrack | 8,597 CSVs of hourly Robinhood holder counts, May 2018 to August 2020, one per Robinhood instrument, named by its ticker at export time |
| CRSP `stocknames_v2` | ticker, name, CUSIP and classification history per PERMNO |
| CRSP `stkdistributions`, `stkdelists`, `metaflaginfo` | splits, spin-offs, merger and liquidation payments, delistings, and what each code means |
| CRSP `dsi` | the trading calendar |
| CRSP `stkdlysecuritydata` | daily returns, prices, volume and market cap, 2017 onwards |

## 2. From tickers to PERMNOs

Tickers change and get reused, so every reading is matched to the PERMNO that held its ticker **on that date**.

- **Share classes:** until 2020-01-16, Robintrack wrote both classes of a dual-class stock (BRK.A, BRK.B and others) into one file. The second class's readings are split out by their alternating sweep pattern.
- **One instrument per file:** a file's rows take the PERMNO most of its rows match, while that security exists. This catches renames, e.g. `GOLD` was Barrick's `ABX` in 2018.
- **No double counting:** where two files hold one PERMNO at the same time, the date-matched rows win.

Result: 8,042 PERMNOs, with no overlapping segments and no duplicate PERMNO-timestamp pairs. 490 files have no CRSP match (OTC ADRs and ETNs).

## 3. Hourly cleaning (flags; nothing is changed in place)

| Step | Problem | Rule | Rows |
|---|---|---|---|
| C1 | zeros before a PERMNO's first holder: placeholders for instruments not yet trading | drop | 731,813 |
| C2 | a switch to another file is a switch to another Robinhood instrument, which often restarts at 0 and climbs (UTX to RTX) | drop the restart zeros; mark a **break**, and never take a change across it | 277 (83 breaks) |
| C3 | readings less than 5 minutes apart come from overlapping or restarted sweeps (a full sweep takes at least 7 minutes) | keep the last reading of each cluster | 571,048 |
| C4 | sweep-wide outages (no reading for over 1.5 hours) | listed, not dropped; a day without a close gets no change | 97 outages |
| C5 | a count falling to 0 from 10 or more holders: a recording failure (instrument replaced at a reverse split or merger, or a Robintrack fault), not real selling | treat as missing, plus a break | 29 runs, 74,750 rows |

C5 keeps drops from 1 to 9 holders, which look real; any cut-off from 5 to 13 flags the same runs. Of the 29 runs, 23 fall within a week of a CRSP corporate event; the other 6 cluster on shared dates.

## 4. Timing

- **Timestamps are UTC.** The daily pattern shifts by an hour between daylight-saving seasons in raw time but lines up in New York time.
- **Counts lag the reading by about 25 to 40 minutes.** Readings taken up to 9:55 do not yet show the 9:30 open, and readings at 19:40 no longer show after-hours trading, which ended at 18:00.
- **The day's close count is the reading taken 16:40 to 16:59 ET.** That reading shows holdings at about 16:00 to 16:35, the full session. The 15:4x reading would miss the last 30 to 60 minutes. Sweeps run at :40 to :59, so any lag in that range picks the same reading.

## 5. Corporate events (recorded, not applied)

3,634 CRSP events on 1,622 PERMNOs inside the Robintrack period: splits, reverse splits, stock dividends, spin-offs, mergers, liquidations, delistings, and ticker, name and CUSIP changes. Counterparties are listed too, such as the acquirer in a stock merger or the new company in a spin-off. CRSP has no announcement date for mergers or delistings.

Event rows are kept; nothing is excluded around them. On train:

- **Forward splits:** holders rise about 6% on the day and keep climbing. Retail buying the cheaper shares.
- **Reverse splits:** a lasting drop of about 30%, consistent with holders of fewer shares than the ratio being cashed out. Kept anyway.
- **Spin-off parents:** the holder jump tracks the parent's price drop on the ex-date (correlation -0.87, KAR -60% and 15x holders) and comes with heavy volume. Retail buying the apparent drop: flow.
- **Spin-off children:** holders climb smoothly with volume rather than in one delivery step. Flow.
- **Ticker, name and CUSIP changes:** no effect once matched by PERMNO. The large moves are reverse splits and mergers under another label.
- **Stock-merger acquirers:** holders rise one to one with the target's last count over days 0 to 5 (MPC +264 against Andeavor's 263; correlation 0.77). Positions are converted, not bought. Left in, because it touches only a handful of rows above 100 holders and the daily clipping caps them.

## 6. Daily data

- **T1, close counts:** one per PERMNO per trading day, the first reading in the close window. Each close carries a **segment** that changes at every C2 break and after every C5 run. 23 trading days have no close at all (the outages).
- **T2, daily changes:** taken only between consecutive trading days that both have a close in the same segment, with a previous close above 0. Otherwise the change is missing and a status says why. A change never spans an outage.
- **CRSP join:** columns are named by when they are known: `*_prev` (day t-1, known at 15:50), `*_t` (only after day t's close), and `ret_next` (day t's close to the next trading day's close, the return to predict). Prices are as traded, never split-adjusted, because CRSP's adjustment factors carry later splits back in time. Delisting returns are part of CRSP's daily returns. A delisting payoff (`DD`) is kept and flagged, because it can span more than one day.

## 7. Filters from the literature

Applied row by row with day t-1 information:

- a price of at least $1 (Eaton et al.)
- at least 100 holders (Barber et al.)

Recorded but not applied: common stock only (share codes 10/11) and the major exchanges.

Rejected because they decide on a stock's whole history using later information:

- dropping stocks with long gaps or a series that never changes
- dropping every stock that split
- price screens at one fixed date

The holder floor matters most. Stocks under 100 holders cause most of the extreme % changes: 73% of the moves above ±50%.

## 8. Train/test split

- **Cutoff 2019-12-12**, the day Robinhood launched fractional shares. Train runs 2018-05-02 to 2019-12-11 (407 trading days); test runs to 2020-08-13 (169).
- **Embargo:** training return labels that end on or after the cutoff are dropped.
- **Test is never summarized.** Everything below is chosen on train and applied to test with the same fixed rules.

## 9. The two clean panels (`data/intermediate/clean/`)

**Holder panel (`holders_*.parquet`), for nowcasting the change in holders.** Rows: a usable change, a price of at least $1 and at least 100 holders at the previous close. Target `y`:

1. compute log(1 + h_t) - log(1 + h_{t-1});
2. clip at that day's 1st and 99th percentiles;
3. demean within the day.

Units and floor were compared on train: taking logs cuts kurtosis from 389,105 to about 700, and the floor removes the lumpy small-count changes. Clipping per day keeps each day independent of every other.

**Return panel (`returns_*.parquet`), for predicting next-day returns.**

- **Rows:** every stock-day in the **15:50 universe**: a price of at least $1 and at least 100 holders at day t-1's close. Nothing from day t is used to choose the rows; a stock stays even if day t's reading is missing.
- **X:** day t-1's change in holders as a % change, a log change and a change in log(1 + holders). Each is winsorized at that day's 1st and 99th percentiles within the universe, then z-scored that day (`_zw`). Z-scoring after winsorizing keeps the scale robust; z-scoring with the raw standard deviation lets one extreme stock shrink everyone else.
- **y:** `ret_next_dm`, the next-day return minus the universe's mean that day.
- **Upper bound only:** day t's own change (`rh_pct_t*`), known only after the close.

## 10. Checks before writing

Twelve checks gate the write:

- the hourly rows: no missing values, valid and increasing timestamps, whole non-negative counts;
- one row per PERMNO and date;
- every date is a trading day;
- the split boundary and the embargo;
- every row is in its universe;
- `y` and X have a daily mean of 0;
- delisting payoffs are flagged.

Train and test go to separate files, so estimation (`02_robintrack_returns.py`) reads only `returns_train.parquet`.

## Open items

- **2018-10-09:** the close sweep reached only 20% of stocks; it is not yet treated as an outage.
- **Corporate events are not treated** (see section 5). Merger conversions into the acquirer are known not to be flow; they could be netted out with the target's last holder count if they turn out to matter.
- **Market cap** uses CRSP's share count, which can be dated before it was public.
