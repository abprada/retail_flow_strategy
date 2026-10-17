# daily-stock-returns

Research code for predicting next-day cross-sectional US stock returns: how stocks do relative to each other within a universe of liquid equities, not where the market goes. Research only; nothing runs live.

## Layout

Scripts are grouped by source; their number prefix gives the order they run in.

- `scripts/data_acquisition/` downloads raw data into `data/raw/` unchanged: TAQ (the full pull runs on WRDS Cloud, `wrds_cloud/`), Robintrack holder counts, and CRSP names, daily data and corporate events from WRDS. Nothing there modifies the data.
- `scripts/taq/` and `scripts/robintrack/` hold each source's marimo notebooks. `01_*` explore and clean the raw data and write to `data/intermediate/`. `02_*` build on the clean data: the model panel from TAQ (`data/processed/`), and whether Robintrack holder changes predict returns. `scripts/taq/checks/` holds checks of the TAQ pull.
- `scripts/models/` tunes the models with W&B sweeps (03), reruns the chosen settings in a monthly walk-forward (04) and backtests the predictions (05).
- `src/dsr/` holds reusable code and `tests/` its tests (`uv run pytest`).
- Local only, never commit: `data/`, `plan.md`, `literature/`.

## Lookahead bias

Every input to a return prediction must have been known before the decision. Positions are traded with market-on-close orders sent before the 15:50 ET cutoff, so features for day t use only data available before 15:50 ET on day t.

- Date information by when it became public, not by the event it describes. CRSP and other databases are today's corrected versions, not snapshots as of each date, so use announcement or declaration dates. CRSP has no announcement date for mergers, delistings, or ticker, name and CUSIP changes.
- Any setting chosen by looking at the data (holder floors, outlier and winsorizing cutoffs, exclusion windows, target units, model choices) is chosen on the training period only and applied unchanged to the test period. Never tune anything on test-period results.
- Filters work row by row with what was known at the time: yesterday's holders, not a full-sample median.
- Fixing how the data was recorded (PERMNO matching, share-class splits, placeholder zeros, repeated readings) is fine on the full sample, because it does not select by value.
- Robintrack timestamps are retrieval times in UTC, and the counts lag them by about 25 to 40 minutes. A day's count is the first reading taken between 15:40 and 15:50 New York time, the last one before the cutoff; daily changes run from one trading day's count to the next.

## Survivorship bias

- Never pick stocks by whether they survive: no "listed for the whole sample", "still trading today", "present from day one" or current index lists. Decide universe membership on each date with what was known that day.
- Keep delisted stocks, and include delisting returns (CRSP `delret`) when computing returns.
- Identify securities by CRSP PERMNO matched by date, never by ticker alone. Tickers change and get reused, and Robintrack files are named by the ticker at export time.
- Never drop a whole stock because of something that happens later in the sample (for example, removing a stock's 2018 data because it split in 2019).
