# Robintrack and returns

What `scripts/02_robintrack_returns.py` does: does the change in Robinhood holders predict next-day relative returns? Every feature is built in `01_clean_explore_robintrack.py`; 02 picks the universe, standardizes within it and estimates. Only `returns_train.parquet` is read: training labels from 2018-05-04 that end before the 2019-12-12 split. **All results are in sample.**

## Features

g is the daily % change in holders, close to close. Every feature ends at day t-1, so it is known by 15:50 on day t.

| Feature | Definition |
|---|---|
| level | g(t-1) |
| vol | standard deviation of g over the last 20 trading days |
| sqdev | (g(t-1) - mean of g over the 20 trading days before)², the last change as a surprise against the stock's own drift |

Correlations after standardizing: sqdev with vol 0.66, sqdev with level 0.26, level with vol 0.01.

## Winsorization and standardization

Each day, within that day's universe:

- **X:** winsorized at the 1st and 99th percentiles, then z-scored (`_zw`). β is in basis points per standard deviation.
- **y:** `ret_next_dm`, the next-day return minus the universe mean that day.

## Universe

The 15:50 universe from 01 (price of at least $1 and at least 100 holders at day t-1's close), kept to operating companies by CRSP's classification on day t: US common stock, foreign common stock and ADRs. ETFs, closed-end funds, REITs and units are dropped, about 27% of rows. That leaves 1,029,963 stock-days on 3,454 PERMNOs over 392 days.

## Results

- **OLS:** pooled, with Driscoll-Kraay HAC errors (5 lags).
- **Fama-MacBeth:** a cross-sectional OLS each day, averaged over days, with Newey-West t-stats (5 lags).

| X | OLS alone: β (t) | Fama-MacBeth alone: β (t) | OLS together: β (t) | Fama-MacBeth together: β (t) |
|---|---|---|---|---|
| level | -2.88 (-4.21) | -2.84 (-4.15) | -1.76 (-2.76) | -1.82 (-2.93) |
| vol | -3.00 (-4.10) | -3.01 (-4.14) | -1.89 (-1.84) | -1.61 (-1.52) |
| sqdev | -3.39 (-5.24) | -3.35 (-5.12) | -1.70 (-1.89) | -1.88 (-2.13) |

"Together" runs on the 977,735 rows where all three exist, 372 days with a median of 2,654 stocks. The average daily cross-sectional R² is 0.40%.

## Reading

Stocks whose Robinhood flow has been large, erratic or surprising underperform the next day.

- **Alone,** each feature predicts about -3 bp per standard deviation, with t-stats of 4 to 5.
- **Together,** the level stays significant. Vol and sqdev share one effect: the size of recent flow relative to the stock's normal flow. Each takes part of it, and neither is clearly significant on its own.
- **The effect is the sum, not any one coefficient:** about -5 bp a day for a one-standard-deviation move in all three.

## Graveyard

Tried and dropped (numbers in the notebook):

- **Acceleration and the deviation from the 60-day mean:** not significant alone.
- **The 20-day mean:** significant alone, but its correlation with vol is 0.86, and next to vol its sign turns.
- **The squared change g(t-1)²:** replaced by sqdev, which beats it head to head on the same rows (sq t -0.93, sqdev t -3.04).

## Caveats and next steps

- **In sample, and heavily searched:** seven features were tried on the same training data and the three kept were chosen after seeing results. Sqdev was the last one tried, so its t of -5.2 is the most data-mined number here. The t-stats overstate.
- **No controls yet** for the previous day's return, the stock's own return volatility, or size. Vol and sqdev may partly stand in for stocks with few holders, whose % changes are noisy by construction.
- **Next:**
  - add those controls;
  - run a portfolio sort with the sign fixed and an expanding-window test inside train;
  - use the test period once, at the end.
