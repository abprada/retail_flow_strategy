# Robintrack and returns

What `scripts/02_robintrack_returns.py` does: does the change in Robinhood holders predict next-day relative returns? Every feature is built in `01_clean_explore_robintrack.py`; 02 picks the universe, standardizes within it and estimates.

- **Estimation sections:** read only `returns_train.parquet` (labels from 2018-05-04 that end before the 2019-12-12 split).
- **The backtest at the end:** the one use of `returns_test.parquet`.

## Features

The holder count for a day is the first Robintrack reading retrieved between 15:40 and 15:50 New York time, the last one before the cutoff. g is the daily % change in holders, from one trading day's count to the next. Days on which a stock received another stock's holders (the acquirer in a stock merger, a new spin-off; trading days 0 to 5 from the event) have no change, because that rise is delivery, not trading. Every feature ends at day t-1, so it is known by 15:50 on day t.

| Feature | Definition |
|---|---|
| level | g(t-1) |
| vol60 | standard deviation of g over the last 60 trading days |
| sqdev20 | (g(t-1) - mean of g over the 20 trading days before)², the last change as a surprise against the stock's own drift |

- **Windows:** 01 builds vol and sqdev over 5, 10, 20, 40 and 60 days. The features use 60 and 20. The lasso cross-validation below puts vol40 with sqdev5 first, but every pair scores within one standard error of the best, so the windows were not changed.
- **Correlations after standardizing:** sqdev20 with vol60 0.41, sqdev20 with level 0.27, level with vol60 -0.03.

## Winsorization, standardization, universe

- **X:** each day, within the universe, winsorized at the 1st and 99th percentiles, then z-scored (`_zw`). β is in basis points per standard deviation.
- **y:** `ret_next_dm`, the next-day return minus the universe mean that day.
- **Universe:** the 15:50 universe from 01 (price of at least $1 and at least 100 holders at day t-1's count), kept to US common stock, foreign common stock and ADRs by CRSP's classification on day t. ETFs, closed-end funds, REITs and units are dropped, about 27% of rows. On train: 1,013,290 stock-days, 3,451 PERMNOs, 387 days.

## OLS and Fama-MacBeth (train, in sample)

- **OLS:** pooled, with Driscoll-Kraay HAC errors (5 lags).
- **Fama-MacBeth:** a cross-sectional OLS each day, averaged, with Newey-West t-stats (5 lags).

| X | OLS alone | Fama-MacBeth alone | OLS together | Fama-MacBeth together |
|---|---|---|---|---|
| level | -3.10 (-4.13) | -2.93 (-3.90) | -2.04 (-2.74) | -1.76 (-2.46) |
| vol60 | -1.71 (-2.47) | -1.39 (-1.91) | -1.03 (-1.40) | -0.46 (-0.54) |
| sqdev20 | -3.17 (-4.32) | -3.02 (-4.14) | -1.97 (-2.58) | -2.34 (-2.83) |

β in bp per sd, t in brackets. "Together" runs on 901,139 rows: 344 days with a median of 2,650 stocks, and an average daily R² of 0.42%.

## Lasso, 10-fold cross-validation (train)

- **What is chosen:** λ and both windows, together.
- **Folds:** 10 contiguous blocks of days, with 5 days left out on each side of each block.
- **Result:**
  - λ goes to the bottom of the grid, so it is effectively OLS: with three non-redundant features there is nothing to select.
  - The best pair is vol40 with sqdev5, with a cross-validated out-of-sample R² of 0.75 bp (standard error 0.43). vol60 with sqdev20 scores 0.73. That is positive but small, and the window pairs can't be told apart.

## Backtest: ridge refitted daily, through the test period

- **Status:** the single use of the test file. The features come straight from Robintrack, which will not exist live, so this is an upper bound, not a tradable Model 2.
- **Refit:** each day, a ridge on level, vol60 and sqdev20 is fitted on every row whose label had ended. That is an expanding window, starting once it holds 120 training days. λ comes from 5-fold blocked cross-validation each day.
- **Strategy:** long the top fifth of predictions and short the bottom fifth, equal-weighted, gross.

| | Inside train (223 days, pseudo out of sample) | Test (141 days, 2019-12-12 to 2020-08-12) |
|---|---|---|
| Mean, bp/day (NW t) | 3.2 (1.43) | 2.1 (0.22) |
| Sharpe | 1.64 | 0.32 |
| Annual volatility | 4.9% | 16.4% |
| Maximum drawdown | -3.5% | -16.6% |
| Hit rate | 51% | 55% |
| Turnover per day (gross 2) | 2.41 | 2.30 |
| Break-even cost per unit of turnover | 1.3 bp | 0.90 bp |
| Alpha against the universe, bp/day (t) | 3.7 (1.79) | 3.0 (0.32) |
| Mean daily IC (NW t) | 0.0070 (3.69) | 0.0046 (0.82) |

**Betas on the test period only, together:**

| X | Pooled OLS (t) | Fama-MacBeth (NW t) |
|---|---|---|
| level | -4.98 (-2.04) | -4.91 (-1.55) |
| vol60 | +6.01 (+2.16) | +6.81 (+2.34) |
| sqdev20 | -10.44 (-4.83) | -10.06 (-3.60) |

## Reading

- **sqdev20 passes out of sample.** On data never used to choose it, its beta is significant and about five times its in-sample beta together (-10.4 against -2.0). Surprise retail flow predicts lower next-day returns, more strongly during the 2020 retail surge.
- **level keeps its sign.** It is negative on test too, significant in pooled OLS but not in Fama-MacBeth.
- **vol60 flips sign in the test period,** and the expanding-window ridge keeps its old negative weight.
- **The combined strategy is not significant on test.** March and April 2020 lose 4.8% and 6.4%, the worst days are 23 and 24 March, and May to August recover about 17%.
- **Costs dominate.** Even inside train, the break-even is 1.3 bp per unit of turnover at about 2.4 turnover a day.
- **For the plan:** the target passes the check "does it predict returns at all?". That justifies building Model 1. Its value has to show as incremental IC in Model 2 over the tape features, measured on data after 2020, since the Robintrack test period is now used.

## Graveyard

Tried on train and dropped (numbers in the notebook):

- **Acceleration and the deviation from the 60-day mean:** weak alone (only the deviation reaches t -2.05, in Fama-MacBeth), and mostly the level again (correlations 0.55 and 0.82).
- **The 20-day mean:** significant alone, but its correlation with vol20 is 0.86, and next to vol its sign turns.
- **The squared change g(t-1)²:** replaced by sqdev, which beats it head to head next to the level (-2.35 against -0.57).
- **A small neural network,** fitted monthly on an expanding window: tried and reverted. It did not beat OLS out of sample.

## Caveats

- **Heavily searched on train:** seven features, a window grid, and the selection after seeing results. Only the test-period betas are free of that.
- **No controls** for the tape's attention measures: the previous day's return, its absolute size, abnormal volume, return volatility, size. sqdev may partly proxy for them, which is the main risk for Model 1 adding value over the tape features.
