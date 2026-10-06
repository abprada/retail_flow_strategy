# Robintrack and returns

What `scripts/02_robintrack_returns.py` does: does the change in Robinhood holders predict next-day relative returns? Every feature is built in `01_clean_explore_robintrack.py`; 02 picks the universe, standardizes within it and estimates.

- **Estimation sections:** read only `returns_train.parquet` (labels from 2018-05-04 that end before the 2019-12-12 split).
- **The backtest at the end:** the one use of `returns_test.parquet`.

## Features

g is the daily % change in holders, close to close. Every feature ends at day t-1, so it is known by 15:50 on day t.

| Feature | Definition |
|---|---|
| level | g(t-1) |
| vol60 | standard deviation of g over the last 60 trading days |
| sqdev20 | (g(t-1) - mean of g over the 20 trading days before)², the last change as a surprise against the stock's own drift |

- **Windows:** 01 builds vol and sqdev over 5, 10, 20, 40 and 60 days. The 60/20 pair was chosen by the lasso cross-validation below, but every pair scored within one standard error of the best, so the choice is weak.
- **Correlations after standardizing:** sqdev20 with vol60 0.41, sqdev20 with level 0.28, level with vol60 -0.03.

## Winsorization, standardization, universe

- **X:** each day, within the universe, winsorized at the 1st and 99th percentiles, then z-scored (`_zw`). β is in basis points per standard deviation.
- **y:** `ret_next_dm`, the next-day return minus the universe mean that day.
- **Universe:** the 15:50 universe from 01 (price of at least $1 and at least 100 holders at day t-1's close), kept to US common stock, foreign common stock and ADRs by CRSP's classification on day t. ETFs, closed-end funds, REITs and units are dropped, about 27% of rows. On train: 1,029,963 stock-days, 3,454 PERMNOs, 392 days.

## OLS and Fama-MacBeth (train, in sample)

- **OLS:** pooled, with Driscoll-Kraay HAC errors (5 lags).
- **Fama-MacBeth:** a cross-sectional OLS each day, averaged, with Newey-West t-stats (5 lags).

| X | OLS alone | Fama-MacBeth alone | OLS together | Fama-MacBeth together |
|---|---|---|---|---|
| level | -2.88 (-4.21) | -2.84 (-4.15) | -1.55 (-2.42) | -1.69 (-2.76) |
| vol60 | -1.78 (-2.65) | -1.86 (-2.81) | -1.05 (-1.47) | -1.11 (-1.52) |
| sqdev20 | -3.39 (-5.24) | -3.35 (-5.12) | -2.22 (-3.24) | -2.14 (-3.05) |

β in bp per sd, t in brackets. "Together" runs on 927,813 rows: 353 days with a median of 2,663 stocks, and an average daily R² of 0.38%.

## Lasso, 10-fold cross-validation (train)

- **What is chosen:** λ and both windows, together.
- **Folds:** 10 contiguous blocks of days, with 5 days left out on each side of each block.
- **Result:**
  - λ goes to the bottom of the grid, so it is effectively OLS: with three non-redundant features there is nothing to select.
  - The cross-validated out-of-sample R² is 0.64 bp (standard error 0.40). That is positive but small, and the window pairs can't be told apart.

## Backtest: ridge refitted daily, through the test period

- **Status:** the single use of the test file, with everything above fixed beforehand. The features come straight from Robintrack, which will not exist live, so this is an upper bound, not a tradable Model 2.
- **Refit:** each day, a ridge on level, vol60 and sqdev20 is fitted on every row whose label had ended. That is an expanding window, starting once it holds 120 training days. λ comes from 5-fold blocked cross-validation each day.
- **Strategy:** long the top fifth of predictions and short the bottom fifth, equal-weighted, gross.

| | Inside train (232 days, pseudo out of sample) | Test (157 days, 2019-12-12 to 2020-08-12) |
|---|---|---|
| Mean, bp/day (NW t) | 3.8 (1.74) | -0.1 (-0.01) |
| Sharpe | 1.87 | -0.01 |
| Annual volatility | 5.1% | 18.5% |
| Maximum drawdown | -3.0% | -24.9% |
| Hit rate | 55% | 57% |
| Turnover per day (gross 2) | 2.30 | 2.18 |
| Break-even cost per unit of turnover | 1.7 bp | negative |
| Alpha against the universe, bp/day (t) | 4.7 (2.30) | 2.2 (0.22) |
| Mean daily IC (NW t) | 0.0089 (4.55) | 0.0064 (1.11) |

**Betas on the test period only, together:**

| X | Pooled OLS (t) | Fama-MacBeth (NW t) |
|---|---|---|
| level | -1.46 (-0.38) | -1.37 (-0.30) |
| vol60 | +5.10 (+1.97) | +4.82 (+1.84) |
| sqdev20 | -9.11 (-4.76) | -8.68 (-3.38) |

## Reading

- **sqdev20 passes out of sample.** On data never used to choose it, its beta is significant and about four times its training size. Surprise retail flow predicts lower next-day returns, more strongly during the 2020 retail surge.
- **The combined strategy does not survive 2020.** vol60 flips sign in the test period, and the expanding-window ridge keeps its old negative weight. March and April 2020 lose about 8.5% each, the worst days are 18 and 23 March, and May to August recover about 18%.
- **Costs dominate.** Even inside train, the break-even is 1.7 bp per unit of turnover at about 2.3 turnover a day.
- **For the plan:** the target passes the check "does it predict returns at all?". That justifies building Model 1. Its value has to show as incremental IC in Model 2 over the tape features, measured on data after 2020, since the Robintrack test period is now used.

## Graveyard

Tried on train and dropped (numbers in the notebook):

- **Acceleration and the deviation from the 60-day mean:** not significant alone.
- **The 20-day mean:** significant alone, but its correlation with vol20 is 0.86, and next to vol its sign turns.
- **The squared change g(t-1)²:** replaced by sqdev, which beats it head to head on the same rows.
- **A small neural network,** fitted monthly on an expanding window: tried and reverted. It did not beat OLS out of sample.

## Caveats

- **Heavily searched on train:** seven features, a window grid, and the selection after seeing results. Only the test-period betas are free of that.
- **No controls** for the tape's attention measures: the previous day's return, its absolute size, abnormal volume, return volatility, size. sqdev may partly proxy for them, which is the main risk for Model 1 adding value over the tape features.
