# Robintrack and returns

What `scripts/02_robintrack_returns.py` does: does the change in Robinhood holders predict next-day relative returns? It only estimates. Every variable is built in `01_clean_explore_robintrack.py`, and only `returns_train.parquet` is read: training labels from 2018-05-02 that end before the 2019-12-12 split. **All results are in sample.**

## Setup

- **Rows:** the 15:50 universe, meaning a price of at least $1 and at least 100 holders at day t-1's close. About 1.38 million stock-days over 404 days.
- **y:** `ret_next_dm`, the return from day t's close to the next close, minus the universe mean that day.
- **X:** features of g, the daily % change in holders, all ending at day t-1, so known by 15:50 on day t. Each is used as `_zw`: winsorized at the day's 1st and 99th percentiles, then z-scored within the universe that day. β is in basis points per standard deviation.
- **Standard errors:** Driscoll-Kraay HAC with 5 lags, robust to correlation across stocks on a day and over days.

## Features

| Feature | Definition |
|---|---|
| level | g(t-1) |
| acceleration | g(t-1) - g(t-2) |
| squared change | g(t-1)² |
| volatility | standard deviation of g over 20 trading days |
| deviation | g(t-1) minus its mean over the previous 60 days |

**Correlations:** deviation and level 0.82, level and acceleration 0.59, squared change and volatility 0.37. The squared change has a rank correlation of 0.01 with the level, because it measures size, not direction.

## Results

| X | alone: β (HAC t) | three together: β (HAC t) | lasso β | kept in bootstrap | post-lasso HAC t |
|---|---|---|---|---|---|
| level | -2.17 (-3.92) | -1.02 (-1.83) | -1.50 | 98% | -2.50 |
| acceleration | -0.24 (-0.46) | | +0.91 | 96% | 2.06 |
| squared change | -3.35 (-4.69) | -1.52 (-1.98) | -1.20 | 94% | -1.44 |
| volatility | -2.68 (-4.10) | -2.22 (-3.42) | -1.86 | 100% | -3.21 |
| deviation | -0.36 (-0.65) | | 0 | 48% | |

- **Three together:** joint Wald χ²(3) = 28.3, p = 3e-6.
- **Lasso:** λ is chosen by forward-chaining cross-validation over 5 blocks of days, with a 1-day gap. The cross-validated out-of-sample R² is 0.79 bp. Significance comes from post-lasso OLS and a bootstrap of 5-day blocks with 200 draws. Only volatility has a bootstrap 95% interval clear of 0 with room to spare: [-2.96, -0.69].

## Reading

Stocks with persistent Robinhood activity underperform the next day.

- The 20-day volatility is the only robust feature.
- A single day's move, or its size, adds little once volatility is in.
- The positive acceleration only means the change two days ago also predicts negatively.
- The deviation from a stock's own mean carries nothing.

## Caveats and next steps

- In sample, and the features were chosen after seeing results, so the t-stats overstate.
- No controls yet for the previous day's return or the stock's own return volatility.
- Next: add those controls, then a portfolio sort with the sign fixed and an expanding-window test inside train. The test period is used once, at the end.
