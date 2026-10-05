# Holder-change features and next-day returns

Do features of the daily change in Robinhood holders predict next-day relative returns? Built in `scripts/01_clean_explore_robintrack.py`, estimated in `scripts/02_robintrack_returns.py`. Everything here is **in sample on the training period** (2018-05-02 to 2019-12-11); the test file is untouched.

## Features (built in 01)

With g the daily % change in holders, every feature ends at day t-1's change, so it is known by 15:50 on day t:

| Feature | Definition |
|---|---|
| level `rh_pct_lag1` | g(t-1) |
| acceleration `rh_pct_accel_lag1` | g(t-1) - g(t-2) |
| squared change `rh_pct_sq_lag1` | g(t-1)² |
| volatility `rh_pct_vol_lag1` | standard deviation of g over the last 20 trading days (at least 10 changes) |
| deviation `rh_pct_dev_lag1` | g(t-1) minus its mean over the 60 trading days before (at least 30 changes) |

- **Built in 01, not 02,** because the rolling windows need each stock's full daily history. The return panel keeps only universe rows, so a window computed there would skip days.
- **Used as `_zw`:** winsorized at the day's 1st and 99th percentiles within the 15:50 universe, then z-scored that day. β is in return per standard deviation.

## Method (02)

- y is `ret_next_dm`, the next-day return minus the universe mean that day.
- Standard errors are Driscoll-Kraay HAC with 5 lags, robust to correlation across stocks on a day and over days.

## Results

**Correlations.** The deviation is nearly the level again (0.82), because a stock's 60-day mean change is tiny next to its daily change. The squared change measures size, not direction: its rank correlation with the level is 0.01. It overlaps with volatility (0.37).

**One feature at a time:**

| X | β (bp per sd) | HAC t |
|---|---|---|
| level | -2.17 | -3.92 |
| acceleration | -0.24 | -0.46 |
| squared change | -3.35 | -4.69 |
| volatility | -2.68 | -4.10 |
| deviation | -0.36 | -0.65 |

**Lasso on all five.** λ is chosen by forward-chaining cross-validation over 5 blocks of days, with a 1-day gap. Significance comes from post-lasso OLS and a bootstrap of 5-day blocks with 200 draws.

| X | lasso β | kept in bootstrap | bootstrap 95% interval | post-lasso HAC t |
|---|---|---|---|---|
| volatility | -1.86 | 100% | [-2.96, -0.69] | -3.21 |
| level | -1.50 | 98% | [-3.46, -0.11] | -2.50 |
| acceleration | +0.91 | 96% | [0.00, 2.14] | 2.06 |
| squared change | -1.20 | 94% | [-2.91, 0.03] | -1.44 |
| deviation | 0 | 48% | [-0.66, 1.33] | - |

The cross-validated out-of-sample R² is 0.79 bp, positive across the λ grid. Shrinkage barely helps: with five features and 1.25 million rows, the lasso is close to OLS.

## Reading

- **Stocks with persistent Robinhood activity underperform the next day.** The 20-day volatility is the only robust feature.
- **A single day's move adds little.** The squared change loses significance once volatility is in.
- **The positive acceleration is not a signal of its own.** Next to the level it means the change two days ago also predicts negatively.
- **The deviation from a stock's own mean carries nothing.**

## Caveats

- All in sample. The features were chosen after seeing these results, so the t-stats overstate.
- Not yet controlled for the previous day's return or the stock's own return volatility. High-attention stocks are also volatile stocks, which tend to underperform on their own.
- Next: those controls, then a portfolio sort with the sign fixed and an expanding-window test inside train. The test period is used once, at the end.
