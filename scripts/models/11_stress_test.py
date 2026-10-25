import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stress test of the best long-shorts

    The three best long-shorts of `explanations/summary.md`, as 05 trades them: each day the ≥ $1B stocks are ranked by prediction, the top decile bought and the bottom one sold short, equal-weighted with $1 a side, at the close and held to the next close, before costs.

    - **LightGBM, refit yearly** (`lgbm_yearly`): the final model.
    - **The two-branch model after step 3 at λ = 0.1** (`two_branch_day_cnn_lam0.1/comb_s3`): the CNN family's pick (it replaced λ = 0.5, the model the success rule judged, on 2026-10-25).
    - **The two-branch combined head after step 2** (`comb_s2_lr0.0003_h16-8`, the setting kept): the best Sharpe.

    All three rank the same stock-days over 2019-2023, the validation years their settings were chosen on, so every number here is optimistic. The test years (2024-2025) are not read.
    """)
    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from dsr.backtest import books, daily, positions, stress
    from dsr.evaluation import daily_rank_ic, ic_summary, r2_summary

    return alt, books, daily, daily_rank_ic, ic_summary, mo, np, pd, positions, r2_summary, stress


@app.cell
def _(mo):
    WALKFORWARD = mo.notebook_dir().parents[1] / "data" / "walkforward"
    MODELS = {"LightGBM": "lgbm_yearly", "two-branch, step 3 (λ 0.1)": "two_branch_day_cnn_lam0.1/comb_s3",
              "two-branch, step 2": "two_branch_day_cnn/comb_s2_lr0.0003_h16-8"}  # step 2's kept setting (choices.json)
    BUCKETS = 10  # deciles
    DROP = 0.01  # the share of best days (and of best winning days) removed
    return BUCKETS, DROP, MODELS, WALKFORWARD


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The books

    For each model, five daily return series, each summed over the days like 05's long-short (the book is reset to $1 a side every day):

    | Book | Daily return | Turnover |
    |---|---|---|
    | long-short | long leg minus short leg | the mean of the two legs': the share of the $2 book replaced |
    | long leg | the long leg's average return | the long leg's |
    | short leg | minus the short leg's average return (what the short earns) | the short leg's |
    | long leg − average | the long leg minus the day's average | the long leg's |
    | average − short leg | the day's average minus the short leg | the short leg's |

    The legs as held carry the market's direction; against the average they are what each adds to the long-short (the two add up to it). **The universe:** the day's equal-weighted average return of every ≥ $1B stock ranked, held long, for reference; the correlations are with it.
    """)
    return


@app.cell
def _(BUCKETS, DROP, MODELS, WALKFORWARD, books, daily, daily_rank_ic, ic_summary, pd, positions, r2_summary,
      stress):
    _rows, _signal, series = [], [], {}
    for _model, _folder in MODELS.items():
        _p = pd.read_parquet(WALKFORWARD / _folder / "predictions.parquet",
                             columns=["day", "permno", "pred", "target_demeaned", "ret_next", "large_cap"])
        _p = _p[_p["large_cap"]]
        _ic = ic_summary(daily_rank_ic(_p["day"], _p["pred"], _p["ret_next"]))
        _signal.append({"model": _model, "ic": _ic["ic_mean"], "ic_t": _ic["ic_t"],
                        **r2_summary(_p["day"], _p["pred"], _p["target_demeaned"])})
        _held, average = positions(_p["day"], _p["permno"], _p["pred"], _p["ret_next"], BUCKETS)
        for _book, (_r, _turnover) in books(daily(_held, average)).items():
            series[_model, _book] = _r
            _rows.append({"model": _model, "book": _book, **stress(_r, average, DROP),
                          "turnover": float(_turnover.mean())})
    _rows.append({"model": "universe", "book": "held long", **stress(average, average, DROP)})  # the same days for all
    stats, signal = pd.DataFrame(_rows), pd.DataFrame(_signal)
    return series, signal, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The predictions

    On every ≥ $1B stock-day, not only the deciles traded. **IC:** the mean daily rank correlation of the prediction with the next day's return. **R²:** against a prediction of 0, on the target the models train on (`target_demeaned`), in %, pooled over the days like 03's. Both with Newey-West t-statistics over days (5 lags); the R²'s is that of each day's fall in squared error.
    """)
    return


@app.cell
def _(mo, signal):
    mo.ui.table(signal.round(4), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Risk and return

    Mean daily return (bp), the same a year (%, 252 days, summed) and its Newey-West t-statistic, and annualized volatility (%); the Sharpe ratio, annualized, and its t-statistic (Mertens' standard error: it allows for the skewness and fat tails but not for autocorrelation, so it is close to the mean's t); the Sortino ratio, annualized (it divides by the root mean square of the losing days, counted over every day); the largest fall of the summed return (% of $1 a side); the number of drawdowns (runs of days below the running peak) and their mean and longest length in trading days; the mean share of the book replaced a day; the daily correlation with the universe.
    """)
    return


@app.cell
def _(mo, stats):
    mo.ui.table(stats[["model", "book", "mean_bp", "annual_mean_pct", "t", "annual_vol_pct", "sharpe", "sharpe_t",
                       "sortino", "max_drawdown_pct",
                       "drawdowns", "drawdown_days_mean", "drawdown_days_max", "turnover", "corr_universe"]].round(3),
                selection=None, page_size=20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The distribution of daily returns

    Skewness and excess kurtosis (0 for a normal distribution); the 1st, 5th, 50th, 95th and 99th percentiles and the worst and best day (bp); the share of winning days and the mean of the winning and of the losing days (bp).
    """)
    return


@app.cell
def _(mo, stats):
    mo.ui.table(stats[["model", "book", "skew", "excess_kurtosis", "p1_bp", "p5_bp", "p50_bp", "p95_bp", "p99_bp",
                       "worst_bp", "best_bp", "win_days", "average_win_bp", "average_loss_bp"]].round(3),
                selection=None, page_size=20)
    return


@app.cell
def _(MODELS, alt, np, pd, series):
    _edges = np.arange(-400, 401, 20)
    _x = np.linspace(-400, 400, 161)
    _charts = []
    for _m in MODELS:
        _bp = 1e4 * series[_m, "long-short"].dropna()
        _bars = pd.DataFrame({"left": _edges[:-1], "right": _edges[1:],
                              "days": np.histogram(_bp.clip(-399, 399), _edges)[0]})
        _normal = pd.DataFrame({"bp": _x, "days": len(_bp) * 20 * np.exp(-0.5 * ((_x - _bp.mean()) / _bp.std()) ** 2)
                                / (_bp.std() * np.sqrt(2 * np.pi))})
        _hist = alt.Chart(_bars).mark_bar(color="#019df1").encode(
            x=alt.X("left:Q", title="bp a day (clipped at ±400)"), x2="right:Q", y=alt.Y("days:Q", title="days"))
        _line = alt.Chart(_normal).mark_line(color="#002438", strokeWidth=1.5).encode(x="bp:Q", y="days:Q")
        _charts.append(alt.layer(_hist, _line).properties(width=240, height=220, title=_m))
    (alt.hconcat(*_charts).resolve_scale(y="shared")
     .configure(background="white").configure_axis(grid=False)
     .properties(title="Daily long-short returns, with a normal of the same mean and volatility"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Without the best days

    The mean (bp) and Sharpe ratio after removing the best 1% of all days (13 of 1,257), and after removing the best 1% of the winning days instead (about 7). The days are dropped, not set to 0.
    """)
    return


@app.cell
def _(mo, stats):
    mo.ui.table(stats[["model", "book", "mean_bp", "sharpe", "mean_bp_without_best_days", "sharpe_without_best_days",
                       "mean_bp_without_best_wins", "sharpe_without_best_wins"]].round(3), selection=None, page_size=20)
    return


if __name__ == "__main__":
    app.run()
