import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Robintrack and returns

    Does the change in Robinhood holders, or features of it, predict next-day returns? Only the training return panel is read here: `data/intermediate/clean/returns_train.parquet`, one row per stock and day in the universe known at 15:50 (a price of at least $1 and 100+ holders at the previous close), with return labels ending before the 2019-12-12 split. The test file stays closed until every choice made here is frozen.

    Timing reminder: the `*_prev` columns, `holders_close_prev` and `rh_pct_lag1*` are known before the 15:50 cutoff on day t; the `rh_pct_t*` columns only after the close; `ret_next` is the close-to-close return from day t to the next trading day.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    return mo, pd, plt, sm


@app.cell
def _(mo, pd):
    PANEL_TRAIN_PATH = mo.notebook_dir().parent / "data" / "intermediate" / "clean" / "returns_train.parquet"
    panel = pd.read_parquet(PANEL_TRAIN_PATH)
    return (panel,)


@app.cell
def _(mo, panel):
    _cols = ["permno", "date", "holders_close_prev", "rh_pct_lag1", "rh_pct_lag1_dm", "rh_pct_lag1_wdm", "ret_next", "ret_next_dm", "ret_next_is_delisting", "price_prev", "cap_prev"]
    _mid = len(panel) // 2
    mo.vstack([
        mo.md(f"**Loaded:** {len(panel):,} stock-days in the 15:50 universe, {panel['permno'].nunique():,} PERMNOs, {panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}, {panel.shape[1]} columns"),
        mo.md("100 rows from the middle of the panel:"),
        mo.ui.table(panel[_cols].iloc[_mid - 50:_mid + 50]),
    ])
    return


@app.cell
def _(mo, panel, pd, plt):
    _versions = {
        "rh_pct_lag1_zw": "X: % change in holders on day t-1",
        "rh_logchg_lag1_zw": "X: log change in holders on day t-1",
        "rh_log1pchg_lag1_zw": "X: change in log(1 + holders) on day t-1",
        "ret_next_dm": "y: next-day return minus the universe's mean that day",
    }
    _d = panel[["date", *_versions]].dropna()


    def _stats(s):
        q = s.quantile([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
        return pd.Series({
            "count": len(s), "mean": s.mean(), "std": s.std(), "skew": s.skew(), "excess_kurtosis": s.kurt(),
            "min": s.min(), **{f"p{100 * k:g}": v for k, v in q.items()}, "max": s.max(),
        })


    _summary = pd.DataFrame({_c: _stats(_d[_c]) for _c in _versions})
    _fig, _axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for _ax, (_c, _label) in zip(_axes, _versions.items()):
        _lo, _hi = _d[_c].quantile([0.01, 0.99])
        _ax.hist(_d.loc[_d[_c].between(_lo, _hi), _c], bins=120)
        _ax.set_title(_c, fontsize=9)
    _fig.tight_layout()
    mo.vstack([
        mo.md("**The regression sample** (rows with all four values). X are z-scores of the winsorized values within the 15:50 universe each day (`_zw`). " + "; ".join(f"`{k}`: {v}" for k, v in _versions.items())),
        mo.ui.table(_summary.reset_index(names="statistic").round(6), selection=None, page_size=20),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## OLS: next-day return on the previous day's change in holders

    $r^{dm}_{t \to t+1} = \alpha + \beta X_{t-1} + \varepsilon$, once for each unit of the previous day's change in Robinhood holders, all known by 15:50 on day t:

    1. `rh_pct_lag1_zw`: % change, $h_{t-1}/h_{t-2} - 1$
    2. `rh_logchg_lag1_zw`: log change, $\log(h_{t-1}/h_{t-2})$
    3. `rh_log1pchg_lag1_zw`: change in $\log(1 + h)$, $\log(1 + h_{t-1}) - \log(1 + h_{t-2})$

    Each X is winsorized at that day's 1st and 99th percentiles within the 15:50 universe, then z-scored with that day's mean and standard deviation, so $\beta$ is the next-day return for a one-standard-deviation move. y is `ret_next_dm`, the next-day return minus the universe's mean that day. All three use the same rows. Standard errors are Driscoll-Kraay HAC (statsmodels' `hac-groupsum`, 5 lags): robust to correlation across stocks on the same day and to autocorrelation over days.
    """)
    return


@app.cell
def _(mo, panel, pd, sm):
    HAC_LAGS = 5  # Newey-West rule of thumb, 4 * (T / 100) ** (2 / 9), for about 400 days
    X_VERSIONS = ["rh_pct_lag1_zw", "rh_logchg_lag1_zw", "rh_log1pchg_lag1_zw"]

    _d = panel[["date", "ret_next_dm", *X_VERSIONS]].dropna().sort_values("date")
    _time = pd.factorize(_d["date"], sort=True)[0]
    ols_fits = {
        _x: sm.OLS(_d["ret_next_dm"], sm.add_constant(_d[_x])).fit(
            cov_type="hac-groupsum", cov_kwds={"time": _time, "maxlags": HAC_LAGS}
        )
        for _x in X_VERSIONS
    }
    ols_comparison = pd.DataFrame([
        {"X": _x, "beta_bp_per_sd": _m.params[_x] * 1e4, "hac_t": _m.tvalues[_x], "p": _m.pvalues[_x], "r_squared": _m.rsquared, "n": int(_m.nobs)}
        for _x, _m in ols_fits.items()
    ])
    mo.vstack([
        mo.md(f"**{len(_d):,} stock-days on {_d['date'].nunique():,} days**"),
        mo.ui.table(ols_comparison.round(6), selection=None),
        *[mo.plain_text(_m.summary().as_text()) for _m in ols_fits.values()],
    ])
    return


if __name__ == "__main__":
    app.run()
