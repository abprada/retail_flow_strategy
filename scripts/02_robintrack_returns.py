import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Robintrack and returns

    Does the change in Robinhood holders, or features of it, predict next-day returns? Only the training file is read here: `data/intermediate/clean/panel_train.parquet` (2018-05-02 to 2019-12-11). The test file stays closed until every choice made here is frozen.

    Timing reminder: the `*_prev` columns and `holders_prev` are known before the 15:50 cutoff on day t; `holders`, `change` and the `*_t` columns only after the close; `ret_next` is the close-to-close return from day t to the next trading day.
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
    PANEL_TRAIN_PATH = mo.notebook_dir().parent / "data" / "intermediate" / "clean" / "panel_train.parquet"
    panel = pd.read_parquet(PANEL_TRAIN_PATH)
    return (panel,)


@app.cell
def _(mo, panel):
    _cols = ["permno", "date", "holders_prev", "holders", "y_raw", "y_clipped", "y", "ret_t", "ret_next", "price_prev", "cap_prev"]
    _rows = panel[panel["in_target"]]
    _mid = len(_rows) // 2
    mo.vstack([ 
        mo.md(f"**Loaded:** {len(panel):,} PERMNO-days, {panel['permno'].nunique():,} PERMNOs, {panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}, {panel.shape[1]} columns"),
        mo.md("100 target rows from the middle of the panel, with the holder change, the target and the returns:"),
        mo.ui.table(_rows[_cols].iloc[_mid - 50:_mid + 50]),
    ])
    return


@app.cell
def _(mo, panel, pd, plt):
    _d = panel[["date", "ret_next", "rh_pct_lag1_dm"]].dropna()
    _vars = {
        "rh_pct_lag1_dm": "X: previous day's % change in holders, demeaned within the universe each day",
        "ret_next": "y: next-day close-to-close return (raw, not demeaned)",
    }


    def _stats(s):
        q = s.quantile([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
        return pd.Series({
            "count": len(s), "mean": s.mean(), "std": s.std(), "skew": s.skew(), "excess_kurtosis": s.kurt(),
            "min": s.min(), **{f"p{100 * k:g}": v for k, v in q.items()}, "max": s.max(),
            "largest |daily mean|": s.groupby(_d["date"]).mean().abs().max(),
        })


    _summary = pd.DataFrame({_c: _stats(_d[_c]) for _c in _vars})
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for _ax, (_c, _label) in zip(_axes, _vars.items()):
        _lo, _hi = _d[_c].quantile([0.01, 0.99])
        _ax.hist(_d.loc[_d[_c].between(_lo, _hi), _c], bins=150)
        _ax.set_title(f"{_c}: middle 98%", fontsize=10)
        _ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    _fig.tight_layout()
    mo.vstack([
        mo.md("**The regression sample** (train rows with both values). " + "; ".join(f"`{k}` = {v}" for k, v in _vars.items())),
        mo.ui.table(_summary.reset_index(names="statistic").round(6), selection=None, page_size=20),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A first OLS

    $r_{t \to t+1} = \alpha + \beta \cdot$ `rh_pct_lag1_dm` $+ \varepsilon$: next-day close-to-close return on the previous day's change in Robinhood holders, demeaned within the universe. Known at 15:50 on day t, so this is tradable.

    Standard errors are HAC in the panel form (Driscoll-Kraay, statsmodels' `hac-groupsum`): each day's contributions are summed, then a Newey-West correction with 5 lags is applied across days. This allows for correlation between stocks on the same day and for autocorrelation over the following days.
    """)
    return


@app.cell
def _(mo, panel, pd, sm):
    HAC_LAGS = 5  # Newey-West rule of thumb, 4 * (T / 100) ** (2 / 9), for about 400 days

    _d = panel[["date", "ret_next", "rh_pct_lag1_dm"]].dropna().sort_values("date")
    ols_hac = sm.OLS(_d["ret_next"], sm.add_constant(_d["rh_pct_lag1_dm"])).fit(
        cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(_d["date"], sort=True)[0], "maxlags": HAC_LAGS}
    )
    mo.plain_text(ols_hac.summary().as_text())
    return (HAC_LAGS,)


@app.cell
def _(HAC_LAGS, mo, panel, pd, sm):
    _d = panel[["date", "ret_next_dm", "rh_pct_lag1_dm"]].dropna().sort_values("date")
    ols_hac_dm = sm.OLS(_d["ret_next_dm"], sm.add_constant(_d["rh_pct_lag1_dm"])).fit(
        cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(_d["date"], sort=True)[0], "maxlags": HAC_LAGS}
    )
    mo.vstack([
        mo.md("**Same regression with demeaned returns:** `ret_next_dm`, the next-day return minus that day's universe mean."),
        mo.plain_text(ols_hac_dm.summary().as_text()),
    ])
    return


@app.cell
def _(HAC_LAGS, mo, panel, pd, sm):
    _d = panel[["date", "ret_next_dm", "rh_pct_lag1_wdm"]].dropna().sort_values("date")
    ols_hac_w = sm.OLS(_d["ret_next_dm"], sm.add_constant(_d["rh_pct_lag1_wdm"])).fit(
        cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(_d["date"], sort=True)[0], "maxlags": HAC_LAGS}
    )
    mo.vstack([
        mo.md("**With X winsorized:** `rh_pct_lag1_wdm`, the previous day's change clipped at each day's 1st and 99th percentiles within the universe, then demeaned; y is `ret_next_dm`."),
        mo.plain_text(ols_hac_w.summary().as_text()),
    ])
    return


if __name__ == "__main__":
    app.run()
