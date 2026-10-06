import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Robintrack and returns

    Does the change in Robinhood holders, or features of it, predict next-day returns? Only the training return panel is read: `data/intermediate/clean/returns_train.parquet`, one row per stock and day in the universe known at 15:50 (a price of at least $1 and 100+ holders at the previous close), with return labels ending before the 2019-12-12 split. The test file stays closed until every choice made here is frozen.

    Timing: the `*_prev` columns, `holders_close_prev` and the `rh_pct_*_lag1` features are known before the 15:50 cutoff on day t; `ret_next` is the close-to-close return from day t to the next trading day.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    return mo, np, pd, sm


@app.cell
def _(mo, pd):
    PANEL_TRAIN_PATH = mo.notebook_dir().parent / "data" / "intermediate" / "clean" / "returns_train.parquet"
    panel_raw = pd.read_parquet(PANEL_TRAIN_PATH)
    return (panel_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Features

    Five features of the daily % change in holders $g$, built in 01 from each stock's full daily history and ending at day t-1, so known by 15:50 on day t:

    | Feature | Column | Definition |
    |---|---|---|
    | level | `rh_pct_lag1` | $g_{t-1}$ |
    | accel | `rh_pct_accel_lag1` | $g_{t-1} - g_{t-2}$ |
    | sq | `rh_pct_sq_lag1` | $g_{t-1}^2$, a one-day variance |
    | vol | `rh_pct_vol_lag1` | standard deviation of $g$ over the last 20 trading days (at least 10 changes) |
    | dev | `rh_pct_dev_lag1` | $g_{t-1}$ minus its mean over the 60 trading days before (at least 30 changes) |
    """)
    return


@app.cell
def _():
    FEATURES = {
        "level": "rh_pct_lag1",
        "accel": "rh_pct_accel_lag1",
        "sq": "rh_pct_sq_lag1",
        "vol": "rh_pct_vol_lag1",
        "dev": "rh_pct_dev_lag1",
    }
    X = {_name: f"{_col}_zw" for _name, _col in FEATURES.items()}  # the standardized column each regression uses
    return FEATURES, X


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Winsorization and standardization

    Each day, within that day's universe (next section), so every step uses only the row's own day:

    - **X:** each feature is winsorized at the day's 1st and 99th percentiles, then z-scored with the day's mean and standard deviation (`_zw`). A beta is then the next-day return per one-standard-deviation move.
    - **y:** `ret_next_dm`, the next-day return minus the universe's mean that day.
    """)
    return


@app.cell
def _(FEATURES):
    WINSOR_QUANTILES = (0.01, 0.99)


    def standardize(df):
        """Winsorize and z-score each feature, and demean the next-day return, within each day of df."""
        df = df.copy()
        day = df["date"]
        for col in FEATURES.values():
            q = df[col].groupby(day).quantile(list(WINSOR_QUANTILES)).unstack()
            w = df[col].clip(day.map(q[WINSOR_QUANTILES[0]]), day.map(q[WINSOR_QUANTILES[1]]))
            df[f"{col}_zw"] = (w - w.groupby(day).transform("mean")) / w.groupby(day).transform("std")
        df["ret_next_dm"] = df["ret_next"] - df["ret_next"].groupby(day).transform("mean")
        return df

    return (standardize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Universe

    The 15:50 universe from 01, restricted to operating companies by CRSP's classification on day t: US common stock, foreign common stock and ADRs. ETFs, closed-end funds, REITs and units are dropped. X and y are standardized within this universe.
    """)
    return


@app.cell
def _(mo, panel_raw, standardize):
    UNIVERSE_TYPES = ["common stock", "foreign common stock", "ADR"]
    panel = standardize(panel_raw[panel_raw["security_type"].isin(UNIVERSE_TYPES)]).reset_index(drop=True)

    _types = panel_raw["security_type"].value_counts().rename_axis("security_type").reset_index(name="stock-days")
    mo.vstack([
        mo.md(f"**Kept:** {len(panel):,} of {len(panel_raw):,} stock-days ({len(panel) / len(panel_raw):.0%}), {panel['permno'].nunique():,} PERMNOs, "
              f"{panel['date'].nunique():,} days, {panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}"),
        mo.ui.table(_types.assign(kept=_types["security_type"].isin(UNIVERSE_TYPES)), selection=None),
    ])
    return (panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OLS, one feature at a time

    $r^{dm}_{t \to t+1} = \alpha + \beta X_{t-1} + \varepsilon$, pooled over stocks and days, for each feature on the rows where it exists. Standard errors are Driscoll-Kraay HAC (statsmodels' `hac-groupsum`, 5 lags): robust to correlation across stocks on the same day and to autocorrelation over days.
    """)
    return


@app.cell
def _(X, mo, panel, pd, sm):
    HAC_LAGS = 5  # Newey-West rule of thumb, 4 * (T / 100) ** (2 / 9), for about 400 days


    def ols_hac(cols):
        """Pooled OLS of ret_next_dm on cols, with Driscoll-Kraay HAC errors over days."""
        d = panel[["date", "ret_next_dm", *cols]].dropna().sort_values("date")
        return sm.OLS(d["ret_next_dm"], sm.add_constant(d[cols])).fit(
            cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(d["date"], sort=True)[0], "maxlags": HAC_LAGS}
        )


    ols_single = {_name: ols_hac([_col]) for _name, _col in X.items()}
    ols_single_results = pd.DataFrame([
        {"feature": _name, "X": X[_name], "beta_bp_per_sd": _m.params[X[_name]] * 1e4, "hac_t": _m.tvalues[X[_name]],
         "p": _m.pvalues[X[_name]], "r_squared": _m.rsquared, "n": int(_m.nobs)}
        for _name, _m in ols_single.items()
    ])
    mo.vstack([
        mo.ui.table(ols_single_results.round(4), selection=None),
        mo.accordion({"Full OLS output": mo.vstack([mo.plain_text(_m.summary().as_text()) for _m in ols_single.values()])}),
    ])
    return HAC_LAGS, ols_hac


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## OLS, together

    $r^{dm}_{t \to t+1} = \alpha + \beta_1\, \text{level}_{t-1} + \beta_2\, \text{sq}_{t-1} + \beta_3\, \text{vol}_{t-1} + \varepsilon$ on the rows where all three exist, with the same errors. Acceleration and the deviation from the 60-day mean are left out.
    """)
    return


@app.cell
def _(X, mo, ols_hac, pd):
    JOINT = ["level", "sq", "vol"]

    ols_joint = ols_hac([X[_name] for _name in JOINT])
    mo.vstack([
        mo.ui.table(pd.DataFrame({
            "feature": JOINT,
            "beta_bp_per_sd": [ols_joint.params[X[_name]] * 1e4 for _name in JOINT],
            "hac_t": [ols_joint.tvalues[X[_name]] for _name in JOINT],
            "p": [ols_joint.pvalues[X[_name]] for _name in JOINT],
        }).round(4), selection=None),
        mo.plain_text(ols_joint.summary().as_text()),
    ])
    return (JOINT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fama-MacBeth, one feature at a time

    Each training day, a cross-sectional OLS of `ret_next_dm` on one feature; the daily slopes are then averaged over days. The t-stat is the mean slope over its standard error across days: plain (Fama-MacBeth) and Newey-West with 5 lags, since the daily slopes can be autocorrelated. The average slope is the daily return of a long-short portfolio with unit exposure to the feature.
    """)
    return


@app.cell
def _(HAC_LAGS, X, mo, np, panel, pd, sm):
    def fama_macbeth(cols):
        """Daily cross-sectional OLS of ret_next_dm on cols; returns the daily slopes and their averages with t-stats."""
        d = panel[["date", "ret_next_dm", *cols]].dropna().sort_values("date")
        rows = []
        for day, g in d.groupby("date"):
            xmat = np.column_stack([np.ones(len(g)), g[cols].to_numpy()])
            b, *_ = np.linalg.lstsq(xmat, g["ret_next_dm"].to_numpy(), rcond=None)
            resid = g["ret_next_dm"].to_numpy() - xmat @ b
            r2 = 1 - (resid**2).sum() / ((g["ret_next_dm"] - g["ret_next_dm"].mean()) ** 2).sum()
            rows.append({"date": day, "n": len(g), "r_squared": r2, **dict(zip(["const", *cols], b))})
        slopes = pd.DataFrame(rows).set_index("date")

        def summarize(s):
            nw = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
            return {"mean_slope_bp": s.mean() * 1e4, "fm_t": s.mean() / (s.std() / np.sqrt(len(s))), "nw_t": nw.tvalues[0],
                    "share_days_negative": (s < 0).mean()}

        results = pd.DataFrame([{"X": c, **summarize(slopes[c])} for c in ["const", *cols]])
        return slopes, results


    fm_single = {_name: fama_macbeth([_col]) for _name, _col in X.items()}
    fm_single_results = pd.DataFrame([
        {"feature": _name, **_res.iloc[1].to_dict(), "days": len(_slopes), "avg_r_squared": _slopes["r_squared"].mean()}
        for _name, (_slopes, _res) in fm_single.items()
    ])
    mo.ui.table(fm_single_results.round(4), selection=None)
    return (fama_macbeth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fama-MacBeth, together

    Each training day, a cross-sectional OLS of `ret_next_dm` on level, squared change and volatility together (the same three as the pooled OLS); the slopes are averaged over days as above. Each average slope is the daily return of a long-short portfolio with unit exposure to that feature and zero to the other two.
    """)
    return


@app.cell
def _(JOINT, X, fama_macbeth, mo):
    fm_joint_slopes, fm_joint_results = fama_macbeth([X[_name] for _name in JOINT])
    mo.vstack([
        mo.md(f"**{len(fm_joint_slopes):,} days, {int(fm_joint_slopes['n'].median()):,} stocks on the median day, "
              f"average cross-sectional R2 {fm_joint_slopes['r_squared'].mean():.2%}**"),
        mo.ui.table(fm_joint_results.assign(feature=["const", *JOINT]).round(4), selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
