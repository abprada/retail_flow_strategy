import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Robintrack and returns

    Does the change in Robinhood holders, or features of it, predict next-day returns? Only the training return panel is read here: `data/intermediate/clean/returns_train.parquet`, one row per stock and day in the universe known at 15:50 (a price of at least $1 and 100+ holders at the previous close), with return labels ending before the 2019-12-12 split. The universe is narrowed to operating companies, and X and y are standardized each day, in the next section. The test file stays closed until every choice made here is frozen.

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
    from sklearn.linear_model import Lasso, lasso_path
    from sklearn.model_selection import TimeSeriesSplit

    return Lasso, TimeSeriesSplit, lasso_path, mo, np, pd, plt, sm


@app.cell
def _(mo, pd):
    PANEL_TRAIN_PATH = mo.notebook_dir().parent / "data" / "intermediate" / "clean" / "returns_train.parquet"
    panel_raw = pd.read_parquet(PANEL_TRAIN_PATH)
    return (panel_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Universe and standardization

    The panel from 01 holds raw values only, because the right demeaning and scaling depends on which stocks are in the universe. Both happen here, each day within the universe:

    - **Universe:** the 15:50 universe from 01, restricted to operating companies by CRSP's classification on day t: US common stock, foreign common stock and ADRs. ETFs, closed-end funds, REITs and units are dropped.
    - **X:** each holder-change feature is winsorized at that day's 1st and 99th percentiles, then z-scored with that day's mean and standard deviation (`_zw`).
    - **y:** `ret_next_dm`, the next-day return minus the universe's mean that day.

    Every step uses only the row's own day, so nothing crosses days or the split.
    """)
    return


@app.cell
def _(panel_raw):
    UNIVERSE_TYPES = ["common stock", "foreign common stock", "ADR"]
    WINSOR_QUANTILES = (0.01, 0.99)
    RAW_FEATURES = ["rh_pct_lag1", "rh_pct_accel_lag1", "rh_pct_sq_lag1", "rh_pct_vol_lag1", "rh_pct_dev_lag1"]

    _p = panel_raw[panel_raw["security_type"].isin(UNIVERSE_TYPES)].copy()
    _day = _p["date"]
    for _f in RAW_FEATURES:
        _q = _p[_f].groupby(_day).quantile(list(WINSOR_QUANTILES)).unstack()
        _w = _p[_f].clip(_day.map(_q[WINSOR_QUANTILES[0]]), _day.map(_q[WINSOR_QUANTILES[1]]))
        _p[f"{_f}_zw"] = (_w - _w.groupby(_day).transform("mean")) / _w.groupby(_day).transform("std")
    _p["ret_next_dm"] = _p["ret_next"] - _p["ret_next"].groupby(_day).transform("mean")
    panel = _p.reset_index(drop=True)
    return UNIVERSE_TYPES, panel


@app.cell
def _(UNIVERSE_TYPES, mo, panel, panel_raw):
    _cols = ["permno", "date", "security_type", "holders_close_prev", "rh_pct_lag1", "rh_pct_lag1_zw", "ret_next", "ret_next_dm", "ret_next_is_delisting", "price_prev", "cap_prev"]
    _mid = len(panel) // 2
    mo.vstack([
        mo.md(f"**Loaded:** {len(panel_raw):,} stock-days in the 15:50 universe. **Kept** ({', '.join(UNIVERSE_TYPES)}): {len(panel):,} stock-days ({len(panel) / len(panel_raw):.0%}), {panel['permno'].nunique():,} PERMNOs, {panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}, {panel.shape[1]} columns"),
        mo.ui.table(panel_raw["security_type"].value_counts().rename_axis("security_type").reset_index(name="stock-days")
                    .assign(kept=lambda d: d["security_type"].isin(UNIVERSE_TYPES)), selection=None),
        mo.md("100 rows from the middle of the panel:"),
        mo.ui.table(panel[_cols].iloc[_mid - 50:_mid + 50]),
    ])
    return


@app.cell
def _(mo, panel, pd, plt):
    _versions = {
        "rh_pct_lag1_zw": "X: % change in holders on day t-1",
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
    _fig, _axes = plt.subplots(1, len(_versions), figsize=(4 * len(_versions), 3.2))
    for _ax, (_c, _label) in zip(_axes, _versions.items()):
        _lo, _hi = _d[_c].quantile([0.01, 0.99])
        _ax.hist(_d.loc[_d[_c].between(_lo, _hi), _c], bins=120)
        _ax.set_title(_c, fontsize=9)
    _fig.tight_layout()
    mo.vstack([
        mo.md("**The regression sample** (rows with both values). X is the z-score of the winsorized values within the universe each day (`_zw`). " + "; ".join(f"`{k}`: {v}" for k, v in _versions.items())),
        mo.ui.table(_summary.reset_index(names="statistic").round(6), selection=None, page_size=20),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## OLS: next-day return on the previous day's change in holders

    $r^{dm}_{t \to t+1} = \alpha + \beta X_{t-1} + \varepsilon$, with X the previous day's change in Robinhood holders, known by 15:50 on day t:

    `rh_pct_lag1_zw`: % change, $h_{t-1}/h_{t-2} - 1$

    X is winsorized at that day's 1st and 99th percentiles within the universe, then z-scored with that day's mean and standard deviation, so $\beta$ is the next-day return for a one-standard-deviation move. y is `ret_next_dm`, the next-day return minus the universe's mean that day. Standard errors are Driscoll-Kraay HAC (statsmodels' `hac-groupsum`, 5 lags): robust to correlation across stocks on the same day and to autocorrelation over days.
    """)
    return


@app.cell
def _(mo, panel, pd, sm):
    HAC_LAGS = 5  # Newey-West rule of thumb, 4 * (T / 100) ** (2 / 9), for about 400 days
    X_VERSIONS = ["rh_pct_lag1_zw"]

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
    return (HAC_LAGS,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Features of the holder-change history

    Four features of the daily % change in holders $g$, all built in 01 from each stock's full daily history and ending at day t-1, so known by 15:50 on day t:

    1. `rh_pct_accel_lag1`: acceleration, $g_{t-1} - g_{t-2}$
    2. `rh_pct_sq_lag1`: squared change, $g_{t-1}^2$, a one-day variance
    3. `rh_pct_vol_lag1`: volatility, the standard deviation of $g$ over the last 20 trading days (at least 10 changes)
    4. `rh_pct_dev_lag1`: deviation from the long-run mean, $g_{t-1}$ minus its mean over the 60 trading days before (at least 30 changes)

    Each is used as `_zw`, like `rh_pct_lag1_zw`: winsorized at the day's 1st and 99th percentiles within the universe, then z-scored that day (see the universe section). The matrix is the pooled Pearson correlation on rows with all five values.
    """)
    return


@app.cell
def _(mo, np, panel, plt):
    HOLDER_FEATURES = ["rh_pct_lag1_zw", "rh_pct_accel_lag1_zw", "rh_pct_sq_lag1_zw", "rh_pct_vol_lag1_zw", "rh_pct_dev_lag1_zw"]

    _d = panel[HOLDER_FEATURES].dropna()
    holder_feature_corr = _d.corr()
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _im = _ax.imshow(holder_feature_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    _labels = [_c.removeprefix("rh_pct_").removesuffix("_zw") for _c in HOLDER_FEATURES]
    _ax.set_xticks(range(len(_labels)), _labels, rotation=45, ha="right")
    _ax.set_yticks(range(len(_labels)), _labels)
    for (_i, _j), _v in np.ndenumerate(holder_feature_corr.to_numpy()):
        _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=9)
    _fig.colorbar(_im, ax=_ax, shrink=0.8)
    _fig.tight_layout()
    mo.vstack([mo.md(f"**{len(_d):,} stock-days on {panel.loc[_d.index, 'date'].nunique():,} days**"), _fig])
    return (HOLDER_FEATURES,)


@app.cell
def _(mo):
    mo.md(r"""
    ### OLS on each feature, one at a time

    $r^{dm}_{t \to t+1} = \alpha + \beta X_{t-1} + \varepsilon$ for each of the five features above, each on the rows where it exists. All X are the `_zw` versions (winsorized at the day's 1st and 99th percentiles, then demeaned and scaled within the universe that day), so $\beta$ is in return per standard deviation. y and the Driscoll-Kraay HAC errors are as in the OLS above.
    """)
    return


@app.cell
def _(HAC_LAGS, HOLDER_FEATURES, mo, panel, pd, sm):
    _fits = {}
    for _x in HOLDER_FEATURES:
        _d = panel[["date", "ret_next_dm", _x]].dropna().sort_values("date")
        _fits[_x] = sm.OLS(_d["ret_next_dm"], sm.add_constant(_d[_x])).fit(
            cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(_d["date"], sort=True)[0], "maxlags": HAC_LAGS}
        )
    feature_ols_fits = _fits
    feature_ols_comparison = pd.DataFrame([
        {"X": _x, "beta_bp_per_sd": _m.params[_x] * 1e4, "hac_t": _m.tvalues[_x], "p": _m.pvalues[_x], "r_squared": _m.rsquared, "n": int(_m.nobs)}
        for _x, _m in feature_ols_fits.items()
    ])
    mo.vstack([
        mo.ui.table(feature_ols_comparison.round(6), selection=None),
        *[mo.plain_text(_m.summary().as_text()) for _m in feature_ols_fits.values()],
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### OLS on the three significant features together

    $r^{dm}_{t \to t+1} = \alpha + \beta_1\, \text{level}_{t-1} + \beta_2\, \text{sq}_{t-1} + \beta_3\, \text{vol}_{t-1} + \varepsilon$, in sample on the training period, on the rows where all three exist. Same `_zw` units, y and Driscoll-Kraay HAC errors as above.
    """)
    return


@app.cell
def _(HAC_LAGS, mo, panel, pd, sm):
    JOINT_FEATURES = ["rh_pct_lag1_zw", "rh_pct_sq_lag1_zw", "rh_pct_vol_lag1_zw"]

    _d = panel[["date", "ret_next_dm", *JOINT_FEATURES]].dropna().sort_values("date")
    joint_ols_fit = sm.OLS(_d["ret_next_dm"], sm.add_constant(_d[JOINT_FEATURES])).fit(
        cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(_d["date"], sort=True)[0], "maxlags": HAC_LAGS}
    )
    mo.vstack([
        mo.md(f"**{len(_d):,} stock-days on {_d['date'].nunique():,} days**"),
        mo.plain_text(joint_ols_fit.summary().as_text()),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Lasso on all five features

    $\min_{\alpha, \beta}\ \frac{1}{2n}\sum (r^{dm}_{t \to t+1} - \alpha - X_{t-1}\beta)^2 + \lambda \lVert \beta \rVert_1$ with the five `_zw` features above, on the rows where all five exist. The X are already on one scale (z-scores within each day), so the penalty treats them alike.

    $\lambda$ is chosen by forward-chaining cross-validation over days: the training days are cut into 6 consecutive blocks, and each of the last 5 is predicted by a fit on all days before it, with a 1-day gap so no training label overlaps a validation day. The score is the out-of-sample $R^2$ against predicting the training mean, which is about 0 since y is demeaned each day. The final fit uses the chosen $\lambda$ on all training rows; it is still in sample.
    """)
    return


@app.cell
def _(
    HOLDER_FEATURES,
    Lasso,
    TimeSeriesSplit,
    lasso_path,
    mo,
    np,
    panel,
    pd,
    plt,
):
    LASSO_FOLDS = 5
    _d = panel[["date", "ret_next_dm", *HOLDER_FEATURES]].dropna().sort_values("date").reset_index(drop=True)
    _X, _y = _d[HOLDER_FEATURES].to_numpy(), _d["ret_next_dm"].to_numpy()
    _days = np.sort(_d["date"].unique())
    _lambdas, _, _ = lasso_path(_X - _X.mean(0), _y - _y.mean(), alphas=60, eps=1e-4)

    # forward-chaining folds over days, with a 1-day gap between the last training day and the validation block
    _r2 = []
    for _tr_days, _va_days in TimeSeriesSplit(n_splits=LASSO_FOLDS, gap=1).split(_days):
        _tr, _va = _d["date"].isin(_days[_tr_days]).to_numpy(), _d["date"].isin(_days[_va_days]).to_numpy()
        _xm, _ym = _X[_tr].mean(0), _y[_tr].mean()
        _, _coefs, _ = lasso_path(_X[_tr] - _xm, _y[_tr] - _ym, alphas=_lambdas)
        _pred = _ym + (_X[_va] - _xm) @ _coefs  # one column per lambda
        _sse = ((_y[_va, None] - _pred) ** 2).sum(0)
        _r2.append(1 - _sse / ((_y[_va] - _ym) ** 2).sum())
    lasso_cv = pd.DataFrame({"lambda": _lambdas, "oos_r2_mean": np.mean(_r2, axis=0), "oos_r2_se": np.std(_r2, axis=0, ddof=1) / np.sqrt(LASSO_FOLDS)})
    LASSO_LAMBDA = lasso_cv.loc[lasso_cv["oos_r2_mean"].idxmax(), "lambda"]

    _, _path, _ = lasso_path(_X - _X.mean(0), _y - _y.mean(), alphas=_lambdas)
    lasso_fit = Lasso(alpha=LASSO_LAMBDA).fit(_X, _y)
    lasso_coefs = pd.DataFrame({"X": HOLDER_FEATURES, "beta_bp_per_sd": lasso_fit.coef_ * 1e4})

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(13, 4))
    for _i, _x in enumerate(HOLDER_FEATURES):
        _a1.plot(_lambdas, _path[_i] * 1e4, label=_x.removeprefix("rh_pct_").removesuffix("_zw"))
    _a1.set(xscale="log", xlabel="lambda", ylabel="beta (bp per sd)", title="Lasso path, all training rows")
    _a1.legend(fontsize=8)
    _a2.errorbar(lasso_cv["lambda"], lasso_cv["oos_r2_mean"] * 1e4, yerr=lasso_cv["oos_r2_se"] * 1e4, fmt="o-", ms=3)
    _a2.axhline(0, color="grey", lw=0.8)
    _a2.set(xscale="log", xlabel="lambda", ylabel="out-of-sample R2 (bp)", title=f"Forward-chaining CV, {LASSO_FOLDS} folds")
    for _ax in (_a1, _a2):
        _ax.axvline(LASSO_LAMBDA, color="k", ls="--", lw=0.8)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"**{len(_d):,} stock-days on {len(_days):,} days.** Chosen lambda {LASSO_LAMBDA:.2e}, CV out-of-sample R2 {lasso_cv['oos_r2_mean'].max() * 1e4:.2f} bp (dashed line)."),
        mo.ui.table(lasso_coefs.round(4), selection=None),
        _fig,
    ])
    return LASSO_LAMBDA, lasso_fit


@app.cell
def _(mo):
    mo.md(r"""
    ### Significance of the lasso betas

    Lasso coefficients have no standard errors of their own: the penalty biases them towards 0 and the selection depends on the data. Two ways around it:

    1. **Post-lasso OLS:** OLS with Driscoll-Kraay HAC errors on the features the lasso kept. The t-stats ignore the selection step, so they overstate significance when the selection is unstable; the bootstrap below shows how often each feature is kept.
    2. **Block bootstrap of the lasso itself:** resample blocks of 5 consecutive training days (the HAC lag length) with replacement, refit the lasso at the chosen $\lambda$, and record each $\beta$. Resampling whole days keeps the correlation across stocks on a day; blocks keep the autocorrelation over days. Reported: how often each feature is kept, and the 2.5th and 97.5th percentiles of $\beta$. $\lambda$ stays fixed at the cross-validated value.
    """)
    return


@app.cell
def _(
    HAC_LAGS,
    HOLDER_FEATURES,
    LASSO_LAMBDA,
    Lasso,
    lasso_fit,
    mo,
    np,
    panel,
    pd,
    sm,
):
    LASSO_BOOT_DRAWS, LASSO_BOOT_BLOCK = 200, HAC_LAGS
    _d = panel[["date", "ret_next_dm", *HOLDER_FEATURES]].dropna().sort_values("date").reset_index(drop=True)
    _X, _y = _d[HOLDER_FEATURES].to_numpy(), _d["ret_next_dm"].to_numpy()

    # post-lasso OLS on the kept features
    LASSO_SELECTED = [_x for _x, _b in zip(HOLDER_FEATURES, lasso_fit.coef_) if _b != 0]
    post_lasso_fit = sm.OLS(_d["ret_next_dm"], sm.add_constant(_d[LASSO_SELECTED])).fit(
        cov_type="hac-groupsum", cov_kwds={"time": pd.factorize(_d["date"], sort=True)[0], "maxlags": HAC_LAGS}
    )

    # moving-block bootstrap over days
    _day_rows = np.split(np.arange(len(_d)), np.flatnonzero(np.diff(pd.factorize(_d["date"])[0])) + 1)
    _n_days = len(_day_rows)
    _rng = np.random.default_rng(0)
    _draws = []
    for _ in range(LASSO_BOOT_DRAWS):
        _starts = _rng.integers(0, _n_days - LASSO_BOOT_BLOCK + 1, size=-(-_n_days // LASSO_BOOT_BLOCK))
        _rows = np.concatenate([_day_rows[_k] for _s in _starts for _k in range(_s, _s + LASSO_BOOT_BLOCK)])
        _draws.append(Lasso(alpha=LASSO_LAMBDA).fit(_X[_rows], _y[_rows]).coef_)
    _draws = np.array(_draws) * 1e4
    lasso_inference = pd.DataFrame({
        "X": HOLDER_FEATURES,
        "lasso_beta_bp": lasso_fit.coef_ * 1e4,
        "boot_kept_share": (_draws != 0).mean(0),
        "boot_p2.5_bp": np.percentile(_draws, 2.5, axis=0),
        "boot_p97.5_bp": np.percentile(_draws, 97.5, axis=0),
        "post_lasso_beta_bp": [post_lasso_fit.params.get(_x, np.nan) * 1e4 for _x in HOLDER_FEATURES],
        "post_lasso_hac_t": [post_lasso_fit.tvalues.get(_x, np.nan) for _x in HOLDER_FEATURES],
    })
    mo.vstack([
        mo.md(f"**{len(_d):,} stock-days on {_n_days:,} days; {LASSO_BOOT_DRAWS} bootstrap draws of {LASSO_BOOT_BLOCK}-day blocks; lambda {LASSO_LAMBDA:.2e}**"),
        mo.ui.table(lasso_inference.round(4), selection=None),
        mo.plain_text(post_lasso_fit.summary().as_text()),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fama-MacBeth

    Each training day, a cross-sectional OLS of `ret_next_dm` on three features: level, squared change and volatility, all `_zw`. Acceleration and the deviation from the 60-day mean are left out. The daily slopes are then averaged over days. The t-stat is the mean slope over its standard error across days: plain (Fama-MacBeth) and Newey-West with 5 lags (the daily slopes can be autocorrelated). The average daily slope is the daily return of a long-short portfolio with unit exposure to that feature and zero to the others.
    """)
    return


@app.cell
def _(HAC_LAGS, mo, np, panel, pd, sm):
    FM_FEATURES = ["rh_pct_lag1_zw", "rh_pct_sq_lag1_zw", "rh_pct_vol_lag1_zw"]

    _d = panel[["date", "ret_next_dm", *FM_FEATURES]].dropna().sort_values("date")
    _rows = []
    for _day, _g in _d.groupby("date"):
        _X = np.column_stack([np.ones(len(_g)), _g[FM_FEATURES].to_numpy()])
        _b, *_ = np.linalg.lstsq(_X, _g["ret_next_dm"].to_numpy(), rcond=None)
        _resid = _g["ret_next_dm"].to_numpy() - _X @ _b
        _r2 = 1 - (_resid**2).sum() / ((_g["ret_next_dm"] - _g["ret_next_dm"].mean()) ** 2).sum()
        _rows.append({"date": _day, "n": len(_g), "r_squared": _r2, **dict(zip(["const", *FM_FEATURES], _b))})
    fm_slopes = pd.DataFrame(_rows).set_index("date")


    def _fm_row(s):
        _nw = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        return {"mean_slope_bp": s.mean() * 1e4, "fm_t": s.mean() / (s.std() / np.sqrt(len(s))), "nw_t": _nw.tvalues[0], "share_days_negative": (s < 0).mean()}


    fm_results = pd.DataFrame([{"X": _x, **_fm_row(fm_slopes[_x])} for _x in ["const", *FM_FEATURES]])
    mo.vstack([
        mo.md(f"**{len(fm_slopes):,} days, {int(fm_slopes['n'].median()):,} stocks on the median day, average cross-sectional R2 {fm_slopes['r_squared'].mean():.2%}**"),
        mo.ui.table(fm_results.round(4), selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
