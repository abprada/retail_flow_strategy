import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Robintrack and returns

    Does the change in Robinhood holders, or features of it, predict next-day returns? Only the training return panel is read: `data/intermediate/clean/returns_train.parquet`, one row per stock and day in the universe known at 15:50 (a price of at least $1 and 100+ holders at the previous close), with return labels ending before the 2019-12-12 split. The test file stays closed until every choice made here is frozen; it is opened once, in the backtest at the end.

    Timing: the `*_prev` columns, `holders_close_prev` and the `rh_pct_*_lag1` features are known before the 15:50 cutoff on day t; `ret_next` is the close-to-close return from day t to the next trading day.
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

    return Lasso, lasso_path, mo, np, pd, plt, sm


@app.cell
def _(mo, pd):
    PANEL_TRAIN_PATH = mo.notebook_dir().parent / "data" / "intermediate" / "clean" / "returns_train.parquet"
    panel_raw = pd.read_parquet(PANEL_TRAIN_PATH)
    return PANEL_TRAIN_PATH, panel_raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Features

    Three features of the daily % change in holders $g$, built in 01 from each stock's full daily history and ending at day t-1, so known by 15:50 on day t:

    | Feature | Column | Definition |
    |---|---|---|
    | level | `rh_pct_lag1` | $g_{t-1}$ |
    | vol | `rh_pct_vol60_lag1` | standard deviation of $g$ over the last 60 trading days (at least 30 changes) |
    | sqdev | `rh_pct_sqdev20_lag1` | $(g_{t-1} - \bar g)^2$, with $\bar g$ the mean of $g$ over the 20 trading days before (at least 10 changes): the size of the last change as a surprise against the stock's own drift |

    Vol and sqdev are also built over rolling windows of 5, 10, 20, 40 and 60 trading days (at least half the window); the windows above, 60 days for vol and 20 for sqdev, are the ones the 10-fold cross-validation in the lasso section chose on train. The choice is weak: every window pair scored within one standard error of the best.

    Four more, acceleration, the deviation from the 60-day mean, the 20-day mean and the plain squared change, were dropped; see the graveyard at the bottom.
    """)
    return


@app.cell
def _():
    FEATURES = {
        "level": "rh_pct_lag1",
        "vol": "rh_pct_vol60_lag1",  # windows chosen by the 10-fold cross-validation in the lasso section
        "sqdev": "rh_pct_sqdev20_lag1",
    }
    X = {_name: f"{_col}_zw" for _name, _col in FEATURES.items()}  # the standardized column each regression uses
    # candidate rolling windows for vol and sqdev (built in 01), chosen by cross-validation in the lasso section
    WINDOW_GRID = [5, 10, 20, 40, 60]
    WINDOW_COLUMNS = [f"rh_pct_{_k}{_n}_lag1" for _k in ("vol", "sqdev") for _n in WINDOW_GRID]
    return FEATURES, WINDOW_COLUMNS, WINDOW_GRID, X


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
def _(FEATURES, WINDOW_COLUMNS):
    WINSOR_QUANTILES = (0.01, 0.99)


    def standardize(df):
        """Winsorize and z-score each feature, and demean the next-day return, within each day of df."""
        df = df.copy()
        day = df["date"]
        for col in [*FEATURES.values(), *WINDOW_COLUMNS]:
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
    return UNIVERSE_TYPES, panel


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


    def ols_hac(cols, data=None):
        """Pooled OLS of ret_next_dm on cols, with Driscoll-Kraay HAC errors over days; on the training panel unless data is given."""
        d = (panel if data is None else data)[["date", "ret_next_dm", *cols]].dropna().sort_values("date")
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

    $r^{dm}_{t \to t+1} = \alpha + \beta_1\, \text{level}_{t-1} + \beta_2\, \text{sqdev}_{t-1} + \beta_3\, \text{vol}_{t-1} + \varepsilon$ on the rows where all three exist, with the same errors.
    """)
    return


@app.cell
def _(X, mo, ols_hac, pd):
    JOINT = ["level", "sqdev", "vol"]

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
    def fama_macbeth(cols, data=None):
        """Daily cross-sectional OLS of ret_next_dm on cols; returns the daily slopes and their averages with t-stats.

        On the training panel unless data is given."""
        d = (panel if data is None else data)[["date", "ret_next_dm", *cols]].dropna().sort_values("date")
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

    Each training day, a cross-sectional OLS of `ret_next_dm` on level, squared deviation and volatility together (the same three as the pooled OLS); the slopes are averaged over days as above. Each average slope is the daily return of a long-short portfolio with unit exposure to that feature and zero to the other two.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lasso, with 10-fold cross-validation

    $\min_{\alpha, \beta}\ \frac{1}{2n}\sum (r^{dm}_{t \to t+1} - \alpha - X_{t-1}\beta)^2 + \lambda \lVert \beta \rVert_1$ on level, vol and sqdev. Three hyperparameters are chosen together by cross-validation on train: $\lambda$, the rolling window of vol and the rolling window of sqdev (each 5, 10, 20, 40 or 60 trading days).

    - **Folds:** the training days are cut into 10 consecutive blocks. Each block is predicted by a fit on all other days, leaving out 5 days on each side of the block (the HAC lag length), so neighbouring days' correlated returns do not leak into the fit. Folds are by day, never by row, because stocks on the same day share their shocks.
    - **Same rows for every candidate:** rows where all window choices exist, so the 25 window pairs are compared on one sample.
    - **Score:** the out-of-sample $R^2$ in each validation block against predicting the training mean, averaged over the 10 folds. The chosen triple has the highest average; the final lasso refits it on all those rows, so its betas are still in sample.
    """)
    return


@app.cell
def _(
    HAC_LAGS,
    Lasso,
    WINDOW_COLUMNS,
    WINDOW_GRID,
    lasso_path,
    mo,
    np,
    panel,
    pd,
    plt,
):
    CV_FOLDS, CV_GAP_DAYS = 10, HAC_LAGS

    # rows where every candidate window exists
    _cols = ["rh_pct_lag1_zw", *[f"{_c}_zw" for _c in WINDOW_COLUMNS]]
    _d = panel[["date", "ret_next_dm", *_cols]].dropna().sort_values("date").reset_index(drop=True)
    _y = _d["ret_next_dm"].to_numpy()
    _days = np.sort(_d["date"].unique())
    _day_no = np.searchsorted(_days, _d["date"].to_numpy())

    # 10 consecutive blocks of days; the training rows leave out CV_GAP_DAYS on each side of the block
    _folds = []
    for _block in np.array_split(np.arange(len(_days)), CV_FOLDS):
        _va = np.isin(_day_no, _block)
        _tr = (_day_no < _block[0] - CV_GAP_DAYS) | (_day_no > _block[-1] + CV_GAP_DAYS)
        _folds.append((_tr, _va))

    # one lambda grid for every window pair: from the largest lambda that keeps any feature down to 1/1000 of it
    _Xall = _d[_cols].to_numpy()
    _lambda_max = np.abs((_Xall - _Xall.mean(0)).T @ (_y - _y.mean())).max() / len(_y)
    LASSO_LAMBDAS = np.geomspace(_lambda_max, _lambda_max * 1e-3, 40)

    _rows = []
    for _nv in WINDOW_GRID:
        for _ns in WINDOW_GRID:
            _X = _d[["rh_pct_lag1_zw", f"rh_pct_vol{_nv}_lag1_zw", f"rh_pct_sqdev{_ns}_lag1_zw"]].to_numpy()
            _r2 = []
            for _tr, _va in _folds:
                _xm, _ym = _X[_tr].mean(0), _y[_tr].mean()
                _, _coefs, _ = lasso_path(_X[_tr] - _xm, _y[_tr] - _ym, alphas=LASSO_LAMBDAS)
                _pred = _ym + (_X[_va] - _xm) @ _coefs  # one column per lambda
                _r2.append(1 - ((_y[_va, None] - _pred) ** 2).sum(0) / ((_y[_va] - _ym) ** 2).sum())
            _r2 = np.array(_r2)
            for _i, _lam in enumerate(LASSO_LAMBDAS):
                _rows.append({"n_vol": _nv, "n_sqdev": _ns, "lambda": _lam, "cv_r2_bp": _r2[:, _i].mean() * 1e4,
                              "cv_r2_se_bp": _r2[:, _i].std(ddof=1) / np.sqrt(CV_FOLDS) * 1e4})
    lasso_cv = pd.DataFrame(_rows)
    lasso_best = lasso_cv.loc[lasso_cv["cv_r2_bp"].idxmax()]

    # refit the chosen triple on all the rows
    LASSO_COLUMNS = ["rh_pct_lag1_zw", f"rh_pct_vol{int(lasso_best['n_vol'])}_lag1_zw", f"rh_pct_sqdev{int(lasso_best['n_sqdev'])}_lag1_zw"]
    lasso_fit = Lasso(alpha=lasso_best["lambda"]).fit(_d[LASSO_COLUMNS].to_numpy(), _y)
    lasso_coefs = pd.DataFrame({"feature": ["level", "vol", "sqdev"], "X": LASSO_COLUMNS, "beta_bp_per_sd": lasso_fit.coef_ * 1e4})

    # best CV R2 over lambda for each window pair, and the lambda curve of the chosen pair
    _by_pair = lasso_cv.groupby(["n_vol", "n_sqdev"])["cv_r2_bp"].max().unstack().round(3)
    _by_pair.columns = [f"sqdev {_n} days" for _n in _by_pair.columns]
    _by_pair.index = [f"vol {_n} days" for _n in _by_pair.index]
    _curve = lasso_cv[(lasso_cv["n_vol"] == lasso_best["n_vol"]) & (lasso_cv["n_sqdev"] == lasso_best["n_sqdev"])]
    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.errorbar(_curve["lambda"], _curve["cv_r2_bp"], yerr=_curve["cv_r2_se_bp"], fmt="o-", ms=3)
    _ax.axhline(0, color="grey", lw=0.8)
    _ax.axvline(lasso_best["lambda"], color="k", ls="--", lw=0.8)
    _ax.set(xscale="log", xlabel="lambda", ylabel="CV out-of-sample R2 (bp)",
            title=f"vol {int(lasso_best['n_vol'])} days, sqdev {int(lasso_best['n_sqdev'])} days: 10-fold CV")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"**{len(_d):,} stock-days on {len(_days):,} days.** Chosen: vol over {int(lasso_best['n_vol'])} days, sqdev over "
              f"{int(lasso_best['n_sqdev'])} days, lambda {lasso_best['lambda']:.2e}; CV out-of-sample R2 {lasso_best['cv_r2_bp']:.2f} bp "
              f"(standard error {lasso_best['cv_r2_se_bp']:.2f} bp across folds)."),
        mo.md("**Lasso betas on all rows** (bp per standard deviation)"),
        mo.ui.table(lasso_coefs.round(4), selection=None),
        mo.md("**Best CV R2 (bp) for each window pair** (rows: vol window, columns: sqdev window)"),
        mo.ui.table(_by_pair.reset_index(names="window"), selection=None),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backtest: ridge refitted daily, through the test period

    **This is the one use of the test file.** Everything above (the three features, their windows, the universe, the winsorization) was fixed on train before it was opened; changing any of it after seeing these results would make the test period in-sample. The features come straight from Robintrack, which will not exist live, so this is the upper bound from the plan's check ("does the target predict returns at all?"), not a tradable Model 2.

    - **Data:** the training and test return panels, the same universe, X and y standardized within each day as above.
    - **Daily refit, expanding window:** trading starts on the first day with at least 120 training days (about six months). From then on, each day d, a ridge regression of `ret_next_dm` on level, vol and sqdev is fitted on every row whose label had ended by 15:50 on day d (the label of day t ends at the close of t+1, so the last training day is d-2), then predicts day d's stocks.
    - **$\lambda$ each day by 5-fold cross-validation:** the training days are cut into 5 consecutive blocks; each is predicted by a fit on the other days, leaving out 5 days on each side; the $\lambda$ with the lowest total squared error wins ($\lambda = 0$ is plain OLS). The penalty is $\lambda n \lVert \beta \rVert^2$ on the slopes only, not the intercept.
    - **Strategy:** each day, long the top fifth of predicted returns and short the bottom fifth, equal-weighted, dollar-neutral, traded close to close with market-on-close orders. Returns are gross: no costs, no borrow fees.
    - **Reported separately:** the traded days inside train (pseudo out of sample: the features were chosen on them) and the test period (2019-12-12 to 2020-08-12, out of sample). The day 2019-12-11 is missing: its label ends on the first test day, so it is in neither file.
    """)
    return


@app.cell
def _(
    HAC_LAGS,
    PANEL_TRAIN_PATH,
    UNIVERSE_TYPES,
    X,
    mo,
    np,
    panel,
    pd,
    standardize,
):
    BT_INPUTS = [X["level"], X["vol"], X["sqdev"]]
    BT_MIN_TRAIN_DAYS = 120  # trading starts once the expanding window holds about six months of training days
    BT_FOLDS, BT_GAP_DAYS = 5, HAC_LAGS
    BT_LAMBDAS = np.r_[0.0, np.geomspace(1e-4, 1e2, 25)]
    PANEL_TEST_PATH = PANEL_TRAIN_PATH.with_name("returns_test.parquet")

    _test_raw = pd.read_parquet(PANEL_TEST_PATH)
    TEST_START = _test_raw["date"].min()
    bt_panel = pd.concat(
        [panel, standardize(_test_raw[_test_raw["security_type"].isin(UNIVERSE_TYPES)])], ignore_index=True
    )
    _d = bt_panel[["date", "next_day", "permno", "ret_next", "ret_next_dm", *BT_INPUTS]].dropna().sort_values(["date", "permno"]).reset_index(drop=True)

    # per-day sufficient statistics of [1, x] and y, so each fit is a 4 x 4 solve
    _Z = np.column_stack([np.ones(len(_d)), _d[BT_INPUTS].to_numpy()])
    _y = _d["ret_next_dm"].to_numpy()
    _days, _first = np.unique(_d["date"].to_numpy(), return_index=True)
    _next = _d["next_day"].to_numpy()[_first]
    _cum = lambda a: np.concatenate([np.zeros((1, *a.shape[1:])), np.cumsum(a, axis=0)])
    _ZZ = _cum(np.add.reduceat(_Z[:, :, None] * _Z[:, None, :], _first, axis=0))
    _Zy = _cum(np.add.reduceat(_Z * _y[:, None], _first, axis=0))
    _yy = _cum(np.add.reduceat(_y**2, _first))
    _n = _cum(np.diff(np.r_[_first, len(_y)]).astype(float))
    _P = np.diag([0.0, 1.0, 1.0, 1.0])  # no penalty on the intercept


    def _stats(ranges):
        """Sums of the statistics over day ranges [a, b)."""
        return [sum(s[b] - s[a] for a, b in ranges if b > a) for s in (_ZZ, _Zy, _yy, _n)]


    def _ridge(zz, zy, n, lam):
        return np.linalg.solve(zz + lam * n * _P, zy)


    _pred = np.full(len(_y), np.nan)
    _log = []
    for _j in range(len(_days)):
        _m = int(np.searchsorted(_next, _days[_j], side="left"))  # days 0.._m-1 have labels ending before day j
        if _m < BT_MIN_TRAIN_DAYS:
            continue
        _sse = np.zeros(len(BT_LAMBDAS))
        for _block in np.array_split(np.arange(_m), BT_FOLDS):
            _a, _b = _block[0], _block[-1] + 1
            _zz, _zy, _, _nn = _stats([(0, max(_a - BT_GAP_DAYS, 0)), (min(_b + BT_GAP_DAYS, _m), _m)])
            _vzz, _vzy, _vyy, _ = _stats([(_a, _b)])
            for _k, _lam in enumerate(BT_LAMBDAS):
                _beta = _ridge(_zz, _zy, _nn, _lam)
                _sse[_k] += _vyy - 2 * _beta @ _vzy + _beta @ _vzz @ _beta
        _lam = BT_LAMBDAS[_sse.argmin()]
        _zz, _zy, _, _nn = _stats([(0, _m)])
        _beta = _ridge(_zz, _zy, _nn, _lam)
        _rows = slice(_first[_j], _first[_j + 1] if _j + 1 < len(_first) else len(_y))
        _pred[_rows] = _Z[_rows] @ _beta
        _log.append({"date": pd.Timestamp(_days[_j]), "train_days": _m, "lambda": _lam,
                     **{f"beta_{_name}_bp": _v * 1e4 for _name, _v in zip(["const", "level", "vol", "sqdev"], _beta)}})

    bt_fits = pd.DataFrame(_log).set_index("date")
    bt_predictions = _d.assign(pred=_pred).dropna(subset=["pred"])[["date", "permno", "ret_next", "ret_next_dm", "pred"]]
    mo.md(f"**{len(bt_fits):,} daily refits from {bt_fits.index.min():%Y-%m-%d} to {bt_fits.index.max():%Y-%m-%d}; "
          f"{len(bt_predictions):,} predicted stock-days.** Chosen lambda: 0 (plain OLS) on {(bt_fits['lambda'] == 0).mean():.0%} of days, "
          f"median {bt_fits['lambda'].median():.1e}.")
    return BT_INPUTS, TEST_START, bt_fits, bt_panel, bt_predictions


@app.cell
def _(HAC_LAGS, TEST_START, bt_fits, bt_predictions, mo, np, pd, plt, sm):
    # the daily long-short portfolio: top fifth of predictions long, bottom fifth short, equal-weighted
    _p = bt_predictions.copy()
    _p["quintile"] = _p.groupby("date")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    _p["weight"] = 0.0
    for _q, _sign in ((4, 1.0), (0, -1.0)):
        _in = _p["quintile"] == _q
        _p.loc[_in, "weight"] = _sign / _p[_in].groupby("date")["permno"].transform("size")
    _daily = pd.DataFrame({
        "long_short": _p.groupby("date").apply(lambda g: (g["weight"] * g["ret_next"]).sum()),
        "market": _p.groupby("date")["ret_next"].mean(),  # the universe, equal-weighted
        "ic": _p.groupby("date").apply(lambda g: g["pred"].corr(g["ret_next_dm"], method="spearman")),
    })
    # turnover: the sum of absolute weight changes from the previous day (gross exposure is 2)
    _w = _p.pivot_table(index="date", columns="permno", values="weight", fill_value=0.0)
    _daily["turnover"] = _w.diff().abs().sum(axis=1)
    _daily.iloc[0, _daily.columns.get_loc("turnover")] = np.nan
    bt_daily = _daily


    def _nw_t(s):
        s = s.dropna()
        return sm.OLS(s.to_numpy(), np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}).tvalues[0]


    def _metrics(g):
        r = g["long_short"]
        cum = r.cumsum()
        capm = sm.OLS(r.to_numpy(), sm.add_constant(g["market"].to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        return pd.Series({
            "days": len(r),
            "mean_bp_per_day": r.mean() * 1e4,
            "nw_t": _nw_t(r),
            "annual_return_pct": r.mean() * 252 * 100,
            "annual_vol_pct": r.std() * np.sqrt(252) * 100,
            "sharpe": r.mean() / r.std() * np.sqrt(252),
            "hit_rate": (r > 0).mean(),
            "max_drawdown_pct": (cum - cum.cummax()).min() * 100,
            "worst_day_pct": r.min() * 100,
            "mean_daily_turnover": g["turnover"].mean(),
            "breakeven_cost_bp": r.mean() / g["turnover"].mean() * 1e4,
            "market_beta": capm.params[1],
            "alpha_bp_per_day": capm.params[0] * 1e4,
            "alpha_nw_t": capm.tvalues[0],
            "mean_ic": g["ic"].mean(),
            "ic_nw_t": _nw_t(g["ic"]),
        })


    _period = np.where(bt_daily.index < TEST_START, "inside train (pseudo out of sample)", "test (out of sample)")
    bt_metrics = pd.concat([bt_daily.groupby(_period).apply(_metrics).T, _metrics(bt_daily).rename("all")], axis=1)

    _fig, (_a1, _a2, _a3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    _a1.plot(bt_daily.index, bt_daily["long_short"].cumsum() * 100)
    _a1.set(ylabel="cumulative %", title="Long-short, top minus bottom fifth (gross, not compounded)")
    for _name in ("level", "vol", "sqdev"):
        _a2.plot(bt_fits.index, bt_fits[f"beta_{_name}_bp"], label=_name)
    _a2.set(ylabel="bp per sd", title="Daily refitted ridge betas")
    _a2.legend()
    _a3.plot(bt_fits.index, bt_fits["lambda"])
    _a3.set_yscale("symlog", linthresh=1e-4)
    _a3.set(ylabel="lambda", title="Lambda chosen by 5-fold CV")
    for _ax in (_a1, _a2, _a3):
        _ax.axvline(TEST_START, color="k", ls="--", lw=0.8)
        _ax.axhline(0, color="grey", lw=0.5)
    _fig.tight_layout()
    mo.vstack([
        mo.md("**Strategy metrics** (gross; the dashed line in the plots is the start of the test period). `breakeven_cost_bp` is the cost per unit of turnover that would take the mean return to zero."),
        mo.ui.table(bt_metrics.round(4).reset_index(names="metric"), selection=None, page_size=20),
        _fig,
    ])
    return


@app.cell
def _(BT_INPUTS, TEST_START, bt_panel, fama_macbeth, mo, ols_hac, pd):
    # significance of the betas on the test period only, with the methods above
    _test = bt_panel[bt_panel["date"] >= TEST_START]
    bt_test_ols = ols_hac(BT_INPUTS, data=_test)
    bt_test_fm_slopes, bt_test_fm = fama_macbeth(BT_INPUTS, data=_test)
    _names = ["level", "vol", "sqdev"]
    bt_test_betas = pd.DataFrame({
        "feature": _names,
        "ols_beta_bp": [bt_test_ols.params[_c] * 1e4 for _c in BT_INPUTS],
        "ols_hac_t": [bt_test_ols.tvalues[_c] for _c in BT_INPUTS],
        "fm_mean_slope_bp": bt_test_fm.set_index("X").loc[BT_INPUTS, "mean_slope_bp"].to_numpy(),
        "fm_nw_t": bt_test_fm.set_index("X").loc[BT_INPUTS, "nw_t"].to_numpy(),
    })
    mo.vstack([
        mo.md(f"**Betas on the test period only** ({int(bt_test_ols.nobs):,} stock-days on {len(bt_test_fm_slopes):,} days; together, bp per standard deviation)"),
        mo.ui.table(bt_test_betas.round(4), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Graveyard

    Features tried and dropped, with their train results at the time, in the same universe and units as above (bp per standard deviation).

    - **accel**, `rh_pct_accel_lag1` $= g_{t-1} - g_{t-2}$: OLS alone $-0.69$ (HAC t $-1.07$); Fama-MacBeth alone $-0.89$ (Newey-West t $-1.35$). Correlation 0.58 with the level. It only looked useful next to the level, in a lasso on all five features (kept in 82% of bootstrap draws, +0.58, post-lasso t 1.36): the positive sign just says the change two days ago also predicts negatively. Dropped.
    - **dev**, `rh_pct_dev_lag1` $= g_{t-1}$ minus its mean over the 60 trading days before: OLS alone $-0.74$ (HAC t $-1.19$); Fama-MacBeth alone $-0.82$ (Newey-West t $-1.34$). Correlation 0.82 with the level, so it is mostly the level again; the lasso set it to 0. Dropped.
    - **mean**, `rh_pct_mean_lag1`, the mean of $g$ over the 20 trading days ending at t-1 (same window as the volatility): OLS alone $-2.51$ (HAC t $-3.47$); Fama-MacBeth alone $-2.48$ (Newey-West t $-3.42$). But its correlation with the volatility is 0.86 after winsorizing and z-scoring: the holder change is right-skewed, so big inflows raise the mean and the spread together. Next to level, sq and vol it turns positive, $+1.52$ (HAC t $1.27$) in pooled OLS and $+1.43$ (Newey-West t $1.13$) in Fama-MacBeth, while the volatility gets stronger. It adds nothing beyond the volatility. Dropped.
    - **sq**, `rh_pct_sq_lag1` $= g_{t-1}^2$: OLS alone $-3.84$ (HAC t $-4.32$); Fama-MacBeth alone $-3.78$ (Newey-West t $-4.35$). With level and vol: $-1.42$ (HAC t $-1.51$) in pooled OLS, $-1.66$ (Newey-West t $-1.69$) in Fama-MacBeth. Replaced by sqdev, the squared deviation from the 20-day mean: correlation 0.75, and head to head on the same rows sq falls to $-1.12$ (HAC t $-0.93$) while sqdev keeps $-2.57$ (HAC t $-3.04$). The size of a change measured against the stock's own drift is the cleaner surprise. Dropped.

    All four are still built in `01_clean_explore_robintrack.py` and written to the return panel.
    """)
    return


if __name__ == "__main__":
    app.run()
