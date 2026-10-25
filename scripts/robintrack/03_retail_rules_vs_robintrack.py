import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Do the retail-signing rules track Robinhood?

    A quick check: is the retail flow that the midpoint rule and BJZZ pick out of TAQ correlated with what Robinhood customers did, as Robintrack counted it? Both rules' level, vol and sqdev are built exactly as 01 builds Robintrack's, and compared stock by stock within each day.

    **A nowcast:** everything ends on the same day d.

    - **Robintrack:** g(d), the % change in holders into day d's count. The count is the first reading retrieved 15:40 to 15:50 ET and shows holdings at about 15:00 to 15:25. These are the `*_lag1` columns of the return panel rows whose previous day is d, built in 01 from each stock's full daily history.
    - **TAQ:** each rule's retail imbalance in trades, x(d) = (buys − sells) ÷ (buys + sells), over the window from 15:50 on the previous trading day to 15:50 on d. It is missing with fewer than 5 signed retail trades, as in the model panel.

    Both are known by the 15:50 cutoff on day d.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from dsr.evaluation import daily_rank_ic, ic_summary
    from dsr.features import winsorize_standardize
    from dsr.panel import MIN_RETAIL_TRADES

    return MIN_RETAIL_TRADES, daily_rank_ic, ic_summary, mo, np, pd, pq, winsorize_standardize


@app.cell
def _(mo, pd):
    ROOT = mo.notebook_dir().parents[1]
    CLEAN = ROOT / "data" / "intermediate" / "clean"
    TAQ_DAYS = ROOT / "data" / "intermediate" / "taq" / "train"  # 2014-2023: covers all of Robintrack
    FIRST_TAQ_DAY = pd.Timestamp("2017-12-01")  # 60+ trading days of history before Robintrack's first change
    LAST_TAQ_DAY = pd.Timestamp("2020-08-14")
    UNIVERSE_TYPES = ["common stock", "foreign common stock", "ADR"]  # as in 02
    RULES = ("mid", "bjzz")
    FEATURES = {"level": "rh_pct_lag1", "vol": "rh_pct_vol60_lag1", "sqdev": "rh_pct_sqdev20_lag1"}  # 02's three
    return CLEAN, FEATURES, FIRST_TAQ_DAY, LAST_TAQ_DAY, RULES, TAQ_DAYS, UNIVERSE_TYPES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The rules' level, vol and sqdev

    From each TAQ day folder (universe stock-days only, so a day out of the universe counts as missing), each rule's imbalance x(d), then, on each stock's daily series over every trading day, as 01 builds Robintrack's:

    | Feature | Definition |
    |---|---|
    | level | x(d) |
    | vol | standard deviation of x over the 60 trading days to d (at least 30) |
    | sqdev | (x(d) − the mean of x over the 20 trading days before d)², at least 10 of them |
    """)
    return


@app.cell
def _(FIRST_TAQ_DAY, LAST_TAQ_DAY, MIN_RETAIL_TRADES, RULES, TAQ_DAYS, pd, pq):
    def _imbalances(day_dir):
        """One day's retail imbalance in trades by each rule, per permno; missing with fewer than 5 signed trades."""
        columns = [f"{r}_{s}_trades" for r in RULES for s in ("buy", "sell")]
        sums = pq.read_table(day_dir / "bars.parquet", columns=["permno", *columns]).to_pandas()
        sums = sums.groupby("permno")[columns].sum()
        out = {}
        for r in RULES:
            buy, sell = sums[f"{r}_buy_trades"], sums[f"{r}_sell_trades"]
            out[r] = ((buy - sell) / (buy + sell)).where(buy + sell >= MIN_RETAIL_TRADES)
        return pd.DataFrame(out)


    def _features(wide):
        """level, vol and sqdev of a days x permnos frame of imbalances, as 01 builds Robintrack's (ending on each day)."""
        return {"level": wide,
                "vol": wide.rolling(60, min_periods=30).std(),
                "sqdev": (wide - wide.rolling(20, min_periods=10).mean().shift(1)) ** 2}


    _dirs = sorted(p for p in TAQ_DAYS.iterdir() if FIRST_TAQ_DAY <= pd.Timestamp(p.name) <= LAST_TAQ_DAY)
    _long = pd.concat({pd.Timestamp(p.name): _imbalances(p) for p in _dirs}, names=["day", "permno"])
    _calendar = pd.DatetimeIndex([pd.Timestamp(p.name) for p in _dirs])  # every trading day has a folder
    _parts = {}
    for _r in RULES:
        _wide = _long[_r].unstack("permno").reindex(_calendar)
        for _name, _f in _features(_wide).items():
            _parts[f"{_r}_{_name}"] = _f.stack()
    taq = pd.DataFrame(_parts).rename_axis(["day", "permno"]).reset_index()
    return (taq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The sample

    Robintrack's 15:50 universe (price at least $1, 100+ holders) as 02 restricts it (US common stock, foreign common stock and ADRs), matched to the TAQ universe by PERMNO and day. The row of day d+1 carries Robintrack's features ending on d, so its universe is set by day d's count, the one g(d) ends on. Train runs to 2019-12-11 and test from 2019-12-12 (Robintrack's split, at fractional shares); they are reported apart, and nothing is chosen here.

    Only stock-days with all three Robintrack features and all six of the rules' are kept, so every comparison uses the same rows.
    """)
    return


@app.cell
def _(CLEAN, FEATURES, RULES, UNIVERSE_TYPES, mo, pd, taq):
    _cols = ["permno", "prev_day", "security_type", "cap_prev", *FEATURES.values()]
    _rt = pd.concat({_p: pd.read_parquet(CLEAN / f"returns_{_p}.parquet", columns=_cols) for _p in ("train", "test")},
                    names=["period"]).reset_index("period")
    _rt = _rt[_rt["security_type"].isin(UNIVERSE_TYPES)]
    _rt = _rt.rename(columns={"prev_day": "day", **{c: f"rt_{n}" for n, c in FEATURES.items()}})
    _rt["day"] = pd.to_datetime(_rt["day"])
    _needed = [f"rt_{n}" for n in FEATURES] + [f"{r}_{n}" for r in RULES for n in FEATURES] + ["cap_prev"]
    sample = _rt.merge(taq, on=["day", "permno"], how="inner", validate="one_to_one").dropna(subset=_needed)
    sample = sample.sort_values(["period", "day", "permno"], ignore_index=True)

    _counts = sample.groupby("period", sort=False).agg(stock_days=("permno", "size"), stocks=("permno", "nunique"),
                                                       days=("day", "nunique"), first=("day", "min"), last=("day", "max"))
    mo.vstack([mo.md(f"Robintrack universe rows: {len(_rt):,}; matched to TAQ with every feature: {len(sample):,}."),
               mo.ui.table(_counts.reset_index(), selection=None)])
    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How closely they move

    For each Robintrack feature and each rule, by day, then averaged over the days (Newey-West t, 5 lags):

    - **Rank correlation:** the Spearman correlation between Robintrack's feature and the rule's same feature.
    - **Without size:** the same after removing market cap (the ranks of both on the rank of the previous day's cap, the residuals correlated). Few holders make holder changes jumpy and few retail trades make imbalances jumpy, so vol and sqdev could agree only because both mark small stocks.
    - **R²:** Robintrack's feature regressed on the rule's three features, each winsorized at the day's 1st and 99th percentiles and standardized within the day.
    - **Midpoint − BJZZ:** the daily difference, its mean and t.
    """)
    return


@app.cell
def _(np, pd):
    def partial_rank_corr(day, x, y, z):
        """Each day's correlation between the ranks of x and y after regressing both on the ranks of z."""
        df = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(z)})
        d = np.asarray(day)
        c = df.groupby(d).rank(pct=True)
        c = c - c.groupby(d).transform("mean")
        zz = (c["z"] ** 2).groupby(d).transform("sum")
        e = {k: c[k] - (c[k] * c["z"]).groupby(d).transform("sum") / zz * c["z"] for k in ("x", "y")}
        num = (e["x"] * e["y"]).groupby(d).sum()
        return num / np.sqrt((e["x"] ** 2).groupby(d).sum() * (e["y"] ** 2).groupby(d).sum())


    def daily_r2(day, y, X):
        """Each day's R² of y on the columns of X (with an intercept)."""
        out = {}
        for d, idx in pd.Series(np.arange(len(y))).groupby(np.asarray(day)).groups.items():
            A = np.column_stack([np.ones(len(idx)), X[idx]])
            coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
            resid = y[idx] - A @ coef
            out[d] = 1 - (resid ** 2).sum() / ((y[idx] - y[idx].mean()) ** 2).sum()
        return pd.Series(out)

    return daily_r2, partial_rank_corr


@app.cell
def _(FEATURES, RULES, daily_r2, daily_rank_ic, ic_summary, mo, partial_rank_corr, pd, sample,
      winsorize_standardize):
    def _mean_t(s):
        """A daily series' mean and Newey-West t, as 'mean (t)'."""
        found = ic_summary(s)
        return f"{found['ic_mean']:.3f} ({found['ic_t']:.1f})"


    _rows = []
    for _period, _s in sample.groupby("period", sort=False):
        _day = _s["day"]
        _z = winsorize_standardize(_s[[c for c in _s.columns if c.startswith(("rt_", *[f"{r}_" for r in RULES]))]],
                                   _day)
        for _name in FEATURES:
            _y = _s[f"rt_{_name}"]
            _corr = {r: daily_rank_ic(_day, _s[f"{r}_{_name}"], _y) for r in RULES}
            _part = {r: partial_rank_corr(_day, _s[f"{r}_{_name}"], _y, _s["cap_prev"]) for r in RULES}
            _r2 = {r: daily_r2(_day.to_numpy(), _z[f"rt_{_name}"].to_numpy(),
                               _z[[f"{r}_{n}" for n in FEATURES]].to_numpy()) for r in RULES}
            _rows.append({
                "period": _period, "Robintrack feature": _name,
                "rank corr: midpoint": _mean_t(_corr["mid"]), "rank corr: BJZZ": _mean_t(_corr["bjzz"]),
                "rank corr: midpoint − BJZZ": _mean_t(_corr["mid"] - _corr["bjzz"]),
                "without size: midpoint": _mean_t(_part["mid"]), "without size: BJZZ": _mean_t(_part["bjzz"]),
                "without size: midpoint − BJZZ": _mean_t(_part["mid"] - _part["bjzz"]),
                "R²: midpoint": _mean_t(_r2["mid"]), "R²: BJZZ": _mean_t(_r2["bjzz"]),
                "R²: midpoint − BJZZ": _mean_t(_r2["mid"] - _r2["bjzz"]),
            })
    results = pd.DataFrame(_rows)
    mo.ui.table(results, selection=None)
    return (results,)


if __name__ == "__main__":
    app.run()
