import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The model panel

    Builds the panels the ML script reads from the clean TAQ data (`scripts/01_clean_and_explore_TAQ.py`, section 5): one row per stock and decision day, built up in steps: (1) the keys, (2) the targets, (3) family A of the features, (4) family B, (5) family C, (6) family D; then (7) joined, with each feature winsorized at the day's 1st and 99th percentiles and standardized within the day (except the 0/1 flags; `dsr.features.winsorize_standardize`); (8) screened for near-copies; (9) saved to `data/processed/`. The plan is in `explanations/features/features_nn_linear_tree.md`.

    **Input:** `data/intermediate/taq/panel_train.parquet` and `panel_test.parquet` (one row per day and stock in the universe, with what is known before the cutoff and the target `ret_next`), the day folders' session `bars`, and CRSP daily for past returns.

    **Timing:** every feature for day t uses only what is known before 15:50 on day t (12:50 on early closes). CRSP's day-t return is known only after the close; it reaches day t+1's row as `ret_prev`.

    **Train and test, built the same way; only train is looked at.** Every step builds both periods with the same functions (`dsr.panel`), so the test panel is made exactly like the training one. Every output shown is from train; of test, only row counts. Every setting chosen from the data is chosen on the training days and applied unchanged to the test days.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    import dsr.panel as build
    from dsr.features import winsorize_standardize

    return build, mo, np, pd, winsorize_standardize


@app.cell
def _(mo, pd):
    TAQ_DIR = mo.notebook_dir().parent / "data" / "intermediate" / "taq"
    CRSP_DAILY = mo.notebook_dir().parent / "data" / "raw" / "crsp" / "stkdlysecuritydata_from_2013.csv"
    PERIODS = ["train", "test"]
    DAYS = {_p: sorted(_d.name for _d in (TAQ_DIR / _p).iterdir() if _d.name.isdigit()) for _p in PERIODS}
    panels = {_p: pd.read_parquet(TAQ_DIR / f"panel_{_p}.parquet") for _p in PERIODS}
    CODES = pd.read_csv(TAQ_DIR / "codes.csv")
    mo.md(f"**Train:** {len(panels['train']):,} stock-days on {panels['train']['day'].nunique():,} days "
          f"({panels['train']['day'].min():%Y-%m-%d} to {panels['train']['day'].max():%Y-%m-%d}). "
          f"**Test:** {len(panels['test']):,} stock-days on {len(DAYS['test']):,} days.")
    return CRSP_DAILY, DAYS, PERIODS, TAQ_DIR, panels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Keys
    """)
    return


@app.cell
def _(PERIODS, panels):
    panel_keys = {_p: panels[_p][["day", "permno"]] for _p in PERIODS}
    panel_keys["train"].head()
    return (panel_keys,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Targets

    All three are made from `ret_next` within each day and size group: the stocks with previous cap of $1B or more (`large_cap`, the ones traded) among themselves, the others among themselves.

    - **`target_demeaned`** (baseline, regression): `ret_next` winsorized at the group's 1st and 99th percentiles that day, minus the group's average of the winsorized returns. Unclipped, a few news-driven stock-days (up to +650%) would steer a squared-error fit.
    - **`target_rank`** (challenger, regression): each stock's position u = (rank − 0.5) ÷ n (tied returns share their average rank), mapped to a standard normal.
    - **`target_class`** (challenger, three classes): 1 in the top quintile (u ≥ 0.8), −1 in the bottom (u < 0.2), 0 in the middle 60%. The models sort on P(top) − P(bottom).

    `ret_next` and `cap_prev` stay too, to compute the long-short's return and value weights. Rows without `ret_next` stay with no target: dropping them would pick stocks by what happened after the decision.
    """)
    return


@app.cell
def _(PERIODS, build, panels):
    targets = {_p: build.targets(panels[_p]) for _p in PERIODS}
    targets["train"].groupby("large_cap").agg(
        demeaned_mean=("target_demeaned", "mean"), demeaned_sd=("target_demeaned", "std"),
        rank_mean=("target_rank", "mean"), rank_sd=("target_rank", "std"),
        top=("target_class", lambda s: s.eq(1).fillna(False).mean()),
        middle=("target_class", lambda s: s.eq(0).fillna(False).mean()),
        bottom=("target_class", lambda s: s.eq(-1).fillna(False).mean()),
    ).round(4)
    return (targets,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Family A: returns

    The baseline every retail feature has to beat. Day t is the decision day; the cutoff is 15:50 (12:50 on early closes).

    - **Today, in three pieces that compound to `ret_1550`:** `ret_overnight` (yesterday's cutoff to today's open, so it includes yesterday's last 10 minutes and the evening; taken out of `ret_1550`, so splits and dividends are already handled), `ret_open_1450` (today's open to the last trade at or before 14:50; 11:50 on early closes), `ret_last_hour` (from there to the last trade before the cutoff). The open is CRSP's `open_t`; the other prices are the closes of the session bars.
    - **Past days, in windows that don't overlap**, from CRSP daily (which has a row for every listed stock every day, in the universe or not): `ret_prev` (t−1), `ret_2_5` (t−5 to t−2, at least 3 of the 4 days), `ret_6_20` (t−20 to t−6, at least 12 of 15).
    - **Risk:** `vol_20`, the standard deviation of daily log returns over t−20 to t−1 (at least 15 days); `range_today`, log(high ÷ low) of the session up to the cutoff.
    """)
    return


@app.cell
def _(CRSP_DAILY, PERIODS, build, panels, pd):
    _permnos = pd.concat([panels[_p]["permno"] for _p in PERIODS]).unique()
    _crsp = pd.read_csv(CRSP_DAILY, usecols=["permno", "dlycaldt", "dlyret"], engine="pyarrow").rename(
        columns={"dlycaldt": "date", "dlyret": "ret"})
    _crsp = _crsp[_crsp["permno"].isin(_permnos)].assign(date=lambda d: pd.to_datetime(d["date"]))
    crsp_windows = build.crsp_windows(_crsp)  # one table for both periods: every window looks only backwards
    return (crsp_windows,)


@app.cell
def _(DAYS, PERIODS, TAQ_DIR, build, mo, panels, pd):
    session_prices = {}
    for _p in PERIODS:
        _early = panels[_p].groupby("day")["early_close"].first()
        session_prices[_p] = pd.concat([
            build.session_prices(TAQ_DIR / _p / _d / "bars.parquet", bool(_early[pd.Timestamp(_d)])).assign(day=pd.Timestamp(_d))
            for _d in mo.status.progress_bar(DAYS[_p], title=f"{_p}: session bars")
        ], ignore_index=True)
    return (session_prices,)


@app.cell
def _(PERIODS, build, crsp_windows, panels, session_prices):
    family_a = {_p: build.family_a(panels[_p], session_prices[_p], crsp_windows) for _p in PERIODS}
    family_a["train"].describe(percentiles=[0.01, 0.5, 0.99]).T.assign(missing=family_a["train"].isna().mean()).round(4)
    return (family_a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Family B: retail flow

    Everything over one window per stock: from the previous trading day's cutoff to this day's cutoff, all four pieces of the day folder added up (the window sums, shared with family C) (outside the session, almost no retail trades print at sub-penny prices, so the pieces are too thin to keep apart). Retail trades are the bars' off-exchange trades under $100,000 at sub-penny prices, signed by the midpoint rule (main) or BJZZ (benchmark).

    - **Imbalances**, (buys − sells) ÷ (buys + sells): `imb_mid_trades`, `imb_mid_dollars`, `imb_bjzz_trades`, `imb_bjzz_dollars`; missing with fewer than 5 retail trades.
    - **Activity:** `retail_trades`, log(1 + retail trades); `retail_share`, retail dollars ÷ all dollars; `retail_abnormal`, log((1 + retail trades) ÷ (1 + their average over t−20 to t−1)), at least 15 of the 20 days.
    - **Persistence:** `imb_mid_5d`, the average `imb_mid_trades` over t−5 to t−1, at least 4 of the 5 days.
    - **Odd lots:** `odd_lot_share`, off-exchange trades under 100 shares ÷ off-exchange trades.

    The histories come from the day folders, which hold only universe stock-days: a day out of the universe counts as missing. Train and test are stitched together for them, since they look only backwards.
    """)
    return


@app.cell
def _(DAYS, PERIODS, TAQ_DIR, build, mo, pd):
    _rows = []
    for _p in PERIODS:
        for _d in mo.status.progress_bar(DAYS[_p], title=f"{_p}: window sums"):
            _rows.append(build.window_sums(TAQ_DIR / _p / _d / "bars.parquet").assign(day=pd.Timestamp(_d)))
    window_sums = pd.concat(_rows, ignore_index=True)  # both periods: the histories look only backwards
    CALENDAR = pd.DatetimeIndex(pd.to_datetime([_d for _p in PERIODS for _d in DAYS[_p]]))  # every trading day
    return CALENDAR, window_sums


@app.cell
def _(CALENDAR, PERIODS, build, panels, window_sums):
    family_b = {_p: build.family_b(panels[_p], window_sums, CALENDAR) for _p in PERIODS}
    family_b["train"].describe(percentiles=[0.01, 0.5, 0.99]).T.assign(missing=family_b["train"].isna().mean()).round(4)
    return (family_b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Family C: activity and liquidity

    Over the same window as family B, from the previous cutoff to this one:

    - `dollars` and `trades`: log(1 + all dollars traded), log(1 + all trades);
    - `dollars_abnormal`: log((1 + dollars) ÷ (1 + their average over t−20 to t−1)), at least 15 of the 20 days;
    - `spread`: log(1 + `spread_bp`), the average quoted spread over the day's session trades up to the cutoff, in basis points.
    """)
    return


@app.cell
def _(CALENDAR, PERIODS, build, panels, window_sums):
    family_c = {_p: build.family_c(panels[_p], window_sums, CALENDAR) for _p in PERIODS}
    family_c["train"].describe(percentiles=[0.01, 0.5, 0.99]).T.assign(missing=family_c["train"].isna().mean()).round(4)
    return (family_c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Family D: context

    - `log_cap`, log(`cap_prev`); `log_price`, log(|`price_prev`|) (a negative CRSP price is a bid/ask midpoint).
    - 0/1 flags: `is_foreign`, `is_adr`, `is_reit` (the base is common stock), `early_close`. They are not winsorized or standardized: that would turn a rare 1 into 0, and `early_close`, the same for every stock that day, into 0. Trees and networks can use them to tell where the other features work. `large_cap` is already in the panel.
    """)
    return


@app.cell
def _(PERIODS, build, panels):
    family_d = {_p: build.family_d(panels[_p]) for _p in PERIODS}
    family_d["train"].describe().T.round(3)
    return (family_d,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. The panel

    Keys, targets and the features side by side: every feature winsorized and standardized within the day, except the 0/1 flags (`dsr.panel.BINARY_FEATURES`). A missing feature becomes 0, the day's centre.
    """)
    return


@app.cell
def _(
    PERIODS,
    build,
    family_a,
    family_b,
    family_c,
    family_d,
    mo,
    panel_keys,
    panels,
    pd,
    targets,
    winsorize_standardize,
):
    def _features(p):
        _f = pd.concat([family_a[p], family_b[p], family_c[p], family_d[p]], axis=1)
        _flags = list(build.BINARY_FEATURES)
        return pd.concat([winsorize_standardize(_f.drop(columns=_flags), panels[p]["day"]), _f[_flags]], axis=1)


    panel = {_p: pd.concat([panel_keys[_p], targets[_p], _features(_p)], axis=1) for _p in PERIODS}
    mo.md(f"**Train:** {len(panel['train']):,} rows. **Test:** {len(panel['test']):,} rows. "
          f"**Columns:** {', '.join(f'`{_c}`' for _c in panel['train'].columns)}")
    return (panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Near-copies dropped, then screened (train only)

    Of each pair of features that move together on the training days, the more basic one stays and the other is dropped (correlations on the transformed training features):

    | Dropped | Moves with (kept) | Correlation |
    |---|---|---|
    | `trades`, `dollars` | `retail_trades`, `log_cap`, `spread` (themselves dropped below) | VIF 28 and 43 |
    | `spread` | `log_cap` | −0.79 |
    | `log_price` | `log_cap`; `odd_lot_share` | 0.69; 0.53 |
    | `retail_trades` | `log_cap` | 0.65 |
    | `dollars_abnormal` | `retail_abnormal` (the project's retail attention) | 0.68 |
    | `imb_mid_dollars`, `imb_bjzz_dollars` | `imb_mid_trades`, `imb_bjzz_trades` | 0.64 |
    | `range_today` | `vol_20` | 0.58 |

    **The screen**, on the remaining training features (BJZZ left out, since it is swapped in for the midpoint imbalance, not used with it; `early_close` left out, since it is the same for every stock on a day): **VIF** of a feature is 1 ÷ (1 − R²) of a regression of it on all the others, so 1 means it shares nothing with them; it equals the diagonal of the inverse of the correlation matrix. The pairs listed are those with |correlation| ≥ 0.3. The list is decided on train and applied to test unchanged.
    """)
    return


@app.cell
def _(PERIODS, mo, np, panel, pd):
    FEATURES_DROPPED = ["trades", "dollars", "spread", "log_price", "retail_trades", "dollars_abnormal",
                        "imb_mid_dollars", "imb_bjzz_dollars", "range_today"]  # chosen on train, dropped from test unchanged
    _keys = ["day", "permno", "ret_next", "cap_prev", "large_cap", "target_demeaned", "target_rank", "target_class"]
    model_panel = {_p: panel[_p].drop(columns=FEATURES_DROPPED) for _p in PERIODS}
    _use = [_c for _c in model_panel["train"].columns if _c not in _keys and "bjzz" not in _c and _c != "early_close"]
    feature_corr = model_panel["train"][_use].astype("float64").corr()
    _pairs = feature_corr.where(np.triu(np.ones(feature_corr.shape, dtype=bool), k=1)).stack()
    _high = _pairs[_pairs.abs() >= 0.3].sort_values(key=abs, ascending=False)
    feature_vif = pd.Series(np.diag(np.linalg.inv(feature_corr.to_numpy())), index=feature_corr.columns, name="VIF")
    mo.hstack([
        feature_vif.sort_values(ascending=False).round(2).to_frame(),
        _high.round(2).rename("correlation").rename_axis(["feature", "with"]).reset_index(),
    ], justify="start", gap=2)
    return (model_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. The final panel, saved

    Written to `data/processed/model_panel_train.parquet` and `model_panel_test.parquet`; test is written, not looked at. `imb_bjzz_trades` stays in the panel: it is the benchmark the ML script swaps in for `imb_mid_trades`, not a feature it uses alongside it.
    """)
    return


@app.cell
def _(PERIODS, mo, model_panel):
    PROCESSED_DIR = mo.notebook_dir().parent / "data" / "processed"
    PROCESSED_DIR.mkdir(exist_ok=True)
    for _p in PERIODS:
        model_panel[_p].to_parquet(PROCESSED_DIR / f"model_panel_{_p}.parquet", index=False, compression="zstd")
    mo.md(f"**Written:** {', '.join(f'`model_panel_{_p}.parquet` ({len(model_panel[_p]):,} rows)' for _p in PERIODS)}; "
          f"{model_panel['train'].shape[1]} columns: {', '.join(f'`{_c}`' for _c in model_panel['train'].columns)}")
    return


if __name__ == "__main__":
    app.run()
