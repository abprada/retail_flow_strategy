import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backtest: long-short portfolios on each model's predictions

    Trades each model's walk-forward predictions over 2019-2023 on the ≥ $1B stocks: the tabular models' monthly refits (`scripts/models/04_walkforward.py`) and the CNN on one-day chart images, refit yearly (`cnn_one_day`, `scripts/models/06_cnn_walkforward.py`). Each day, the portfolio buys the top quintile (or decile) of the stocks by prediction and sells the bottom one short, equal-weighted with $1 a side, at the close, and holds them to the next close. The plan is in `explanations/models/models.md`.

    - **Before costs** (user decision): no spreads, commissions or borrow fees. Turnover is shown: it is what a cost per dollar traded would multiply.
    - **Hit rates are counted per trade:** a stock's stay in one leg, from the close it enters to the close it leaves.

    **The test years (2024-2025) are not read.** The models' settings were chosen on these same years, so the returns below are not out of sample for the settings.
    """)
    return


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    from dsr.backtest import daily, positions, summary, trades, universe, yearly

    return alt, daily, mo, pd, positions, summary, trades, universe, yearly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Settings
    """)
    return


@app.cell
def _(mo):
    WALKFORWARD = mo.notebook_dir().parents[1] / "data" / "walkforward"  # 04's predictions, one folder per model
    PANEL = mo.notebook_dir().parents[1] / "data" / "processed" / "model_panel_train.parquet"  # the universe's caps
    MODELS = ("linear", "mlp", "lgbm", "lgbm_yearly", "lgbm_yearly_refit", "cnn_one_day",  # the benchmark: lgbm_yearly_refit
              "two_branch/comb_s3", "two_branch/tab_s3", "two_branch/price_s3", "two_branch/price_s1",
              "two_branch/tab_s1", "two_branch_lambda0/comb_s3")
    _choices = WALKFORWARD / "two_branch" / "choices.json"
    if _choices.exists():  # step 2's chosen setting, once `choose` has run
        import json  # noqa: PLC0415
        _step2 = json.loads(_choices.read_text()).get("step2")
        MODELS = MODELS + ((f"two_branch/comb_s2_{_step2}",) if _step2 else ())
    CUTS = {"quintiles": 5, "deciles": 10}  # long the top bucket, short the bottom one
    return CUTS, MODELS, PANEL, WALKFORWARD


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The backtest

    For each model and cut, on the ≥ $1B rows of its predictions (`dsr.backtest`):

    - **Positions:** each day the stocks are ranked by prediction and cut into equal buckets, as 04 cuts deciles. Every stock with a prediction is ranked, so whether it is held never depends on its return; the few held without a next-day return (about 40 stock-days) count as 0. The last day, 2023-12-29, has no returns (its target is embargoed) and is left out.
    - **Returns:** each leg's equal-weighted average `ret_next` (close to next close, delisting returns included). The long-short is the long leg minus the short leg; the book is $1 a side every day, so returns are summed, not compounded.
    - **Turnover:** the share of a leg's value replaced each day: half the sum of the absolute differences between the day's equal weights and the previous day's weights after that day's returns moved them (0: the same stocks; 1: all new).
    - **The universe, for reference:** every ≥ $1B stock held long, equal-weighted and weighted by the previous day's market cap (`cap_prev`, known before the decision), with $1 invested every day and the returns summed like the long-short's. It carries the market's direction, which the long-short does not.
    """)
    return


@app.cell
def _(CUTS, MODELS, PANEL, WALKFORWARD, daily, pd, positions, summary, trades, universe, yearly):
    results = {}
    for _kind in MODELS:
        _path = WALKFORWARD / _kind / "predictions.parquet"
        if not _path.exists():
            continue
        _p = pd.read_parquet(_path, columns=["day", "permno", "pred", "ret_next", "large_cap"])
        _p = _p[_p["large_cap"]]
        for _cut, _buckets in CUTS.items():
            _held, _average = positions(_p["day"], _p["permno"], _p["pred"], _p["ret_next"], _buckets)
            _days = daily(_held, _average)
            _trades = trades(_held, _average.index)
            results[_kind, _cut] = {"days": _days, "summary": summary(_days, _trades, _held),
                                    "yearly": yearly(_days, _trades)}
    _u = pd.read_parquet(PANEL, columns=["day", "ret_next", "cap_prev", "large_cap"])
    _u = _u[_u["large_cap"] & _u["day"].between(_p["day"].min(), _p["day"].max())] if results else _u.iloc[:0]
    market = universe(_u["day"], _u["ret_next"], _u["cap_prev"])  # the >= $1B stocks of the backtest's days
    return market, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Results

    Over 2019-2023, before costs. In the table:

    - **`mean_bp`, `t`, `positive_days`:** the long-short's mean daily return, its Newey-West t-statistic and its share of positive days. **`annual_*` and `sharpe`:** the same annualized over 252 days; the Sharpe ratio has no risk-free rate, since the $1 long pays for the $1 short. **`max_drawdown_pct`:** the largest fall of the summed returns, in % of $1 a side.
    - **`long_vs_average_bp`, `short_vs_average_bp`:** each leg against the day's equal-weighted ≥ $1B average, in bp a day; they add up to `mean_bp`.
    - **`turnover_long`, `turnover_short`:** the share of each leg replaced a day. **`trades`, `holding_days`:** the number of trades and their mean length in days.
    - **`trade_hit`:** the share of trades that made money: a long whose stock rose over the stay, a short whose stock fell. **`trade_hit_vs_average`:** the share that beat the ≥ $1B average compounded over the same days (a long above it, a short below it). Each over both legs and per leg. A leg's raw hit rate carries the market's direction (in a rising market, longs win more often and shorts less); over both legs that mostly cancels, and the rate against the average removes it. **`day_hit*`:** the same per stock-day held.
    """)
    return


@app.cell
def _(mo, pd, results):
    _table = pd.DataFrame({f"{_k}, {_c}": _r["summary"] for (_k, _c), _r in results.items()})
    _table.round(4) if results else mo.md("No predictions yet: run `scripts/models/04_walkforward.py` first.")
    return


@app.cell
def _(CUTS, MODELS, alt, market, pd, results):
    alt.data_transformers.disable_max_rows()
    _universe = {"universe, equal-weighted": "equal", "universe, value-weighted": "value"}
    _cum = pd.concat(
        [(100 * _r["days"]["long_short"].fillna(0).cumsum()).rename("cum_pct").reset_index()
         .assign(series=_k, cut=_c) for (_k, _c), _r in results.items()]
        + [(100 * market[_col].cumsum()).rename("cum_pct").reset_index().assign(series=_name, cut=_c)
           for _name, _col in _universe.items() for _c in CUTS]) if results else None
    (alt.Chart(_cum).mark_line(strokeWidth=1).encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("cum_pct:Q", title="summed return, % of $1 a side"),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=[*MODELS, *_universe],
                                        range=["#4c78a8", "#f58518", "#e45756", "#a6a6a6", "#4d4d4d"])))
     .properties(width=320, height=250)
     .facet(column=alt.Column("cut:N", title=None, sort=["quintiles", "deciles"]),
            title="Cumulative long-short return against the >= $1B universe held long, before costs")
     if results else None)
    return


@app.cell
def _(alt, pd, results):
    _legs = pd.concat([(100 * _r["days"][[f"{_l}_vs_average" for _l in ("long", "short")]].fillna(0).cumsum())
                       .reset_index().melt(id_vars="day", var_name="leg", value_name="cum_pct")
                       .assign(model=_k, cut=_c) for (_k, _c), _r in results.items()]) if results else None
    (alt.Chart(_legs).mark_line(strokeWidth=1).encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("cum_pct:Q", title="summed return against the average, %"), color="model:N", strokeDash="leg:N")
     .properties(width=320, height=250)
     .facet(column=alt.Column("cut:N", title=None, sort=["quintiles", "deciles"]),
            title="Each leg against the >= $1B average (the short leg as average minus short)")
     if results else None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### By year

    The long-short's mean daily return (bp), Sharpe ratio and share of positive days, each leg against the average, and the hit rate of the trades entered that year.
    """)
    return


@app.cell
def _(pd, results):
    (pd.concat({(_k, _c): _r["yearly"] for (_k, _c), _r in results.items()}, names=["model", "cut"]).round(3)
     if results else None)
    return


if __name__ == "__main__":
    app.run()
