import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Round-lot prices by decision day

    **Purpose:** put the price-only pull (`data/raw/taq/prices/<YYYYMMDD>/prices.parquet`, explanations/taq/taq_pull.md section 6.4b) next to each decision day's bars, as `data/intermediate/taq/train|test/<YYYYMMDD>/prices.parquet`. The chart images draw prices from it; volume and the retail columns stay in `bars`. Nothing already written changes.

    **What a decision day gets:** the previous WCT day's last ten minutes (`prev_last10`) and its own session to 15:50 (`session`), for its panel stocks' primary symbols, with the PERMNO each symbol had on the day it traded: the same rows the bars have, by `permno`, `piece` and `bar`. A bar whose trades were all odd lots has no row. Three price sets, each `open`, `high`, `low`, `close` and `trades`:
    - `rl_`: trades not flagged as odd lots, by the round-lot rule of that day;
    - `rl100_`: of those, trades of 100 shares or more;
    - `px_`, **the one to draw:** the round-lot rule before 2025-11-03 held fixed. Before that WCT day it is `rl_`. From it, when round lots started depending on price (40 shares above $250, 10 above $1,000), it is `rl100_`, except for the few stocks whose round lot was already 10 or 1 share on 2025-10-31 (`rl_`). The rule comes from the regulation's date, not from the data.

    **The links** (symbol to PERMNO, primary symbol) are built exactly as in `01_clean_and_explore_TAQ.py` section 4: `dsr.taq_link.link_permnos` on every symbol in the pull's `bars`, CRSP's ticker in force that day, `mark_primary`.

    **What we check:** every price row has its bar in that day's `bars`; its round-lot high and low lie within the bar's high and low; its round lots are at most the bar's trades; every PERMNO is in that day's panel; `px_` equals `rl_` before 2025-11-03. On the test days only these mechanical checks are run.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from dsr.crsp import class_on, read_name_records
    from dsr.taq_link import link_permnos, mark_primary
    from dsr.taq_split import ROUND_LOT_TIERS_FROM, write_decision_day_prices

    return (
        ROUND_LOT_TIERS_FROM,
        class_on,
        link_permnos,
        mark_primary,
        mo,
        np,
        pd,
        pq,
        read_name_records,
        write_decision_day_prices,
    )


@app.cell
def _(mo, pd):
    DATA_DIR = mo.notebook_dir().parents[1] / "data"
    PULL_DIR = DATA_DIR / "raw" / "taq" / "pull"
    PRICES_DIR = DATA_DIR / "raw" / "taq" / "prices"
    MASTER_DIR = DATA_DIR / "raw" / "taq" / "master"
    DECISION_DIR = DATA_DIR / "intermediate" / "taq"
    TEST_START = "20240102"  # as in 01_clean_and_explore_TAQ.py

    ALL_DAYS = sorted(_p.name for _p in PULL_DIR.iterdir() if _p.name.isdigit())
    _priced = sorted(_p.name for _p in PRICES_DIR.iterdir() if _p.name.isdigit())
    _log = pd.read_csv(PRICES_DIR / "downloaded.csv", dtype=str).fillna("")
    mo.stop(_priced != ALL_DAYS, mo.md(f"**Stopped:** {len(_priced):,} days of prices against {len(ALL_DAYS):,} pulled."))
    mo.md(f"{len(ALL_DAYS):,} WCT days with bars and prices, {ALL_DAYS[0]} to {ALL_DAYS[-1]}; "
          f"{int((_log['against_bars'] != '').sum())} days whose prices differed from their bars when downloaded.")
    return ALL_DAYS, DATA_DIR, DECISION_DIR, MASTER_DIR, PRICES_DIR, PULL_DIR, TEST_START


@app.cell
def _(ALL_DAYS, DATA_DIR, MASTER_DIR, PULL_DIR, class_on, link_permnos, mark_primary, mo, pd, pq, read_name_records):
    def _traded(day):
        return pq.read_table(PULL_DIR / day / "bars.parquet", columns=["symbol"]).to_pandas().drop_duplicates().assign(
            day=pd.Timestamp(day))

    def _master(day):
        _path = MASTER_DIR / f"mastm_{day}.parquet"
        _cols = [c for c in ["symbol_15", "cusip", "round_lot"] if c in pq.read_schema(_path).names]
        return pq.read_table(_path, columns=_cols).to_pandas().assign(day=pd.Timestamp(day))

    _days = mo.status.progress_bar(ALL_DAYS, title="Reading traded symbols and master files", remove_on_exit=True)
    _both = [(_traded(_d), _master(_d)) for _d in _days]
    traded = pd.concat([_t for _t, _ in _both], ignore_index=True)
    masters = pd.concat([_m for _, _m in _both], ignore_index=True).rename(columns={"symbol_15": "symbol"})
    _names = pd.read_csv(DATA_DIR / "raw" / "crsp" / "stocknames_v2.csv", dtype=str, keep_default_na=False)
    _links = link_permnos(traded, masters, _names)
    _linked = _links["permno"].notna()
    _ticker = class_on(_links.loc[_linked, ["day", "permno"]].astype({"permno": "int64"}).rename(columns={"day": "date"}),
                       read_name_records(DATA_DIR / "raw" / "crsp" / "stocknames_v2.csv"))["ticker"]
    symbols = mark_primary(_links[["day", "symbol", "permno"]].assign(ticker=_ticker))
    links = {_d.strftime("%Y%m%d"): _g[["symbol", "permno", "primary"]] for _d, _g in symbols.groupby("day")}
    mo.md(f"{len(traded):,} traded symbol-days; {int(symbols['primary'].sum()):,} primary symbol-days with a PERMNO.")
    return links, masters


@app.cell
def _(ROUND_LOT_TIERS_FROM, ALL_DAYS, links, masters, mo, pd):
    _before = max(_d for _d in ALL_DAYS if _d < ROUND_LOT_TIERS_FROM)  # the last WCT day of the old rule
    _m = masters[masters["day"] == pd.Timestamp(_before)][["symbol", "round_lot"]]
    _m = _m[pd.to_numeric(_m["round_lot"], errors="coerce") < 100].merge(links[_before], on="symbol")
    small_lot = set(_m.loc[_m["primary"], "permno"].astype("int64"))
    mo.vstack([mo.md(f"**Round lot below 100 shares on {_before}** (keep `rl_` after the change): {len(small_lot)} stocks"),
               _m[_m["primary"]][["symbol", "permno", "round_lot"]]])
    return (small_lot,)


@app.cell
def _(DECISION_DIR, pd):
    _panel = pd.concat([pd.read_parquet(DECISION_DIR / f"panel_{_p}.parquet", columns=["day", "permno"])
                        for _p in ["train", "test"]], ignore_index=True)
    panel_stocks = {_d.strftime("%Y%m%d"): set(_g["permno"]) for _d, _g in _panel.groupby("day")}
    return (panel_stocks,)


@app.cell
def _(
    ALL_DAYS,
    DECISION_DIR,
    PRICES_DIR,
    TEST_START,
    links,
    mo,
    panel_stocks,
    small_lot,
    write_decision_day_prices,
):
    written = write_decision_day_prices(
        PRICES_DIR, DECISION_DIR, ALL_DAYS, TEST_START, links.get, lambda _day: panel_stocks.get(_day, set()), small_lot,
        progress=lambda _pairs: mo.status.progress_bar(_pairs, title="Writing prices by decision day", remove_on_exit=True),
    )
    written.groupby("period").agg(days=("decision_day", "size"), first=("decision_day", "min"),
                                  last=("decision_day", "max"), rows=("prices_rows", "sum"))
    return (written,)


@app.cell
def _(DECISION_DIR, ROUND_LOT_TIERS_FROM, mo, panel_stocks, pd, pq, written):
    _keys = ["permno", "piece", "bar"]
    _out = []
    for _day, _period in mo.status.progress_bar(list(zip(written["decision_day"], written["period"])), title="Checking",
                                                remove_on_exit=True):
        _folder = DECISION_DIR / _period / _day
        _p = pq.read_table(_folder / "prices.parquet").to_pandas()
        _b = pq.read_table(_folder / "bars.parquet", columns=[*_keys, "high", "low", "trades"]).to_pandas()
        _b = _b[_b["piece"].isin(["prev_last10", "session"])]
        _m = _p.merge(_b, on=_keys, how="left", indicator=True)
        _has = _m["_merge"] == "both"
        _old_rule = _day < ROUND_LOT_TIERS_FROM  # every row of the day traded under the old rule, so px_ is rl_
        _out.append({
            "decision_day": _day, "period": _period, "price_rows": len(_p), "bar_rows": len(_b),
            "without_bar": int((~_has).sum()),
            "outside_bar": int((_m.loc[_has, "rl_high"].astype(float) > _m.loc[_has, "high"].astype(float) + 1e-9).sum()
                               + (_m.loc[_has, "rl_low"].astype(float) < _m.loc[_has, "low"].astype(float) - 1e-9).sum()),
            "more_than_bar": int((_m.loc[_has, "rl_trades"] > _m.loc[_has, "trades"]).sum()),
            "not_in_panel": int((~_p["permno"].isin(panel_stocks.get(_day, set()))).sum()),
            "px_not_rl_before_change": int((_m["px_trades"] != _m["rl_trades"]).sum()) if _old_rule else 0,
        })
    price_checks = pd.DataFrame(_out)
    _sums = price_checks.drop(columns=["decision_day", "period"]).groupby(price_checks["period"]).sum()
    _sums["bars with round-lot prices %"] = (100 * _sums["price_rows"] / _sums["bar_rows"]).round(2)
    _passed = not price_checks[["without_bar", "outside_bar", "more_than_bar", "not_in_panel",
                                "px_not_rl_before_change"]].to_numpy().any()
    mo.vstack([mo.md(f"**Checks {'pass' if _passed else 'FAIL'}** on {len(price_checks):,} decision days:"), _sums])
    return (price_checks,)


if __name__ == "__main__":
    app.run()
