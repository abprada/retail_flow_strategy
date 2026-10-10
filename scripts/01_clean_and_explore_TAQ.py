import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TAQ: cleaning the full pull

    ## Table heads

    The first rows of each table on one WCT day, 2019-06-04: the four tables the pull wrote (`data/intermediate/taq/by_wct_day/20190604/`, defined in `explanations/taq_pull.md`) and the day's master file (`data/raw/taq/master/`).
    """)
    return


@app.cell
def _():
    import exchange_calendars as xcals
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyarrow.parquet as pq
    from matplotlib.colors import LogNorm

    from dsr.taq_link import link_permnos

    return LogNorm, link_permnos, mo, pd, plt, pq, xcals


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parent / "data"
    PULL_DIR = DATA_DIR / "intermediate" / "taq" / "by_wct_day"  # one folder per WCT day: daily, cells, bars, audit
    MASTER_DIR = DATA_DIR / "raw" / "taq" / "master"  # mastm_<YYYYMMDD>.parquet, every column as text
    DAY = "20190604"
    return DATA_DIR, DAY, MASTER_DIR, PULL_DIR


@app.cell
def _(DAY, MASTER_DIR, PULL_DIR, pq):
    def _head(path, n=10):
        """The first n rows of a parquet file, with its total rows and columns, without reading the rest."""
        _file = pq.ParquetFile(path)
        return next(_file.iter_batches(batch_size=n)).to_pandas(), _file.metadata.num_rows, _file.metadata.num_columns


    heads = {name: _head(PULL_DIR / DAY / f"{name}.parquet") for name in ["daily", "cells", "bars", "audit"]}
    heads["master"] = _head(MASTER_DIR / f"mastm_{DAY}.parquet")
    return (heads,)


@app.cell
def _(heads, mo):
    mo.vstack([
        mo.vstack([mo.md(f"### `{_name}`: {_rows:,} rows, {_cols} columns"), _df])
        for _name, (_df, _rows, _cols) in heads.items()
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploring the pull: setup

    What the pull wrote, looked at table by table. Values (anything computed from `daily`, `cells` or `bars`) are looked at on training days only; the manifests and `audit` are mechanical and cover all days.

    - `ALL_DAYS`: every WCT day pulled, 2013-12-31 to 2025-12-31.
    - `TRAIN_DAYS`: the WCT days whose rows all feed training decision days. The test period starts with decision day 2023-01-03, whose last bar and evening piece come from WCT day 2022-12-30, so the training days end on 2022-12-29.
    - `SAMPLE_DAYS`: one training day a month from 2014, drawn at random with a fixed seed, for distributions that do not need all 2,266 days.
    - `per_day`: reads the chosen columns of one table for each day in a list, applies a function to each day, and stacks the results. It refuses days outside the training period for `daily`, `cells` and `bars`.
    """)
    return


@app.cell
def _(PULL_DIR, mo, pd):
    ALL_DAYS = sorted(_p.name for _p in PULL_DIR.iterdir() if _p.name.isdigit())  # every WCT day pulled, YYYYMMDD
    TEST_START = "20230103"  # first decision day of the test period
    TRAIN_DAYS = ALL_DAYS[: ALL_DAYS.index(TEST_START) - 1]  # the WCT day before TEST_START already feeds it
    _train = pd.Series(TRAIN_DAYS)
    _train = _train[_train >= "20140101"]  # 2013-12-31 is pulled only for the first decision day's pieces
    SAMPLE_DAYS = sorted(_train.groupby(_train.str[:6]).sample(1, random_state=0))


    def _span(days):
        return f"| {len(days):,} | {pd.Timestamp(days[0]):%Y-%m-%d} | {pd.Timestamp(days[-1]):%Y-%m-%d} |"


    mo.md(f"""
    | List | Days | First | Last |
    |---|---|---|---|
    | `ALL_DAYS` {_span(ALL_DAYS)}
    | `TRAIN_DAYS` {_span(TRAIN_DAYS)}
    | `SAMPLE_DAYS` {_span(SAMPLE_DAYS)}
    """)
    return ALL_DAYS, TRAIN_DAYS


@app.cell
def _(PULL_DIR, TRAIN_DAYS, mo, pd, pq):
    def per_day(table, columns, fn, days):
        """Apply fn to one table on each day in days and stack the results, with the day as the first column.

        Reads only `columns` of each day's file, so a pass over every training day fits in memory. fn takes the day's
        DataFrame and returns a DataFrame. The pull's values (daily, cells, bars) are read on training days only.
        """
        if table != "audit" and not set(days) <= set(TRAIN_DAYS):
            raise ValueError(f"{table}: values are looked at on training days only")
        parts = []
        for day in mo.status.progress_bar(days, title=f"Reading {table}", remove_on_exit=True):
            result = fn(pq.read_table(PULL_DIR / day / f"{table}.parquet", columns=columns).to_pandas())
            result.insert(0, "day", pd.Timestamp(day))
            parts.append(result)
        return pd.concat(parts, ignore_index=True)

    return (per_day,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inventory

    The manifests: one row per WCT day, written by the WRDS job, with the rows of each table, the MB of raw trades read from WRDS (`wct_rows_mb`, CSV) and written (`out_mb`, parquet), the day's close, and whether it has nanosecond timestamps and trade IDs. Row counts are mechanical, so every day is shown, with the test period shaded.

    A day that differs from its neighbours is found by dividing each count by the median of the 21 days around it, so the growth over the years does not hide it. Early closes (20 regular bars instead of 38) are marked and listed apart.
    """)
    return


@app.cell
def _(ALL_DAYS, PULL_DIR, TRAIN_DAYS, pd, xcals):
    _manifests = pd.concat(
        [pd.read_csv(_f, dtype={"version": str}) for _f in sorted((PULL_DIR / "manifests").glob("manifest_*.csv"))],
        ignore_index=True,
    )
    # a day pulled again has a row per code version; keep the one whose tables are on disk: downloaded.csv's last entry
    _on_disk = pd.read_csv(PULL_DIR / "downloaded.csv", dtype=str).drop_duplicates("day", keep="last")
    manifest = (
        _manifests[_manifests["checks_passed"]]
        .assign(day=lambda _m: pd.to_datetime(_m["day"]))
        .merge(_on_disk[["day", "version"]].assign(day=lambda _d: pd.to_datetime(_d["day"])), on=["day", "version"])
        .sort_values("day")
        .reset_index(drop=True)
    )
    _sessions = xcals.get_calendar("XNYS", start=ALL_DAYS[0], end=ALL_DAYS[-1]).sessions
    assert list(manifest["day"].dt.strftime("%Y%m%d")) == ALL_DAYS, "the manifests and the day folders differ"
    assert manifest["day"].tolist() == _sessions.tolist(), "a NYSE session is missing or extra"

    _groups = {
        "every day": manifest["day"].notna(),
        "no nanosecond timestamps": ~manifest["has_nano"],
        "no trade ID column in ctm": ~manifest["has_tr_id"],
        "early close (13:00)": manifest["close"] == "13:00",
        "test period": ~manifest["day"].dt.strftime("%Y%m%d").isin(TRAIN_DAYS),
        **{f"code version {_v}": manifest["version"] == _v for _v in manifest["version"].unique()},
    }
    pd.DataFrame(
        {
            _name: {"days": _m.sum(), "first": manifest.loc[_m, "day"].min().date(), "last": manifest.loc[_m, "day"].max().date()}
            for _name, _m in _groups.items()
        }
    ).T
    return (manifest,)


@app.cell
def _(ALL_DAYS, TRAIN_DAYS, manifest, pd, plt):
    _panels = [
        ("daily_rows", 1, "daily: symbols traded in the session"),
        ("bars_rows", 1e3, "bars: rows, thousands"),
        ("cells_rows", 1e6, "cells: rows, millions"),
        ("audit_rows", 1, "audit: rows"),
        ("wct_rows_mb", 1e3, "raw trades read from WRDS: GB of CSV"),
        ("out_mb", 1, "written: MB of parquet"),
    ]
    _early = manifest["close"] == "13:00"
    _test_start = pd.Timestamp(ALL_DAYS[len(TRAIN_DAYS)])
    _fig, _axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for _ax, (_col, _unit, _title) in zip(_axes.flat, _panels):
        _ax.plot(manifest["day"], manifest[_col] / _unit, lw=0.6, color="C0")
        _ax.scatter(manifest.loc[_early, "day"], manifest.loc[_early, _col] / _unit, s=10, color="C3", zorder=3,
                    label="early close")
        _ax.axvspan(_test_start, manifest["day"].max(), color="0.9", zorder=0)
        _ax.set_title(_title, fontsize=10)
    _axes[0, 0].legend(loc="upper left", fontsize=8)
    _fig.suptitle("Each WCT day, from the manifests (test period shaded)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(manifest, mo, pd):
    _counts = ["daily_rows", "bars_rows", "cells_rows", "wct_rows_mb"]
    _ratio = manifest[_counts] / manifest[_counts].rolling(21, center=True, min_periods=11).median()
    _table = pd.concat([manifest["day"].dt.date, manifest["close"], _ratio.round(2)], axis=1)
    _far = (manifest["close"] == "16:00") & ((_ratio < 2 / 3) | (_ratio > 1.5)).any(axis=1)
    mo.vstack([
        mo.md("Each count divided by the median of the 21 days around it. **Normal-close days at least 1.5 times above or "
              "below that median** in any count:"),
        _table[_far],
        mo.md("**The early closes:**"),
        _table[manifest["close"] == "13:00"],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Audit

    What the cleaning on WRDS removed, every day (`audit`; mechanical, so all days). Four sections:

    - `step`: the trades read (`raw`), the trades removed at each step (`records`, `cancelled_or_erroneous`, `sale_condition`, `price_or_size`) and those `kept`; `restored_original` counts corrected trades given back their original data, which stay.
    - `ctm`: the trade file's flagged rows by correction code (`01` corrected, `07` erroneous, `08` cancelled, and their records `12`, `11`, `10`), and three counts: `unmatched` (not found in `wct`), `record_after_cutoff` (the record came at or after the cutoff, so the trade stays as printed) and `unpaired` (no record with the same trade ID; trade IDs exist from 2016).
    - `condition`: trades by sale-condition string, by venue, and whether they were kept.
    - `venue_code`: trades by venue code and whether they were kept.
    """)
    return


@app.cell
def _(ALL_DAYS, pd, per_day):
    audit = per_day("audit", ["section", "item", "venue", "kept", "trades", "shares", "dollars"], lambda d: d, ALL_DAYS)
    audit["month"] = audit["day"].dt.to_period("M").dt.to_timestamp()
    audit["year"] = audit["day"].dt.year
    _removed = ["records", "cancelled_or_erroneous", "sale_condition", "price_or_size"]
    _steps = audit[audit["section"] == "step"].pivot_table(
        index=["day", "venue"], columns="item", values="trades", aggfunc="sum", fill_value=0
    )
    assert (_steps["raw"] - _steps[_removed].sum(axis=1) == _steps["kept"]).all(), "raw minus the removed steps is not kept"
    _total = _steps.groupby("venue").sum().T.loc[["raw", *_removed, "restored_original", "kept"]]
    pd.DataFrame({
        "exchange, millions": _total[0] / 1e6,
        "exchange, % of raw": _total[0] / _total.loc["raw", 0] * 100,
        "off-exchange, millions": _total[1] / 1e6,
        "off-exchange, % of raw": _total[1] / _total.loc["raw", 1] * 100,
    }).round(3)
    return (audit,)


@app.cell
def _(ALL_DAYS, TRAIN_DAYS, audit, pd, plt):
    _m = audit[audit["section"] == "step"].pivot_table(
        index=["venue", "month"], columns="item", values="trades", aggfunc="sum", fill_value=0
    )
    _test_start = pd.Timestamp(ALL_DAYS[len(TRAIN_DAYS)])
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for _ax, (_venue, _name) in zip(_axes, [(0, "exchange"), (1, "off-exchange")]):
        _v = _m.loc[_venue]
        for _step in ["sale_condition", "price_or_size", "cancelled_or_erroneous", "records"]:
            _ax.plot(_v.index, _v[_step] / _v["raw"] * 100, lw=1, label=_step)
        _ax.set_yscale("log")
        _ax.axvspan(_test_start, _v.index.max(), color="0.9", zorder=0)
        _ax.set_title(f"{_name}: % of raw trades removed at each step, by month", fontsize=10)
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(audit, mo):
    _c = audit[audit["section"] == "ctm"].pivot_table(
        index="year", columns="item", values="trades", aggfunc="sum", fill_value=0
    )
    mo.vstack([
        mo.md("**The trade file's flagged rows by year**, in trades:"),
        _c[["01", "07", "08", "12", "11", "10", "unmatched", "record_after_cutoff", "unpaired"]],
    ])
    return


@app.cell
def _(audit, mo):
    _raw = audit[(audit["section"] == "step") & (audit["item"] == "raw")].groupby("venue")["trades"].sum()
    _c = audit[audit["section"] == "condition"]
    _c = (
        _c.assign(condition=_c["item"].str.replace(" ", "·"))
        .groupby(["condition", "venue", "kept"])
        .agg(first_day=("day", "min"), last_day=("day", "max"), days=("day", "nunique"), trades=("trades", "sum"))
        .reset_index()
    )
    _c["% of the venue's raw trades"] = _c["trades"] / _c["venue"].map(_raw) * 100
    _c["first_day"], _c["last_day"] = _c["first_day"].dt.date, _c["last_day"].dt.date
    mo.vstack([
        mo.md("**Every sale-condition string** (spaces shown as ·), by venue (0 exchange, 1 off-exchange) and whether its "
              "trades were kept. A string can be both: `T` is kept outside the session and excluded inside it."),
        _c.sort_values("trades", ascending=False),
    ])
    return


@app.cell
def _(ALL_DAYS, LogNorm, TRAIN_DAYS, audit, mo, pd, plt):
    _names = {
        "A": "NYSE American", "B": "Nasdaq BX", "C": "NYSE National", "D": "FINRA (off-exchange)", "H": "MIAX Pearl",
        "J": "Cboe EDGA", "K": "Cboe EDGX", "L": "LTSE", "M": "NYSE Chicago", "N": "NYSE", "P": "NYSE Arca",
        "Q": "Nasdaq (Tape C)", "T": "Nasdaq (Tapes A, B)", "U": "MEMX", "V": "IEX", "X": "Nasdaq PSX", "Y": "Cboe BYX",
        "Z": "Cboe BZX",
    }
    _v = audit[audit["section"] == "venue_code"]
    _share = _v.pivot_table(index="item", columns="month", values="trades", aggfunc="sum", fill_value=0)
    _share = _share / _share.sum()
    _fig, _ax = plt.subplots(figsize=(13, 5))
    _mesh = _ax.pcolormesh(_share.columns, range(len(_share)), _share.where(_share > 0), norm=LogNorm(1e-6, 1),
                           shading="nearest")
    _ax.set_yticks(range(len(_share)), [f"{_code} {_names.get(_code, '')}" for _code in _share.index], fontsize=8)
    _ax.axvline(pd.Timestamp(ALL_DAYS[len(TRAIN_DAYS)]), color="k", lw=1)
    _ax.set_title("Each venue code's share of the month's trades (blank: none; line: test period starts)", fontsize=10)
    _fig.colorbar(_mesh, ax=_ax)
    _fig.tight_layout()

    _t = _v.groupby("item").agg(first_day=("day", "min"), last_day=("day", "max"), days=("day", "nunique"),
                                trades=("trades", "sum"))
    _t["% of all trades"] = _t["trades"] / _t["trades"].sum() * 100
    _t["% kept"] = _v[_v["kept"]].groupby("item")["trades"].sum().reindex(_t.index, fill_value=0) / _t["trades"] * 100
    _t["first_day"], _t["last_day"] = _t["first_day"].dt.date, _t["last_day"].dt.date
    _t.insert(0, "venue", _t.index.map(lambda _code: _names.get(_code, "")))
    mo.vstack([_fig, _t.round(3)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Symbols, PERMNOs and security groups

    Every symbol of every day's master file (`data/raw/taq/master/`) linked to a CRSP PERMNO by `dsr.taq_link.link_permnos`, never by ticker: the master's CUSIP against the CUSIP (`cusip9`) of CRSP's name record in force that day (`stocknames_v2`); failing that, a CUSIP used by a single PERMNO in all of CRSP; a symbol missing from a day's master takes its CUSIP from the latest earlier master. The link fixes how the data was recorded and selects nothing by value, so it covers every day. The master's own security type (`sec_type`) exists only from 2016; CRSP's (`securitytype`, `securitysubtype`, `sharetype`) is dated and covers every year.
    """)
    return


@app.cell
def _(ALL_DAYS, DATA_DIR, MASTER_DIR, mo, pd, pq):
    def _read_master(day):
        """One day's master file: symbol, CUSIP and, from 2016, the master's security type."""
        path = MASTER_DIR / f"mastm_{day}.parquet"
        columns = [c for c in ["symbol_15", "cusip", "sec_type"] if c in pq.read_schema(path).names]
        return pq.read_table(path, columns=columns).to_pandas().assign(day=pd.Timestamp(day))


    masters = pd.concat(
        [_read_master(_day) for _day in mo.status.progress_bar(ALL_DAYS, title="Reading master files", remove_on_exit=True)],
        ignore_index=True,
    ).rename(columns={"symbol_15": "symbol"})
    crsp_names = pd.read_csv(DATA_DIR / "raw" / "crsp" / "stocknames_v2.csv", dtype=str, keep_default_na=False)
    mo.md(f"{len(masters):,} master rows over {masters['day'].nunique():,} days; {len(crsp_names):,} CRSP name records")
    return crsp_names, masters


@app.cell
def _(crsp_names, link_permnos, masters, mo, pd):
    master_links = link_permnos(masters[["day", "symbol"]], masters, crsp_names).merge(
        masters[["day", "symbol", "sec_type"]].drop_duplicates(["day", "symbol"]), on=["day", "symbol"], how="left"
    )
    _by_year = pd.crosstab(master_links["day"].dt.year, master_links["rule"], normalize="index") * 100
    mo.vstack([
        mo.md("**How each master symbol-day was linked**, % of the year's symbol-days:"),
        _by_year.round(2),
    ])
    return (master_links,)


@app.cell
def _(master_links, mo, pd):
    _from_2016 = master_links[master_links["day"] >= "2016-01-01"]
    _types = pd.crosstab(_from_2016["sec_type"].replace("", "blank"), _from_2016["rule"])
    _types = _types.loc[_types.sum(axis=1).sort_values(ascending=False).index]
    _common = (
        _from_2016[(_from_2016["sec_type"] == "A") & _from_2016["permno"].isna()]
        .groupby(["symbol", "cusip"], dropna=False)["day"].agg(days="size", first="min", last="max")
        .sort_values("days", ascending=False)
        .reset_index()
    )
    mo.vstack([
        mo.md("**Master symbol-days by the master's security type and how they were linked**, 2016 onwards (the "
              "master has no type before): `A` common stock, `H` ADR, `B` preferred, `C` warrant, `R` unit, `D` right, "
              "`G` structured product, `U` unknown, `P` fund"),
        _types,
        mo.md("**Common stocks (`A`) without a PERMNO**, by symbol and CUSIP:"),
        _common,
    ])
    return


@app.cell
def _(master_links, mo):
    _linked = master_links[master_links["permno"].notna()]
    _shared = _linked[_linked.groupby(["day", "permno"])["symbol"].transform("size") > 1]
    _pairs = (
        _shared.groupby(["permno", "day"])["symbol"].agg(lambda _s: " / ".join(sorted(_s)))
        .reset_index()
        .groupby(["permno", "symbol"])["day"].agg(days="size", first="min", last="max")
        .sort_values("first")
        .reset_index()
    )
    mo.vstack([
        mo.md(f"**PERMNOs reached by two symbols on the same day**: {len(_shared):,} symbol-days, "
              f"{_shared['day'].dt.year.value_counts().sort_index().to_dict()} by year. Mostly a when-issued (`WI`) or "
              "when-distributed (`WD`) symbol trading next to the regular one, and ticker changes or spin-offs "
              "listed under both symbols for a day:"),
        _pairs,
    ])
    return


if __name__ == "__main__":
    app.run()
