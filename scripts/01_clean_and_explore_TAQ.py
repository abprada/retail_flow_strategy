import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TAQ: cleaning the full pull

    ## 0. Table heads

    **Purpose:** see what each table holds before using it: the first rows of the four tables the pull wrote for one WCT day, 2019-06-04 (`data/intermediate/taq/by_wct_day/20190604/`, defined in `explanations/taq_pull.md`), and that day's master file (`data/raw/taq/master/`).

    **What to look at:** the keys (`symbol`, `bucket`, `bar`, the `cells` codes), the units (prices in dollars, `last_trade_time` in nanoseconds after midnight) and the size of each table.
    """)
    return


@app.cell
def _():
    import exchange_calendars as xcals
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from matplotlib.colors import LogNorm

    from dsr.crsp import class_on, is_common_stock, read_name_records, security_type
    from dsr.taq_link import WHEN_ISSUED_SUFFIXES, link_permnos, mark_primary

    return (
        WHEN_ISSUED_SUFFIXES,
        class_on,
        is_common_stock,
        link_permnos,
        mark_primary,
        mo,
        np,
        pd,
        plt,
        pq,
        read_name_records,
        security_type,
        xcals,
    )


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
    ## Setup: day lists and the reader

    **Purpose:** make the test-period rule structural instead of a matter of care. Values (anything computed from `daily`, `cells` or `bars`) are read on training days only; keys (which symbol traded, in which bucket and bar), the manifests and `audit` are mechanical and read on every day.

    - `ALL_DAYS`: every WCT day pulled, 2013-12-31 to 2025-12-31.
    - `TRAIN_DAYS`: the WCT days whose rows all feed training decision days. The test period starts with decision day 2023-01-03, whose last bar and evening piece come from WCT day 2022-12-30, so the training days end on 2022-12-29.
    - `SAMPLE_DAYS`: one training day a month from 2014, drawn at random with a fixed seed, for distributions that do not need all 2,266 days.
    - `per_day`: reads chosen columns of one table for each day in a list, applies a function to each day and stacks the results. It refuses value columns of `daily`, `cells` and `bars` outside the training days.
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
        DataFrame and returns a DataFrame. The pull's values (daily, cells, bars) are read on training days only; their
        keys (which symbol traded, in which bucket and bar) and the audit are mechanical and readable on every day.
        """
        if table != "audit" and set(columns) - {"symbol", "bucket", "bar"} and not set(days) <= set(TRAIN_DAYS):
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
    ## 1. Inventory

    **Purpose:** check that the pull is complete and that each day looks like the days around it, from the manifests (one row per WCT day, written by the WRDS job; mechanical, so every day, the test period shaded).

    **What we check, and what the output shows:**

    1. **No day missing or extra:** the days pulled are exactly NYSE's sessions (the cell stops otherwise). Yes: 3,019 days, 2013-12-31 to 2025-12-31.
    2. **Every day passed its checks**, under a known code version. Yes. 1,108 days were pulled again with the corrections paired on sequence numbers (code version `242c83d5bae3`); a day pulled again has a row per version, and the one on disk is `downloaded.csv`'s last entry.
    3. **Less trading on early close days:** on the 26 days that close at 13:00, `bars` has about 55% of a normal day's rows (20 regular bars instead of 38) and every count dips (red dots). Yes.
    4. **No day breaks from its neighbours:** the counts move smoothly. The lasting shifts are the market's: symbols from about 7,500 a day to 11,000 (the 2020-2021 rise and 2022 fall are SPAC units and warrants), and `cells` nearly doubling from 2020. The one-day spikes are busy markets (2015-08-24, Brexit, the 2016 election, April 2025) and the dips holiday weeks.
    5. **2014 and 2015 are the older format:** no nanosecond timestamps and no trade-ID column (505 days).
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Audit

    **Purpose:** check what the cleaning on WRDS removed, every day (`audit`; mechanical, so all days). The cleaning drops correction records and cancelled trades (as known at the cutoff), trades with excluded sale conditions, and prices under $1.

    **What we check, and what the output shows:**

    1. **Nothing is lost silently:** raw trades minus each removal step equal the kept trades, every day and venue (the cell stops otherwise). Yes: 166.8 billion raw trades, 161.2 billion (96.6%) kept.
    2. **How much each step removes, and whether that changes over time** (plot): sale conditions remove about 1% of off-exchange trades in 2014 and 3 to 4% in 2022, mostly derivatively priced odd lots, which the whitelist exists to exclude; prices under $1 about 1 to 2%; cancellations a tiny share. In November 2021 late-reported odd lots double the off-exchange exclusions, which will show as a dip in retail counts.
    3. **Every correction pairs with its record** (table): `unpaired` is 0 except on 2015-02-11, whose 28,148 corrections have records on another venue (left as printed, see `explanations/taq_pull.md`, section 10). `record_after_cutoff` counts corrections ignored because they were published after 15:50.

    Codes: `step` items are `raw`, the removals (`records`, `cancelled_or_erroneous`, `sale_condition`, `price_or_size`), `kept` and `restored_original` (corrected trades given back their original print). `ctm` items are the correction codes (`01` corrected, `07` erroneous, `08` cancelled; their records `12`, `11`, `10`), `unmatched` (not found in `wct`), `record_after_cutoff` and `unpaired`.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Master files and CRSP names

    **Purpose:** load the two sources of the symbol-to-PERMNO link: every day's master file (symbol, CUSIP and, from 2016, the master's security type) and CRSP's name history (`stocknames_v2`: for each PERMNO, the CUSIP, ticker and classification in force over each period).
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The symbols table

    **Purpose:** one row per WCT day and traded symbol, built once and written to `data/intermediate/taq/symbols.parquet`: its PERMNO, whether it is the PERMNO's primary symbol that day, CRSP's classification and the Tick Size Pilot group. Everything after (the train/test split, the exploration, the features) joins it on day and symbol instead of matching again.

    ### 4.1 Traded symbols

    **What we check:** which symbols traded each day (a row in `bars`, any bucket; `daily` misses symbols whose session trades all met crossed or missing quotes), and how many are not in their day's master file under their own name. Most of those are in 2014 to 2016: Nasdaq class shares that the old master writes without the space (`LVNTA` where the trades say `LVNT A`), which the link below handles.
    """)
    return


@app.cell
def _(ALL_DAYS, masters, mo, per_day):
    traded = per_day("bars", ["symbol"], lambda d: d.drop_duplicates(), ALL_DAYS)
    _in_master = traded.merge(masters[["day", "symbol"]].drop_duplicates(), on=["day", "symbol"], how="left", indicator=True)
    _missing = _in_master[_in_master["_merge"] == "left_only"]
    mo.vstack([
        mo.md(f"{len(traded):,} traded symbol-days over {traded['day'].nunique():,} days; **{len(_missing):,} are not in "
              "their day's master file**, by year:"),
        _missing.groupby(_missing["day"].dt.year).size().rename("symbol-days").to_frame().T.rename(columns=str),
    ])
    return (traded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 PERMNOs

    **How:** `dsr.taq_link.link_permnos`, never by ticker. The CUSIP comes from the day's master (under the symbol, or written without its space), or else from the latest master in the 5 trading days before (a cap, so a reused ticker cannot take up an old CUSIP); it is matched on its first 8 characters to the CUSIP of CRSP's name record in force that day, or else to a CUSIP used by a single PERMNO in all of CRSP.

    **What we check, and what the output shows:**

    1. **Where each CUSIP comes from:** almost all from the same day's master; about 106,000 symbol-days in 2014 to 2016 only without the space; 3,693 from an earlier master, nearly all one day back (a stock's last trading day, missing from that morning's master); 5,665 from none (SIP test symbols, ETNs). The cap turned away 95 symbol-days looking back 6 to 167 days, none of them in CRSP.
    2. **How many get a PERMNO:** 87 to 91% of traded symbol-days each year. The rest are preferreds, warrants, units and ETNs, which CRSP does not cover; by trades, about 99% get one.
    """)
    return


@app.cell
def _(ALL_DAYS, crsp_names, link_permnos, masters, mo, np, pd, traded):
    symbol_links = link_permnos(traded, masters, crsp_names)
    _lag = pd.Series(pd.DatetimeIndex(ALL_DAYS).get_indexer(symbol_links["day"]) - pd.DatetimeIndex(ALL_DAYS).get_indexer(
        symbol_links["master_day"].fillna(symbol_links["day"])), index=symbol_links.index)
    symbol_links["cusip_from"] = np.select(
        [symbol_links["master_day"].isna(), _lag.eq(0) & symbol_links["master_symbol"].ne(symbol_links["symbol"]), _lag.eq(0)],
        ["no master", "same day, without the space", "same day"], default="earlier master",
    )
    symbol_links["master_lag"] = _lag.where(symbol_links["cusip_from"].eq("earlier master"))  # trading days back
    mo.vstack([
        mo.md("**Where each traded symbol-day's CUSIP comes from**, symbol-days by year:"),
        pd.crosstab(symbol_links["day"].dt.year, symbol_links["cusip_from"]).rename(index=str),
        mo.md("**How it was linked**, % of the year's traded symbol-days:"),
        (pd.crosstab(symbol_links["day"].dt.year, symbol_links["rule"], normalize="index") * 100).round(2).rename(index=str),
    ])
    return (symbol_links,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Primary symbol

    **Purpose:** count each stock once. When two traded symbols map to one PERMNO on a day, only one may carry it.

    **Rule** (`dsr.taq_link.mark_primary`): a symbol alone on its PERMNO that day is primary, whatever its suffix (`MKC V` is McCormick's voting class, not when-issued shares). Among several, the symbols without a when-issued suffix (`WI`, `WD`, `V`: shares of a split, spin-off or new issue trading before they exist, at their own price) are preferred; if two regular ones remain, the one matching CRSP's ticker that day.

    **What we check, and what the output shows:**

    1. **How often two symbols share a PERMNO:** 550 PERMNO-days in 12 years. 539 are a regular symbol next to its when-issued twin.
    2. **The 11 with two regular symbols** are each a merger's last day (ACE and CB on 2016-01-14, TYC and JCI on 2016-09-02): the master already gives the target's old ticker the acquirer's CUSIP. CRSP's ticker picks the acquirer every time; the target's last-day trades are kept but not primary, so they never reach the acquirer.
    3. **At most one primary symbol per PERMNO and day** (the cell stops otherwise), and every linked PERMNO-day has one.
    """)
    return


@app.cell
def _(
    DATA_DIR,
    WHEN_ISSUED_SUFFIXES,
    class_on,
    mark_primary,
    pd,
    read_name_records,
    symbol_links,
):
    crsp_name_records = read_name_records(DATA_DIR / "raw" / "crsp" / "stocknames_v2.csv")
    _linked = symbol_links["permno"].notna()
    crsp_class = class_on(
        symbol_links.loc[_linked, ["day", "permno"]].astype({"permno": "int64"}).rename(columns={"day": "date"}),
        crsp_name_records,
    )  # the name record in force that day, for every linked symbol-day
    symbol_primary = mark_primary(symbol_links[["day", "symbol", "permno"]].assign(ticker=crsp_class["ticker"]))
    _primaries = symbol_primary[symbol_primary["permno"].notna()].groupby(["day", "permno"])["primary"].sum()
    assert _primaries.eq(1).all(), "a linked PERMNO-day has no primary symbol, or two"
    _linked_rows = symbol_primary[symbol_primary["permno"].notna()]
    _twin = _linked_rows["symbol"].str.split(" ").str[1].isin(WHEN_ISSUED_SUFFIXES)
    _n = _linked_rows.assign(twin=_twin).groupby(["day", "permno"]).agg(
        symbols=("symbol", "size"), twins=("twin", "sum"), primary=("primary", "sum"))
    _shared = _n["symbols"] > 1
    pd.DataFrame({"PERMNO-days": [
        len(_n), _shared.sum(), (_shared & (_n["symbols"] - _n["twins"] == 1)).sum(),
        (_shared & (_n["symbols"] - _n["twins"] > 1)).sum(), (_n["primary"] == 0).sum(),
    ]}, index=["linked", "with two symbols", "  a regular symbol and its when-issued twin",
               "  two regular symbols (decided by CRSP's ticker)", "with no primary symbol"])
    return crsp_class, symbol_primary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Classification

    **Purpose:** say what each symbol is, with the definitions both models share.

    - **CRSP's class on the day** (`dsr.crsp`, the same functions as the Robintrack panels): `security_type` (common stock, foreign common stock, ADR, REIT, ETF, closed-end fund, unit, other; unclassified when no name record is in force that day) and `is_common` (share codes 10 and 11). Symbols without a PERMNO have neither.
    - **The Tick Size Pilot group** (`dsr.taq_link.pilot_groups`): C (control), 1, 2 or 3, from 2016-10-03 to 2018-09-28 only. From 2016-10-31 as that day's master shows it; before, the group the stock's CUSIP shows on 2016-10-31, because the master leaves it blank until 10-21 and shows group 3 stocks as C until each was phased in. FINRA published the assignments before the pilot began. After the pilot the master keeps writing C for former pilot stocks, which means nothing.

    **What we check, and what the output shows:**

    1. **Primary symbols by security type:** about 3,500 common stocks a day, peaking in 2021-2022; ETFs triple from about 1,330 a day to 4,180; foreign common doubles from 2021; under 310 unclassified symbol-days in any year.
    2. **Pilot group sizes:** in October 2016, 393, 388 and 386 traded stocks in groups 1 to 3 and 1,167 controls (designed: 400 and 1,200), shrinking to about 330 and 1,040 by September 2018 as stocks leave.
    """)
    return


@app.cell
def _(
    PILOT_END,
    PILOT_START,
    crsp_class,
    is_common_stock,
    masters,
    mo,
    pd,
    pilot_groups,
    security_type,
    symbol_links,
    symbol_primary,
):
    _pilot = pilot_groups(masters).rename(columns={"day": "master_day", "symbol": "master_symbol"})
    symbols = (
        symbol_primary[["day", "symbol", "permno", "symbols_on_permno", "primary"]]
        .join(symbol_links[["master_day", "master_symbol", "cusip", "rule"]].rename(columns={"rule": "link_rule"}))
        .join(pd.DataFrame({"ticker": crsp_class["ticker"], "security_type": security_type(crsp_class),
                            "is_common": is_common_stock(crsp_class)}))
        .merge(_pilot[["master_day", "master_symbol", "pilot_group"]].drop_duplicates(["master_day", "master_symbol"]),
               on=["master_day", "master_symbol"], how="left")
    )
    _p = symbols[symbols["primary"]]
    _in_pilot = _p[_p["day"].between(PILOT_START, PILOT_END)]
    _per_day = _in_pilot.groupby(["day", "pilot_group"]).size().unstack()
    mo.vstack([
        mo.md("**Primary symbols by CRSP security type**, symbol-days by year:"),
        pd.crosstab(_p["day"].dt.year, _p["security_type"]).rename(index=str),
        mo.md("**Tick Size Pilot groups**, primary symbols per day (median by month):"),
        _per_day.groupby(_per_day.index.to_period("M").astype(str)).median().iloc[[0, 1, 6, 12, 18, -1]],
    ])
    return (symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.5 Checks and writing the table

    **Purpose:** gate the write. `data/intermediate/taq/symbols.parquet` is written only when every check passes and the button is pressed.

    **What we check:**

    1. One row per day and traded symbol, and every traded symbol has a row.
    2. Every linked PERMNO-day has exactly one primary symbol, and a primary symbol always has a PERMNO.
    3. Every day is a pulled WCT day, and pilot groups appear only from 2016-10-03 to 2018-09-28.
    4. **Coverage** (training days only, since it reads trades and dollars): the share of all trades and dollars that land on a primary symbol with a PERMNO, by year (about 98 to 99%; the rest are preferreds, warrants, ETNs and ADRs CRSP does not carry), and, separately, the share of all trades that are US common stock (about two thirds; the rest is ETFs, foreign stocks, ADRs, REITs and funds). Of the trades in symbols the master calls common stock, 99.97% get a PERMNO.
    """)
    return


@app.cell
def _(TRAIN_DAYS, pd, per_day, symbols):
    _totals = per_day("bars", ["symbol", "trades", "dollars"],
                       lambda d: d.groupby("symbol", as_index=False)[["trades", "dollars"]].sum(), TRAIN_DAYS)
    _t = _totals.merge(symbols[["day", "symbol", "primary", "is_common"]].astype({"day": _totals["day"].dtype}),
                       on=["day", "symbol"], how="left")
    _t["year"] = _t["day"].dt.year
    _all = _t.groupby("year")[["trades", "dollars"]].sum()
    _on = _t[_t["primary"]].groupby("year")[["trades", "dollars"]].sum()
    _common = _t[_t["primary"] & _t["is_common"]].groupby("year")[["trades", "dollars"]].sum()
    symbol_coverage = pd.DataFrame({
        "% trades on a primary PERMNO": _on["trades"] / _all["trades"] * 100,
        "% dollars on a primary PERMNO": _on["dollars"] / _all["dollars"] * 100,
        "% of all trades that are US common stock": _common["trades"] / _all["trades"] * 100,
    }).round(2).rename(index=str)
    symbol_coverage
    return


@app.cell
def _(ALL_DAYS, PILOT_END, PILOT_START, mo, pd, symbols, traded):
    _linked = symbols["permno"].notna()
    _primaries = symbols[_linked].groupby(["day", "permno"])["primary"].sum()
    _pilot_days = symbols.loc[symbols["pilot_group"].notna(), "day"]
    symbol_checks = pd.DataFrame([
        ("one row per day and traded symbol", not symbols.duplicated(["day", "symbol"]).any()),
        ("every traded symbol has a row", len(symbols) == len(traded)),
        ("every linked PERMNO-day has exactly one primary symbol", bool(_primaries.eq(1).all())),
        ("a primary symbol always has a PERMNO", bool(_linked[symbols["primary"]].all())),
        ("every day is a pulled WCT day", bool(symbols["day"].dt.strftime("%Y%m%d").isin(ALL_DAYS).all())),
        ("pilot groups only from 2016-10-03 to 2018-09-28", bool(_pilot_days.between(PILOT_START, PILOT_END).all())),
    ], columns=["check", "passed"])
    write_symbols = mo.ui.run_button(label="Write data/intermediate/taq/symbols.parquet")
    mo.vstack([mo.ui.table(symbol_checks, selection=None), write_symbols])
    return symbol_checks, write_symbols


@app.cell
def _(DATA_DIR, mo, symbol_checks, symbols, write_symbols):
    mo.stop(not symbol_checks["passed"].all(), mo.md("**Not written:** a check failed."))
    mo.stop(not write_symbols.value, mo.md("Press the button to write."))
    _path = DATA_DIR / "intermediate" / "taq" / "symbols.parquet"
    _part = _path.with_name(_path.name + ".part")
    symbols.to_parquet(_part, index=False)
    _part.rename(_path)
    mo.md(f"Written: `{_path.relative_to(DATA_DIR.parent)}`, {len(symbols):,} rows, {_path.stat().st_size / 1e6:,.0f} MB")
    return


if __name__ == "__main__":
    app.run()
