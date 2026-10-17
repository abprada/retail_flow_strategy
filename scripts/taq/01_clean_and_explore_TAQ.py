import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TAQ: cleaning the full pull

    ## 0. Table heads

    **Purpose:** see what each table holds before using it: the first rows of the four tables the pull wrote for one WCT day, 2019-06-04 (`data/raw/taq/pull/20190604/`, the WRDS pull as downloaded, defined in `explanations/taq/taq_pull.md`), and that day's master file (`data/raw/taq/master/`).

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
    from dsr.taq_link import PILOT_END, PILOT_START, WHEN_ISSUED_SUFFIXES, link_permnos, mark_primary, pilot_groups
    from dsr.crsp import DAILY_COLUMNS as CRSP_COLUMNS
    from dsr.taq_split import CAP_MIN, CODES, DOLLAR_VOLUME_MIN, DOLLARS_BEFORE_CUTOFF_MIN, PRICE_MIN, UNIVERSE_TYPES, stack_panels, stock_days, write_decision_days

    return (
        CAP_MIN,
        CODES,
        DOLLARS_BEFORE_CUTOFF_MIN,
        DOLLAR_VOLUME_MIN,
        PILOT_END,
        PILOT_START,
        PRICE_MIN,
        UNIVERSE_TYPES,
        WHEN_ISSUED_SUFFIXES,
        class_on,
        is_common_stock,
        link_permnos,
        mark_primary,
        mo,
        np,
        pd,
        pilot_groups,
        plt,
        pq,
        read_name_records,
        security_type,
        stack_panels,
        stock_days,
        write_decision_days,
        xcals,
    )


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parents[1] / "data"
    PULL_DIR = DATA_DIR / "raw" / "taq" / "pull"  # the WRDS pull, one folder per WCT day: daily, cells, bars, audit
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
    - `TRAIN_DAYS`: the WCT days whose rows all feed training decision days. The test period is decision days 2024-01-02 to 2025-12-31; its first day's last bar and evening piece come from WCT day 2023-12-29, so the training days end on 2023-12-28.
    - `SAMPLE_DAYS`: one training day a month from 2014, drawn at random with a fixed seed, for distributions that do not need every training day.
    - `per_day`: reads chosen columns of one table for each day in a list, applies a function to each day and stacks the results. It refuses value columns of `daily`, `cells` and `bars` outside the training days.
    """)
    return


@app.cell
def _(PULL_DIR, mo, pd):
    ALL_DAYS = sorted(_p.name for _p in PULL_DIR.iterdir() if _p.name.isdigit())  # every WCT day pulled, YYYYMMDD
    TEST_START = "20240102"  # first decision day of the test period (test: 2024 and 2025)
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
    return ALL_DAYS, SAMPLE_DAYS, TEST_START, TRAIN_DAYS


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
    3. **Every correction pairs with its record** (table): `unpaired` is 0 except on 2015-02-11, whose 28,148 corrections have records on another venue (left as printed, see `explanations/taq/taq_pull.md`, section 10). `record_after_cutoff` counts corrections ignored because they were published after 15:50.

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
    return crsp_class, crsp_name_records, symbol_primary


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The clean data: the universe, and what is known at 15:50

    **Purpose:** rewrite the pull so it holds, for the stocks in each day's universe, exactly what is known at that day's 15:50 cutoff, split into train (decision days 2014-01-02 to 2023-12-29) and test (2024-01-02 to 2025-12-31). Written to `data/intermediate/taq/`:
    - `panel_train.parquet` and `panel_test.parquet`: one row per decision day and stock (`day`, `permno`);
    - `train/<YYYYMMDD>/` and `test/<YYYYMMDD>/`: that day's `bars` and `cells`;
    - `codes.csv`, which says what every code means.

    **The universe** (`dsr.taq_split`), decided per day with what was known at its cutoff. The rules were chosen on the training days (section 6) and apply unchanged to the test days:
    - from the day before, in `stock_days`: CRSP's security type that day is common stock, foreign common stock, ADR or REIT (no ETFs); the previous close is at least $1 (Eaton et al.) and the market cap at least $100M; rule 1, the previous day's dollar volume is at least $100,000;
    - from the tape, while writing: rule 2, at least $50,000 traded before the cutoff (pre-open and session) on the decision day and on the trading day before.

    Nothing about later days is used, so no stock is dropped for a later split or delisting.

    **The files:**
    - **`bars` and `cells`:** the rows of the universe stocks' primary symbols. They cover the previous trading day's last 10 minutes and evening, then the day's pre-open and session to 15:50, in time order. Each row has `permno` (the one its symbol had on the day it traded) and `piece`. Trade rows are otherwise copied unchanged: counts as integers, prices as exact decimals.
    - **The panel:**
      - who the stock is: `symbol`, `security_type`, `is_common`, `pilot_group`;
      - the day: `early_close`;
      - the day before, from CRSP: `price_prev`, `cap_prev`, `dollar_volume_prev`, `ret_prev`;
      - today before 15:50: `open_t`, CRSP's official open; `facprc_t`, the day's split factor; `spread_bp`; and `ret_1550`, the return to 15:50: (last trade before today's cutoff × `facprc_t` + the day's dividend) / last trade before yesterday's cutoff − 1, CRSP's return formula ending at 15:50;
      - the target: `ret_next`, close to next close with delisting returns, kept when it spans one trading day; `ret_next_is_delisting`. The last training day's target ends in 2024 and is left empty (the embargo).

    **What we check** (after writing):
    - one panel row per stock and day;
    - every panel row meets the universe rule and rule 1, and traded at least $50,000 before the cutoff;
    - every `bars` and `cells` stock is in its day's panel;
    - for the kept stocks, rows and trades equal the pull's, piece by piece;
    - 2014-2023 in train and 2024-2025 in test;
    - no training target reaches into 2024.
    """)
    return


@app.cell
def _(
    ALL_DAYS,
    DATA_DIR,
    TEST_START,
    crsp_name_records,
    manifest,
    mo,
    pd,
    stock_days,
):
    _crsp_cols = {"dlycaldt": "date", "dlyprc": "price", "dlycap": "cap", "dlyprcvol": "dollar_volume", "dlyret": "ret",
                  "dlyretdurflg": "ret_span", "dlyopen": "open", "dlyfacprc": "facprc"}
    _crsp = pd.read_csv(DATA_DIR / "raw" / "crsp" / "stkdlysecuritydata_from_2013.csv", usecols=["permno", *_crsp_cols],
                        engine="pyarrow").rename(columns=_crsp_cols).assign(date=lambda d: pd.to_datetime(d["date"]))
    _dist = pd.read_csv(DATA_DIR / "raw" / "crsp" / "stkdistributions.csv", usecols=["permno", "disexdt", "disdivamt", "disfacpr"],
                        engine="pyarrow")
    _dividends = (_dist[_dist["disdivamt"] > 0].rename(columns={"disexdt": "date", "disdivamt": "div", "disfacpr": "facpr"})
                  .assign(date=lambda d: pd.to_datetime(d["date"])))  # stock_days keeps the amounts CRSP adds to its return
    _early = set(manifest.loc[manifest["close"] == "13:00", "day"].dt.strftime("%Y%m%d"))
    universe = stock_days(_crsp, crsp_name_records, ALL_DAYS, TEST_START, _early, dividends=_dividends)
    _per_day = universe.groupby(["decision_day", "security_type"]).size().unstack(fill_value=0)
    mo.vstack([
        mo.md(f"**The universe before rule 2** (which needs the tape, so the writer applies it): {len(universe):,} stock-days; stocks per day by security type (median by year):"),
        _per_day.groupby(_per_day.index.year).median().astype(int).rename(index=str),
    ])
    return (universe,)


@app.cell
def _(mo):
    write_by_decision_day = mo.ui.run_button(label="Write data/intermediate/taq/ (one to four hours)")
    write_by_decision_day
    return (write_by_decision_day,)


@app.cell
def _(
    ALL_DAYS,
    CODES,
    DATA_DIR,
    PULL_DIR,
    TEST_START,
    mo,
    stack_panels,
    symbols,
    universe,
    write_by_decision_day,
    write_decision_days,
):
    mo.stop(not write_by_decision_day.value, mo.md("Press the button to write."))
    DECISION_DIR = DATA_DIR / "intermediate" / "taq"
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    CODES.to_csv(DECISION_DIR / "codes.csv", index=False)
    _links = {_d.strftime("%Y%m%d"): _g[["symbol", "permno", "primary", "pilot_group"]] for _d, _g in symbols.groupby("day")}
    _universe = {_d.strftime("%Y%m%d"): _g for _d, _g in universe.groupby("decision_day")}
    _empty = universe.iloc[0:0]
    decision_days = write_decision_days(
        PULL_DIR, DECISION_DIR, ALL_DAYS, TEST_START, _links.get, lambda _day: _universe.get(_day, _empty),
        progress=lambda _pairs: mo.status.progress_bar(_pairs, title="Writing decision days", remove_on_exit=True),
    )
    stack_panels(DECISION_DIR)
    decision_days.groupby("period").agg(days=("decision_day", "size"), first=("decision_day", "min"), last=("decision_day", "max"),
                                        bars_rows=("bars_rows", "sum"), cells_rows=("cells_rows", "sum"))
    return DECISION_DIR, decision_days


@app.cell
def _(
    ALL_DAYS,
    CAP_MIN,
    DECISION_DIR,
    DOLLARS_BEFORE_CUTOFF_MIN,
    DOLLAR_VOLUME_MIN,
    PRICE_MIN,
    PULL_DIR,
    TEST_START,
    UNIVERSE_TYPES,
    decision_days,
    mo,
    pd,
    pq,
    symbols,
):
    _panels = {_p: pd.read_parquet(DECISION_DIR / f"panel_{_p}.parquet") for _p in ["train", "test"]}
    _panel = pd.concat(_panels.values(), ignore_index=True)
    _bad, _orphans, _thin = [], 0, 0
    _prim = symbols[symbols["primary"]][["day", "symbol", "permno"]]
    _prim = {_d.strftime("%Y%m%d"): _g.set_index("symbol")["permno"] for _d, _g in _prim.groupby("day")}
    _by_day = {_d.strftime("%Y%m%d"): set(_g["permno"]) for _d, _g in _panel.groupby("day")}
    for _prev, _day, _period in mo.status.progress_bar(
        list(zip(ALL_DAYS[:-1], ALL_DAYS[1:], decision_days["period"])), title="Checking", remove_on_exit=True
    ):
        _folder = DECISION_DIR / _period / _day
        _in = _by_day.get(_day, set())
        for _name in ["cells", "bars"]:
            _w = pq.read_table(_folder / f"{_name}.parquet", columns=["permno", "piece", "trades", "dollars"]).to_pandas()
            _orphans += int((~_w["permno"].isin(_in)).sum())
            if _name == "bars":
                _today = _w[_w["piece"].isin(["pre_open", "session"])].groupby("permno")["dollars"].sum()
                _thin += int(_today.reindex(list(_in)).fillna(0).lt(DOLLARS_BEFORE_CUTOFF_MIN).sum())
            _written = _w.groupby("piece")["trades"].agg(["size", "sum"])
            _expected = []
            for _src, _buckets, _pieces in [(_prev, [2, 3], {2: "prev_last10", 3: "prev_evening"}),
                                            (_day, [0, 1], {0: "pre_open", 1: "session"})]:
                _p = pq.read_table(PULL_DIR / _src / f"{_name}.parquet", columns=["symbol", "bucket", "trades"]).to_pandas()
                _p = _p[_p["bucket"].isin(_buckets) & _p["symbol"].map(_prim.get(_src, pd.Series(dtype="Int64"))).isin(_in)]
                _expected.append(_p.assign(piece=_p["bucket"].map(_pieces)).groupby("piece")["trades"].agg(["size", "sum"]))
            if not _written.sort_index().equals(pd.concat(_expected).sort_index()):
                _bad.append((_day, _name))
    _outside = ~(_panel["price_prev"].abs().ge(PRICE_MIN) & _panel["cap_prev"].ge(CAP_MIN)
                 & _panel["dollar_volume_prev"].ge(DOLLAR_VOLUME_MIN) & _panel["security_type"].isin(UNIVERSE_TYPES))
    _last_train = _panels["train"]["day"].max()
    decision_checks = pd.DataFrame([
        ("one folder per decision day, 2014-01-02 to 2025-12-31", list(decision_days["decision_day"]) == ALL_DAYS[1:]),
        ("every decision day has stocks in its panel", sorted(_by_day) == ALL_DAYS[1:]),
        ("2014-2023 in panel_train, 2024-2025 in panel_test",
         bool(_panels["train"]["day"].lt(pd.Timestamp(TEST_START)).all() and _panels["test"]["day"].ge(pd.Timestamp(TEST_START)).all())),
        ("one panel row per day and stock", not _panel.duplicated(["day", "permno"]).any()),
        ("every panel row meets the universe rule and rule 1", not _outside.any()),
        ("rule 2: every panel stock traded at least $50,000 before today's cutoff", _thin == 0),
        ("every bars and cells stock is in its day's panel", _orphans == 0),
        ("for the kept stocks, rows and trades equal the pull's, piece by piece", not _bad),
        ("no training target reaches into 2024", bool(_panels["train"].loc[_panels["train"]["day"] == _last_train, "ret_next"].isna().all())),
    ], columns=["check", "passed"])
    mo.vstack([decision_checks, mo.md(f"{len(_panel):,} stock-days; `ret_1550` empty on {int(_panel['ret_1550'].isna().sum()):,}, "
                                      f"`ret_next` empty on {int(_panel['ret_next'].isna().sum()):,}")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Training cleaning

    **Purpose:** look at the training days (`data/intermediate/taq/train/`, decision days 2014-01-02 to 2023-12-29) and decide what still needs cleaning before features are built. Every rule found here is chosen on training days only, written as tested code, and applied unchanged to the test days when features are built. The test folder stays closed.

    **Questions, in order:**

    1. **Bad prints:** are there broken prices (highs or lows far from the close and the quote), and does any feature need a rule for them?
    2. **The Tick Size Pilot (2016-10 to 2018-09):** do the pilot stocks' retail counts work as they are, or do they need the half-cent rule or masking?
    3. **Retail measures over time:** do the sub-penny (BJZZ) and midpoint rules behave sensibly, including around fractional shares (2021-02-16) and the late reports of November 2021?
    4. **Usable stocks:** how many stocks trade enough each day to be usable, as input to the universe.
    5. **The trading universe:** which stocks are liquid enough to trade, measured by what trading costs rather than by market cap (6.5).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 The training data: shape and health

    **Purpose:** before looking at any values, check what the training data holds: `panel_train.parquet` (one row per day and stock) and every day's `bars` and `cells` in `train/`, how many rows each table has per day, each column's type, how many values are empty, and each column's smallest and largest value. Read from the parquet files' own statistics, so nothing is loaded.

    **What we expect:** 2,516 days in the panel and in `train/`; empty values only where they are meaningful (`last_mid` when a bar had no valid quote; in the panel, `spread_bp` without a valid quote, `open_t` when CRSP has no open, `ret_next` when the target is missing or embargoed, `pilot_group` outside the pilot); no negative counts or prices.
    """)
    return


@app.cell
def _(DATA_DIR, mo, pd, pq):
    TRAIN_DIR = DATA_DIR / "intermediate" / "taq" / "train"
    TRAIN_PANEL = DATA_DIR / "intermediate" / "taq" / "panel_train.parquet"  # one row per training day and stock
    TRAIN_DECISION_DAYS = sorted(_p.name for _p in TRAIN_DIR.iterdir() if _p.name.isdigit())


    def _file_stats(path):
        """Rows of a parquet file and, per column, its type, empty values, smallest and largest value (from metadata)."""
        _meta = pq.ParquetFile(path).metadata
        _schema = pq.read_schema(path)
        _rows = []
        for _i, _field in enumerate(_schema):
            _nulls, _lo, _hi = 0, None, None
            for _g in range(_meta.num_row_groups):
                _s = _meta.row_group(_g).column(_i).statistics
                if _s is None:
                    continue
                _nulls += _s.null_count
                if _s.has_min_max:
                    _lo = _s.min if _lo is None else min(_lo, _s.min)
                    _hi = _s.max if _hi is None else max(_hi, _s.max)
            _rows.append({"column": _field.name, "type": str(_field.type), "nulls": _nulls, "min": _lo, "max": _hi})
        return _meta.num_rows, _rows


    _missing = [_d for _d in TRAIN_DECISION_DAYS if not all((TRAIN_DIR / _d / f"{_n}.parquet").exists() for _n in ["bars", "cells"])]
    _per_day, _cols = [], {}
    for _day in mo.status.progress_bar(TRAIN_DECISION_DAYS, title="Reading file statistics", remove_on_exit=True):
        for _name in ["bars", "cells"]:
            _n, _stats = _file_stats(TRAIN_DIR / _day / f"{_name}.parquet")
            _per_day.append({"day": _day, "table": _name, "rows": _n})
            for _r in _stats:
                _c = _cols.setdefault((_name, _r["column"]), {"type": _r["type"], "nulls": 0, "min": None, "max": None})
                _c["nulls"] += _r["nulls"]
                if _r["min"] is not None:
                    _c["min"] = _r["min"] if _c["min"] is None else min(_c["min"], _r["min"])
                    _c["max"] = _r["max"] if _c["max"] is None else max(_c["max"], _r["max"])
    _panel_days = pq.read_table(TRAIN_PANEL, columns=["day"]).to_pandas()["day"].dt.strftime("%Y%m%d").value_counts()
    _per_day += [{"day": _d, "table": "panel", "rows": int(_n)} for _d, _n in _panel_days.items()]
    _n, _stats = _file_stats(TRAIN_PANEL)
    for _r in _stats:
        _cols[("panel", _r["column"])] = {"type": _r["type"], "nulls": _r["nulls"], "min": _r["min"], "max": _r["max"]}
    train_rows = pd.DataFrame(_per_day)
    _totals = train_rows.groupby("table", sort=False)["rows"]
    train_shape = pd.DataFrame({"days": _totals.size(), "rows": _totals.sum(), "rows per day, median": _totals.median(),
                                "min": _totals.min(), "max": _totals.max()})
    train_columns = pd.DataFrame([{"table": _t, "column": _c, **_v} for (_t, _c), _v in _cols.items()])
    train_columns["% empty"] = (train_columns["nulls"] / train_columns["table"].map(_totals.sum()) * 100).round(3)
    train_columns[["min", "max"]] = train_columns[["min", "max"]].astype(str)
    mo.vstack([
        mo.md(f"**{len(TRAIN_DECISION_DAYS):,} training days; days missing `bars` or `cells`: {len(_missing)}; days in the panel: {len(_panel_days):,}**"),
        train_shape,
        mo.md("**Every column**: type, empty values, smallest and largest value over all training days"),
        mo.ui.table(train_columns[["table", "column", "type", "nulls", "% empty", "min", "max"]], selection=None, page_size=60),
    ])
    return TRAIN_DECISION_DAYS, TRAIN_DIR, TRAIN_PANEL


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Table heads

    The first rows of each table on one training day, 2019-06-04: `daily` (one row per universe stock), `bars` and `cells` (rows by stock, then in time order). Codes are explained in `data/intermediate/taq/codes.csv`.
    """)
    return


@app.cell
def _(TRAIN_DIR, TRAIN_PANEL, mo, pd, pq):
    _day = TRAIN_DIR / "20190604"
    _panel_day = pq.read_table(TRAIN_PANEL, filters=[("day", "=", pd.Timestamp("2019-06-04"))]).to_pandas()
    mo.vstack([
        mo.vstack([mo.md(f"**`panel_train`** on 2019-06-04: {len(_panel_day):,} rows"), _panel_day.head(15)]),
        *[mo.vstack([mo.md(f"**`{_name}`**: {pq.ParquetFile(_day / f'{_name}.parquet').metadata.num_rows:,} rows"),
                     pq.read_table(_day / f"{_name}.parquet").slice(0, 15).to_pandas()])
          for _name in ["bars", "cells"]],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 The market over time

    **Purpose:** see how trading, spreads, market value and the retail measures move over the training years, and whether anything breaks. One pass over the 2,516 training days sums `bars` across all universe stocks per day and summarises `daily`; every plot reads that table (`market_days`). A decision day's totals include the previous day's last 10 minutes and evening: everything known by its 15:50.

    **Marked on every plot:** the Tick Size Pilot (2016-10 to 2018-09, shaded), zero commissions (2019-10-07), COVID (2020-03), fractional shares printing as one-share trades (2021-02-16) and the meme surge (late January 2021).

    **Columns** (meanings in `explanations/taq/pulled_data_full.md`): `bjzz_*` retail trades by the sub-penny rule, `mid_*` by the midpoint rule, `off_*` all off-exchange, `qr_*` the quote rule on every venue and `exch_qr_*` on exchanges only, `both_signed` and `same_sign` how often the two retail rules sign the same trade and agree.
    """)
    return


@app.cell
def _(TRAIN_DECISION_DAYS, TRAIN_DIR, TRAIN_PANEL, mo, pd, pq):
    _bar_sums = ["trades", "shares", "dollars", "bjzz_buy_trades", "bjzz_buy_dollars", "bjzz_sell_trades", "bjzz_sell_dollars",
                 "mid_buy_trades", "mid_buy_dollars", "mid_sell_trades", "mid_sell_dollars", "off_trades", "off_dollars",
                 "off_odd_trades", "off_one_trades", "qr_buy_shares", "qr_sell_shares", "exch_qr_buy_shares",
                 "exch_qr_sell_shares", "both_signed_trades", "same_sign_trades"]
    _panel = pd.read_parquet(TRAIN_PANEL, columns=["day", "is_common", "cap_prev", "dollar_volume_prev", "spread_bp"])
    _panel_by_day = dict(tuple(_panel.groupby(_panel["day"].dt.strftime("%Y%m%d"))))
    _rows = []
    for _day in mo.status.progress_bar(TRAIN_DECISION_DAYS, title="Summing the training days", remove_on_exit=True):
        _b = pq.read_table(TRAIN_DIR / _day / "bars.parquet", columns=_bar_sums).to_pandas().sum()
        _d = _panel_by_day[_day]
        _w = _d["dollar_volume_prev"].where(_d["spread_bp"].notna())
        _rows.append({"day": pd.Timestamp(_day), **_b.to_dict(), "stocks": len(_d),
                      "cap_all": _d["cap_prev"].sum() * 1e3, "cap_common": _d.loc[_d["is_common"], "cap_prev"].sum() * 1e3,
                      "spread_median": _d["spread_bp"].median(), "spread_dollar_weighted": (_d["spread_bp"] * _w).sum() / _w.sum()})
    market_days = pd.DataFrame(_rows).set_index("day")
    market_days
    return (market_days,)


@app.cell
def _(PILOT_END, PILOT_START, pd):
    MARKET_EVENTS = {"zero commissions": "2019-10-07", "COVID": "2020-03-16", "meme surge": "2021-01-27",
                     "fractional shares print": "2021-02-16"}


    def mark_events(ax):
        """Shade the Tick Size Pilot and mark the market events on a time-series axis."""
        ax.axvspan(PILOT_START, PILOT_END, color="0.92", zorder=0)
        for _name, _day in MARKET_EVENTS.items():
            ax.axvline(pd.Timestamp(_day), color="0.5", lw=0.7, ls="--", zorder=1)


    def two_axes(ax, left, right, left_label, right_label, title):
        """Series on two y axes: `left` and `right` are {label: series}; left in solid lines, right in dashed."""
        mark_events(ax)
        for _i, (_label, _s) in enumerate(left.items()):
            ax.plot(_s.index, _s, lw=0.6, color=f"C{_i}", label=f"{_label} (left)")
        ax.set_ylabel(left_label)
        _r = ax.twinx()
        for _i, (_label, _s) in enumerate(right.items()):
            _r.plot(_s.index, _s, lw=0.6, ls="--", color=f"C{_i + len(left)}", label=f"{_label} (right)")
        _r.set_ylabel(right_label)
        _h = ax.get_legend_handles_labels()[0] + _r.get_legend_handles_labels()[0]
        ax.legend(_h, [_x.get_label() for _x in _h], fontsize=7, loc="upper left")
        ax.set_title(title, fontsize=10)

    return mark_events, two_axes


@app.cell
def _(mark_events, market_days, plt, two_axes):
    _m = market_days
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    two_axes(_axes[0, 0], {"trades, millions": _m["trades"] / 1e6}, {"dollars, $ billions": _m["dollars"] / 1e9},
             "trades (millions)", "dollars ($ billions)", "All trading by the universe stocks per day")
    two_axes(_axes[0, 1], {"stocks in the universe": _m["stocks"]}, {},
             "stocks", "", "Stocks in the universe")
    mark_events(_axes[1, 0])
    _axes[1, 0].plot(_m.index, _m["spread_median"], lw=0.6, label="median stock")
    _axes[1, 0].plot(_m.index, _m["spread_dollar_weighted"], lw=0.6, label="weighted by previous dollar volume")
    _axes[1, 0].set_ylabel("bp"); _axes[1, 0].legend(fontsize=7); _axes[1, 0].set_title("Quoted spread to 15:50", fontsize=10)
    two_axes(_axes[1, 1], {"all universe stocks": _m["cap_all"] / 1e12, "US common stock": _m["cap_common"] / 1e12}, {},
             "$ trillions", "", "Total market cap, previous close")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(market_days, plt, two_axes):
    _m = market_days
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    two_axes(_axes[0], {"buys": _m["bjzz_buy_trades"] / 1e6, "sells": _m["bjzz_sell_trades"] / 1e6},
             {"buys": _m["bjzz_buy_dollars"] / 1e9, "sells": _m["bjzz_sell_dollars"] / 1e9},
             "trades (millions)", "dollars ($ billions)", "Retail by the sub-penny rule (BJZZ)")
    two_axes(_axes[1], {"buys": _m["mid_buy_trades"] / 1e6, "sells": _m["mid_sell_trades"] / 1e6},
             {"buys": _m["mid_buy_dollars"] / 1e9, "sells": _m["mid_sell_dollars"] / 1e9},
             "trades (millions)", "dollars ($ billions)", "Retail by the midpoint rule")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(market_days, plt, two_axes):
    _m = market_days
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    two_axes(_axes[0, 0], {"trades, millions": _m["off_trades"] / 1e6}, {"dollars, $ billions": _m["off_dollars"] / 1e9},
             "trades (millions)", "dollars ($ billions)", "Off-exchange trading")
    two_axes(_axes[0, 1], {"odd lots (1-99 shares)": _m["off_odd_trades"] / 1e6, "single shares": _m["off_one_trades"] / 1e6}, {},
             "trades (millions)", "", "Off-exchange odd lots and single-share trades")
    two_axes(_axes[1, 0], {"buys": _m["qr_buy_shares"] / 1e9, "sells": _m["qr_sell_shares"] / 1e9}, {},
             "shares (billions)", "", "Quote rule, every venue")
    two_axes(_axes[1, 1], {"buys": _m["exch_qr_buy_shares"] / 1e9, "sells": _m["exch_qr_sell_shares"] / 1e9}, {},
             "shares (billions)", "", "Quote rule, exchanges only")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(market_days, plt, two_axes):
    _m = market_days
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    two_axes(_axes[0], {"signed by both rules": _m["both_signed_trades"] / 1e6, "same sign": _m["same_sign_trades"] / 1e6},
             {"agreement, same / both": _m["same_sign_trades"] / _m["both_signed_trades"]},
             "trades (millions)", "share", "Agreement of the two retail rules")
    _ratios = {
        "off-exchange / all trades": _m["off_trades"] / _m["trades"],
        "BJZZ-signed / off-exchange": (_m["bjzz_buy_trades"] + _m["bjzz_sell_trades"]) / _m["off_trades"],
        "midpoint-signed / off-exchange": (_m["mid_buy_trades"] + _m["mid_sell_trades"]) / _m["off_trades"],
        "odd lots / off-exchange": _m["off_odd_trades"] / _m["off_trades"],
        "single shares / off-exchange": _m["off_one_trades"] / _m["off_trades"],
        "BJZZ buys / (buys + sells)": _m["bjzz_buy_trades"] / (_m["bjzz_buy_trades"] + _m["bjzz_sell_trades"]),
    }
    two_axes(_axes[1], _ratios, {}, "share of trades", "", "Shares of trading")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Comparing the retail measures.** Four measures of retail trading in `bars`: the sub-penny rule (BJZZ) and the midpoint rule, each as buys and sells, and two that do not use sub-penny prices: off-exchange odd lots (1 to 99 shares; retail trades small) and off-exchange single-share trades (mostly fractional orders from 2021). The quote rule is left out: it signs all trades, not retail ones.

    1. **Levels**: each measure divided by its 2014 average (21-day average, log scale), with the median stock's spread on the right axis.
    2. **Day-to-day agreement**: the correlation of the measures' daily percentage changes, over a rolling quarter (63 days).
    3. **Direction**: the buy-sell imbalance, (buys - sells) / (buys + sells), by each rule (21-day average), which is what a retail flow signal would use.

    Below, the correlation of the daily percentage changes over all training days.
    """)
    return


@app.cell
def _(market_days, mo, pd, plt, two_axes):
    _m = market_days
    _retail = pd.DataFrame({
        "BJZZ buys": _m["bjzz_buy_trades"], "BJZZ sells": _m["bjzz_sell_trades"],
        "midpoint buys": _m["mid_buy_trades"], "midpoint sells": _m["mid_sell_trades"],
        "odd lots": _m["off_odd_trades"], "single shares": _m["off_one_trades"],
    })
    _level = _retail.rolling(21).mean() / _retail[_retail.index.year == 2014].mean()
    _change = _retail.pct_change()
    _fig, _axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    two_axes(_axes[0], {_c: _level[_c] for _c in _level}, {"median spread, bp": _m["spread_median"].rolling(21).mean()},
             "multiple of the 2014 average (log)", "spread (bp)", "Retail measures, level")
    _axes[0].set_yscale("log")
    _pairs = {"BJZZ vs midpoint (buys + sells)": (_retail["BJZZ buys"] + _retail["BJZZ sells"], _retail["midpoint buys"] + _retail["midpoint sells"]),
              "BJZZ vs odd lots": (_retail["BJZZ buys"] + _retail["BJZZ sells"], _retail["odd lots"]),
              "midpoint vs odd lots": (_retail["midpoint buys"] + _retail["midpoint sells"], _retail["odd lots"])}
    two_axes(_axes[1], {_k: _a.pct_change().rolling(63).corr(_b.pct_change()) for _k, (_a, _b) in _pairs.items()}, {},
             "correlation of daily % changes", "", "Day-to-day agreement of the measures (rolling 63 days)")
    _imb = {"BJZZ": (_retail["BJZZ buys"] - _retail["BJZZ sells"]) / (_retail["BJZZ buys"] + _retail["BJZZ sells"]),
            "midpoint": (_retail["midpoint buys"] - _retail["midpoint sells"]) / (_retail["midpoint buys"] + _retail["midpoint sells"])}
    two_axes(_axes[2], {_k: _v.rolling(21).mean() for _k, _v in _imb.items()}, {}, "imbalance", "",
             "Retail buy-sell imbalance, (buys - sells) / (buys + sells)")
    _axes[2].axhline(0, color="0.3", lw=0.6)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("**Correlation of the daily % changes, all training days**"), _change.corr().round(2)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.4 Empirical distributions

    **A. The stock-day variables in `daily`**, over every training stock-day (14.2 million): the previous close, market cap and dollar volume (log scale), the spread to 15:50, the previous-day return, the target `ret_next`, and the overnight gap (CRSP's open over the previous close). Returns are shown in the middle 98% with the tails counted in the table, which gives each variable's quantiles from 0.01% to 99.99% with the minimum and maximum.
    """)
    return


@app.cell
def _(TRAIN_PANEL, mo, pd):
    train_daily = pd.read_parquet(TRAIN_PANEL, columns=[
        "day", "permno", "security_type", "pilot_group", "price_prev", "cap_prev", "dollar_volume_prev", "ret_prev", "open_t",
        "facprc_t", "spread_bp", "ret_1550", "ret_next"])
    train_daily["gap"] = train_daily["open_t"] / train_daily["price_prev"] - 1
    mo.md(f"{len(train_daily):,} training stock-days")
    return (train_daily,)


@app.cell
def _(mo, pd, plt, train_daily):
    _q = [0.0001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.9999]
    _vars = {"previous close ($)": train_daily["price_prev"], "previous market cap ($ millions)": train_daily["cap_prev"] / 1e3,
             "previous dollar volume ($ millions)": train_daily["dollar_volume_prev"] / 1e6, "spread to 15:50 (bp)": train_daily["spread_bp"],
             "previous-day return": train_daily["ret_prev"], "target: next-day return": train_daily["ret_next"],
             "overnight gap: open / previous close - 1": train_daily["gap"]}
    _skewed = {"previous close ($)", "previous market cap ($ millions)", "previous dollar volume ($ millions)", "spread to 15:50 (bp)"}
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 7))
    for _ax, (_name, _s) in zip(_axes.flat, _vars.items()):
        _s = _s.dropna()
        if _name in _skewed:
            _hi = _s.quantile(0.95)
            _ax.hist(_s, bins=100, range=(0, _hi), color="C0")
            _ax.set_xlabel(f"up to its 95th percentile ({_hi:,.1f}); the top 5% not shown")
        else:
            _lo, _hi = _s.quantile([0.01, 0.99])
            _ax.hist(_s.clip(_lo, _hi), bins=100, color="C1")
            _ax.set_xlabel(f"{_name} (1% and 99% tails clipped)")
        _ax.set_yticks([])
        _ax.set_title(_name, fontsize=9)
    _axes.flat[-1].axis("off")
    _fig.tight_layout()
    _table = pd.DataFrame({_n: _s.quantile(_q) for _n, _s in _vars.items()}).T
    _table.columns = [f"{_x:.2%}" for _x in _q]
    _table["min"] = [_s.min() for _s in _vars.values()]
    _table["max"] = [_s.max() for _s in _vars.values()]
    _table["empty"] = [int(_s.isna().sum()) for _s in _vars.values()]
    mo.vstack([_fig, _table.round(4)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Cleaning rule 1: a floor on the previous day's dollar volume, $100,000** (`dsr.taq_split.DOLLAR_VOLUME_MIN`, chosen on the training days). Of the universe's stock-days, 2.5% had under $10,000 traded the day before and 114,643 had none: above $50M of market value, but not trading. The floor keeps 92% of stock-days and 99.8% of market value (about 4,900 stocks a day) and cuts the quoted spreads above 400 bp by two thirds; higher floors start removing ordinary small caps, where retail trades. Every later look at the training data, and the final clean data, use only the stock-days that pass it.

    **The universe's market cap floor is now $100M** (`dsr.taq_split.CAP_MIN`, raised from $50M on 2026-10-11). The written data still holds the $50M universe, so the line below applies both. The higher floor keeps 92.7% of the rule-1 stock-days (about 4,600 stocks a day instead of 4,900) and 99.9% of market value; ETFs lose the most (85% kept). It removes few SPACs, whose trusts are mostly $100M to $350M.

    **ETFs are out of the universe** (`dsr.taq_split.UNIVERSE_TYPES`, since 2026-10-11): they were 22% of the stock-days, largely duplicates of one another (SPY, IVV, VOO) and market or sector exposure rather than stocks, with leveraged and inverse funds that CRSP's issuer names cannot single out. The written data still holds them, so the line below drops them.

    **Cleaning rule 2: at least $50,000 traded before 15:50, on the decision day and on the trading day before** (`dsr.taq_split.DOLLARS_BEFORE_CUTOFF_MIN`, chosen on the training days; pre-open and session, from the pull's `bars` and each day's primary symbol, so a stock new to the universe is measured too). Known at the cutoff, so no lookahead. Today's half drops stocks that have barely traded by 15:50 (about 1% of stock-days); yesterday's half guarantees a price at both ends of `ret_1550` (rule 1 alone does not: CRSP's full-day volume counts trades after 15:50, as on halted days or the 2022-09-30 SPAC liquidations).

    SPACs (blank check companies, near their $10 trust value with near-zero returns) stay in: CRSP classifies them as common stock and codes them reliably as such only until about 2019, and no rule in this data catches them cleanly; their pinned price shows in ordinary features (volatility, return history).
    """)
    return


@app.cell
def _(ALL_DAYS, PULL_DIR, TEST_START, mo, pd, pq, symbols):
    _prim = symbols.loc[symbols["primary"], ["day", "symbol", "permno"]]
    _days = ALL_DAYS[: ALL_DAYS.index(TEST_START)]  # every WCT day whose pre-open and session belong to a training decision day
    _parts = []
    for _d in mo.status.progress_bar(_days, title="Reading the pull's bars", remove_on_exit=True):
        _b = pq.read_table(PULL_DIR / _d / "bars.parquet", columns=["symbol", "bucket", "dollars"],
                           filters=[("bucket", "in", [0, 1])]).to_pandas()
        _parts.append(_b.groupby("symbol", as_index=False)["dollars"].sum().assign(day=pd.Timestamp(_d)))
    _w = pd.concat(_parts, ignore_index=True).merge(_prim, on=["day", "symbol"])[["day", "permno", "dollars"]]
    _next = pd.Series(pd.to_datetime(_days[1:]), index=pd.to_datetime(_days[:-1]))
    train_dollars_before_cutoff = (  # per training decision day and PERMNO: dollars before the cutoff that day and the day before
        _w.rename(columns={"dollars": "dollars_before_cutoff"})
        .merge(_w.assign(day=_w["day"].map(_next)).dropna(subset=["day"]).rename(columns={"dollars": "dollars_before_cutoff_prev"}),
               on=["day", "permno"], how="outer")
    )
    return (train_dollars_before_cutoff,)


@app.cell
def _(mo, train_daily, train_dollars_before_cutoff):
    train_daily_clean = (
        train_daily.merge(train_dollars_before_cutoff, on=["day", "permno"], how="left")
        .fillna({"dollars_before_cutoff": 0, "dollars_before_cutoff_prev": 0})
        .query("dollar_volume_prev >= @DOLLAR_VOLUME_MIN and cap_prev >= @CAP_MIN and dollars_before_cutoff >= @DOLLARS_BEFORE_CUTOFF_MIN"
               " and dollars_before_cutoff_prev >= @DOLLARS_BEFORE_CUTOFF_MIN"
               " and security_type in @UNIVERSE_TYPES")
    )
    mo.md(f"**After rules 1 and 2, the $100M floor and without ETFs:** {len(train_daily_clean):,} of {len(train_daily):,} training stock-days "
          f"({len(train_daily_clean) / len(train_daily):.1%}), {int(train_daily_clean.groupby('day').size().median()):,} stocks a day")
    return (train_daily_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **B. Activity per stock and day, from `bars`**, on the 120 sample days (one training day a month), for the stock-days that pass rule 1. Everything known by 15:50 (all four pieces), except the session bars with a trade (of 38, or 20 on early closes):
    - trades and dollars traded;
    - retail trades signed by each rule (buys + sells), and each rule's buy-sell imbalance, (buys - sells) / (buys + sells), for stocks with at least one signed trade;
    - the off-exchange share of trades, and the odd-lot share of off-exchange trades.
    """)
    return


@app.cell
def _(SAMPLE_DAYS, TRAIN_DIR, mo, pd, plt, pq, train_daily_clean):
    _cols = ["permno", "piece", "trades", "dollars", "bjzz_buy_trades", "bjzz_sell_trades", "mid_buy_trades", "mid_sell_trades",
             "off_trades", "off_odd_trades"]
    _keep = train_daily_clean[["day", "permno"]]
    _parts = []
    for _d in mo.status.progress_bar(SAMPLE_DAYS, title="Reading sample days", remove_on_exit=True):
        _b = pq.read_table(TRAIN_DIR / _d / "bars.parquet", columns=_cols).to_pandas()
        _s = _b.groupby("permno")[_cols[2:]].sum()
        _s["session_bars"] = _b[_b["piece"] == "session"].groupby("permno").size()
        _parts.append(_s.reset_index().assign(day=pd.Timestamp(_d)))
    stock_activity = pd.concat(_parts, ignore_index=True).merge(_keep, on=["day", "permno"])
    stock_activity["session_bars"] = stock_activity["session_bars"].fillna(0)
    _a = stock_activity
    _bjzz, _mid = _a["bjzz_buy_trades"] + _a["bjzz_sell_trades"], _a["mid_buy_trades"] + _a["mid_sell_trades"]
    _vars = {
        "trades per stock-day": _a["trades"], "dollars traded per stock-day ($ millions)": _a["dollars"] / 1e6,
        "BJZZ-signed retail trades": _bjzz, "midpoint-signed retail trades": _mid,
        "BJZZ imbalance": ((_a["bjzz_buy_trades"] - _a["bjzz_sell_trades"]) / _bjzz).where(_bjzz > 0),
        "midpoint imbalance": ((_a["mid_buy_trades"] - _a["mid_sell_trades"]) / _mid).where(_mid > 0),
        "off-exchange share of trades": _a["off_trades"] / _a["trades"],
        "odd-lot share of off-exchange": (_a["off_odd_trades"] / _a["off_trades"]).where(_a["off_trades"] > 0),
        "session bars with a trade": _a["session_bars"],
    }
    _skewed = {"trades per stock-day", "dollars traded per stock-day ($ millions)", "BJZZ-signed retail trades", "midpoint-signed retail trades"}
    _fig, _axes = plt.subplots(3, 3, figsize=(15, 10))
    for _ax, (_name, _s) in zip(_axes.flat, _vars.items()):
        _s = _s.dropna()
        if _name in _skewed:
            _hi = _s.quantile(0.95)
            _ax.hist(_s, bins=100, range=(0, _hi))
            _ax.set_xlabel(f"up to its 95th percentile ({_hi:,.1f}); the top 5% not shown; {(_s == 0).mean():.1%} are 0")
        else:
            _ax.hist(_s, bins=100, color="C1")
        _ax.set_yticks([])
        _ax.set_title(_name, fontsize=9)
    _fig.tight_layout()
    _q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    _table = pd.DataFrame({_n: _s.quantile(_q) for _n, _s in _vars.items()}).T
    _table.columns = [f"{_x:.0%}" for _x in _q]
    _table["mean"] = [_s.mean() for _s in _vars.values()]
    mo.vstack([mo.md(f"{len(_a):,} stock-days on {_a['day'].nunique()} sample days"), _fig, _table.round(3)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **C. Prices within a bar: bad prints and broken quotes**, on the 120 sample days, for the stock-days that pass rule 1. Per bar:
    - **close against the midpoint**, |log(close / last_mid)|: how far the last trade is from the quote's midpoint at that trade. Large values are either a bad print or a broken (stub) quote;
    - **the bar's range**, log(high / low): large values in a bar of a few trades suggest a bad print.

    By piece, since the evening and pre-open trade thinly against wider quotes. The table counts bars beyond 5%, 10%, 50% and 3 times, and the extremes are listed to see which side is wrong.
    """)
    return


@app.cell
def _(SAMPLE_DAYS, TRAIN_DIR, mo, np, pd, plt, pq, train_daily_clean):
    _cols = ["permno", "piece", "bar", "open", "high", "low", "close", "last_mid", "trades", "dollars"]
    _keep = train_daily_clean[["day", "permno"]]
    _parts = [pq.read_table(TRAIN_DIR / _d / "bars.parquet", columns=_cols).to_pandas().assign(day=pd.Timestamp(_d))
              for _d in mo.status.progress_bar(SAMPLE_DAYS, title="Reading sample days", remove_on_exit=True)]
    bar_prices = pd.concat(_parts, ignore_index=True).merge(_keep, on=["day", "permno"])
    for _c in ["open", "high", "low", "close"]:
        bar_prices[_c] = bar_prices[_c].astype(float)
    bar_prices["close_vs_mid"] = np.log(bar_prices["close"] / bar_prices["last_mid"]).abs()
    bar_prices["range"] = np.log(bar_prices["high"] / bar_prices["low"])
    _pieces = ["prev_last10", "prev_evening", "pre_open", "session"]
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 6))
    for _j, _p in enumerate(_pieces):
        _x = bar_prices[bar_prices["piece"] == _p]
        for _i, _v in enumerate(["close_vs_mid", "range"]):
            _s = _x[_v].dropna()
            _hi = _s.quantile(0.99) * 100
            _axes[_i, _j].hist(_s * 100, bins=100, range=(0, _hi), color=f"C{_i}")
            _axes[_i, _j].set_yticks([])
            _axes[_i, _j].set_title(f"{_p}: {_v}", fontsize=9)
            _axes[_i, _j].set_xlabel(f"% (about), up to its 99th percentile ({_hi:.2f}%); {(_s == 0).mean():.0%} are 0")
    _fig.tight_layout()
    _rows = []
    for _p in _pieces:
        _x = bar_prices[bar_prices["piece"] == _p]
        for _v in ["close_vs_mid", "range"]:
            _s = _x[_v]
            _rows.append({"piece": _p, "measure": _v, "bars": len(_x), "empty": int(_s.isna().sum()),
                          **{f"beyond {_lab}": int((_s > np.log(_k)).sum()) for _lab, _k in
                             [("5%", 1.05), ("10%", 1.10), ("50%", 1.5), ("3x", 3.0)]},
                          "median": _s.median(), "99.9%": _s.quantile(0.999), "max": _s.max()})
    bar_price_tails = pd.DataFrame(_rows)
    mo.vstack([mo.md(f"{len(bar_prices):,} bars"), _fig, bar_price_tails.round(4)])
    return (bar_prices,)


@app.cell
def _(bar_prices, mo):
    _top = lambda _v: bar_prices.nlargest(12, _v)[["day", "permno", "piece", "bar", "open", "high", "low", "close", "last_mid", "trades", "dollars"]]
    mo.vstack([mo.md("**Largest close against midpoint**"), _top("close_vs_mid"), mo.md("**Largest range**"), _top("range")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **D. Where retail-sized off-exchange trades are priced**, from `cells` on the 120 sample days, for the stock-days that pass rule 1: off-exchange trades under $100,000 (the pool both retail rules sign), all four pieces. Shares of trades:

    1. **within the cent** (`z`, the fraction of a cent in the price), by year: BJZZ signs a buy when the fraction is above .6 and a sell below .4;
    2. **within the quote** (`q`, the position between bid and ask), by year, for sub-penny trades with a valid quote: the midpoint rule signs a buy above .6 of the spread and a sell below .4;
    3. **both at once**, at one-cent spreads and at wider ones: where the two rules can disagree;
    4. **the Tick Size Pilot**: `z` for the pilot's groups on the sample days inside it. Groups 2 and 3 quoted in 5-cent steps and could improve retail prices only by half a cent, so their retail trades should pile up at exactly .5 (code 6).
    """)
    return


@app.cell
def _(CODES, SAMPLE_DAYS, TRAIN_DIR, mo, pd, plt, pq, train_daily_clean):
    _cols = ["permno", "venue", "z", "q", "spread", "size", "trades"]
    _parts = []
    for _d in mo.status.progress_bar(SAMPLE_DAYS, title="Reading sample days", remove_on_exit=True):
        _c = pq.read_table(TRAIN_DIR / _d / "cells.parquet", columns=_cols,
                           filters=[("venue", "=", 1), ("size", "!=", 4)]).to_pandas()
        _c = _c.merge(train_daily_clean.loc[train_daily_clean["day"] == pd.Timestamp(_d), ["permno", "pilot_group"]], on="permno")
        _parts.append(_c.groupby(["z", "q", "spread", "pilot_group"], dropna=False)["trades"].sum().reset_index()
                      .assign(day=pd.Timestamp(_d)))
    retail_cells = pd.concat(_parts, ignore_index=True)
    retail_cells["year"] = retail_cells["day"].dt.year
    _codes = lambda _col: CODES.loc[CODES["column"] == _col].set_index("code")["label"]


    def _share_heatmap(_ax, _table, _title, _labels):
        _sh = _table.div(_table.sum(axis=1), axis=0)
        _im = _ax.imshow(_sh.to_numpy(), aspect="auto", cmap="viridis")
        _ax.set_yticks(range(len(_sh)), [str(_i) for _i in _sh.index], fontsize=7)
        _ax.set_xticks(range(len(_sh.columns)), [_labels.get(_c, _c) for _c in _sh.columns], rotation=90, fontsize=7)
        _ax.set_title(_title, fontsize=10)
        plt.colorbar(_im, ax=_ax, fraction=0.03)


    _r = retail_cells
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 11))
    _share_heatmap(_axes[0, 0], _r.pivot_table(index="year", columns="z", values="trades", aggfunc="sum"),
                   "Within the cent (z), share of the year's trades", _codes("z").to_dict())
    _sub = _r[(_r["z"] != 0) & (_r["q"] != 0)]
    _share_heatmap(_axes[0, 1], _sub.pivot_table(index="year", columns="q", values="trades", aggfunc="sum"),
                   "Within the quote (q), sub-penny trades with a valid quote", _codes("q").to_dict())
    for _ax, (_name, _m) in zip(_axes[1], [("one-cent spreads", _sub["spread"] == 1), ("spreads over one cent", _sub["spread"] > 1)]):
        _share_heatmap(_ax, _sub[_m].pivot_table(index="z", columns="q", values="trades", aggfunc="sum").rename(index=_codes("z").to_dict()),
                       f"z (rows) by q (columns), {_name}, share within each z", _codes("q").to_dict())
    _fig.tight_layout()
    _fig
    return (retail_cells,)


@app.cell
def _(CODES, PILOT_END, PILOT_START, mo, retail_cells):
    _p = retail_cells[retail_cells["day"].between(PILOT_START, PILOT_END)]
    _t = _p.assign(group=_p["pilot_group"].fillna("not in the pilot")).pivot_table(
        index="group", columns="z", values="trades", aggfunc="sum").fillna(0)
    _t = (_t.div(_t.sum(axis=1), axis=0) * 100).round(1).rename(columns=CODES.loc[CODES["column"] == "z"].set_index("code")["label"])
    mo.vstack([mo.md(f"**z inside the Tick Size Pilot**, % of each group's off-exchange trades under $100,000 "
                     f"({_p['day'].nunique()} sample days, 2016-10 to 2018-09)"), _t])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **E. The return to 15:50, `ret_1550`**, planned for the clean write: the last trade before the cutoff today, times today's split factor, over the last trade before the cutoff on the previous trading day, minus 1. On the 120 sample days, for the stock-days that pass the rules. Prices from the pull's `bars` (pre-open and session, the primary symbol of each day), so a stock that entered the universe today still gets yesterday's price.

    What we check: how often it is empty (no trade before 15:50 on either day); that the split factor goes the right way (CRSP's own return equals close × `facprc_t` / previous close - 1); and how close it is to CRSP's close-to-close return, which it should track except for the last 10 minutes.
    """)
    return


@app.cell
def _(
    ALL_DAYS,
    PULL_DIR,
    SAMPLE_DAYS,
    mo,
    pd,
    plt,
    pq,
    symbols,
    train_daily,
    train_daily_clean,
):
    def _last_before_cutoff(_day):
        """Per PERMNO, the price of the last trade before the day's cutoff (pre-open and session), from the pull."""
        _b = pq.read_table(PULL_DIR / _day / "bars.parquet", columns=["symbol", "bucket", "bar", "close", "trades"],
                           filters=[("bucket", "in", [0, 1])]).to_pandas()
        _b = _b[_b["trades"] > 0].sort_values(["symbol", "bucket", "bar"]).groupby("symbol").tail(1)
        _p = symbols.loc[symbols["primary"] & (symbols["day"] == pd.Timestamp(_day)), ["symbol", "permno"]]
        return _b.merge(_p, on="symbol")[["permno", "close"]].astype({"close": float})


    _rows = []
    for _d in mo.status.progress_bar(SAMPLE_DAYS, title="Reading sample days", remove_on_exit=True):
        _i = ALL_DAYS.index(_d)
        _u = train_daily_clean.loc[train_daily_clean["day"] == pd.Timestamp(_d), ["day", "permno", "price_prev", "facprc_t"]]
        _crsp = train_daily.loc[train_daily["day"] == pd.Timestamp(ALL_DAYS[_i + 1]), ["permno", "ret_prev", "price_prev"]]
        _rows.append(_u.merge(_last_before_cutoff(_d).rename(columns={"close": "price_1550"}), on="permno", how="left")
                     .merge(_last_before_cutoff(ALL_DAYS[_i - 1]).rename(columns={"close": "price_1550_prev"}), on="permno", how="left")
                     .merge(_crsp.rename(columns={"ret_prev": "crsp_ret", "price_prev": "crsp_close"}), on="permno", how="left"))
    ret_1550_check = pd.concat(_rows, ignore_index=True)
    ret_1550_check["ret_1550"] = ret_1550_check["price_1550"] * ret_1550_check["facprc_t"] / ret_1550_check["price_1550_prev"] - 1
    _r = ret_1550_check
    _ok = _r.dropna(subset=["ret_1550", "crsp_ret"])
    _gap = (_ok["ret_1550"] - _ok["crsp_ret"]).abs()
    _factor = (_r["crsp_close"].abs() * _r["facprc_t"] / _r["price_prev"].abs() - 1 - _r["crsp_ret"]).abs()
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _lim = 0.3
    _ax.hexbin(_ok["crsp_ret"].clip(-_lim, _lim), _ok["ret_1550"].clip(-_lim, _lim), gridsize=120, bins="log", cmap="viridis")
    _ax.plot([-_lim, _lim], [-_lim, _lim], color="w", lw=0.6)
    _ax.set_xlabel("CRSP close-to-close return (clipped at ±30%)")
    _ax.set_ylabel("ret_1550 (clipped at ±30%)")
    _summary = pd.DataFrame([
        ("stock-days", len(_r)),
        ("no trade before 15:50 today", int(_r["price_1550"].isna().sum())),
        ("no trade before 15:50 yesterday", int(_r["price_1550_prev"].isna().sum())),
        ("split-factor days (facprc_t ≠ 1)", int(_r["facprc_t"].ne(1).sum())),
        ("CRSP's return = close × facprc_t / previous close - 1, median gap", float(_factor.median())),
        ("correlation with CRSP's return (Pearson)", round(_ok["ret_1550"].corr(_ok["crsp_ret"]), 3)),
        ("correlation with CRSP's return (Spearman)", round(_ok["ret_1550"].corr(_ok["crsp_ret"], method="spearman"), 3)),
        ("|ret_1550 - CRSP's return| > 10%", int((_gap > 0.10).sum())),
        ("|ret_1550 - CRSP's return| > 50%", int((_gap > 0.50).sum())),
    ], columns=["check", "value"])
    _largest = _ok.assign(gap=_gap).nlargest(8, "gap")[["day", "permno", "price_prev", "price_1550_prev", "price_1550", "crsp_close",
                                                          "facprc_t", "ret_1550", "crsp_ret"]]
    mo.vstack([mo.hstack([_fig, _summary], justify="start"), mo.md("**Largest gaps from CRSP's return**"), _largest])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.5 The trading universe: what makes a stock liquid

    **Purpose:** the models were scored and traded on the stocks whose previous market cap is at least $1B (until v3), and the target was demeaned separately within that group. Market cap stands in for what matters to a strategy that trades at the close every day: what a stock costs to trade, and how much of it can be traded. This section measures those directly, per stock and day, and compares candidate rules for the trading universe. The rules are compared on liquidity only, not on returns or model results, so the choice is made on cost grounds rather than on which universe backtests best. Training days only, on the stock-days that pass the cleaning rules (`train_daily_clean`).

    **The measures**, each known by the decision day's 15:50:
    - `cap`: the previous close's market cap, the current rule;
    - `spread_today`: today's relative quoted spread, (ask − bid) / midpoint, averaged over the session's trades up to 15:50 (`spread_bp`). The spread at exactly 15:50 is not in the clean data (`bars` keeps only the midpoint) and would need a new pull;
    - `spread_20d`: the median of `spread_today` over the last 20 trading days, today included, with at least 15 of them in the panel; steadier than a single day;
    - `dollar_volume_20d`: the average CRSP dollar volume over the 20 trading days before, closing auctions included (the pull leaves auctions out): how much can be traded;
    - `amihud_20d`: the average of |return| / dollar volume over the same days (Amihud 2002), in bp per $1M traded: how far trading moves the price.

    **The rules compared** each take, every day, as many stocks as the $1B rule does, the most liquid by one measure, so all have the same breadth. For each rule: how many of its stocks are also above $1B, the spreads it admits (median and 90th percentile), the share of the day's dollar volume it covers, and how many of its names change from one day to the next, which a strategy pays for in trades.
    """)
    return


@app.cell
def _(pd, train_daily, train_daily_clean):
    _calendar = pd.DatetimeIndex(sorted(train_daily["day"].unique()))
    _history = train_daily[["day", "permno"]].assign(  # every written stock-day, so the windows see the whole history
        spread_today=train_daily["spread_bp"],
        dollar_volume=train_daily["dollar_volume_prev"],  # the day before's, so a window ending today covers t-20 to t-1
        amihud=train_daily["ret_prev"].abs() / train_daily["dollar_volume_prev"] * 1e10,  # bp per $1M traded
    )
    _keys = pd.MultiIndex.from_frame(train_daily_clean[["day", "permno"]])


    def _trailing(_column, _how, _days=20, _at_least=15):
        """Each stock's mean or median of a column over its last `_days` rows on the calendar, the current one
        included, with at least `_at_least` of them in the panel (days out of the panel count as missing)."""
        _wide = _history.pivot(index="day", columns="permno", values=_column).reindex(_calendar)
        _rolled = getattr(_wide.rolling(_days, min_periods=_at_least), _how)()
        return _rolled.stack().reindex(_keys).to_numpy()


    liquidity = train_daily_clean[["day", "permno", "security_type", "price_prev", "dollar_volume_prev"]].assign(
        cap=train_daily_clean["cap_prev"] / 1e3,  # $ millions
        large=train_daily_clean["cap_prev"] >= 1_000_000,  # the $1B rule (CRSP cap in $ thousands)
        spread_today=train_daily_clean["spread_bp"],
        spread_20d=_trailing("spread_today", "median"),
        dollar_volume_20d=_trailing("dollar_volume", "mean"),
        amihud_20d=_trailing("amihud", "mean"),
    ).reset_index(drop=True)
    _q = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    _measures = {"cap ($ millions)": liquidity["cap"], "spread today (bp)": liquidity["spread_today"],
                 "spread, 20-day median (bp)": liquidity["spread_20d"],
                 "dollar volume, 20-day average ($ millions)": liquidity["dollar_volume_20d"] / 1e6,
                 "Amihud, 20-day average (bp per $1M)": liquidity["amihud_20d"]}
    _table = pd.DataFrame({_n: _s.quantile(_q) for _n, _s in _measures.items()}).T
    _table.columns = [f"{_x:.0%}" for _x in _q]
    _table["empty"] = [int(_s.isna().sum()) for _s in _measures.values()]
    _table["rank correlation with cap"] = [_s.corr(liquidity["cap"], method="spearman") for _s in _measures.values()]
    _table.round(3)
    return (liquidity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Market cap against spread.** Each stock-day's cap against its 20-day spread, with the $1B line and spreads of 10 and 20 bp marked. The table counts stocks per day (the average over the training days) in each cell of cap by spread: the $1B rule takes the three right-hand columns whatever their spread, and leaves out the two left-hand ones however tight theirs.
    """)
    return


@app.cell
def _(liquidity, mo, np, pd, plt):
    _l = liquidity.dropna(subset=["spread_20d"])
    _l = _l[_l["spread_20d"] > 0]
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _hb = _ax.hexbin(np.log10(_l["cap"]), np.log10(_l["spread_20d"]), gridsize=120, bins="log", cmap="viridis", mincnt=1)
    _ax.axvline(np.log10(1_000), color="w", lw=0.8)
    for _bp in (10, 20):
        _ax.axhline(np.log10(_bp), color="w", lw=0.6, ls="--")
    _ax.set_xticks(np.log10([100, 300, 1_000, 3_000, 10_000, 100_000, 1_000_000]),
                   ["$100M", "$300M", "$1B", "$3B", "$10B", "$100B", "$1T"])
    _ax.set_yticks(np.log10([1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000]), ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1,000"])
    _ax.set_xlabel("previous market cap")
    _ax.set_ylabel("spread, 20-day median (bp)")
    _ax.set_title("Market cap against spread, training stock-days", fontsize=10)
    plt.colorbar(_hb, ax=_ax, label="stock-days")
    _cap = pd.cut(_l["cap"], [0, 300, 1_000, 3_000, 10_000, np.inf], right=False,
                  labels=["$100M-300M", "$300M-1B", "$1B-3B", "$3B-10B", "over $10B"])
    _spread = pd.cut(_l["spread_20d"], [0, 5, 10, 20, 50, np.inf], right=False,
                     labels=["under 5 bp", "5-10 bp", "10-20 bp", "20-50 bp", "50 bp or more"])
    _per_day = (pd.crosstab(_spread, _cap) / liquidity["day"].nunique()).round(0).astype(int)
    _per_day.loc["all"] = _per_day.sum()
    mo.hstack([_fig, mo.vstack([mo.md("**Stocks per day**, 20-day spread (rows) by cap (columns)"), _per_day])], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Over time.** Fixed thresholds give a universe whose size moves with the market: spreads widen in stress (COVID) and narrow as trading grows. Top: stocks per day under the $1B rule and under fixed spread and dollar-volume thresholds. Bottom: the median stock's spread today, above and below $1B.
    """)
    return


@app.cell
def _(liquidity, mark_events, pd, plt):
    _counts = pd.DataFrame({
        "cap ≥ $1B": liquidity["large"],
        "20-day spread ≤ 10 bp": liquidity["spread_20d"] <= 10,
        "20-day spread ≤ 20 bp": liquidity["spread_20d"] <= 20,
        "20-day dollar volume ≥ $10M": liquidity["dollar_volume_20d"] >= 1e7,
    }).groupby(liquidity["day"]).sum().iloc[19:]  # the 20-day measures need 15 days of history; from the 20th day
    _medians = liquidity.groupby(["day", "large"])["spread_today"].median().unstack()
    _fig, _axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for _ax in _axes:
        mark_events(_ax)
    for _c in _counts:
        _axes[0].plot(_counts.index, _counts[_c], lw=0.7, label=_c)
    _axes[0].set_ylabel("stocks")
    _axes[0].legend(fontsize=7, loc="upper left")
    _axes[0].set_title("Stocks per day under each rule", fontsize=10)
    _axes[1].plot(_medians.index, _medians[True], lw=0.6, label="cap ≥ $1B")
    _axes[1].plot(_medians.index, _medians[False], lw=0.6, label="cap under $1B")
    _axes[1].set_yscale("log")
    _axes[1].set_ylabel("bp (log)")
    _axes[1].legend(fontsize=7, loc="upper left")
    _axes[1].set_title("Median spread today", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The rules side by side**, each taking the day's $1B count of the most liquid stocks by its measure (a stock without the measure that day, such as one with under 15 days of history, is left out). Averages over the training days; spreads and caps over the rule's stock-days.
    """)
    return


@app.cell
def _(liquidity, pd):
    _days = pd.DatetimeIndex(sorted(liquidity["day"].unique()))
    _n = liquidity.groupby("day")["large"].transform("sum")
    _rank = lambda _m, _ascending: liquidity.groupby("day")[_m].rank(method="first", ascending=_ascending)
    _rules = {
        "cap ≥ $1B (current)": liquidity["large"],
        "tightest spread today": _rank("spread_today", True) <= _n,
        "tightest 20-day spread": _rank("spread_20d", True) <= _n,
        "largest 20-day dollar volume": _rank("dollar_volume_20d", False) <= _n,
        "lowest 20-day Amihud": _rank("amihud_20d", True) <= _n,
    }
    _dollars = liquidity.groupby("day")["dollar_volume_prev"].sum()
    _rows = []
    for _name, _in in _rules.items():
        _s = liquidity[_in]
        _member = _s.pivot(index="day", columns="permno", values="cap").reindex(_days).notna()
        _new = (_member & ~_member.shift(1, fill_value=False)).sum(axis=1) / _member.sum(axis=1)
        _rows.append({
            "rule": _name,
            "stocks a day": int(_s.groupby("day").size().median()),
            "% also ≥ $1B": 100 * _s["large"].mean(),
            "spread today, median (bp)": _s["spread_today"].median(),
            "spread today, 90th percentile (bp)": _s["spread_today"].quantile(0.9),
            "cap, median ($ millions)": _s["cap"].median(),
            "% of the day's dollar volume": 100 * (_s.groupby("day")["dollar_volume_prev"].sum() / _dollars).mean(),
            "% of names new each day": 100 * _new.iloc[1:].mean(),
        })
    liquidity_rules = pd.DataFrame(_rows).set_index("rule").round(1)
    liquidity_rules
    return (liquidity_rules,)


if __name__ == "__main__":
    app.run()
