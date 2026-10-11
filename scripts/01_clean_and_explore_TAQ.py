import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TAQ: cleaning the full pull

    ## 0. Table heads

    **Purpose:** see what each table holds before using it: the first rows of the four tables the pull wrote for one WCT day, 2019-06-04 (`data/raw/taq/pull/20190604/`, the WRDS pull as downloaded, defined in `explanations/taq_pull.md`), and that day's master file (`data/raw/taq/master/`).

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
    from dsr.taq_split import CAP_MIN, CODES, PRICE_MIN, UNIVERSE_TYPES, stock_days, write_decision_days

    return (
        CAP_MIN,
        CODES,
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
        stock_days,
        write_decision_days,
        xcals,
    )


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parent / "data"
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
    return ALL_DAYS, TEST_START, TRAIN_DAYS


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
    ## 5. The data by decision day: the universe, and what is known at 15:50

    **Purpose:** rewrite the pull so each day's folder holds, for the stocks in that day's universe, exactly what is known at its 15:50 cutoff, split into train (decision days 2014-01-02 to 2023-12-29) and test (2024-01-02 to 2025-12-31): `data/intermediate/taq/train/<YYYYMMDD>/` and `test/<YYYYMMDD>/`, three files each.

    **The universe** (`dsr.taq_split.stock_days`), decided per day with what was known the day before: CRSP's security type that day is common stock, foreign common stock, ADR, ETF or REIT; the previous trading day's close is at least $1 (Eaton et al.) and its market cap at least $50M. Nothing about later days is used, so no stock is dropped for a later split or delisting.

    **The files** (`dsr.taq_split`):
    - `bars` and `cells`: the rows of the universe stocks' primary symbols: the previous trading day's last 10 minutes and evening, then the day's pre-open and session to 15:50, in time order. Each row has `permno` (the one its symbol had on the day it traded) and `piece`; trade rows are otherwise copied unchanged, counts as integers, prices as exact decimals.
    - `daily`: one row per universe stock, traded or not: who it is (`symbol`, `security_type`, `is_common`, `pilot_group`), the day (`early_close`), the day before from CRSP (`price_prev`, `cap_prev`, `dollar_volume_prev`, `ret_prev`), today before 15:50 (`open_t`, CRSP's official open, usable when `traded_before_cutoff`; `facprc_t`, the day's split or distribution factor; `spread_bp`), and the target (`ret_next`, close to next close with delisting returns, kept when it spans one trading day; `ret_next_is_delisting`). The last training day's target ends in 2024 and is left empty (the embargo).
    - `data/intermediate/taq/codes.csv` says what every code means.

    **What we check** (after writing): one `daily` row per stock and day; every `bars` and `cells` stock is in its day's `daily`; every `daily` row meets the universe rule; for the kept stocks, rows and trades equal the pull's, piece by piece; 2014-2023 in train and 2024-2025 in test; no training target reaches into 2024.
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
    _early = set(manifest.loc[manifest["close"] == "13:00", "day"].dt.strftime("%Y%m%d"))
    universe = stock_days(_crsp, crsp_name_records, ALL_DAYS, TEST_START, _early)
    _per_day = universe.groupby(["decision_day", "security_type"]).size().unstack(fill_value=0)
    mo.vstack([
        mo.md(f"**The universe:** {len(universe):,} stock-days; stocks per day by security type (median by year):"),
        _per_day.groupby(_per_day.index.year).median().astype(int).rename(index=str),
    ])
    return (universe,)


@app.cell
def _(mo):
    write_by_decision_day = mo.ui.run_button(label="Write data/intermediate/taq/train/ and test/ (about an hour)")
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
    symbols,
    universe,
    write_by_decision_day,
    write_decision_days,
):
    mo.stop(not write_by_decision_day.value, mo.md("Press the button to write."))
    DECISION_DIR = DATA_DIR / "intermediate" / "taq"
    CODES.to_csv(DECISION_DIR / "codes.csv", index=False)
    _links = {_d.strftime("%Y%m%d"): _g[["symbol", "permno", "primary", "pilot_group"]] for _d, _g in symbols.groupby("day")}
    _universe = {_d.strftime("%Y%m%d"): _g for _d, _g in universe.groupby("decision_day")}
    _empty = universe.iloc[0:0]
    decision_days = write_decision_days(
        PULL_DIR, DECISION_DIR, ALL_DAYS, TEST_START, _links.get, lambda _day: _universe.get(_day, _empty),
        progress=lambda _pairs: mo.status.progress_bar(_pairs, title="Writing decision days", remove_on_exit=True),
    )
    decision_days.groupby("period").agg(days=("decision_day", "size"), first=("decision_day", "min"), last=("decision_day", "max"))
    return DECISION_DIR, decision_days


@app.cell
def _(
    ALL_DAYS,
    CAP_MIN,
    DECISION_DIR,
    PRICE_MIN,
    PULL_DIR,
    TEST_START,
    UNIVERSE_TYPES,
    decision_days,
    mo,
    np,
    pd,
    pq,
    symbols,
):
    _bad, _orphans, _dups, _outside, _no_trade, _stock_days = [], 0, 0, 0, 0, 0
    _prim = symbols[symbols["primary"]][["day", "symbol", "permno"]]
    _prim = {_d.strftime("%Y%m%d"): _g.set_index("symbol")["permno"] for _d, _g in _prim.groupby("day")}
    for _prev, _day, _period in mo.status.progress_bar(
        list(zip(ALL_DAYS[:-1], ALL_DAYS[1:], decision_days["period"])), title="Checking", remove_on_exit=True
    ):
        _folder = DECISION_DIR / _period / _day
        _d = pq.read_table(_folder / "daily.parquet", columns=["permno", "price_prev", "cap_prev", "security_type",
                                                               "traded_before_cutoff"]).to_pandas()
        _dups += int(_d["permno"].duplicated().sum())
        _outside += int((~(_d["price_prev"].abs().ge(PRICE_MIN) & _d["cap_prev"].ge(CAP_MIN)
                           & _d["security_type"].isin(UNIVERSE_TYPES))).sum())
        _no_trade += int((~_d["traded_before_cutoff"]).sum())
        _stock_days += len(_d)
        _in = set(_d["permno"])
        for _name in ["cells", "bars"]:
            _w = pq.read_table(_folder / f"{_name}.parquet", columns=["permno", "piece", "trades"]).to_pandas()
            _orphans += int((~_w["permno"].isin(_in)).sum())
            _written = _w.groupby("piece")["trades"].agg(["size", "sum"])
            _expected = []
            for _src, _buckets, _pieces in [(_prev, [2, 3], {2: "prev_last10", 3: "prev_evening"}),
                                            (_day, [0, 1], {0: "pre_open", 1: "session"})]:
                _p = pq.read_table(PULL_DIR / _src / f"{_name}.parquet", columns=["symbol", "bucket", "trades"]).to_pandas()
                _p = _p[_p["bucket"].isin(_buckets) & _p["symbol"].map(_prim.get(_src, pd.Series(dtype="Int64"))).isin(_in)]
                _expected.append(_p.assign(piece=_p["bucket"].map(_pieces)).groupby("piece")["trades"].agg(["size", "sum"]))
            if not _written.sort_index().equals(pd.concat(_expected).sort_index()):
                _bad.append((_day, _name))
    _last_train = DECISION_DIR / "train" / ALL_DAYS[ALL_DAYS.index(TEST_START) - 1] / "daily.parquet"
    decision_checks = pd.DataFrame([
        ("one folder per decision day, 2014-01-02 to 2025-12-31", list(decision_days["decision_day"]) == ALL_DAYS[1:]),
        ("2014-2023 in train, 2024-2025 in test", bool((decision_days["period"] == np.where(
            decision_days["decision_day"] < TEST_START, "train", "test")).all())),
        ("one daily row per stock and day", _dups == 0),
        ("every daily row meets the universe rule", _outside == 0),
        ("every bars and cells stock is in its day's daily", _orphans == 0),
        ("for the kept stocks, rows and trades equal the pull's, piece by piece", not _bad),
        ("no training target reaches into 2024", bool(pd.read_parquet(_last_train)["ret_next"].isna().all())),
    ], columns=["check", "passed"])
    mo.vstack([decision_checks, mo.md(f"{_stock_days:,} stock-days; {_no_trade:,} ({_no_trade / _stock_days:.2%}) "
                                      "without a trade before 15:50")])
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 The training data: shape and health

    **Purpose:** before looking at any values, check what `train/` holds: every day has its three files, how many rows each table has per day, each column's type, how many values are empty, and each column's smallest and largest value. Read from the parquet files' own statistics, so nothing is loaded.

    **What we expect:** 2,516 days with three files each; empty values only where they are meaningful (`last_mid` when a bar had no valid quote; in `daily`, the trading columns of a stock that did not trade before 15:50, `open_t` when CRSP has no open, `ret_next` when the target is missing or embargoed, `pilot_group` outside the pilot); no negative counts or prices.
    """)
    return


@app.cell
def _(DATA_DIR, mo, pd, pq):
    TRAIN_DIR = DATA_DIR / "intermediate" / "taq" / "train"
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


    _missing = [_d for _d in TRAIN_DECISION_DAYS if not all((TRAIN_DIR / _d / f"{_n}.parquet").exists() for _n in ["daily", "bars", "cells"])]
    _per_day, _cols = [], {}
    for _day in mo.status.progress_bar(TRAIN_DECISION_DAYS, title="Reading file statistics", remove_on_exit=True):
        for _name in ["daily", "bars", "cells"]:
            _n, _stats = _file_stats(TRAIN_DIR / _day / f"{_name}.parquet")
            _per_day.append({"day": _day, "table": _name, "rows": _n})
            for _r in _stats:
                _c = _cols.setdefault((_name, _r["column"]), {"type": _r["type"], "nulls": 0, "min": None, "max": None})
                _c["nulls"] += _r["nulls"]
                if _r["min"] is not None:
                    _c["min"] = _r["min"] if _c["min"] is None else min(_c["min"], _r["min"])
                    _c["max"] = _r["max"] if _c["max"] is None else max(_c["max"], _r["max"])
    train_rows = pd.DataFrame(_per_day)
    _totals = train_rows.groupby("table", sort=False)["rows"]
    train_shape = pd.DataFrame({"days": _totals.size(), "rows": _totals.sum(), "rows per day, median": _totals.median(),
                                "min": _totals.min(), "max": _totals.max()})
    train_columns = pd.DataFrame([{"table": _t, "column": _c, **_v} for (_t, _c), _v in _cols.items()])
    train_columns["% empty"] = (train_columns["nulls"] / train_columns["table"].map(_totals.sum()) * 100).round(3)
    train_columns[["min", "max"]] = train_columns[["min", "max"]].astype(str)
    mo.vstack([
        mo.md(f"**{len(TRAIN_DECISION_DAYS):,} training days; days missing a file: {len(_missing)}**"),
        train_shape,
        mo.md("**Every column**: type, empty values, smallest and largest value over all training days"),
        mo.ui.table(train_columns[["table", "column", "type", "nulls", "% empty", "min", "max"]], selection=None, page_size=60),
    ])
    return TRAIN_DECISION_DAYS, TRAIN_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 Table heads

    The first rows of each table on one training day, 2019-06-04: `daily` (one row per universe stock), `bars` and `cells` (rows by stock, then in time order). Codes are explained in `data/intermediate/taq/codes.csv`.
    """)
    return


@app.cell
def _(TRAIN_DIR, mo, pq):
    _day = TRAIN_DIR / "20190604"
    mo.vstack([
        mo.vstack([mo.md(f"**`{_name}`**: {pq.ParquetFile(_day / f'{_name}.parquet').metadata.num_rows:,} rows"),
                   pq.read_table(_day / f"{_name}.parquet").slice(0, 15).to_pandas()])
        for _name in ["daily", "bars", "cells"]
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 The market over time

    **Purpose:** see how trading, spreads, market value and the retail measures move over the training years, and whether anything breaks. One pass over the 2,516 training days sums `bars` across all universe stocks per day and summarises `daily`; every plot reads that table (`market_days`). A decision day's totals include the previous day's last 10 minutes and evening: everything known by its 15:50.

    **Marked on every plot:** the Tick Size Pilot (2016-10 to 2018-09, shaded), zero commissions (2019-10-07), COVID (2020-03), fractional shares printing as one-share trades (2021-02-16) and the meme surge (late January 2021).

    **Columns** (meanings in `explanations/TAQ/pulled_data_full.md`): `bjzz_*` retail trades by the sub-penny rule, `mid_*` by the midpoint rule, `off_*` all off-exchange, `qr_*` the quote rule on every venue and `exch_qr_*` on exchanges only, `both_signed` and `same_sign` how often the two retail rules sign the same trade and agree.
    """)
    return


@app.cell
def _(TRAIN_DECISION_DAYS, TRAIN_DIR, mo, pd, pq):
    _bar_sums = ["trades", "shares", "dollars", "bjzz_buy_trades", "bjzz_buy_dollars", "bjzz_sell_trades", "bjzz_sell_dollars",
                 "mid_buy_trades", "mid_buy_dollars", "mid_sell_trades", "mid_sell_dollars", "off_trades", "off_dollars",
                 "off_odd_trades", "off_one_trades", "qr_buy_shares", "qr_sell_shares", "exch_qr_buy_shares",
                 "exch_qr_sell_shares", "both_signed_trades", "same_sign_trades"]
    _rows = []
    for _day in mo.status.progress_bar(TRAIN_DECISION_DAYS, title="Summing the training days", remove_on_exit=True):
        _b = pq.read_table(TRAIN_DIR / _day / "bars.parquet", columns=_bar_sums).to_pandas().sum()
        _d = pq.read_table(TRAIN_DIR / _day / "daily.parquet", columns=[
            "is_common", "cap_prev", "dollar_volume_prev", "spread_bp", "traded_before_cutoff"]).to_pandas()
        _w = _d["dollar_volume_prev"].where(_d["spread_bp"].notna())
        _rows.append({"day": pd.Timestamp(_day), **_b.to_dict(), "stocks": len(_d), "traded": int(_d["traded_before_cutoff"].sum()),
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
    two_axes(_axes[0, 1], {"stocks in the universe": _m["stocks"], "traded before 15:50": _m["traded"]}, {},
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


if __name__ == "__main__":
    app.run()
