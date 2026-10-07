import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # TAQ: raw files (one day)

    The first rows of each file that `00_fetch_TAQ.py` downloads into `data/raw/taq/<date>/`, as text and unchanged. Only the first rows are read, not the whole day.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    return mo, np, pa, pc, pd, pq


@app.cell
def _(mo):
    DATE = "2019-06-04"
    RAW_DIR = mo.notebook_dir().parent / "data" / "raw" / "taq" / DATE
    CRSP_NAMES_PATH = mo.notebook_dir().parent / "data" / "raw" / "crsp" / "stocknames_v2.csv"
    return CRSP_NAMES_PATH, DATE, RAW_DIR


@app.cell
def _(RAW_DIR, pq):
    def _head(name, n=20):
        """The first n rows of a raw parquet file, without reading the rest."""
        return next(pq.ParquetFile(RAW_DIR / f"{name}.parquet").iter_batches(batch_size=n)).to_pandas()


    wct_raw = _head("wct")
    master_raw = _head("mastm")
    return master_raw, wct_raw


@app.cell
def _(mo):
    mo.md(r"""
    ## WRDS Consolidated Trades (`wct`)

    Every trade with the NBBO prevailing just before it, built by WRDS. It replaces the trade (`ctm`) and NBBO (`complete_nbbo`) tables; see `explanations/TAQ/wct.md`.
    """)
    return


@app.cell
def _(wct_raw):
    wct_raw
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Master (`mastm`)
    """)
    return


@app.cell
def _(master_raw):
    master_raw
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## CUSIP on every trade

    Each WCT trade gets the CUSIP its symbol had that day in the master file. The symbol is `sym_root` and `sym_suffix` joined by a space, which is the master file's `symbol_15` (`BRK B`; `A` when there is no suffix). The PERMNO is joined to the CUSIP later, through CRSP's name records valid on that date.
    """)
    return


@app.cell
def _(RAW_DIR, pc, pq):
    def _symbol(root, suffix):
        """sym_root and sym_suffix as the master file writes them."""
        return pc.if_else(pc.equal(suffix, ""), root, pc.binary_join_element_wise(root, suffix, " "))


    _wct = pq.read_table(RAW_DIR / "wct.parquet")
    _wct = _wct.append_column("symbol", _symbol(_wct["sym_root"], _wct["sym_suffix"]))
    _master = pq.read_table(RAW_DIR / "mastm.parquet", columns=["symbol_15", "cusip"]).rename_columns(["symbol", "cusip"])
    assert len(pc.unique(_master["symbol"])) == len(_master), "a symbol appears twice in the master file"
    wct_cusip = _wct.join(_master, "symbol", join_type="left outer")
    return (wct_cusip,)


@app.cell
def _(mo, pc, wct_cusip):
    _has = pc.and_kleene(pc.is_valid(wct_cusip["cusip"]), pc.not_equal(wct_cusip["cusip"], ""))  # Kleene: a missing CUSIP counts as false, not null
    _trades = wct_cusip.select(["symbol"]).append_column("has_cusip", _has).group_by(["symbol", "has_cusip"]).aggregate([([], "count_all")])
    _no = _trades.filter(pc.invert(_trades["has_cusip"])).sort_by([("count_all", "descending")]).to_pandas()
    mo.vstack([
        mo.md(f"""
    - **Trades:** {len(wct_cusip):,}; with a CUSIP: {pc.sum(_has).as_py():,} ({pc.mean(_has).as_py():.4%})
    - **Symbols traded:** {len(_trades):,}; without a CUSIP: {len(_no):,}, with {_no["count_all"].sum():,} trades between them
    """),
        mo.md("**Symbols without a CUSIP, by trades**"),
        _no.rename(columns={"count_all": "trades"})[["symbol", "trades"]].head(20),
        mo.md("**WCT with its CUSIP, first rows**"),
        wct_cusip.slice(0, 20).to_pandas(),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PERMNO on every trade

    The CUSIP is matched to the PERMNO whose CRSP name record (`stocknames_v2`) is valid on the day and has the same CUSIP. TAQ's CUSIP has 9 characters and CRSP's 8 (no check digit), so the first 8 are compared. A CUSIP that matches two PERMNOs on the same day would be left unmatched rather than guessed.
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, DATE, RAW_DIR, pa, pd, pq, wct_cusip):
    _names = pd.read_csv(CRSP_NAMES_PATH, dtype=str, usecols=["permno", "namedt", "nameenddt", "cusip"])
    _names = _names[(_names["namedt"] <= DATE) & (_names["nameenddt"] >= DATE)][["cusip", "permno"]].drop_duplicates()
    cusip_ambiguous = _names[_names.groupby("cusip")["permno"].transform("size") > 1]
    _cusip_permno = _names[~_names["cusip"].isin(cusip_ambiguous["cusip"])].rename(columns={"cusip": "cusip8"})

    symbol_permno = (
        pq.read_table(RAW_DIR / "mastm.parquet", columns=["symbol_15", "cusip", "sec_type", "sec_desc"]).to_pandas()
        .rename(columns={"symbol_15": "symbol"})
        .assign(cusip8=lambda d: d["cusip"].str[:8])
        .merge(_cusip_permno, on="cusip8", how="left")
    )
    wct_permno = wct_cusip.join(
        pa.Table.from_pandas(symbol_permno[["symbol", "permno"]], preserve_index=False).cast(
            pa.schema([("symbol", pa.string()), ("permno", pa.string())])
        ),
        "symbol",
        join_type="left outer",
    )
    return cusip_ambiguous, symbol_permno, wct_permno


@app.cell(hide_code=True)
def _(cusip_ambiguous, mo, pa, pc, symbol_permno, wct_permno):
    _has = pc.is_valid(wct_permno["permno"])
    _dollars = pc.multiply(pc.cast(wct_permno["price"], pa.float64()), pc.cast(wct_permno["size"], pa.float64()))
    _by_symbol = (
        wct_permno.select(["symbol"]).append_column("dollars", _dollars).group_by("symbol")
        .aggregate([("dollars", "sum"), ([], "count_all")]).to_pandas()
        .rename(columns={"dollars_sum": "dollars", "count_all": "trades"})
        .merge(symbol_permno, on="symbol", how="left")
    )
    _no = _by_symbol[_by_symbol["permno"].isna()].sort_values("trades", ascending=False)
    _multi = _by_symbol.dropna(subset=["permno"]).groupby("permno")["symbol"].nunique()
    mo.vstack([
        mo.md(f"""
    - **Trades with a PERMNO:** {pc.sum(_has).as_py():,} of {len(wct_permno):,} ({pc.mean(_has).as_py():.2%}); by dollars {_by_symbol.loc[_by_symbol["permno"].notna(), "dollars"].sum() / _by_symbol["dollars"].sum():.2%}
    - **Symbols traded:** {len(_by_symbol):,}; with a PERMNO {_by_symbol["permno"].notna().sum():,}, without {len(_no):,}
    - **CUSIPs matching two PERMNOs that day:** {cusip_ambiguous["cusip"].nunique():,}; **PERMNOs reached by more than one symbol:** {int((_multi > 1).sum()):,}
    - **Symbols without a PERMNO by TAQ security type** (trades): {", ".join(f"{k or 'blank'} {v:,}" for k, v in _no.groupby(_no["sec_type"].fillna(""))["trades"].sum().sort_values(ascending=False).items())}
    """),
        mo.md("**Symbols without a PERMNO, by trades**"),
        _no[["symbol", "sec_type", "sec_desc", "cusip", "trades"]].head(25),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checks: negative prices, bid above ask, missing values

    Every trade of the day (all hours), on the WCT columns plus the symbol, CUSIP and PERMNO joined above. Values are stored as text, so a missing value is a null or an empty string. A price, bid or ask that is not a number counts as missing.
    """)
    return


@app.cell
def _(mo, pa, pc, pd, wct_permno):
    _n = len(wct_permno)
    _num = {c: pc.cast(pc.if_else(pc.equal(wct_permno[c], ""), None, wct_permno[c]), pa.float64()) for c in ["price", "nbb", "nbo"]}
    _valid_quote = pc.and_(pc.is_valid(_num["nbb"]), pc.is_valid(_num["nbo"]))
    wct_checks = pd.DataFrame([
        ("price below 0", pc.sum(pc.less(_num["price"], 0)).as_py() or 0),
        ("price equal to 0", pc.sum(pc.equal(_num["price"], 0)).as_py() or 0),
        ("bid (nbb) below 0", pc.sum(pc.less(_num["nbb"], 0)).as_py() or 0),
        ("ask (nbo) below 0", pc.sum(pc.less(_num["nbo"], 0)).as_py() or 0),
        ("bid above ask (crossed)", pc.sum(pc.greater(_num["nbb"], _num["nbo"])).as_py() or 0),
        ("bid equal to ask (locked)", pc.sum(pc.equal(_num["nbb"], _num["nbo"])).as_py() or 0),
    ], columns=["check", "trades"]).assign(share=lambda d: d["trades"] / _n)
    wct_missing = pd.DataFrame([
        {"column": c, "missing": pc.sum(pc.or_kleene(pc.is_null(wct_permno[c]), pc.equal(wct_permno[c], ""))).as_py()} for c in wct_permno.column_names
    ]).assign(share=lambda d: d["missing"] / _n)
    mo.vstack([
        mo.md(f"**{_n:,} trades**"),
        mo.ui.table(wct_checks, selection=None),
        mo.md("**Missing values by column** (null or empty)"),
        mo.ui.table(wct_missing, selection=None, page_size=20),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cleaning one day

    The rules in `explanations/TAQ/cleaning_spec.md`, one section per step. Each step keeps a table of trades and adds a row to the counts: trades, dollars and off-exchange trades left after it.

    ## Load

    WCT's columns are text. Here they become numbers, a symbol as the master file writes it, and `t_ns`, the SIP time in nanoseconds after midnight (`time_m` plus `time_m_nano`).
    """)
    return


@app.cell
def _(RAW_DIR, np, pd):
    def typed(raw):
        """Numbers, symbol and time from TAQ's text columns."""
        suffix = raw["sym_suffix"].fillna("")
        out = raw.assign(
            symbol=np.where(suffix.eq(""), raw["sym_root"], raw["sym_root"] + " " + suffix),
            t_ns=pd.to_timedelta(raw["time_m"]).to_numpy().astype("timedelta64[ns]").astype("int64")
            + pd.to_numeric(raw["time_m_nano"]).astype("int64").to_numpy(),
        )
        for col in ["price", "size", "nbb", "nbo", "tr_seqnum"]:
            if col in out:
                out[col] = pd.to_numeric(out[col].replace("", np.nan))
        return out


    def count_step(step, t):
        """One row of the step counts."""
        return {"step": step, "trades": len(t), "dollars": float((t["price"] * t["size"]).sum()),
                "off_exchange_trades": int(t["ex"].eq("D").sum())}


    c0_trades = typed(pd.read_parquet(RAW_DIR / "wct.parquet"))
    c0_counts = [count_step("raw WCT", c0_trades)]

    return c0_counts, c0_trades, count_step, typed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: corrected and cancelled trades

    WCT has no correction flag, so the flagged trades come from the trade file (`ctm_corrections.parquet`). They are matched to WCT on symbol, `tr_seqnum` and `t_ns`, which is unique per trade in both tables. Dropped: `08` later cancelled, `10` cancel record, `07` and `11` erroneous, `12` correction record (it carries the wrong original data). Kept: `01`, the trade as corrected.
    """)
    return


@app.cell
def _(RAW_DIR, c0_counts, c0_trades, count_step, pd, typed):
    TRADE_KEY = ["sym_root", "sym_suffix", "tr_seqnum", "t_ns"]
    CORRECTIONS_TO_DROP = {"07", "08", "10", "11", "12"}

    c1_corrections = typed(pd.read_parquet(RAW_DIR / "ctm_corrections.parquet"))
    _drop = c1_corrections.loc[c1_corrections["tr_corr"].isin(CORRECTIONS_TO_DROP), TRADE_KEY].drop_duplicates()
    _m = c0_trades.merge(_drop.assign(_correction=True), on=TRADE_KEY, how="left")
    c1_matched = int(_m["_correction"].notna().sum())
    c1_to_drop = len(_drop)
    c1_trades = _m[_m["_correction"].isna()].drop(columns="_correction")
    c1_counts = c0_counts + [count_step("without corrected and cancelled trades", c1_trades)]

    return c1_corrections, c1_counts, c1_matched, c1_to_drop, c1_trades


@app.cell(hide_code=True)
def _(c1_corrections, c1_matched, c1_to_drop, mo):
    mo.vstack([
        mo.md(f"**{c1_matched:,} of {c1_to_drop:,} trades to drop found in WCT.** By correction code and venue:"),
        c1_corrections.assign(off_exchange=c1_corrections["ex"].eq("D")).groupby(["tr_corr", "off_exchange"]).size().rename("trades").reset_index(),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: PERMNO

    Symbol → CUSIP from that day's master file → PERMNO from the CRSP name record valid that day, on the first 8 CUSIP characters. A CUSIP valid for two PERMNOs that day is left unmatched. Trades without a PERMNO are dropped: test symbols, ETNs, preferreds, warrants, units, structured products, ADRs that CRSP lacks.
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, DATE, RAW_DIR, c1_counts, c1_trades, count_step, pd):
    _names = pd.read_csv(CRSP_NAMES_PATH, dtype=str, usecols=["permno", "namedt", "nameenddt", "cusip"])
    _names = _names.loc[(_names["namedt"] <= DATE) & (_names["nameenddt"] >= DATE), ["cusip", "permno"]].drop_duplicates()
    _names = _names[~_names["cusip"].duplicated(keep=False)].rename(columns={"cusip": "cusip8"})
    _master = pd.read_parquet(RAW_DIR / "mastm.parquet")
    c2_symbols = (
        _master.assign(symbol=_master["symbol_15"], cusip8=_master["cusip"].str[:8],
                       tick_pilot_indicator=_master["tick_pilot_indicator"].fillna(""))
        .merge(_names, on="cusip8", how="left")[["symbol", "cusip", "permno", "listed_exchange", "tick_pilot_indicator", "sec_type", "sec_desc"]]
    )
    _t = c1_trades.merge(c2_symbols, on="symbol", how="left")
    c2_dropped = _t[_t["permno"].isna()]
    c2_trades = _t[_t["permno"].notna()]
    c2_counts = c1_counts + [count_step("with a PERMNO", c2_trades)]

    return c2_counts, c2_dropped, c2_symbols, c2_trades


@app.cell(hide_code=True)
def _(c2_dropped, mo):
    mo.vstack([
        mo.md("**Dropped, by the master file's security type**"),
        c2_dropped.groupby(c2_dropped["sec_type"].fillna("not in master"))["symbol"].agg(symbols="nunique", trades="size").sort_values("trades", ascending=False).reset_index(),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: sale conditions and buckets

    Each trade's conditions put it in one class:
    - **normal:** only blank / `@`, `I`, `F`;
    - **extended:** normal codes plus `T` or `U`;
    - **open:** an `O` (opening) print;
    - **excluded:** anything else.

    Buckets on a normal day, by `t_ns`:
    - **`pre`** [04:00, 09:30), normal or extended, feeds this day;
    - **`open_auction`**: `O` prints on the stock's listing exchange, feeds this day;
    - **`regular`** [09:30, 15:50), normal, feeds this day;
    - **`last`** [15:50, 16:00), normal, feeds the next trading day;
    - **`after`** from 16:00, normal or extended, feeds the next trading day.

    Early-close days (13:00 close, 12:50 cutoff) move the times; 2019-06-04 is not one. The opening auction becomes one price (its largest print) and one volume per stock; it is not counted as trades. `Q` prints (each venue's own official open) and `O` prints off the listing exchange are excluded.
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, DATE, c2_counts, c2_trades, count_step, np, pd):
    EARLY_CLOSE_DAYS = set(pd.to_datetime([
        "2018-07-03", "2018-11-23", "2018-12-24", "2019-07-03", "2019-11-29", "2019-12-24", "2020-11-27", "2020-12-24",
        "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
        "2025-07-03", "2025-11-28", "2025-12-24",
    ]))  # NYSE calendar, written from memory; checked against the data when the full sample is built
    TRADE_BUCKETS = ["pre", "regular", "last", "after"]


    def condition_class(code):
        codes = set(code) - {" "}
        if "O" in codes:
            return "open"
        if not codes <= set("@IFTU"):
            return "excluded"
        return "extended" if codes & set("TU") else "normal"


    _dsi = pd.read_csv(CRSP_NAMES_PATH.parent / "dsi.csv", usecols=["date"], parse_dates=["date"])["date"]
    NEXT_DAY = _dsi[_dsi > pd.Timestamp(DATE)].min()
    _close = pd.Timedelta(hours=13 if pd.Timestamp(DATE) in EARLY_CLOSE_DAYS else 16)
    SESSION = {"open": pd.Timedelta(hours=9, minutes=30), "cutoff": _close - pd.Timedelta(minutes=10), "close": _close}

    _codes = c2_trades["tr_scond"].fillna("")
    _unique = _codes.unique()
    _cls = _codes.map(dict(zip(_unique, map(condition_class, _unique))))
    _t, _n, _e = c2_trades["t_ns"], _cls.eq("normal"), _cls.eq("extended")
    _bucket = np.select(
        [
            _cls.eq("open") & c2_trades["ex"].eq(c2_trades["listed_exchange"]),
            (_n | _e) & (_t < SESSION["open"].value),
            _n & (_t >= SESSION["open"].value) & (_t < SESSION["cutoff"].value),
            _n & (_t >= SESSION["cutoff"].value) & (_t < SESSION["close"].value),
            (_n | _e) & (_t >= SESSION["close"].value),
        ],
        ["open_auction", "pre", "regular", "last", "after"],
        default="",
    )
    c3_all = c2_trades.assign(
        condition_class=_cls.to_numpy(), bucket=_bucket,
        decision_day=np.where(np.isin(_bucket, ["last", "after"]), NEXT_DAY, pd.Timestamp(DATE)),
    )
    _open = c3_all[c3_all["bucket"].eq("open_auction")]
    opening_auctions = pd.DataFrame({
        "open_price": _open.sort_values("size").groupby("permno").tail(1).set_index("permno")["price"],
        "open_volume": _open.groupby("permno")["size"].sum(),
    }).reset_index()
    c3_trades = c3_all[c3_all["bucket"].isin(TRADE_BUCKETS)]
    c3_counts = c2_counts + [count_step("normal sale conditions, in a bucket", c3_trades)]

    return (
        NEXT_DAY,
        SESSION,
        TRADE_BUCKETS,
        c3_all,
        c3_counts,
        c3_trades,
        opening_auctions,
    )


@app.cell(hide_code=True)
def _(TRADE_BUCKETS, c2_symbols, c3_all, c3_trades, mo, opening_auctions):
    _x = c3_all[~c3_all["bucket"].isin(TRADE_BUCKETS)]
    mo.vstack([
        mo.md(f"**Opening auctions:** {len(opening_auctions):,} stocks. AAPL's volume: "
              f"{int(opening_auctions.merge(c2_symbols[c2_symbols['symbol'].eq('AAPL')], on='permno')['open_volume'].iloc[0]):,} shares"),
        mo.md("**Kept trades by bucket and decision day**"),
        c3_trades.groupby(["decision_day", "bucket"]).size().rename("trades").reset_index(),
        mo.md("**Not kept as trades, by class and sale condition** (top 20)"),
        _x.groupby(["condition_class", "bucket", "tr_scond"]).size().rename("trades").sort_values(ascending=False).head(20).reset_index(),
    ])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: trade filters

    Prices of at least $1 (Boehmer & Jones 2022), positive price and size, and no stock in Tick Size Pilot groups 2 or 3 (until October 2018; none on this day). An unknown pilot code stops the notebook.

    ## Step 5: flags

    Kept on every trade, not used to drop:
    - **`quote_state`:** `missing` if the bid or offer is empty, `crossed` if bid > offer, `locked` if bid = offer, else `valid`. Midpoint signing needs `valid`.
    - **`outside_quote`:** a valid quote and a price more than one spread below the bid or above the offer. Left out of price-based features.
    - **`large`:** over $100,000.
    """)
    return


@app.cell
def _(c3_counts, c3_trades, count_step, np, pd):
    _pilot = set(c3_trades["tick_pilot_indicator"]) - {"", "C", "1", "2", "3"}
    if _pilot:
        raise ValueError(f"unknown tick_pilot_indicator codes: {sorted(_pilot)}")
    _t = c3_trades[(c3_trades["price"] >= 1) & (c3_trades["size"] > 0)]
    _counts = c3_counts + [count_step("price at least $1", _t)]
    _t = _t[~_t["tick_pilot_indicator"].isin({"2", "3"})]
    _counts = _counts + [count_step("outside Tick Size Pilot groups 2 and 3", _t)]

    _bid, _ask, _px = _t["nbb"], _t["nbo"], _t["price"]
    _state = np.select([_bid.isna() | _ask.isna(), _bid > _ask, _bid == _ask], ["missing", "crossed", "locked"], default="valid")
    _spread = _ask - _bid
    clean_trades = _t.assign(
        quote_state=_state,
        outside_quote=(_state == "valid") & ((_px < _bid - _spread) | (_px > _ask + _spread)),
        large=_px * _t["size"] > 100_000,
    ).reset_index(drop=True)
    clean_counts = pd.DataFrame(_counts).assign(share_of_raw_trades=lambda d: d["trades"] / d["trades"].iloc[0])

    return clean_counts, clean_trades


@app.cell
def _(
    DATE,
    NEXT_DAY,
    SESSION,
    TRADE_BUCKETS,
    c1_matched,
    c1_to_drop,
    clean_counts,
    clean_trades,
    pd,
):
    _own = clean_trades["decision_day"].eq(pd.Timestamp(DATE))
    clean_checks = pd.DataFrame([
        ("every correction to drop matched one WCT trade", c1_matched == c1_to_drop),
        ("every trade has a bucket", clean_trades["bucket"].isin(TRADE_BUCKETS).all()),
        ("every trade feeds this day or the next", clean_trades["decision_day"].isin([pd.Timestamp(DATE), NEXT_DAY]).all()),
        ("this day's trades are before its cutoff", (clean_trades.loc[_own, "t_ns"] < SESSION["cutoff"].value).all()),
        ("no PERMNO under two symbols", clean_trades.groupby("permno")["symbol"].nunique().le(1).all()),
        ("trades never increase from step to step", clean_counts["trades"].is_monotonic_decreasing),
    ], columns=["check", "passed"])

    return (clean_checks,)


@app.cell(hide_code=True)
def _(clean_checks, clean_counts, clean_trades, mo):
    mo.vstack([
        mo.md("## Result"),
        mo.md("**What each step leaves**"),
        mo.ui.table(clean_counts, selection=None),
        mo.md("**Checks**"),
        mo.ui.table(clean_checks, selection=None),
        mo.md("**Flags**"),
        clean_trades["quote_state"].value_counts().rename("trades").reset_index(),
        mo.md(f"outside the quote: {int(clean_trades['outside_quote'].sum()):,}; over $100k: {int(clean_trades['large'].sum()):,}"),
        mo.md(f"**Clean trades: first 20 of {len(clean_trades):,}**"),
        clean_trades.head(20),
    ])

    return


if __name__ == "__main__":
    app.run()
