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
    import exchange_calendars as xcals
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    return mo, np, pa, pc, pd, pq, xcals


@app.cell
def _(mo):
    DATE = "2019-06-04"
    RAW_DIR = mo.notebook_dir().parents[1] / "data" / "raw" / "taq" / DATE
    CRSP_NAMES_PATH = mo.notebook_dir().parents[1] / "data" / "raw" / "crsp" / "stocknames_v2.csv"
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

    Every trade with the NBBO prevailing just before it, built by WRDS. It replaces the trade (`ctm`) and NBBO (`complete_nbbo`) tables; see `explanations/taq/wct.md`.
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

    The rules in `explanations/taq/cleaning_spec.md`, one section per step. Each step keeps a table of trades and adds a row to the counts: trades, dollars and off-exchange trades left after it. The checks at the end stop the notebook if any fails.

    ## Load

    WCT's columns are text. Here they become numbers, a symbol as the master file writes it, and `t_ns`, the SIP time in nanoseconds after midnight (`time_m` plus `time_m_nano`).

    The session times come from NYSE's calendar (`exchange_calendars`, `XNYS`), checked against the closing-auction prints on every trading day from 2014 to 2025 (`scripts/taq/checks/TAQ_closes.py`): the regular session opens at 09:30 and closes at 16:00, and the cutoff is 10 minutes before the close (15:50). On early-close days the close is 13:00 and the cutoff 12:50. Trades are kept at any hour of the day, 00:00 to 24:00. The calendar also gives the next trading day, which the day's trades after the cutoff feed.
    """)
    return


@app.cell
def _(DATE, RAW_DIR, np, pd, xcals):
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


    CUTOFF_OFFSET = pd.Timedelta(minutes=10)  # the decision is taken this long before the close
    _nyse = xcals.get_calendar("XNYS")
    _close_ny = _nyse.session_close(DATE).tz_convert("America/New_York")
    _close = _close_ny - _close_ny.normalize()  # time of day
    SESSION = {
        "open": pd.Timedelta(hours=9, minutes=30),
        "cutoff": _close - CUTOFF_OFFSET,
        "close": _close,
        "day_end": pd.Timedelta(hours=24),
    }
    NEXT_DAY = _nyse.next_session(DATE)

    c0_trades = typed(pd.read_parquet(RAW_DIR / "wct.parquet"))
    c0_counts = [count_step("raw WCT", c0_trades)]
    return NEXT_DAY, SESSION, c0_counts, c0_trades, count_step, typed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: corrected and cancelled trades, as known at the cutoff

    WCT has no correction flag, so the flagged trades come from the trade file (`ctm_corrections.parquet`). Each flagged trade (`08` later cancelled, `07` later erroneous, `01` later corrected) has a record published later (`10`, `11` or `12`), paired on symbol, venue and sequence number (`tr_seqnum`), which the record repeats. Trade IDs (`tr_id`) are not used: they are missing before 2016 and empty on most flagged rows until 2017-11-03. A record carries its own time: when the cancellation or correction was published.

    A record counts for a trade only if it was public before the cutoff of the day the trade feeds:
    - a trade before today's 15:50 feeds today, so a record published at or after 15:50 is ignored. The trade stays as it was printed, at its original price;
    - a trade at or after 15:50 feeds the next trading day, so every same-day record counts.

    So:
    - **records** (`10`, `11`, `12`) are never trades and are always dropped;
    - **cancelled or erroneous trades** are dropped when the record counts, and kept as printed when it does not;
    - **corrected trades:** WCT holds the corrected price at the original time. When the correction does not count yet, the original price, size and conditions are restored from the `12` record.

    WCT rows are found on symbol, venue, `tr_seqnum` and `t_ns`, which is unique per trade in both tables.
    """)
    return


@app.cell
def _(RAW_DIR, SESSION, c0_counts, c0_trades, count_step, pd, typed):
    _TRADE_KEY = ["sym_root", "sym_suffix", "ex", "tr_seqnum", "t_ns"]
    _PAIR_KEY = ["sym_root", "sym_suffix", "ex", "tr_seqnum", "record_code"]
    _RECORD_CODE = {"08": "10", "07": "11", "01": "12"}  # flagged trade -> its record

    c1_corrections = typed(pd.read_parquet(RAW_DIR / "ctm_corrections.parquet"))
    _orig = c1_corrections[c1_corrections["tr_corr"].isin(_RECORD_CODE)]
    _orig = _orig.assign(record_code=_orig["tr_corr"].map(_RECORD_CODE))
    _rec = c1_corrections[c1_corrections["tr_corr"].isin(_RECORD_CODE.values())]
    _rec = _rec.assign(record_code=_rec["tr_corr"])
    c1_pairs = _orig.merge(
        _rec[_PAIR_KEY + ["t_ns", "price", "size", "tr_scond"]], on=_PAIR_KEY, how="left",
        suffixes=("", "_record"), validate="one_to_one",
    )
    _cutoff = SESSION["cutoff"].value
    c1_pairs["record_counts"] = (c1_pairs["t_ns"] >= _cutoff) | (c1_pairs["t_ns_record"] < _cutoff)

    _actions = pd.concat([
        _rec[_TRADE_KEY].assign(_action="drop record"),
        c1_pairs.loc[c1_pairs["tr_corr"].isin(["07", "08"]) & c1_pairs["record_counts"], _TRADE_KEY].assign(_action="drop cancelled"),
        c1_pairs.loc[c1_pairs["tr_corr"].eq("01") & ~c1_pairs["record_counts"], _TRADE_KEY + ["price_record", "size_record", "tr_scond_record"]]
        .assign(_action="restore original"),
    ], ignore_index=True)

    # every flagged trade-file row must be exactly one WCT trade, and every flagged trade must have its record
    _hits = c1_corrections[_TRADE_KEY].merge(c0_trades[_TRADE_KEY], on=_TRADE_KEY, how="left", indicator=True)
    c1_each_matched_once = bool(len(_hits) == len(c1_corrections) and _hits["_merge"].eq("both").all())
    c1_each_paired = bool(len(c1_pairs) == len(_rec) and c1_pairs["t_ns_record"].notna().all())

    _m = c0_trades.merge(_actions, on=_TRADE_KEY, how="left")
    _restore = _m["_action"].eq("restore original")
    for _col in ["price", "size", "tr_scond"]:
        _m.loc[_restore, _col] = _m.loc[_restore, f"{_col}_record"]
    c1_trades = _m[~_m["_action"].isin(["drop record", "drop cancelled"])].drop(
        columns=["_action", "price_record", "size_record", "tr_scond_record"]
    )
    c1_counts = c0_counts + [count_step("corrections and cancellations as known at the cutoff", c1_trades)]
    return c1_counts, c1_each_matched_once, c1_each_paired, c1_pairs, c1_trades


@app.cell(hide_code=True)
def _(c1_pairs, mo):
    mo.vstack([
        mo.md("**Flagged trades by code, whether the record counts, and venue**"),
        c1_pairs.assign(off_exchange=c1_pairs["ex"].eq("D")).groupby(["tr_corr", "record_counts", "off_exchange"]).size().rename("trades").reset_index(),
        mo.md("**Records published after the cutoff of the trade's day** (the trade is kept as printed)"),
        c1_pairs.loc[~c1_pairs["record_counts"], ["symbol", "ex", "tr_corr", "time_m", "price", "price_record", "tr_scond"]]
        .rename(columns={"price": "price_in_wct", "price_record": "price_in_record"}).sort_values("time_m"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: CUSIP and PERMNO, attached but not required

    Symbol → CUSIP from that day's master file → PERMNO from the CRSP name record valid that day, on the first 8 CUSIP characters. A CUSIP valid for two PERMNOs that day is left unmatched. No trade is dropped here: symbols without a CUSIP (test symbols) or without a PERMNO (ETNs, preferreds, warrants, units, structured products, ADRs that CRSP lacks) are kept, as in the full pull, which writes every traded symbol and matches identifiers locally.

    The master file also gives each stock's Tick Size Pilot group (October 2016 to September 2018), kept on every trade: stocks in groups G2 and G3 are not dropped, since their prices and volume are fine and their retail trades can be found from half-cent prices away from the midpoint (`explanations/taq/taq_features.md`). An unknown group code stops the notebook.
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, DATE, RAW_DIR, c1_counts, c1_trades, pd):
    PRICE_MIN = 1.0  # Boehmer & Jones (2022)

    _names = pd.read_csv(CRSP_NAMES_PATH, dtype=str, usecols=["permno", "namedt", "nameenddt", "cusip"])
    _names = _names.loc[(_names["namedt"] <= DATE) & (_names["nameenddt"] >= DATE), ["cusip", "permno"]].drop_duplicates()
    _names = _names[~_names["cusip"].duplicated(keep=False)].rename(columns={"cusip": "cusip8"})
    _master = pd.read_parquet(RAW_DIR / "mastm.parquet")
    _pilot = set(_master["tick_pilot_indicator"].fillna("")) - {"", "C", "1", "2", "3"}
    if _pilot:
        raise ValueError(f"unknown tick_pilot_indicator codes: {sorted(_pilot)}")
    c2_symbols = (
        _master.assign(symbol=_master["symbol_15"], cusip8=_master["cusip"].str[:8],
                       tick_pilot_indicator=_master["tick_pilot_indicator"].fillna(""))
        .merge(_names, on="cusip8", how="left")[["symbol", "cusip", "permno", "tick_pilot_indicator", "sec_type", "sec_desc"]]
    )
    c2_trades = c1_trades.merge(c2_symbols, on="symbol", how="left")
    c2_trades["tick_pilot_indicator"] = c2_trades["tick_pilot_indicator"].fillna("")
    c2_counts = c1_counts  # nothing is dropped here
    return PRICE_MIN, c2_counts, c2_symbols, c2_trades


@app.cell(hide_code=True)
def _(c2_trades, mo):
    _no = c2_trades[c2_trades["permno"].isna()]
    mo.vstack([
        mo.md(f"**Trades without a PERMNO** (kept): {len(_no):,} of {len(c2_trades):,}, by the master file's security type"),
        _no.groupby(_no["sec_type"].fillna("not in master"))["symbol"].agg(symbols="nunique", trades="size").sort_values("trades", ascending=False).reset_index(),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: sale conditions and buckets

    Each trade's conditions put it in one class:
    - **normal:** only blank / `@`, `I`, `F`;
    - **extended:** normal codes plus `T` or `U`;
    - **excluded:** anything else, including the opening and closing auctions (`O`, `Q`, `6`, `M`).

    The auctions are not retail flow: retail orders fill off-exchange, after 09:30 for orders placed overnight. The opening price for the overnight return comes from CRSP (`dlyopen`), as does the previous close.

    Buckets on a normal day, by `t_ns`, covering all 24 hours (the bucket codes of `explanations/taq/taq_pull.md`):
    - **`overnight_morning`** (0) [00:00, 09:30), normal or extended, feeds this day;
    - **`regular`** (1) [09:30, 15:50), normal, feeds this day;
    - **`last`** (2) [15:50, 16:00), normal, feeds the next trading day;
    - **`overnight_evening`** (3) [16:00, 24:00), normal or extended, feeds the next trading day.

    A decision day's `overnight` is the previous day's evening piece and its own morning piece. Early closes (13:00 close, 12:50 cutoff) move the times; 2019-06-04 is not one.
    """)
    return


@app.cell
def _(DATE, NEXT_DAY, SESSION, c2_counts, c2_trades, count_step, np, pd):
    TRADE_BUCKETS = ["overnight_morning", "regular", "last", "overnight_evening"]


    def condition_class(code):
        codes = set(code) - {" "}
        if not codes <= set("@IFTU"):
            return "excluded"
        return "extended" if codes & set("TU") else "normal"


    _codes = c2_trades["tr_scond"].fillna("")
    _unique = _codes.unique()
    _cls = _codes.map(dict(zip(_unique, map(condition_class, _unique))))
    _t, _n, _e = c2_trades["t_ns"], _cls.eq("normal"), _cls.eq("extended")
    _s = {_k: _v.value for _k, _v in SESSION.items()}
    _bucket = np.select(
        [
            (_n | _e) & (_t >= 0) & (_t < _s["open"]),
            _n & (_t >= _s["open"]) & (_t < _s["cutoff"]),
            _n & (_t >= _s["cutoff"]) & (_t < _s["close"]),
            (_n | _e) & (_t >= _s["close"]) & (_t < _s["day_end"]),
        ],
        TRADE_BUCKETS,
        default="",
    )
    c3_all = c2_trades.assign(
        condition_class=_cls.to_numpy(), bucket=_bucket,
        decision_day=np.where(np.isin(_bucket, ["last", "overnight_evening"]), NEXT_DAY, pd.Timestamp(DATE)),
    )
    c3_trades = c3_all[c3_all["bucket"].isin(TRADE_BUCKETS)]
    c3_counts = c2_counts + [count_step("normal sale conditions, in a bucket", c3_trades)]
    return TRADE_BUCKETS, c3_all, c3_counts, c3_trades


@app.cell(hide_code=True)
def _(TRADE_BUCKETS, c3_all, c3_trades, mo):
    _x = c3_all[~c3_all["bucket"].isin(TRADE_BUCKETS)]
    mo.vstack([
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

    Prices of at least $1 (Boehmer & Jones 2022) and positive price and size. Tick Size Pilot stocks are kept, with their group (none on this day: the pilot ended in September 2018).

    ## Step 5: flags

    Kept on every trade, not used to drop:
    - **`quote_state`:** `missing` if the bid or offer is empty or not above 0, `crossed` if bid > offer, `locked` if bid = offer, else `valid`. Midpoint signing needs `valid`.
    - **`large`:** $100,000 or more.

    No trade is dropped or flagged for its price's distance from the quote: most prints far outside it are sweeps through several price levels or off-exchange trades reported after the quote moved (`explanations/taq/taq_features.md`).

    ## Checks

    Every check must pass, or the notebook stops here and `clean_trades` is not created.
    """)
    return


@app.cell
def _(
    DATE,
    NEXT_DAY,
    PRICE_MIN,
    SESSION,
    TRADE_BUCKETS,
    c1_each_matched_once,
    c1_each_paired,
    c3_counts,
    c3_trades,
    count_step,
    np,
    pd,
):
    _t = c3_trades[(c3_trades["price"] >= PRICE_MIN) & (c3_trades["size"] > 0)]
    _counts = c3_counts + [count_step("price at least $1", _t)]

    _bid, _ask, _px = _t["nbb"], _t["nbo"], _t["price"]
    _state = np.select([_bid.isna() | _ask.isna() | (_bid <= 0) | (_ask <= 0), _bid > _ask, _bid == _ask],
                       ["missing", "crossed", "locked"], default="valid")
    _clean = _t.assign(quote_state=_state, large=_px * _t["size"] >= 100_000).reset_index(drop=True)
    clean_counts = pd.DataFrame(_counts).assign(share_of_raw_trades=lambda d: d["trades"] / d["trades"].iloc[0])

    _s = {_k: _v.value for _k, _v in SESSION.items()}
    _own = _clean["decision_day"].eq(pd.Timestamp(DATE))
    clean_checks = pd.DataFrame([
        ("every flagged trade-file row is exactly one WCT trade", c1_each_matched_once),
        ("every flagged trade has its cancel or correction record", c1_each_paired),
        ("every trade has a bucket", _clean["bucket"].isin(TRADE_BUCKETS).all()),
        ("every trade feeds this day or the next", _clean["decision_day"].isin([pd.Timestamp(DATE), NEXT_DAY]).all()),
        ("this day's trades are before its cutoff", (_clean.loc[_own, "t_ns"] < _s["cutoff"]).all()),
        ("every trade is between 00:00 and 24:00", _clean["t_ns"].between(0, _s["day_end"] - 1).all()),
        ("no PERMNO under two symbols", _clean.groupby("permno")["symbol"].nunique().le(1).all()),
        ("trades never increase from step to step", clean_counts["trades"].is_monotonic_decreasing),
        ("no opening or closing auction print among the trades", ~_clean["tr_scond"].fillna("").str.contains("[OQ6M]").any()),
    ], columns=["check", "passed"])
    if not clean_checks["passed"].all():
        raise ValueError("cleaning checks failed: " + "; ".join(clean_checks.loc[~clean_checks["passed"], "check"]))
    clean_trades = _clean
    return clean_checks, clean_counts, clean_trades


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
        mo.md(f"$100k or more: {int(clean_trades['large'].sum()):,}; without a PERMNO: {int(clean_trades['permno'].isna().sum()):,}; "
              f"in a Tick Size Pilot group: {int(clean_trades['tick_pilot_indicator'].ne('').sum()):,}"),
        mo.md(f"**Clean trades: first 20 of {len(clean_trades):,}**"),
        clean_trades.head(20),
    ])
    return


if __name__ == "__main__":
    app.run()
