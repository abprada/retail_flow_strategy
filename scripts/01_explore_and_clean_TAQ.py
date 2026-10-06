import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # TAQ: explore, then clean (one-minute test run)

    Reads the minute that `00_fetch_TAQ.py` downloads into `data/raw/taq/<date>/`; nothing here writes to it. Cleaned output goes to `data/intermediate/taq/`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parent / "data"
    DATE = "2019-06-04"
    HHMM = "1549"
    RAW_DIR = DATA_DIR / "raw" / "taq" / DATE
    CRSP_NAMES_PATH = DATA_DIR / "raw" / "crsp" / "stocknames_v2.csv"
    INTERMEDIATE_DIR = DATA_DIR / "intermediate" / "taq"
    return CRSP_NAMES_PATH, DATA_DIR, DATE, HHMM, INTERMEDIATE_DIR, RAW_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## Raw files

    Everything is read as text, so codes such as `tr_corr = "00"` and leading zeros in CUSIPs survive.
    """)
    return


@app.cell
def _(HHMM, RAW_DIR, pd):
    tables = pd.read_csv(RAW_DIR / "tables.csv")
    trades_raw = pd.read_csv(RAW_DIR / f"ctm_{HHMM}.csv", dtype=str, keep_default_na=False)
    nbbo_raw = pd.read_csv(RAW_DIR / f"complete_nbbo_{HHMM}.csv", dtype=str, keep_default_na=False)
    master_raw = pd.read_csv(RAW_DIR / "mastm.csv", dtype=str, keep_default_na=False)
    return master_raw, nbbo_raw, tables, trades_raw


@app.cell
def _(master_raw, mo, nbbo_raw, tables, trades_raw):
    mo.md(f"""
    - **taqmsec tables that day:** {", ".join(f"`{t}`" for t in tables["table_name"])}
    - **Trades (`ctm`):** {len(trades_raw):,} rows; columns {", ".join(f"`{c}`" for c in trades_raw.columns)}
    - **NBBO (`complete_nbbo`):** {len(nbbo_raw):,} rows; columns {", ".join(f"`{c}`" for c in nbbo_raw.columns)}
    - **Master (`mastm`):** {len(master_raw):,} rows; columns {", ".join(f"`{c}`" for c in master_raw.columns)}
    """)
    return


@app.cell
def _(trades_raw):
    trades_raw.head(20)
    return


@app.cell
def _(nbbo_raw):
    nbbo_raw.head(20)
    return


@app.cell
def _(master_raw):
    master_raw.head(20)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Trades

    The codes that decide which prints are real trades:

    - `tr_corr`: `00` is a regular trade; anything else was corrected or cancelled later.
    - `tr_scond`: sale conditions, e.g. `@` regular, `I` odd lot, `F` intermarket sweep, `T` extended hours, `Z` out of sequence.
    - `ex`: the reporting venue; `D` is FINRA's TRF, where off-exchange (including wholesaler-internalized retail) trades print.
    """)
    return


@app.cell
def _(pd, trades_raw):
    trades = trades_raw.assign(
        price=pd.to_numeric(trades_raw["price"]),
        size=pd.to_numeric(trades_raw["size"]),
        time=pd.to_timedelta(trades_raw["time_m"]),
        symbol=(trades_raw["sym_root"] + " " + trades_raw["sym_suffix"]).str.strip(),
    )
    trades["dollars"] = trades["price"] * trades["size"]
    return (trades,)


@app.cell
def _(mo, trades):
    _per_symbol = trades.groupby("symbol").size()
    mo.md(f"""
    - **Time range:** {trades["time"].min()} to {trades["time"].max()}
    - **Symbols:** {trades["symbol"].nunique():,}; trades per symbol median {_per_symbol.median():.0f}, max {_per_symbol.max():,} (`{_per_symbol.idxmax()}`)
    - **Shares:** {trades["size"].sum():,.0f}; dollars {trades["dollars"].sum() / 1e6:,.0f} million
    - **Size:** median {trades["size"].median():.0f}; {(trades["size"] < 100).mean():.0%} of prints are odd lots (under 100 shares)
    - **Non-positive price or size:** {((trades["price"] <= 0) | (trades["size"] <= 0)).sum():,}
    """)
    return


@app.cell
def _(mo, pd, trades):
    def _share(col):
        _n = trades[col].value_counts()
        return pd.DataFrame({"trades": _n, "share": _n / len(trades)}).head(15)

    mo.hstack([
        mo.vstack([mo.md("**`ex`**"), _share("ex")]),
        mo.vstack([mo.md("**`tr_corr`**"), _share("tr_corr")]),
        mo.vstack([mo.md("**`tr_scond`**"), _share("tr_scond")]),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## NBBO

    `complete_nbbo` has a row each time the national best bid or offer changes. `best_bid` and `best_ask` are the NBBO after the change.
    """)
    return


@app.cell
def _(mo, nbbo_raw, pd):
    nbbo = nbbo_raw.assign(
        bid=pd.to_numeric(nbbo_raw["best_bid"], errors="coerce"),
        ask=pd.to_numeric(nbbo_raw["best_ask"], errors="coerce"),
        time=pd.to_timedelta(nbbo_raw["time_m"]),
        symbol=(nbbo_raw["sym_root"] + " " + nbbo_raw["sym_suffix"]).str.strip(),
    )
    nbbo["mid"] = (nbbo["bid"] + nbbo["ask"]) / 2
    nbbo["spread_bp"] = (nbbo["ask"] - nbbo["bid"]) / nbbo["mid"] * 1e4
    _valid = (nbbo["bid"] > 0) & (nbbo["ask"] > 0)
    mo.md(f"""
    - **Time range:** {nbbo["time"].min()} to {nbbo["time"].max()}
    - **Symbols:** {nbbo["symbol"].nunique():,}
    - **Missing or zero bid or ask:** {(~_valid).sum():,}
    - **Crossed (bid above ask):** {(_valid & (nbbo["bid"] > nbbo["ask"])).sum():,}; locked (bid equals ask): {(_valid & (nbbo["bid"] == nbbo["ask"])).sum():,}
    - **Quoted spread, bp, valid rows:** median {nbbo.loc[_valid, "spread_bp"].median():.1f}, 90th percentile {nbbo.loc[_valid, "spread_bp"].quantile(0.9):.1f}
    """)
    return (nbbo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Matching symbols to PERMNOs

    TAQ's master file gives each symbol's CUSIP that day. Its `symbol_15` is the trade file's `sym_root` and `sym_suffix` joined by a space; TAQ splits some plain Nasdaq tickers that way (`GOOGL` is `GOOG L`, `CMCSA` is `CMCS A`). A symbol is matched to the PERMNO whose CRSP name record, valid on that date, has the same 8-digit CUSIP. Tickers alone are not used.
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, DATE, master_raw, mo, pd, trades):
    _names = pd.read_csv(CRSP_NAMES_PATH, dtype=str, usecols=["permno", "namedt", "nameenddt", "cusip", "ticker"])
    _names = _names[(_names["namedt"] <= DATE) & (_names["nameenddt"] >= DATE)]
    master = master_raw.assign(symbol=master_raw["symbol_15"].str.strip(), cusip8=master_raw["cusip"].str[:8])
    symbol_permno = master.merge(_names, left_on="cusip8", right_on="cusip", how="left", suffixes=("", "_crsp"))[
        ["symbol", "cusip8", "permno", "ticker"]
    ]
    _many = symbol_permno.groupby("symbol")["permno"].nunique()
    symbol_permno = symbol_permno.drop_duplicates("symbol")
    _traded = trades[["symbol", "dollars"]].merge(symbol_permno, on="symbol", how="left")
    _matched = _traded["permno"].notna()
    mo.md(f"""
    - **Master symbols:** {len(master):,}; with a PERMNO: {symbol_permno["permno"].notna().sum():,}; with more than one PERMNO: {(_many > 1).sum():,}
    - **Traded symbols with a PERMNO:** {_traded.loc[_matched, "symbol"].nunique():,} of {_traded["symbol"].nunique():,}
    - **Trades with a PERMNO:** {_matched.mean():.1%}; by dollars {_traded.loc[_matched, "dollars"].sum() / _traded["dollars"].sum():.1%}
    - **TAQ symbol differs from CRSP ticker** (matched by CUSIP): {(symbol_permno["permno"].notna() & (symbol_permno["symbol"].str.replace(" ", "") != symbol_permno["ticker"])).sum():,}
    """)
    return (symbol_permno,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Clean

    Mechanical rules only, nothing chosen by looking at values:

    - **Trades:** keep `tr_corr = 00` with positive price and size. Sale conditions are kept as a column, not filtered, until features decide which ones they use.
    - **NBBO:** keep rows with a positive bid and ask and bid at most ask.
    - Both carry the PERMNO from the CUSIP match; unmatched symbols (ETNs, warrants, preferreds, test symbols) stay with a missing PERMNO.
    """)
    return


@app.cell
def _(mo, nbbo, symbol_permno, trades):
    _keep_cols = ["time", "symbol", "ex", "tr_scond", "price", "size", "dollars"]
    trades_clean = trades.loc[
        (trades["tr_corr"] == "00") & (trades["price"] > 0) & (trades["size"] > 0), _keep_cols
    ].merge(symbol_permno[["symbol", "permno"]], on="symbol", how="left")
    nbbo_clean = nbbo.loc[
        (nbbo["bid"] > 0) & (nbbo["ask"] > 0) & (nbbo["bid"] <= nbbo["ask"]), ["time", "symbol", "bid", "ask", "mid", "spread_bp"]
    ].merge(symbol_permno[["symbol", "permno"]], on="symbol", how="left")
    mo.md(f"""
    - **Trades:** {len(trades):,} raw, {len(trades_clean):,} kept
    - **NBBO:** {len(nbbo):,} raw, {len(nbbo_clean):,} kept
    """)
    return nbbo_clean, trades_clean


@app.cell
def _(mo):
    write_taq = mo.ui.run_button(label="Write the clean minute")
    write_taq
    return (write_taq,)


@app.cell
def _(DATA_DIR, DATE, HHMM, INTERMEDIATE_DIR, mo, nbbo_clean, trades_clean, write_taq):
    mo.stop(not write_taq.value, mo.md("Press the button to write."))
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    _files = {f"trades_{DATE}_{HHMM}": trades_clean, f"nbbo_{DATE}_{HHMM}": nbbo_clean}
    for _name, _df in _files.items():
        _part = INTERMEDIATE_DIR / f"{_name}.parquet.part"
        _df.to_parquet(_part, index=False)
        _part.rename(INTERMEDIATE_DIR / f"{_name}.parquet")
    mo.md(f"Written to `{INTERMEDIATE_DIR.relative_to(DATA_DIR.parent)}`: " + ", ".join(f"`{k}.parquet` ({len(v):,} rows)" for k, v in _files.items()))
    return


if __name__ == "__main__":
    app.run()
