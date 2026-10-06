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
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    return mo, pa, pc, pd, pq


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


if __name__ == "__main__":
    app.run()
