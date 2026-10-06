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
    import pyarrow.parquet as pq

    return mo, pq


@app.cell
def _(mo):
    DATE = "2019-06-04"
    RAW_DIR = mo.notebook_dir().parent / "data" / "raw" / "taq" / DATE
    return (RAW_DIR,)


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


if __name__ == "__main__":
    app.run()
