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
    import marimo as mo
    import pyarrow.parquet as pq

    return mo, pq


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parent / "data"
    PULL_DIR = DATA_DIR / "intermediate" / "taq" / "by_wct_day"  # one folder per WCT day: daily, cells, bars, audit
    MASTER_DIR = DATA_DIR / "raw" / "taq" / "master"  # mastm_<YYYYMMDD>.parquet, every column as text
    DAY = "20190604"
    return DAY, MASTER_DIR, PULL_DIR


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


if __name__ == "__main__":
    app.run()
