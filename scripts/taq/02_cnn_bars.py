import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The CNN's drawing data

    For every large-cap stock-day of the training days (`cap_prev` at least `LARGE_CAP`, 2014-2023), what its chart image draws: where each bar's open, high, low, close and volume fall, and its six color values (`explanations/CNN/cnn.md`). The images themselves are drawn from it on the GPU as the CNN trains (`dsr.cnn.render`), so no image is stored and the drawing can change without a rebuild.

    - **40 bars a stock-day:** the previous trading day's last ten minutes, the night (its evening and today's pre-open), then the session's 38 ten-minute bars (20 on early closes, the rest left empty), 3 pixels each: 4 × 64 × 120 images. The two bars before the session are those of the last section of `notebooks/CNN/test_image.py`.
    - **Prices:** round-lot (`px_` in `prices.parquet`) for the session and the last ten minutes; the night spans the last ten minutes' close to the last trade before 09:30. The scale runs from the stock-day's highest high to its lowest low.
    - **Volume:** every trade, scaled to the session's tallest bar; the two bars before the session are capped at it.
    - **Colors:** each bar's six values, standardized across the large caps' same bar that day. Abnormal retail activity compares a bar's retail trades with the same bar over the previous 20 trading days (at least 15), each day counted for the stocks in the panel that day.
    - **Point in time:** a day's colors use only that day's bars, all before its cutoff, and the 20 days before it.

    **Input:** `data/intermediate/taq/panel_train.parquet` and each training day's `bars.parquet` and `prices.parquet`. **Output:** `data/processed/cnn_bars_train.npz`: `day`, `permno`, and the drawing data `rows` (n × 40 × 4), `volume` (n × 40) and `colors` (n × 40 × 6), all uint8 (`dsr.cnn.drawing`). The test days are not read.
    """)
    return


@app.cell
def _():
    import argparse
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    from dsr.cnn import (FLOW_COLUMNS, FIRST_SESSION_SLOT, PRICES, RetailHistory, day_retail_trades, day_table,
                         drawing, pixel_values)
    from dsr.panel import LARGE_CAP, SESSION_BARS

    return (
        FIRST_SESSION_SLOT,
        FLOW_COLUMNS,
        LARGE_CAP,
        PRICES,
        Path,
        RetailHistory,
        SESSION_BARS,
        argparse,
        day_retail_trades,
        day_table,
        drawing,
        mo,
        np,
        pd,
        pixel_values,
        pq,
        sys,
        time,
    )


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    TAQ_DIR = ROOT / "data" / "intermediate" / "taq"
    OUT = ROOT / "data" / "processed" / "cnn_bars_train.npz"
    return OUT, TAQ_DIR


@app.cell
def _(
    FIRST_SESSION_SLOT,
    FLOW_COLUMNS,
    LARGE_CAP,
    PRICES,
    RetailHistory,
    SESSION_BARS,
    TAQ_DIR,
    day_retail_trades,
    day_table,
    drawing,
    np,
    pd,
    pixel_values,
    pq,
    time,
):
    def build(out, days=None):
        """The drawing data of every large-cap stock-day of the training days, in day order (the first `days` days
        only, as a quick check), written to `out`."""
        panel = pq.read_table(TAQ_DIR / "panel_train.parquet",
                              columns=["day", "permno", "cap_prev", "early_close"]).to_pandas()
        all_days = list(pd.DatetimeIndex(sorted(panel["day"].unique())))[:days]
        members = panel.groupby("day")["permno"].agg(set)
        large = panel[panel["cap_prev"] >= LARGE_CAP].groupby("day")["permno"].agg(sorted)
        early = panel.groupby("day")["early_close"].first()
        history = RetailHistory(panel["permno"].unique())
        parts, start = [], time.perf_counter()
        for i, day in enumerate(all_days):
            folder = TAQ_DIR / "train" / f"{day:%Y%m%d}"
            bars = pq.read_table(folder / "bars.parquet",
                                 columns=["permno", "piece", "bar", "close", *FLOW_COLUMNS]).to_pandas()
            prices = pq.read_table(folder / "prices.parquet",
                                   columns=["permno", "piece", "bar", *[f"px_{p}" for p in PRICES]]).to_pandas()
            stocks = large.get(day, [])
            table = day_table(bars[bars["permno"].isin(stocks)], prices[prices["permno"].isin(stocks)])
            d = drawing(table, pixel_values(table, history), stocks)  # the history holds the 20 days before
            parts.append({"day": np.full(len(stocks), np.datetime64(day.date(), "D")),
                          "permno": np.asarray(stocks, dtype=np.int64), **d})
            history.add(members[day], FIRST_SESSION_SLOT + SESSION_BARS[bool(early[day])], day_retail_trades(bars))
            if (i + 1) % 100 == 0 or i + 1 == len(all_days):
                print(f"{day:%Y-%m-%d}: {i + 1} of {len(all_days)} days, "
                      f"{(time.perf_counter() - start) / (i + 1):.2f} s a day", flush=True)
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        return {k: v.shape for k, v in arrays.items()}

    return (build,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it (from the terminal)

    ```
    uv run python scripts/taq/02_cnn_bars.py build                                      # every training day
    uv run python scripts/taq/02_cnn_bars.py build --days 30 --out /tmp/cnn_check.npz   # a quick check
    ```

    Opened in marimo, this cell only shows the commands; the cell below reads what was built.
    """)
    return


@app.cell
def _(OUT, Path, argparse, build, mo, sys):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="02_cnn_bars.py")
        sub = parser.add_subparsers(dest="command", required=True)
        b = sub.add_parser("build", help="build the drawing data of every large-cap training stock-day")
        b.add_argument("--days", type=int, help="only the first N days, as a quick check")
        b.add_argument("--out", type=Path, default=OUT)
        args = parser.parse_args(argv)
        print(build(args.out, args.days), flush=True)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


@app.cell
def _(OUT, mo, np, pd):
    if OUT.exists():
        with np.load(OUT) as _z:
            _keys = pd.DataFrame({"day": _z["day"], "permno": _z["permno"]})
            _drawn = (_z["volume"] > 0).mean(axis=0)
        _by_year = _keys.groupby(_keys["day"].dt.year).agg(days=("day", "nunique"), stock_days=("permno", "size"))
        _out = mo.vstack([mo.md(f"**Built:** {len(_keys):,} large-cap stock-days on {_keys['day'].nunique():,} days."),
                          _by_year,
                          mo.md(f"Share of stock-days with a volume bar: the last ten minutes {_drawn[0]:.1%}, the "
                                f"night {_drawn[1]:.1%}, the first session bar {_drawn[2]:.1%}, the last "
                                f"{_drawn[39]:.1%} (early closes stop at bar 20).")])
    else:
        _out = mo.md("Not built yet: run `uv run python scripts/taq/02_cnn_bars.py build`.")
    _out
    return


if __name__ == "__main__":
    app.run()
