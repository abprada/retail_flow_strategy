import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The daily CNN's data

    For every stock-day of the training days in the panel (2014-2023), its daily bar, from which the CNN's 20-day images are drawn on the GPU as it trains (`dsr.cnn.DayWindows`): Jiang, Kelly and Xiu's 20-day image (64 × 60), built from our own round-lot prices and with the six retail colors. Every stock in the panel, not only the large caps: a stock of $1B or more today may have been smaller in the 39 days its image and moving average look back.

    **Two stored daily bars a stock-day** (`dsr.cnn.daily_table`), each with its round-lot open, high, low and close, its shares (every trade) and its six color pixels:

    - **0, the session to the cutoff** (09:30-15:50; 12:50 on early closes), drawn for the decision day, so nothing after its cutoff enters its image.
    - **1, the session to the close,** adding the 15:50-16:00 of the next decision day's `prev_last10`, drawn for the past days.

    The night is not a bar: it shows as the jump from one day's close to the next day's open.

    **Prices are split-adjusted** as in `02_cnn_hours.py`: each stock-day's prices times the product of CRSP's `facprc_t` over the stock's panel days up to it.

    **Colors:** the six values of `dsr.cnn.flow` over the day's counts, abnormal retail activity against the same stored bar over the previous 20 trading days (at least 15), each standardized with the percentiles, mean and standard deviation of that day's large caps, applied to every stock.

    **Output:** `data/processed/cnn_days_train.npz`: `day` (position in `days`), `permno`, `prices` (n × 2 × 4), `shares` (n × 2), `colors` (n × 2 × 6), `days` and `early`. The test days are not read.
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

    from dsr.cnn import (D_SLOTS, FLOW_COLUMNS, PRICES, RetailHistory, daily_table, hour_arrays, hour_pixels,
                         hour_retail_trades)
    from dsr.panel import LARGE_CAP

    return (
        D_SLOTS,
        FLOW_COLUMNS,
        LARGE_CAP,
        PRICES,
        Path,
        RetailHistory,
        argparse,
        daily_table,
        hour_arrays,
        hour_pixels,
        hour_retail_trades,
        mo,
        np,
        pd,
        pq,
        sys,
        time,
    )


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    TAQ_DIR = ROOT / "data" / "intermediate" / "taq"
    OUT = ROOT / "data" / "processed" / "cnn_days_train.npz"
    return OUT, TAQ_DIR


@app.cell
def _(
    D_SLOTS,
    FLOW_COLUMNS,
    LARGE_CAP,
    PRICES,
    RetailHistory,
    TAQ_DIR,
    daily_table,
    hour_arrays,
    hour_pixels,
    hour_retail_trades,
    np,
    pd,
    pq,
    time,
):
    def build(out, days=None):
        """The stored daily bars of every panel stock-day of the training days, in day order (the first `days` days only,
        as a quick check), written to `out`."""
        panel = pq.read_table(TAQ_DIR / "panel_train.parquet",
                              columns=["day", "permno", "cap_prev", "early_close", "facprc_t"]).to_pandas()
        panel = panel.sort_values(["permno", "day"])
        panel["factor"] = panel.groupby("permno")["facprc_t"].cumprod()  # prices x factor: one scale over time
        all_days = list(pd.DatetimeIndex(sorted(panel["day"].unique())))[:days]
        members = panel.groupby("day")["permno"].agg(sorted)
        large = panel[panel["cap_prev"] >= LARGE_CAP].groupby("day")["permno"].agg(set)
        early = panel.groupby("day")["early_close"].first()
        factor = panel.set_index(["day", "permno"])[["factor", "facprc_t"]]
        history = RetailHistory(panel["permno"].unique(), D_SLOTS)
        px_cols = [f"px_{p}" for p in PRICES]

        def read(day, pieces):
            folder = TAQ_DIR / "train" / f"{day:%Y%m%d}"
            b = pq.read_table(folder / "bars.parquet", columns=["permno", "piece", "bar", "close", *FLOW_COLUMNS],
                              filters=[("piece", "in", pieces)]).to_pandas()
            p = pq.read_table(folder / "prices.parquet", columns=["permno", "piece", "bar", *px_cols],
                              filters=[("piece", "in", pieces)]).to_pandas()
            return b, p.rename(columns=lambda c: c.removeprefix("px_")).astype({q: float for q in PRICES})

        def merged(b, p, piece):
            on = ["permno", "bar"]
            bb = b[np.asarray(b["piece"], dtype=object) == piece].drop(columns=["piece", "close"])
            pp = p[np.asarray(p["piece"], dtype=object) == piece].drop(columns="piece")
            return bb.merge(pp, on=on, how="outer")

        parts, start = [], time.perf_counter()
        for i, day in enumerate(all_days):
            b, p = read(day, ["session"])
            ten = None
            if i + 1 < len(all_days):  # the day's 15:50-16:00, in the next decision day's files
                b10, p10 = read(all_days[i + 1], ["prev_last10"])
                ten = merged(b10, p10, "prev_last10")
            stocks = members[day]
            f = factor.loc[day].reindex(stocks)
            table = daily_table(merged(b, p, "session"), ten)
            table = table[table.index.get_level_values("permno").isin(stocks)]
            d = hour_arrays(table, hour_pixels(table, history, large.get(day, set())), stocks, D_SLOTS)
            d["prices"] *= f["factor"].to_numpy(np.float32)[:, None, None]
            parts.append({"day": np.full(len(stocks), i, np.int16), "permno": np.asarray(stocks, np.int64), **d})
            history.add(stocks, D_SLOTS, hour_retail_trades(table))  # after it: the 20 days before
            if (i + 1) % 100 == 0 or i + 1 == len(all_days):
                print(f"{day:%Y-%m-%d}: {i + 1} of {len(all_days)} days, "
                      f"{(time.perf_counter() - start) / (i + 1):.2f} s a day", flush=True)
        arrays = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        arrays["days"] = np.array([d.date() for d in all_days], dtype="datetime64[D]")
        arrays["early"] = np.array([bool(early[d]) for d in all_days])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **arrays)
        return {k: v.shape for k, v in arrays.items()}

    return (build,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it (from the terminal)

    ```
    uv run python scripts/taq/02_cnn_days.py build                                      # every training day
    uv run python scripts/taq/02_cnn_days.py build --days 60 --out /tmp/cnn_days.npz   # a quick check
    ```
    """)
    return


@app.cell
def _(OUT, Path, argparse, build, mo, sys):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="02_cnn_days.py")
        sub = parser.add_subparsers(dest="command", required=True)
        b = sub.add_parser("build", help="build the daily bars of every panel stock-day of the training days")
        b.add_argument("--days", type=int, help="only the first N days, as a quick check")
        b.add_argument("--out", type=Path, default=OUT)
        args = parser.parse_args(argv)
        print(build(args.out, args.days), flush=True)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


if __name__ == "__main__":
    app.run()
