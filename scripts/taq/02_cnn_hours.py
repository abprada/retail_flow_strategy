import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The multi-day CNN's hourly data

    For every stock-day of the training days in the panel (2014-2023), its hourly bars, from which the CNN's 21-day images are drawn on the GPU as it trains (`dsr.cnn.HourWindows`; `explanations/CNN/cnn.md`, multi-day images). Every stock in the panel, not only the large caps: a stock of $1B or more today may have been smaller three weeks ago, and its image still shows those days.

    **Nine stored slots a stock-day** (`dsr.cnn.hour_table`), each with its round-lot open, high, low and close, its shares (every trade) and its six color pixels:

    - **0-6, the hours:** 09:30, 10:30, ..., 14:30, then 15:30 to the cutoff (15:50). On early closes the last is 12:30 to 12:50 (slot 3) and slots 4-6 stay empty.
    - **7, the last hour with its last ten minutes:** to 16:00, adding the 15:50-16:00 of the next decision day's `prev_last10`. Drawn for the past days of a window; the decision day's own last bar is slot 6 (or 3), so nothing after its cutoff enters its image.
    - **8, the night before the day:** the previous day's evening and the day's pre-open, from the previous day's last round-lot price to the last trade before 09:30. Drawn for the decision day only, before its first hour.

    **Prices are split-adjusted:** each stock-day's prices times the product of CRSP's `facprc_t` over the stock's panel days up to it, a common scale in which a split is not a jump (the image scales each window to its own range, so the scale's level does not matter).

    **Colors:** the six values of `dsr.cnn.flow`, abnormal retail activity against the same slot over the previous 20 trading days (at least 15), each slot standardized with the percentiles, mean and standard deviation of that day's large caps (`cap_prev` at least $1B), applied to every stock.

    **Output:** `data/processed/cnn_hours_train.npz`: `day` (position in `days`), `permno`, `prices` (n × 9 × 4), `shares` (n × 9), `colors` (n × 9 × 6), and `days` and `early` (each day's early-close flag). The test days are not read.
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

    from dsr.cnn import (DAY_SLOTS, FLOW_COLUMNS, NIGHT_PIECES, PRICES, RetailHistory, hour_arrays,
                         hour_pixels, hour_retail_trades, hour_slots_of, hour_table)
    from dsr.panel import LARGE_CAP, SESSION_BARS

    return (
        DAY_SLOTS,
        FLOW_COLUMNS,
        LARGE_CAP,
        NIGHT_PIECES,
        PRICES,
        Path,
        RetailHistory,
        SESSION_BARS,
        argparse,
        hour_arrays,
        hour_pixels,
        hour_retail_trades,
        hour_slots_of,
        hour_table,
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
    OUT = ROOT / "data" / "processed" / "cnn_hours_train.npz"
    return OUT, TAQ_DIR


@app.cell
def _(
    DAY_SLOTS,
    FLOW_COLUMNS,
    LARGE_CAP,
    NIGHT_PIECES,
    PRICES,
    RetailHistory,
    SESSION_BARS,
    TAQ_DIR,
    hour_arrays,
    hour_pixels,
    hour_retail_trades,
    hour_slots_of,
    hour_table,
    np,
    pd,
    pq,
    time,
):
    def build(out, days=None):
        """The stored slots of every panel stock-day of the training days, in day order (the first `days` days only,
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
        history = RetailHistory(panel["permno"].unique(), DAY_SLOTS)
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
            b, p = read(day, ["session", "prev_last10", *NIGHT_PIECES])
            ten = None
            if i + 1 < len(all_days):  # the day's 15:50-16:00, in the next decision day's files
                b10, p10 = read(all_days[i + 1], ["prev_last10"])
                ten = merged(b10, p10, "prev_last10")
            stocks = members[day]
            f = factor.loc[day].reindex(stocks)
            yesterday = p[np.asarray(p["piece"], dtype=object) == "prev_last10"].set_index("permno")["close"]
            start_price = yesterday.reindex(stocks) / f["facprc_t"]  # the previous day's last price, in today's terms
            n_bars = SESSION_BARS[bool(early[day])]
            table = hour_table(merged(b, p, "session"), ten,
                               b[np.isin(np.asarray(b["piece"], dtype=object), NIGHT_PIECES)], start_price, n_bars)
            table = table[table.index.get_level_values("permno").isin(stocks)]
            d = hour_arrays(table, hour_pixels(table, history, large.get(day, set())), stocks)
            d["prices"] *= f["factor"].to_numpy(np.float32)[:, None, None]
            parts.append({"day": np.full(len(stocks), i, np.int16), "permno": np.asarray(stocks, np.int64), **d})
            history.add(stocks, hour_slots_of(n_bars), hour_retail_trades(table))  # after: the 20 days before
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
    uv run python scripts/taq/02_cnn_hours.py build                                       # every training day
    uv run python scripts/taq/02_cnn_hours.py build --days 40 --out /tmp/cnn_hours.npz   # a quick check
    ```
    """)
    return


@app.cell
def _(OUT, Path, argparse, build, mo, sys):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="02_cnn_hours.py")
        sub = parser.add_subparsers(dest="command", required=True)
        b = sub.add_parser("build", help="build the hourly data of every panel stock-day of the training days")
        b.add_argument("--days", type=int, help="only the first N days, as a quick check")
        b.add_argument("--out", type=Path, default=OUT)
        args = parser.parse_args(argv)
        print(build(args.out, args.days), flush=True)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


if __name__ == "__main__":
    app.run()
