"""The TAQ pull re-keyed by decision day: each day holds what is known at its 15:50 cutoff.

The pull writes one folder per WCT day (a trading day of the tape, data/raw/taq/pull/). A decision
day D uses its own morning piece and regular session (buckets 0 and 1) and the previous WCT day's last 10 minutes and
evening piece (buckets 2 and 3), which come after that day's cutoff. Rows are copied unchanged: nothing is combined,
recomputed, added or dropped. `daily` (the session's spread) is the day's own. `audit` stays by WCT day. Each
cells and bars row gets a readable `piece` (which of the four time pieces it is), and rows run by symbol and in time
order. CODES says what every code of the tables means; it is written next to the data as codes.csv.
"""

import shutil
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TABLES = ["daily", "cells", "bars"]
OWN, PREVIOUS = [0, 1], [2, 3]  # buckets a decision day takes from its own WCT day and from the one before
# the four time pieces of a decision day, in time order: bucket -> piece
PIECES = {2: "prev_last10", 3: "prev_evening", 0: "pre_open", 1: "session"}
_ORDER = {bucket: i for i, bucket in enumerate(PIECES)}

CODES = pd.DataFrame([
    ("bucket", 2, "prev_last10", "previous trading day, from its cutoff (15:50; 12:50 on early closes) to its close"),
    ("bucket", 3, "prev_evening", "previous trading day, from its close to 24:00 (normal and extended-hours trades)"),
    ("bucket", 0, "pre_open", "the decision day, 00:00 to 09:30 (normal and extended-hours trades)"),
    ("bucket", 1, "session", "the decision day, 09:30 to its cutoff (15:50; 12:50 on early closes)"),
    ("venue", 0, "exchange", "an exchange"),
    ("venue", 1, "off-exchange", "a FINRA trade reporting facility (ex = 'D'): wholesalers, dark pools"),
    *[("z", c, label, "the fraction of a cent in the price") for c, label in enumerate([
        "whole cent", "(0, .1)", "[.1, .2)", "[.2, .3)", "[.3, .4)", "[.4, .5)", "exactly .5", "(.5, .6]", "(.6, .7]",
        "(.7, .8]", "(.8, .9]", "(.9, 1)"])],
    *[("q", c, label, "where the price sits in the national best bid and offer, as a fraction of the spread")
      for c, label in enumerate([
          "no valid quote", "below the bid", "at the bid", "(0, .1)", "[.1, .2)", "[.2, .3)", "[.3, .4)", "[.4, .5)",
          "exactly the midpoint", "(.5, .6]", "(.6, .7]", "(.7, .8]", "(.8, .9]", "(.9, 1)", "at the ask",
          "above the ask"])],
    *[("spread", c, label, "the width of the national best bid and offer") for c, label in enumerate([
        "no valid quote", "1 cent or less", "2 cents", "3 to 5 cents", "over 5 cents"])],
    *[("size", c, label, "the trade's size, tested in this order: $100,000 or more first, then shares")
      for c, label in enumerate(["1 share", "2 to 99 shares", "100 to 499 shares", "500 shares or more",
                                 "$100,000 or more"])],
], columns=["column", "code", "label", "meaning"])


def _with_links(table: pd.DataFrame, links: pd.DataFrame | None, day: str) -> pd.DataFrame:
    """The table with traded_day, permno and primary after symbol, from one WCT day (YYYYMMDD) and its links
    (symbol, permno, primary)."""
    if links is None:
        links = pd.DataFrame({"symbol": pd.Series(dtype=str), "permno": pd.array([], dtype="Int64"),
                              "primary": pd.array([], dtype=bool)})
    out = table.drop(columns=["permno", "primary"], errors="ignore").merge(
        links[["symbol", "permno", "primary"]], on="symbol", how="left", validate="many_to_one")
    out["permno"] = out["permno"].astype("Int64")
    out["primary"] = out["primary"].fillna(False).astype(bool)
    out["traded_day"] = pd.Timestamp(day)
    new = ["traded_day", "permno", "primary"]
    at = out.columns.get_loc("symbol") + 1
    return out[[*out.columns[:at], *new, *[c for c in out.columns[at:] if c not in new]]]


def decision_day_tables(previous: dict | None, own: dict, previous_links: pd.DataFrame | None,
                        own_links: pd.DataFrame, previous_day: str | None, own_day: str) -> dict:
    """A decision day's tables from its own WCT day's tables and the previous WCT day's (None for the first day).

    Every row gets traded_day, the WCT day it traded (YYYYMMDD given), and the permno and primary of that day's links
    (symbol, permno, primary; data/intermediate/taq/symbols.parquet), so the previous day's pieces keep that day's
    PERMNO across a ticker change, and anything joined later by (traded_day, symbol) is the right day's.
    """
    out = {"daily": _with_links(own["daily"], own_links, own_day).sort_values("symbol", ignore_index=True)}
    for name in ["cells", "bars"]:
        parts = [_with_links(own[name][own[name]["bucket"].isin(OWN)], own_links, own_day)]
        if previous is not None:
            parts.insert(0, _with_links(previous[name][previous[name]["bucket"].isin(PREVIOUS)], previous_links,
                                        previous_day))
        table = pd.concat(parts, ignore_index=True)
        order = ["symbol", "_order", *[c for c in ["bar", "venue", "z", "q", "spread", "size"] if c in table]]
        table = table.assign(_order=table["bucket"].map(_ORDER)).sort_values(order, ignore_index=True)
        table.insert(table.columns.get_loc("bucket") + 1, "piece", table["bucket"].map(PIECES))
        out[name] = table.drop(columns="_order")
    return out


def write_decision_days(pull_dir: Path, out_dir: Path, wct_days: list[str], test_start: str, links,
                        progress=iter) -> pd.DataFrame:
    """Write every decision day after the first WCT day to out_dir/train/<YYYYMMDD>/ or out_dir/test/<YYYYMMDD>/.

    wct_days: every WCT day pulled, in order (YYYYMMDD); the first only lends its buckets 2 and 3 to the second.
    links: a function from a WCT day (YYYYMMDD) to that day's symbol, permno and primary.
    Days before test_start go to train/. Each day is written to a .part folder and renamed when complete, and days
    already written are skipped. Returns one row per decision day: its period and the rows of each table.
    """
    def read(day):
        return {name: pq.read_table(pull_dir / day / f"{name}.parquet").to_pandas() for name in TABLES}

    rows, previous = [], None  # previous: the last WCT day read, kept for its buckets 2 and 3
    for prev_day, day in progress(list(pairwise(wct_days))):
        period = "train" if day < test_start else "test"
        final = out_dir / period / day
        if final.exists():
            previous = None
        else:
            own = read(day)
            tables = decision_day_tables(previous if previous is not None else read(prev_day), own,
                                         links(prev_day), links(day), prev_day, day)
            part = final.with_name(day + ".part")
            shutil.rmtree(part, ignore_errors=True)
            part.mkdir(parents=True)
            for name, table in tables.items():
                table.to_parquet(part / f"{name}.parquet", index=False, compression="zstd")
            part.rename(final)
            previous = own
        rows.append({"decision_day": day, "period": period,
                     **{f"{name}_rows": pq.ParquetFile(final / f"{name}.parquet").metadata.num_rows for name in TABLES}})
    return pd.DataFrame(rows)
