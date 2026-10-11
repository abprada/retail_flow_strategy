"""The TAQ pull by decision day, for the universe only: each day holds what is known at its 15:50 cutoff.

The pull writes one folder per WCT day (a trading day of the tape, data/raw/taq/pull/). A decision day D uses the
previous WCT day's last 10 minutes and evening piece (buckets 2 and 3, after that day's cutoff) and its own pre-open
and session to the cutoff (buckets 0 and 1). Each row takes the PERMNO its symbol had on the day it traded, so the
previous day's pieces keep that day's PERMNO across a ticker change.

The universe is decided per decision day with what was known the day before (stock_days): CRSP's security type that
day is common stock, foreign common stock, ADR, ETF or REIT, and the previous trading day's close is at least $1 and
its market cap at least $50M. A decision day keeps the bars and cells rows of its universe stocks' primary symbols,
and one daily row per universe stock, whether or not it traded, with what is known before 15:50 and the target.
Trade rows are copied unchanged otherwise; `audit` stays with the pull.
"""

import shutil
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from dsr.crsp import ONE_TRADING_DAY_SPANS, class_on, is_common_stock, security_type

TABLES = ["daily", "cells", "bars"]
OWN, PREVIOUS = [0, 1], [2, 3]  # buckets a decision day takes from its own WCT day and from the one before
# the four time pieces of a decision day, in time order: bucket -> piece
PIECES = {2: "prev_last10", 3: "prev_evening", 0: "pre_open", 1: "session"}
_ORDER = {bucket: i for i, bucket in enumerate(PIECES)}

UNIVERSE_TYPES = ("common stock", "foreign common stock", "ADR", "ETF", "REIT")
PRICE_MIN = 1.0  # previous trading day's close, dollars
CAP_MIN = 50_000  # previous trading day's market cap; CRSP's dlycap is in thousands of dollars, so $50M

DAILY_COLUMNS = ["permno", "symbol", "security_type", "is_common", "pilot_group", "early_close", "price_prev",
                 "cap_prev", "dollar_volume_prev", "ret_prev", "open_t", "facprc_t", "traded_before_cutoff",
                 "spread_bp", "ret_next", "ret_next_is_delisting"]

CODES = pd.DataFrame([
    ("piece", 0, "prev_last10", "previous trading day, from its cutoff (15:50; 12:50 on early closes) to its close"),
    ("piece", 1, "prev_evening", "previous trading day, from its close to 24:00 (normal and extended-hours trades)"),
    ("piece", 2, "pre_open", "the decision day, 00:00 to 09:30 (normal and extended-hours trades)"),
    ("piece", 3, "session", "the decision day, 09:30 to its cutoff (15:50; 12:50 on early closes); bar = 10-minute bar"),
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
], columns=["column", "code", "label", "meaning"])  # piece codes give the time order; the files store the label


def stock_days(crsp: pd.DataFrame, name_records: pd.DataFrame, wct_days: list[str], test_start: str,
               early_closes: set[str]) -> pd.DataFrame:
    """One row per decision day and PERMNO in the universe, with what is known before its 15:50 and the target.

    crsp: CRSP daily rows (permno, date, price, cap in thousands, dollar_volume, ret, ret_span, open, facprc).
    name_records: dsr.crsp.read_name_records. wct_days: every trading day pulled (YYYYMMDD), in order; decision days
    are all but the first. The target ret_next (CRSP's return from the decision day's close to the next trading day's,
    delisting returns included) is kept only when it spans one trading day, and left empty when it ends in the test
    period for a training day (the embargo) or after the last day.
    """
    days = pd.DatetimeIndex(pd.to_datetime(wct_days))
    nxt = pd.Series(days[1:], index=days[:-1])
    prev = crsp[crsp["date"].isin(days[:-1])].rename(columns={
        "price": "price_prev", "cap": "cap_prev", "dollar_volume": "dollar_volume_prev", "ret": "ret_prev"})
    prev = prev[prev["price_prev"].abs().ge(PRICE_MIN) & prev["cap_prev"].ge(CAP_MIN)]
    out = prev.assign(decision_day=prev["date"].map(nxt))[
        ["decision_day", "permno", "price_prev", "cap_prev", "dollar_volume_prev", "ret_prev"]]

    cls = class_on(out[["permno"]].assign(date=out["decision_day"]), name_records)
    out = out.assign(security_type=security_type(cls), is_common=is_common_stock(cls))
    out = out[out["security_type"].isin(UNIVERSE_TYPES)]

    today = crsp[["permno", "date", "open", "facprc"]].rename(
        columns={"date": "decision_day", "open": "open_t", "facprc": "facprc_t"})
    out = out.merge(today, on=["permno", "decision_day"], how="left")
    target = crsp[["permno", "date", "ret", "ret_span"]].rename(columns={"date": "next_day", "ret": "ret_next"})
    out = out.assign(next_day=out["decision_day"].map(nxt)).merge(target, on=["permno", "next_day"], how="left")
    one_day = out["ret_span"].str.fullmatch(ONE_TRADING_DAY_SPANS).fillna(False).astype(bool)
    embargo = out["decision_day"].lt(pd.Timestamp(test_start)) & out["next_day"].ge(pd.Timestamp(test_start))
    out["ret_next"] = out["ret_next"].where(one_day & ~embargo)
    out["ret_next_is_delisting"] = out["ret_span"].eq("DD").fillna(False).astype(bool) & out["ret_next"].notna()
    out["early_close"] = out["decision_day"].dt.strftime("%Y%m%d").isin(early_closes)
    out["permno"] = out["permno"].astype("Int64")
    return out.drop(columns=["next_day", "ret_span"]).sort_values(["decision_day", "permno"], ignore_index=True)


def _keep(table: pd.DataFrame, links: pd.DataFrame | None, universe: set) -> pd.DataFrame:
    """The rows of a WCT day's table whose symbol is the primary symbol of a universe stock that day, with its PERMNO."""
    if links is None:
        return table.iloc[0:0].assign(permno=pd.array([], dtype="Int64"))
    primary = links.loc[links["primary"] & links["permno"].isin(universe), ["symbol", "permno"]]
    return table.merge(primary, on="symbol", how="inner", validate="many_to_one")


def _whole_numbers(table: pd.DataFrame) -> pd.DataFrame:
    """Counts the pull stored as doubles (shares, *_trades, *_shares) as integers; dollars stay doubles."""
    counts = [c for c in table if c == "shares" or c.endswith(("_trades", "_shares"))]
    return table.astype({c: "int64" for c in counts})


def decision_day_tables(previous: dict | None, own: dict, previous_links: pd.DataFrame | None,
                        own_links: pd.DataFrame, previous_day: str | None, own_day: str,
                        universe: pd.DataFrame) -> dict:
    """A decision day's daily, cells and bars from its own WCT day and the previous one (None for the first day).

    links: each WCT day's symbol, permno, primary and pilot_group (from the symbols table of that day). universe: the
    decision day's rows of stock_days. previous_day and own_day (YYYYMMDD) name the two WCT days.
    """
    stocks = set(universe["permno"].dropna())
    out = {}
    for name in ["cells", "bars"]:
        parts = [_keep(own[name][own[name]["bucket"].isin(OWN)], own_links, stocks)]
        if previous is not None:
            parts.insert(0, _keep(previous[name][previous[name]["bucket"].isin(PREVIOUS)], previous_links, stocks))
        table = pd.concat(parts, ignore_index=True)
        table["piece"] = table["bucket"].map(PIECES)
        codes = [c for c in ["bar", "venue", "z", "q", "spread", "size"] if c in table]
        table = table.assign(_order=table["bucket"].map(_ORDER)).sort_values(["permno", "_order", *codes],
                                                                             ignore_index=True)
        keys = ["permno", "piece", *codes]
        rest = [c for c in table if c not in keys and c not in ("symbol", "bucket", "_order")]
        out[name] = _whole_numbers(table[keys + rest])

    spread = _keep(own["daily"], own_links, stocks)[["permno", "spread_bp"]]
    names = pd.concat([links.loc[links["primary"], ["permno", "symbol", "pilot_group"]]
                       for links in (own_links, previous_links) if links is not None]).drop_duplicates("permno")
    traded = set(out["bars"].loc[out["bars"]["piece"].isin(["pre_open", "session"]), "permno"])
    daily = (universe.drop(columns=["decision_day"], errors="ignore")
             .merge(names, on="permno", how="left")
             .merge(spread, on="permno", how="left"))
    daily["traded_before_cutoff"] = daily["permno"].isin(traded)
    daily = daily.reindex(columns=DAILY_COLUMNS).sort_values("permno", ignore_index=True)
    out["daily"] = daily.astype({"symbol": "string", "security_type": "string", "pilot_group": "string"})  # text even
    # on days without a value, so every day's file has the same columns and types
    return {name: out[name] for name in TABLES}


def write_decision_days(pull_dir: Path, out_dir: Path, wct_days: list[str], test_start: str, links, universe,
                        progress=iter) -> pd.DataFrame:
    """Write every decision day after the first WCT day to out_dir/train/<YYYYMMDD>/ or out_dir/test/<YYYYMMDD>/.

    wct_days: every WCT day pulled, in order (YYYYMMDD); the first only lends its buckets 2 and 3 to the second.
    links: a function from a WCT day to its symbol, permno, primary and pilot_group. universe: a function from a
    decision day to its rows of stock_days. Days before test_start go to train/. Each day is written to a .part folder
    and renamed when complete, and days already written are skipped. Returns one row per decision day: its period and
    the rows of each table.
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
                                         links(prev_day), links(day), prev_day, day, universe(day))
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
