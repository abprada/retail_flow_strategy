"""The TAQ pull's tables: bin codes and the conservation checks run on every pulled day (explanations/taq/taq_pull.md).

A day's four tables: `daily` (symbol, spread_bp), `cells` (symbol, bucket, venue, z, q, spread, size; trades, shares,
dollars), `bars` (symbol, bucket, bar; prices, counts and sums) and `audit` (section, item, venue, kept; trades, shares,
dollars). Codes are those of taq_pull.md section 6.
"""

import pandas as pd

BUCKETS = {0: "overnight_morning", 1: "regular", 2: "last", 3: "overnight_evening"}
LARGE_SIZE_CODE = 4  # $100,000 or more

# which cells each retail column of the bars sums: venue 1 (off-exchange), size under $100,000, and these z and q codes
RETAIL_CELLS = {
    "bjzz_buy": {"z": set(range(8, 12)), "q": None},
    "bjzz_sell": {"z": set(range(1, 5)), "q": None},
    "mid_buy": {"z": set(range(1, 12)), "q": set(range(10, 16))},
    "mid_sell": {"z": set(range(1, 12)), "q": set(range(1, 7))},
}
VALUES = ["trades", "shares", "dollars"]
DOLLAR_TOLERANCE = 0.01
RL_PRICES = ["rl_open", "rl_high", "rl_low", "rl_close"]  # the bar's prices from trades other than odd lots
ROUND_SIZE_CODES = [2, 3]  # 100 to 499 and 500 or more shares, under $100,000: never odd lots


def _equal(a: pd.DataFrame, b: pd.DataFrame, what: str) -> tuple[str, bool, str]:
    """Compare two frames of trades, shares, dollars on the same index: counts exactly, dollars within a cent."""
    a, b = a.align(b, fill_value=0)
    bad = (a["trades"] != b["trades"]) | (a["shares"] != b["shares"]) | ((a["dollars"] - b["dollars"]).abs() > DOLLAR_TOLERANCE)
    return what, not bad.any(), f"{int(bad.sum())} of {len(a)} groups differ" if bad.any() else ""


def check_day(daily: pd.DataFrame, cells: pd.DataFrame, bars: pd.DataFrame, audit: pd.DataFrame,
              regular_bars: int) -> pd.DataFrame:
    """The conservation checks of taq_pull.md section 7 for one pulled day; regular_bars is 38, or 20 on an early close.

    Returns one row per check: check, passed, detail.
    """
    results = []
    key = ["symbol", "bucket"]

    for name, df, cols in [("daily", daily, ["symbol"]), ("cells", cells, [*key, "venue", "z", "q", "spread", "size"]),
                           ("bars", bars, [*key, "bar"])]:
        dup = int(df.duplicated(cols).sum())
        results.append((f"{name}: keys are unique", dup == 0, f"{dup} duplicated keys" if dup else ""))

    results.append(_equal(cells.groupby(key)[VALUES].sum(), bars.groupby(key)[VALUES].sum(),
                          "cells and bars have the same trades, shares and dollars per symbol and bucket"))

    retail = cells[(cells["venue"] == 1) & (cells["size"] != LARGE_SIZE_CODE)]
    for col, rule in RETAIL_CELLS.items():
        m = retail["z"].isin(rule["z"])
        if rule["q"] is not None:
            m &= retail["q"].isin(rule["q"])
        from_cells = retail[m].groupby(key)[VALUES].sum()
        from_bars = bars.groupby(key)[[f"{col}_{v}" for v in VALUES]].sum()
        from_bars.columns = VALUES
        results.append(_equal(from_cells, from_bars[(from_bars != 0).any(axis=1)], f"bars' {col} columns equal the matching cells"))

    off_cells = cells[cells["venue"] == 1].groupby(key)[VALUES].sum()
    off_bars = bars.groupby(key)[["off_trades", "off_shares", "off_dollars"]].sum()
    off_bars.columns = VALUES
    results.append(_equal(off_cells, off_bars[(off_bars != 0).any(axis=1)], "bars' off-exchange columns equal the venue-1 cells"))

    reg = bars["bucket"] == 1
    ok = bars.loc[reg, "bar"].between(0, regular_bars - 1).all() and (bars.loc[~reg, "bar"] == 0).all()
    results.append((f"regular bars are numbered 0 to {regular_bars - 1}, other buckets 0", bool(ok), ""))

    if "rl_trades" in bars:  # days pulled before the round-lot columns have none
        has = bars["rl_trades"] > 0
        prices = bars[RL_PRICES]
        bad = (bars["rl_trades"] > bars["trades"]) | prices.notna().any(axis=1).ne(has) | prices.isna().any(axis=1).eq(has)
        results.append(("bars' round-lot prices are there exactly when rl_trades > 0, and rl_trades <= trades",
                        not bad.any(), f"{int(bad.sum())} bars" if bad.any() else ""))
        p = bars[has]
        inside = ((p["low"] <= p["rl_low"]) & (p["rl_low"] <= p["rl_open"]) & (p["rl_low"] <= p["rl_close"])
                  & (p["rl_open"] <= p["rl_high"]) & (p["rl_close"] <= p["rl_high"]) & (p["rl_high"] <= p["high"]))
        results.append(("bars' round-lot prices lie within the bar's low and high, and their open and close within "
                        "their own", bool(inside.all()), f"{int((~inside).sum())} bars" if not inside.all() else ""))
        large = cells[cells["size"].isin(ROUND_SIZE_CODES)].groupby(key)["trades"].sum()
        short = bars.groupby(key)["rl_trades"].sum().reindex(large.index, fill_value=0) < large
        results.append(("bars' rl_trades count at least every trade of 100 shares or more under $100,000",
                        not short.any(), f"{int(short.sum())} symbol-buckets" if short.any() else ""))

    steps = audit[audit["section"] == "step"].groupby(["item", "venue"])["trades"].sum().unstack(fill_value=0)
    removed = steps.reindex(["records", "cancelled_or_erroneous", "sale_condition", "price_or_size"], fill_value=0).sum()
    ok = (steps.loc["raw"] - removed == steps.loc["kept"]).all()
    results.append(("audit: raw minus the removed steps equals kept", bool(ok), ""))
    in_cells = cells.groupby("venue")["trades"].sum().reindex(steps.columns, fill_value=0)
    ok = (steps.loc["kept"] == in_cells).all()
    results.append(("audit: kept equals the trades in cells", bool(ok), ""))

    return pd.DataFrame(results, columns=["check", "passed", "detail"])


def check_prices(prices: pd.DataFrame, regular_bars: int) -> pd.DataFrame:
    """The checks of one day of the price-only pull (taq_sql.prices_sql): one row per check: check, passed, detail."""
    results = []
    dup = int(prices.duplicated(["symbol", "bucket", "bar"]).sum())
    results.append(("prices: keys are unique", dup == 0, f"{dup} duplicated keys" if dup else ""))
    reg = prices["bucket"] == 1
    ok = (prices["bucket"].isin([1, 2]).all() and prices.loc[reg, "bar"].between(0, regular_bars - 1).all()
          and (prices.loc[~reg, "bar"] == 0).all())
    results.append((f"prices: buckets 1 and 2 only, regular bars numbered 0 to {regular_bars - 1}", bool(ok), ""))
    bad = (prices["rl_trades"] <= 0) | prices[RL_PRICES].isna().any(axis=1)
    results.append(("prices: every row has a round lot and all four prices", not bad.any(),
                    f"{int(bad.sum())} rows" if bad.any() else ""))
    p = prices[~bad]
    inside = ((p["rl_low"] <= p["rl_open"]) & (p["rl_low"] <= p["rl_close"]) & (p["rl_open"] <= p["rl_high"])
              & (p["rl_close"] <= p["rl_high"]))
    results.append(("prices: open and close lie within low and high", bool(inside.all()),
                    f"{int((~inside).sum())} rows" if not inside.all() else ""))
    if "rl100_trades" in prices:  # trades of 100 shares or more among them: the round-lot rule before 2025-11-03
        r100 = [c.replace("rl_", "rl100_") for c in RL_PRICES]
        has = prices["rl100_trades"] > 0
        bad100 = ((prices["rl100_trades"] > prices["rl_trades"]) | prices[r100].notna().any(axis=1).ne(has)
                  | prices[r100].isna().any(axis=1).eq(has))
        results.append(("prices: rl100_ prices are there exactly when rl100_trades > 0, and rl100_trades <= rl_trades",
                        not bad100.any(), f"{int(bad100.sum())} rows" if bad100.any() else ""))
        q = prices[has & ~bad100]
        inside = ((q["rl_low"] <= q["rl100_low"]) & (q["rl100_low"] <= q["rl100_open"])
                  & (q["rl100_low"] <= q["rl100_close"]) & (q["rl100_open"] <= q["rl100_high"])
                  & (q["rl100_close"] <= q["rl100_high"]) & (q["rl100_high"] <= q["rl_high"]))
        results.append(("prices: rl100_ prices lie within the rl_ low and high, open and close within their own",
                        bool(inside.all()), f"{int((~inside).sum())} rows" if not inside.all() else ""))
    return pd.DataFrame(results, columns=["check", "passed", "detail"])


def compare_prices_with_bars(prices: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """A price-only day against the same day's full-pull bars: each price row's bar exists, its round-lot high and low
    lie within the bar's high and low, and its round lots are at most the bar's trades. One row per check."""
    keys = ["symbol", "bucket", "bar"]
    m = prices.merge(bars[[*keys, "high", "low", "trades"]], on=keys, how="left")
    missing = m["trades"].isna()
    m = m[~missing]
    outside = (m["rl_high"].astype(float) > m["high"].astype(float) + 1e-9) | (
        m["rl_low"].astype(float) < m["low"].astype(float) - 1e-9)
    more = m["rl_trades"] > m["trades"]
    return pd.DataFrame([
        ("every price row has a bar in the full pull", not missing.any(), f"{int(missing.sum())} rows" if missing.any() else ""),
        ("round-lot high and low within the bar's high and low", not outside.any(),
         f"{int(outside.sum())} rows" if outside.any() else ""),
        ("round lots at most the bar's trades", not more.any(), f"{int(more.sum())} rows" if more.any() else ""),
    ], columns=["check", "passed", "detail"])


KEYS = {"daily": ["symbol"], "cells": ["symbol", "bucket", "venue", "z", "q", "spread", "size"],
        "bars": ["symbol", "bucket", "bar"], "audit": ["section", "item", "venue", "kept"]}
FLOAT_COLUMNS = {"spread_bp", "open", "high", "low", "close", "last_mid", *RL_PRICES}  # within FLOAT_TOLERANCE
FLOAT_TOLERANCE = 1e-6


def compare_tables(name: str, reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row comparison of one pulled table with the reference (the golden test, taq_pull.md section 8).

    Keys must be the same set; integer columns equal; columns named *dollars within a cent; prices, midpoints and
    spreads within FLOAT_TOLERANCE. Returns one row per column: column, rows_differing, detail.
    """
    key = KEYS[name]
    ref, oth = reference.set_index(key).sort_index(), other.set_index(key).sort_index()
    only_ref, only_oth = ref.index.difference(oth.index), oth.index.difference(ref.index)
    rows = [{"column": "(keys)", "rows_differing": len(only_ref) + len(only_oth),
             "detail": f"{len(only_ref)} only in the reference, {len(only_oth)} only in the other"}]
    both = ref.index.intersection(oth.index)
    for col in ref.columns:
        if col not in oth.columns:
            rows.append({"column": col, "rows_differing": len(both), "detail": "missing from the other"})
            continue
        a = pd.to_numeric(ref.loc[both, col], errors="coerce").astype(float)
        b = pd.to_numeric(oth.loc[both, col], errors="coerce").astype(float)
        tol = DOLLAR_TOLERANCE if col.endswith("dollars") else FLOAT_TOLERANCE if col in FLOAT_COLUMNS else 0.0
        bad = ~(((a - b).abs() <= tol) | (a.isna() & b.isna()))
        rows.append({"column": col, "rows_differing": int(bad.sum()),
                     "detail": f"max difference {float((a - b).abs()[bad].max()):g}" if bad.any() else ""})
    rows += [{"column": col, "rows_differing": len(both), "detail": "not in the reference"}
             for col in oth.columns if col not in ref.columns]
    return pd.DataFrame(rows)


TEXT_COLUMNS = {"symbol", "section", "item"}


def read_pulled_csv(path) -> pd.DataFrame:
    """A table as Postgres COPY ... CSV writes it: text kept as text (a symbol `NA` stays `NA`), `t`/`f` as booleans,
    empty numbers as missing, whole numbers as integers."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in df.columns:
        if col in TEXT_COLUMNS:
            continue
        if col == "kept":
            df[col] = df[col].map({"t": True, "f": False, "true": True, "false": False, "True": True, "False": False})
            continue
        num = pd.to_numeric(df[col].replace("", None), errors="coerce")
        df[col] = num.astype("int64") if num.notna().all() and (num == num.round()).all() and "dollars" not in col \
            and col not in FLOAT_COLUMNS else num.astype(float)
    return df


def run_day(conn, sql: dict, out_dir) -> pd.DataFrame:
    """Run one day's statements (taq_sql.day_sql) on an open Postgres connection and write each output table to
    out_dir/<table>.csv with COPY. Returns the time and size of every statement."""
    import time
    from pathlib import Path

    from dsr.data import copy_query

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = []
    with conn.cursor() as cur:
        for i, statement in enumerate(sql["setup"]):
            start = time.perf_counter()
            cur.execute(statement)
            timing.append({"step": f"setup {i + 1}", "seconds": time.perf_counter() - start, "bytes": 0})
    for name in KEYS:
        path = out_dir / f"{name}.csv"
        path.unlink(missing_ok=True)
        start = time.perf_counter()
        copy_query(conn, sql[name], path)
        timing.append({"step": name, "seconds": time.perf_counter() - start, "bytes": path.stat().st_size})
    return pd.DataFrame(timing)
