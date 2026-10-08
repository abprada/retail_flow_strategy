"""The TAQ pull's tables: bin codes and the conservation checks run on every pulled day (explanations/taq_pull.md).

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

    steps = audit[audit["section"] == "step"].groupby(["item", "venue"])["trades"].sum().unstack(fill_value=0)
    removed = steps.reindex(["records", "cancelled_or_erroneous", "sale_condition", "price_or_size"], fill_value=0).sum()
    ok = (steps.loc["raw"] - removed == steps.loc["kept"]).all()
    results.append(("audit: raw minus the removed steps equals kept", bool(ok), ""))
    in_cells = cells.groupby("venue")["trades"].sum().reindex(steps.columns, fill_value=0)
    ok = (steps.loc["kept"] == in_cells).all()
    results.append(("audit: kept equals the trades in cells", bool(ok), ""))

    return pd.DataFrame(results, columns=["check", "passed", "detail"])


KEYS = {"daily": ["symbol"], "cells": ["symbol", "bucket", "venue", "z", "q", "spread", "size"],
        "bars": ["symbol", "bucket", "bar"], "audit": ["section", "item", "venue", "kept"]}
FLOAT_COLUMNS = {"spread_bp", "open", "high", "low", "close", "last_mid"}  # compared within FLOAT_TOLERANCE
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
