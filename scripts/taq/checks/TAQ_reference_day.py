"""Compute the TAQ pull's four tables for 2019-06-04 in Python: the reference the WRDS query must match.

    uv run python scripts/taq/checks/TAQ_reference_day.py

Runs the one-day cleaning notebook (scripts/taq/01_clean_and_explore_TAQ_raw_trades.py) headless, about 5 minutes, then builds
`daily`, `cells`, `bars` and `audit` from its trades exactly as explanations/taq/taq_pull.md (sections 4 to 6) defines them,
and runs the conservation checks of section 7 on them. All bin arithmetic is in whole numbers: prices in millionths of a
dollar, so trades exactly on a bin edge (Z = 0.4 or 0.6) land where the spec says.

Outputs (gitignored), in data/raw/taq/checks/reference/20190604/:
    daily.parquet, cells.parquet, bars.parquet, audit.parquet   the four tables, rows as the query would write them
    checks.csv                                                  the conservation checks
"""

import gc
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dsr.taq import check_day

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "scripts" / "taq" / "01_clean_and_explore_TAQ_raw_trades.py"
BUCKET_CODES = {"overnight_morning": 0, "regular": 1, "last": 2, "overnight_evening": 3}
TEN_MINUTES = 600 * 10**9
LARGE = 100_000 * 10**6  # $100,000 in millionths of a dollar


def isum(inv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Exact integer sums of x per group (np.bincount would sum in floating point)."""
    return pd.Series(np.asarray(x, dtype=np.int64)).groupby(inv).sum().to_numpy()


def micro(x) -> np.ndarray:
    """Dollars to whole millionths of a dollar (prices have at most 6 decimals); NaN as 0."""
    return np.round(np.nan_to_num(np.asarray(x, dtype=float)) * 1e6).astype(np.int64)


def codes(t: pd.DataFrame, open_ns: int) -> dict[str, np.ndarray]:
    """Every trade's integer prices, quote state, bins (taq_pull.md 6.1 and 6.3), bucket and bar."""
    P, size = micro(t["price"]), t["size"].to_numpy().astype(np.int64)
    bid, ask = t["nbb"].to_numpy(dtype=float), t["nbo"].to_numpy(dtype=float)
    have = ~np.isnan(bid) & ~np.isnan(ask) & (bid > 0) & (ask > 0)
    B, A = np.where(have, micro(bid), 0), np.where(have, micro(ask), 0)
    valid = have & (B < A)
    d, w = P - B, np.where(valid, A - B, 1)
    zt = P % 10_000
    D = P * size  # dollars in millionths
    bucket = t["bucket"].map(BUCKET_CODES).to_numpy().astype(np.int64)
    return {
        "P": P, "size": size, "B": B, "A": A, "valid": valid, "locked": have & (B == A), "d": d, "w": w, "zt": zt, "D": D,
        "off": t["ex"].eq("D").to_numpy(),
        "z": np.select([zt == 0, zt == 5000, zt < 5000], [0, 6, 1 + np.minimum(zt // 1000, 4)],
                       6 + np.minimum((zt + 999) // 1000 - 5, 5)),
        "q": np.select([~valid, d < 0, d == 0, d == w, d > w, 2 * d == w, 10 * d < 5 * w],
                       [0, 1, 2, 14, 15, 8, 3 + (10 * d) // w], 8 + (-((-10 * d) // w) - 5)),
        "spread": np.select([~valid, w <= 10_000, w <= 20_000, w <= 50_000], [0, 1, 2, 3], 4),
        "size_code": np.select([D >= LARGE, size == 1, size <= 99, size <= 499], [4, 0, 1, 2], 3),
        "bucket": bucket,
        "bar": np.where(bucket == 1, (t["t_ns"].to_numpy() - open_ns) // TEN_MINUTES, 0),
        "t": t["t_ns"].to_numpy(), "seq": t["tr_seqnum"].to_numpy().astype(np.int64),
        "odd": t["tr_scond"].fillna("").str.contains("I", regex=False).to_numpy(),
    }


def build_daily(symbol: np.ndarray, c: dict) -> pd.DataFrame:
    m = (c["bucket"] == 1) & (c["valid"] | c["locked"])
    bp = (c["A"][m] - c["B"][m]) / ((c["A"][m] + c["B"][m]) / 2) * 1e4
    return pd.DataFrame({"symbol": symbol[m], "spread_bp": bp}).groupby("symbol", as_index=False)["spread_bp"].mean()


def build_cells(sym_code: np.ndarray, sym_names: np.ndarray, c: dict) -> pd.DataFrame:
    parts = [("bucket", 4), ("venue", 2), ("z", 12), ("q", 16), ("spread", 5), ("size", 5)]
    values = {"bucket": c["bucket"], "venue": c["off"].astype(np.int64), "z": c["z"], "q": c["q"],
              "spread": c["spread"], "size": c["size_code"]}
    key = sym_code.astype(np.int64)
    for name, n in parts:
        key = key * n + values[name]
    uniq, inv = np.unique(key, return_inverse=True)
    out, k = {}, uniq.copy()
    for name, n in reversed(parts):
        out[name] = (k % n).astype(np.int8)
        k //= n
    return pd.DataFrame({
        "symbol": sym_names[k], **{name: out[name] for name, _ in parts},
        "trades": np.bincount(inv), "shares": np.bincount(inv, weights=c["size"]).astype(np.int64),
        "dollars": isum(inv, c["D"]) / 1e6,
    })


def build_bars(sym_code: np.ndarray, sym_names: np.ndarray, price: np.ndarray, c: dict) -> pd.DataFrame:
    key = (sym_code.astype(np.int64) * 4 + c["bucket"]) * 64 + c["bar"]
    uniq, inv = np.unique(key, return_inverse=True)
    order = np.lexsort((c["seq"], c["t"], inv))
    first = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
    last = np.r_[first[1:] - 1, len(order) - 1]
    mid = np.where(c["valid"], (c["A"] + c["B"]) / 2e6, np.nan)[order]
    last_valid = np.maximum.reduceat(np.where(np.isnan(mid), -1, np.arange(len(order))), first)
    sums = lambda m, x: isum(inv, np.where(m, x, 0))

    retail = c["off"] & (c["D"] < LARGE)
    bjzz_buy, bjzz_sell = retail & (c["zt"] > 6000), retail & (c["zt"] > 0) & (c["zt"] < 4000)
    mid_signed = retail & (c["zt"] != 0) & c["valid"]
    mid_buy, mid_sell = mid_signed & (10 * c["d"] > 6 * c["w"]), mid_signed & (10 * c["d"] < 4 * c["w"])
    qr_buy = c["valid"] & (2 * c["P"] > c["A"] + c["B"])
    qr_sell = c["valid"] & (2 * c["P"] < c["A"] + c["B"])
    both = (bjzz_buy | bjzz_sell) & (mid_buy | mid_sell)
    same = (bjzz_buy & mid_buy) | (bjzz_sell & mid_sell)
    ones = np.ones(len(inv), dtype=bool)

    # the rl_ prices: trades other than odd lots (sale condition I), which cannot set the official last, high or low
    lot, p_lot = ~c["odd"][order], price[order]
    position = np.arange(len(order))
    first_lot = np.minimum.reduceat(np.where(lot, position, len(order)), first)
    last_lot = np.maximum.reduceat(np.where(lot, position, -1), first)
    has_lot = last_lot >= 0

    bars = pd.DataFrame({
        "symbol": sym_names[uniq // 256], "bucket": ((uniq // 64) % 4).astype(np.int8), "bar": (uniq % 64).astype(np.int8),
        "open": price[order][first], "high": np.maximum.reduceat(price[order], first),
        "low": np.minimum.reduceat(price[order], first), "close": price[order][last],
        "last_mid": np.where(last_valid >= 0, mid[np.maximum(last_valid, 0)], np.nan),
        "last_trade_time": c["t"][order][last],
        "rl_open": np.where(has_lot, p_lot[np.minimum(first_lot, len(order) - 1)], np.nan),
        "rl_high": np.where(has_lot, np.maximum.reduceat(np.where(lot, p_lot, -np.inf), first), np.nan),
        "rl_low": np.where(has_lot, np.minimum.reduceat(np.where(lot, p_lot, np.inf), first), np.nan),
        "rl_close": np.where(has_lot, p_lot[np.maximum(last_lot, 0)], np.nan),
        "rl_trades": sums(~c["odd"], 1).astype(np.int64),
        "trades": np.bincount(inv), "shares": sums(ones, c["size"]).astype(np.int64), "dollars": sums(ones, c["D"]) / 1e6,
    })
    for name, m in [("bjzz_buy", bjzz_buy), ("bjzz_sell", bjzz_sell), ("mid_buy", mid_buy), ("mid_sell", mid_sell)]:
        bars[f"{name}_trades"] = sums(m, 1).astype(np.int64)
        bars[f"{name}_shares"] = sums(m, c["size"]).astype(np.int64)
        bars[f"{name}_dollars"] = sums(m, c["D"]) / 1e6
    bars["off_trades"] = sums(c["off"], 1).astype(np.int64)
    bars["off_shares"] = sums(c["off"], c["size"]).astype(np.int64)
    bars["off_dollars"] = sums(c["off"], c["D"]) / 1e6
    bars["off_odd_trades"] = sums(c["off"] & (c["size"] <= 99), 1).astype(np.int64)
    bars["off_odd_shares"] = sums(c["off"] & (c["size"] <= 99), c["size"]).astype(np.int64)
    bars["off_one_trades"] = sums(c["off"] & (c["size"] == 1), 1).astype(np.int64)
    bars["qr_buy_shares"] = sums(qr_buy, c["size"]).astype(np.int64)
    bars["qr_sell_shares"] = sums(qr_sell, c["size"]).astype(np.int64)
    bars["exch_qr_buy_shares"] = sums(qr_buy & ~c["off"], c["size"]).astype(np.int64)
    bars["exch_qr_sell_shares"] = sums(qr_sell & ~c["off"], c["size"]).astype(np.int64)
    bars["both_signed_trades"] = sums(both, 1).astype(np.int64)
    bars["same_sign_trades"] = sums(same, 1).astype(np.int64)
    return bars


def build_audit(defs: dict, kept_mask_all: np.ndarray) -> pd.DataFrame:
    """What each cleaning step removed (taq_pull.md 6.5), from the notebook's intermediate tables."""
    key = ["sym_root", "sym_suffix", "ex", "tr_seqnum", "t_ns"]
    raw, corr, pairs = defs["c0_trades"], defs["c1_corrections"], defs["c1_pairs"]
    venue = lambda df: df["ex"].eq("D").astype(int)

    def total(section, item, df, kept):
        g = df.assign(venue=venue(df), dollars=micro(df["price"]) * df["size"].astype(np.int64)).groupby("venue")
        return [{"section": section, "item": item, "venue": v, "kept": kept, "trades": len(x),
                 "shares": int(x["size"].sum()), "dollars": x["dollars"].sum() / 1e6} for v, x in g]

    on_raw = lambda rows: raw.merge(rows[key].drop_duplicates(), on=key)
    records = corr[corr["tr_corr"].isin(["10", "11", "12"])]
    cancelled = pairs[pairs["tr_corr"].isin(["07", "08"]) & pairs["record_counts"]]
    restored = pairs[pairs["tr_corr"].eq("01") & ~pairs["record_counts"]]
    after1, in_bucket, clean = defs["c3_all"], defs["c3_trades"], defs["clean_trades"]
    rows = (total("step", "raw", raw, True) + total("step", "records", on_raw(records), False)
            + total("step", "cancelled_or_erroneous", on_raw(cancelled), False)
            + total("step", "restored_original", on_raw(restored), True)
            + total("step", "sale_condition", after1[after1["bucket"].eq("")], False)
            + total("step", "price_or_size", in_bucket[(in_bucket["price"] < defs["PRICE_MIN"]) | (in_bucket["size"] <= 0)], False)
            + total("step", "kept", clean, True))

    a = after1.assign(venue=venue(after1), cond=after1["tr_scond"].fillna(""), kept_cond=after1["bucket"].ne(""),
                      kept_all=kept_mask_all, dollars=micro(after1["price"]) * after1["size"].astype(np.int64))
    for (cond, v, k), x in a.groupby(["cond", "venue", "kept_cond"]):
        rows.append({"section": "condition", "item": cond, "venue": v, "kept": bool(k), "trades": len(x),
                     "shares": int(x["size"].sum()), "dollars": x["dollars"].sum() / 1e6})
    for (ex, v, k), x in a.groupby(["ex", "venue", "kept_all"]):
        rows.append({"section": "venue_code", "item": ex, "venue": v, "kept": bool(k), "trades": len(x),
                     "shares": int(x["size"].sum()), "dollars": x["dollars"].sum() / 1e6})
    for code, x in corr.groupby("tr_corr"):
        rows += total("ctm", code, x, False)
    hits = corr[key].merge(raw[key], on=key, how="left", indicator=True)
    rows += total("ctm", "unmatched", corr[hits["_merge"].eq("left_only").to_numpy()], False)
    paired = pairs["t_ns_record"].notna()
    rows += total("ctm", "record_after_cutoff", pairs[paired & ~pairs["record_counts"]], False)
    rows += total("ctm", "unpaired", pairs[~paired], False)
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("nb", NOTEBOOK)
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    _, defs = nb.app.run()
    day, session = defs["DATE"], {k: v.value for k, v in defs["SESSION"].items()}
    regular_bars = int((session["cutoff"] - session["open"]) // TEN_MINUTES)
    print(f"notebook ran: {len(defs['clean_trades']):,} clean trades ({time.time() - t0:.0f} s)", flush=True)

    after1 = defs["c3_all"]
    kept_all = (after1["bucket"].ne("") & (after1["price"] >= defs["PRICE_MIN"]) & (after1["size"] > 0)).to_numpy()
    audit = build_audit(defs, kept_all)
    # the notebook's intermediate tables are no longer needed: free them before the per-trade arrays (memory)
    t = defs["clean_trades"][["symbol", "price", "size", "nbb", "nbo", "ex", "bucket", "t_ns", "tr_seqnum", "tr_scond"]]
    del defs, after1, kept_all, nb
    gc.collect()

    c = codes(t, session["open"])
    sym_code, sym_names = pd.factorize(t["symbol"])
    sym_names = np.asarray(sym_names)
    tables = {
        "daily": build_daily(t["symbol"].to_numpy(), c),
        "cells": build_cells(sym_code, sym_names, c),
        "bars": build_bars(sym_code, sym_names, t["price"].to_numpy(dtype=float), c),
        "audit": audit,
    }
    out = ROOT / "data" / "raw" / "taq" / "checks" / "reference" / day.replace("-", "")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", compression="zstd", index=False)
        print(f"{name}: {len(df):,} rows, {len(df.columns)} columns, {(out / f'{name}.parquet').stat().st_size / 1e6:.1f} MB")
    checks = check_day(tables["daily"], tables["cells"], tables["bars"], tables["audit"], regular_bars)
    checks.to_csv(out / "checks.csv", index=False)
    print(checks.to_string(index=False))
    print(f"written to {out.relative_to(ROOT)} ({time.time() - t0:.0f} s)")
    if not checks["passed"].all():
        raise SystemExit("conservation checks failed")


if __name__ == "__main__":
    main()
