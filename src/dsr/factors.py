"""Factors built from the trading universe itself: market, size and momentum, daily, before the close of the decision day.

A stock's weight and its sort variable are known before the decision (the previous day's market cap, returns through the
previous close), and the factor earns the stock's `ret_next` (close to the next close) like the models' long-short, so a
factor and a long-short dated the same day cover the same hours."""

import numpy as np
import pandas as pd

SKIP, LOOKBACK = 22, 252  # momentum: the return over the rows 252 back to 22 back (12 months, skipping the last month)
TAIL = 0.3  # momentum: the top and bottom 30% of the day's stocks


def momentum(day, permno, ret) -> pd.Series:
    """Each stock-day's 12-1 month return, compounded over the stock's rows LOOKBACK to SKIP back from the day (a missing
    return counts as 0), indexed by (day, permno); a stock with fewer than LOOKBACK + 1 earlier rows has none. A row's
    return is the close-to-close return dated that day, which is not known at the 15:50 decision, so the window ends SKIP
    rows back and never reaches the day itself."""
    df = pd.DataFrame({"day": np.asarray(day), "permno": np.asarray(permno), "ret": np.asarray(ret, dtype=float)})
    df = df.sort_values(["permno", "day"]).reset_index(drop=True)
    df["log"] = np.log1p(df["ret"].fillna(0.0).clip(lower=-0.9999))
    cum = df.groupby("permno")["log"].cumsum()
    out = np.expm1(cum.groupby(df["permno"]).shift(SKIP) - cum.groupby(df["permno"]).shift(LOOKBACK + 1))
    out[df.groupby("permno").cumcount() < LOOKBACK + 1] = np.nan
    return pd.Series(out.to_numpy(), index=pd.MultiIndex.from_arrays([df["day"], df["permno"]], names=["day", "permno"]))


def universe_factors(day, ret, cap, mom) -> pd.DataFrame:
    """Each day's factors from the stocks passed, indexed by day, as returns of portfolios held one day:

    - `market`: every stock held long, weighted by `cap` (the previous day's market cap);
    - `size`: the stocks under the day's median `cap` minus those over it, each equal-weighted;
    - `momentum`: the top TAIL of the day's `mom` minus the bottom TAIL, equal-weighted (stocks without `mom` are left out).

    A missing return counts as 0, as in `dsr.backtest.positions`; days without any return are dropped."""
    df = pd.DataFrame({"day": np.asarray(day), "ret": np.asarray(ret, dtype=float), "cap": np.asarray(cap, dtype=float),
                       "mom": np.asarray(mom, dtype=float)})
    df = df[df.groupby("day")["ret"].transform("count").gt(0)].assign(ret=lambda d: d["ret"].fillna(0.0))
    g = df.groupby("day")
    out = pd.DataFrame({"market": (df["ret"] * df["cap"]).groupby(df["day"]).sum() / g["cap"].sum()})
    small = df["cap"] < g["cap"].transform("median")
    out["size"] = df["ret"][small].groupby(df["day"][small]).mean() - df["ret"][~small].groupby(df["day"][~small]).mean()
    rank = g["mom"].rank(pct=True)
    top, bottom = rank > 1 - TAIL, rank <= TAIL
    out["momentum"] = df["ret"][top].groupby(df["day"][top]).mean() - df["ret"][bottom].groupby(df["day"][bottom]).mean()
    return out
