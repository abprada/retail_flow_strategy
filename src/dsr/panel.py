"""The model panel's pieces (explanations/features/features_nn_linear_tree.md), built the same way for train and test.

Each function works on one period's `panel_train` or `panel_test` (one row per decision day and PERMNO, with what is
known before the cutoff and the target `ret_next`) and returns columns aligned with its rows, except crsp_windows, which
covers every CRSP row and is joined by day and PERMNO.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from dsr.features import LOWER, UPPER

LARGE_CAP = 1_000_000  # $1B: validate and trade on these; CRSP's cap is in thousands of dollars
SESSION_BARS = {False: 38, True: 20}  # ten-minute bars from 09:30 to the cutoff, 15:50 (12:50 on early closes)
_BARS = pa.schema([("permno", pa.int64()), ("bar", pa.int64()),
                   ("high", pa.float64()), ("low", pa.float64()), ("close", pa.float64())])


def targets(panel: pd.DataFrame) -> pd.DataFrame:
    """ret_next and cap_prev, the large_cap flag, and the three targets, aligned with panel (day, cap_prev, ret_next).

    All are made within each day and size group (large_cap or not). target_demeaned (the baseline) is ret_next
    winsorized at the group's 1st and 99th percentiles that day, minus the group's average of the winsorized returns. For the other two, ret_next is ranked into u = (rank - 0.5) / n, ties sharing their
    average rank: target_rank is u mapped to a standard normal; target_class is 1 for u >= 0.8 (top quintile), -1 for
    u < 0.2 (bottom quintile) and 0 in between. Rows without ret_next get none.
    """
    out = panel[["ret_next", "cap_prev"]].assign(large_cap=panel["cap_prev"] >= LARGE_CAP)
    g = out.groupby([panel["day"], out["large_cap"]])["ret_next"]
    clipped = out["ret_next"].clip(g.transform("quantile", LOWER), g.transform("quantile", UPPER))
    out["target_demeaned"] = clipped - clipped.groupby([panel["day"], out["large_cap"]]).transform("mean")
    u = (g.rank() - 0.5) / g.transform("count")
    out["target_rank"] = stats.norm.ppf(u)
    out["target_class"] = pd.Series(0, index=out.index, dtype="Int8").mask(u >= 0.8, 1).mask(u < 0.2, -1).mask(u.isna())
    return out


def crsp_windows(crsp: pd.DataFrame) -> pd.DataFrame:
    """Past-return windows from CRSP daily (permno, date, ret), keyed by the decision day they feed: the trading day
    after each row's date, in CRSP's own calendar. For decision day t: ret_2_5 compounds t-5 to t-2 (at least 3 of the
    4 days), ret_6_20 t-20 to t-6 (at least 12 of 15), vol_20 is the standard deviation of daily log returns over t-20
    to t-1 (at least 15). Days are the stock's own CRSP rows, so leaving the universe leaves no gap.
    """
    crsp = crsp.sort_values(["permno", "date"], ignore_index=True)
    dates = np.sort(crsp["date"].unique())
    log_ret = np.log1p(crsp["ret"]).groupby(crsp["permno"])

    def window(days, at_least, lag, how="sum"):
        return log_ret.transform(lambda s: getattr(s.rolling(days, min_periods=at_least), how)().shift(lag))

    out = pd.DataFrame({
        "day": crsp["date"].map(pd.Series(dates[1:], index=dates[:-1])),
        "permno": crsp["permno"],
        "ret_2_5": np.expm1(window(4, 3, 1)),
        "ret_6_20": np.expm1(window(15, 12, 5)),
        "vol_20": window(20, 15, 0, "std"),
    })
    return out[out["day"].notna()].reset_index(drop=True)


def session_prices(bars_path: Path, early_close: bool) -> pd.DataFrame:
    """One decision day's prices from its session bars, one row per permno: price_cut, the last trade before the
    cutoff; price_1450, the last trade before the last hour (14:50, or 11:50 on early closes); the session's high and
    low up to the cutoff."""
    bars = (pq.read_table(bars_path, columns=_BARS.names, filters=[("piece", "==", "session")])
            .cast(_BARS).to_pandas().sort_values(["permno", "bar"]))
    last_hour = SESSION_BARS[early_close] - 6  # the last hour's first bar
    g = bars.groupby("permno")
    return pd.DataFrame({
        "price_cut": g["close"].last(),
        "price_1450": bars[bars["bar"] < last_hour].groupby("permno")["close"].last(),
        "high": g["high"].max(),
        "low": g["low"].min(),
    }).rename_axis("permno").reset_index()


def family_a(panel: pd.DataFrame, prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Family A (returns), raw, aligned with panel (day, permno, open_t, ret_1550, ret_prev). prices: session_prices
    with a day column; windows: crsp_windows. Today's three pieces compound to ret_1550: ret_overnight (yesterday's
    cutoff to today's open, taken out of ret_1550 so splits and dividends are handled), ret_open_1450 and
    ret_last_hour. Infinite values (a zero price) become missing."""
    a = (panel[["day", "permno", "open_t", "ret_1550", "ret_prev"]]
         .merge(prices, on=["day", "permno"], how="left")
         .merge(windows, on=["day", "permno"], how="left")
         .set_index(panel.index))
    return pd.DataFrame({
        "ret_overnight": (1 + a["ret_1550"]) * a["open_t"] / a["price_cut"] - 1,
        "ret_open_1450": a["price_1450"] / a["open_t"] - 1,
        "ret_last_hour": a["price_cut"] / a["price_1450"] - 1,
        "ret_prev": a["ret_prev"],
        "ret_2_5": a["ret_2_5"],
        "ret_6_20": a["ret_6_20"],
        "vol_20": a["vol_20"],
        "range_today": np.log(a["high"] / a["low"]),
    }).replace([np.inf, -np.inf], np.nan)


# the bars' columns families B and C sum over the window: retail trades signed by the midpoint rule (main) and BJZZ
# (benchmark), all dollars and trades, and off-exchange trades with their odd lots (under 100 shares)
WINDOW_COLUMNS = (
    "mid_buy_trades", "mid_sell_trades", "mid_buy_dollars", "mid_sell_dollars",
    "bjzz_buy_trades", "bjzz_sell_trades", "bjzz_buy_dollars", "bjzz_sell_dollars",
    "dollars", "trades", "off_trades", "off_odd_trades",
)
MIN_RETAIL_TRADES = 5  # fewer retail trades in the window: no imbalance (a handful of trades gives +-1)


def window_sums(bars_path: Path) -> pd.DataFrame:
    """One decision day's WINDOW_COLUMNS per permno, summed over its whole window: all four pieces, from the
    previous trading day's cutoff to this day's cutoff."""
    bars = pq.read_table(bars_path, columns=["permno", *WINDOW_COLUMNS]).to_pandas()
    return bars.groupby("permno", as_index=False)[list(WINDOW_COLUMNS)].sum()


def _imbalance(r: pd.DataFrame, rule: str, unit: str) -> pd.Series:
    buy, sell = r[f"{rule}_buy_{unit}"], r[f"{rule}_sell_{unit}"]
    enough = r[f"{rule}_buy_trades"] + r[f"{rule}_sell_trades"] >= MIN_RETAIL_TRADES
    return ((buy - sell) / (buy + sell)).where(enough)


def _past(r: pd.DataFrame, column: str, calendar: pd.DatetimeIndex, days: int, at_least: int) -> pd.Series:
    """The average of a column over each stock's previous `days` trading days of the calendar, with at least
    `at_least` of them in r (days out of the universe count as missing); aligned with r."""
    wide = r.pivot(index="day", columns="permno", values=column).reindex(calendar)
    past = wide.rolling(days, min_periods=at_least).mean().shift(1)
    return past.stack().rename(column).reindex(pd.MultiIndex.from_frame(r[["day", "permno"]])).to_numpy()


def family_b(panel: pd.DataFrame, sums: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Family B (retail flow), raw, aligned with panel (day, permno). sums: window_sums per decision day with a
    day column, for every day the histories need (train and test together: they look only backwards); calendar: every
    trading day, in order, so a day missing from sums still counts in the histories. Everything is
    over the window from the previous cutoff to this one; imbalances are (buys - sells) / (buys + sells)."""
    r = sums.sort_values(["day", "permno"], ignore_index=True)
    trades = r["mid_buy_trades"] + r["mid_sell_trades"]
    r = r.assign(imb_mid_trades=_imbalance(r, "mid", "trades"), retail_count=trades)
    r = r.assign(imb_mid_5d=_past(r, "imb_mid_trades", calendar, 5, 4),
                 count_20d=_past(r, "retail_count", calendar, 20, 15))
    b = pd.DataFrame({
        "day": r["day"], "permno": r["permno"],
        "imb_mid_trades": r["imb_mid_trades"],
        "imb_mid_dollars": _imbalance(r, "mid", "dollars"),
        "imb_bjzz_trades": _imbalance(r, "bjzz", "trades"),
        "imb_bjzz_dollars": _imbalance(r, "bjzz", "dollars"),
        "retail_trades": np.log1p(trades),
        "retail_share": (r["mid_buy_dollars"] + r["mid_sell_dollars"]) / r["dollars"],
        "retail_abnormal": np.log((1 + trades) / (1 + r["count_20d"])),
        "imb_mid_5d": r["imb_mid_5d"],
        "odd_lot_share": r["off_odd_trades"] / r["off_trades"],
    })
    out = panel[["day", "permno"]].merge(b, on=["day", "permno"], how="left").set_index(panel.index)
    return out.drop(columns=["day", "permno"]).replace([np.inf, -np.inf], np.nan)


def family_c(panel: pd.DataFrame, sums: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Family C (activity and liquidity), raw, aligned with panel (day, permno, spread_bp). sums and calendar as in
    family_b. dollars and trades are over the window (log of 1 + the sum); dollars_abnormal is log((1 + dollars) /
    (1 + their average over t-20 to t-1)), at least 15 of the 20 days; spread is log(1 + spread_bp), the average
    quoted spread over the day's session trades up to the cutoff."""
    r = sums.sort_values(["day", "permno"], ignore_index=True)
    r = r.assign(dollars_20d=_past(r, "dollars", calendar, 20, 15))
    c = pd.DataFrame({
        "day": r["day"], "permno": r["permno"],
        "dollars": np.log1p(r["dollars"]),
        "dollars_abnormal": np.log((1 + r["dollars"]) / (1 + r["dollars_20d"])),
        "trades": np.log1p(r["trades"]),
    })
    out = panel[["day", "permno"]].merge(c, on=["day", "permno"], how="left").set_index(panel.index)
    out["spread"] = np.log1p(panel["spread_bp"])
    return out.drop(columns=["day", "permno"]).replace([np.inf, -np.inf], np.nan)


# 0/1 features: left out of the winsorizing and standardizing, which would turn a rare 1 into 0 (a 99th percentile of 0)
# and a flag shared by every stock that day (early_close) into 0
BINARY_FEATURES = ("is_foreign", "is_adr", "is_reit", "early_close")


def family_d(panel: pd.DataFrame) -> pd.DataFrame:
    """Family D (context), raw, aligned with panel (cap_prev, price_prev, security_type, early_close): log_cap and
    log_price (a negative CRSP price is a bid/ask midpoint, so its absolute value), and the BINARY_FEATURES as 0 or 1;
    the security type's base is common stock."""
    return pd.DataFrame({
        "log_cap": np.log(panel["cap_prev"]),
        "log_price": np.log(panel["price_prev"].abs()),
        "is_foreign": panel["security_type"].eq("foreign common stock").astype("int8"),
        "is_adr": panel["security_type"].eq("ADR").astype("int8"),
        "is_reit": panel["security_type"].eq("REIT").astype("int8"),
        "early_close": panel["early_close"].astype("int8"),
    }, index=panel.index)
