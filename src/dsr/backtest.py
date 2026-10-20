"""Long-short backtest of a model's predictions, before costs (explanations/models/models.md). Each day the stocks are
ranked by prediction and cut into equal buckets, as `evaluation` cuts deciles; the portfolio buys the top bucket and
sells the bottom one short, equal-weighted with $1 a side, at the close, and holds them to the next close (`ret_next`).
Callers pass the stocks traded (>= $1B)."""

import numpy as np
import pandas as pd

from dsr.evaluation import _group_rank, ic_summary

DAYS_A_YEAR = 252


def positions(day, permno, pred, ret, buckets: int) -> tuple[pd.DataFrame, pd.Series]:
    """Each day's positions: the rows whose prediction ranks in the day's top (leg 1) or bottom (leg -1) of `buckets`
    equal groups; columns day, permno, leg, ret and average (the day's average return over every row ranked). Also
    returns that average, indexed by day.

    Every row with a prediction is ranked, so whether a stock is held never depends on its return; a missing return
    counts as 0. Days without any return (the sample's last, whose target is embargoed) are dropped."""
    df = pd.DataFrame({"day": np.asarray(day), "permno": np.asarray(permno), "pred": np.asarray(pred, dtype=float),
                       "ret": np.asarray(ret, dtype=float)})
    df = df[df["pred"].notna()]
    df = df[df.groupby("day")["ret"].transform("count").gt(0)].assign(ret=lambda d: d["ret"].fillna(0.0))
    codes, days = pd.factorize(df["day"], sort=True)
    n = np.bincount(codes).astype(float)
    bucket = np.clip(np.ceil(_group_rank(codes, df["pred"].to_numpy()) / n[codes] * buckets), 1, buckets)
    average = np.bincount(codes, weights=df["ret"].to_numpy()) / n
    leg = np.where(bucket == buckets, 1, np.where(bucket == 1, -1, 0))
    held = df.assign(leg=leg, average=average[codes])[leg != 0]
    return (held[["day", "permno", "leg", "ret", "average"]].reset_index(drop=True),
            pd.Series(average, index=pd.DatetimeIndex(days, name="day"), name="average"))


def universe(day, ret, cap) -> pd.DataFrame:
    """Each day's return of all the stocks passed, held long, indexed by day: equal-weighted (`equal`) and weighted by
    `cap` (`value`; pass the previous day's market cap, known before the decision). A missing return counts as 0 and
    days without any return are dropped, as in `positions`."""
    df = pd.DataFrame({"day": np.asarray(day), "ret": np.asarray(ret, dtype=float), "cap": np.asarray(cap, dtype=float)})
    df = df[df.groupby("day")["ret"].transform("count").gt(0)].assign(ret=lambda d: d["ret"].fillna(0.0))
    g = df.assign(weighted=df["ret"] * df["cap"]).groupby("day")
    return pd.DataFrame({"equal": g["ret"].mean(), "value": g["weighted"].sum() / g["cap"].sum()})


def turnover(leg: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Each day's share of one leg's value replaced at the close: half the sum of the absolute differences between the
    day's equal weights and the previous day's weights after that day's returns moved them. 0 when the leg keeps the
    same stocks and their returns were equal, 1 when every stock is new; empty on the first day. leg: day, permno and
    ret of one leg's positions; days: every day of the backtest, in order."""
    code = days.get_indexer(leg["day"])
    count = np.bincount(code, minlength=len(days))
    weight = 1.0 / count[code]
    grown = weight * (1 + leg["ret"].to_numpy())
    drifted = grown / np.bincount(code, weights=grown, minlength=len(days))[code]
    permno = leg["permno"].to_numpy()
    now = pd.Series(weight, index=pd.MultiIndex.from_arrays([code, permno]))
    before = pd.Series(drifted, index=pd.MultiIndex.from_arrays([code + 1, permno]))  # what day + 1 starts from
    by_day = 0.5 * now.sub(before, fill_value=0).abs().groupby(level=0).sum()
    out = pd.Series(by_day.reindex(range(len(days))).to_numpy(), index=days, name="turnover")
    out.iloc[0] = np.nan
    return out


def daily(held: pd.DataFrame, average: pd.Series) -> pd.DataFrame:
    """Each day's returns, indexed by day: the long and short legs' equal-weighted average return (`long`, `short`),
    the long-short (`long_short`: long minus short), each leg against the day's average (`long_vs_average`: long minus
    average; `short_vs_average`: average minus short) and each leg's turnover."""
    legs = held.groupby(["day", "leg"])["ret"].mean().unstack("leg").reindex(average.index)
    out = pd.DataFrame({"long": legs.get(1), "short": legs.get(-1)}, index=average.index)
    out["long_short"] = out["long"] - out["short"]
    out["long_vs_average"] = out["long"] - average
    out["short_vs_average"] = average - out["short"]
    for leg, name in ((1, "long"), (-1, "short")):
        out[f"turnover_{name}"] = turnover(held[held["leg"] == leg], average.index)
    return out


def trades(held: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Each trade: a stock's stay in one leg over consecutive days of `days`, from the close it is bought (or sold
    short) to the close it leaves the leg. Columns permno, leg, entry (its first day), days (held), ret (the stock's
    return compounded over the stay) and average (the days' average returns compounded over the same days)."""
    df = held.assign(code=days.get_indexer(held["day"])).sort_values(["leg", "permno", "code"])
    new = (df["leg"].ne(df["leg"].shift()) | df["permno"].ne(df["permno"].shift())
           | df["code"].ne(df["code"].shift() + 1))
    g = df.assign(grown=1 + df["ret"], average_grown=1 + df["average"]).groupby(new.cumsum().to_numpy())
    return pd.DataFrame({"permno": g["permno"].first(), "leg": g["leg"].first(), "entry": g["day"].first(),
                         "days": g.size(), "ret": g["grown"].prod() - 1,
                         "average": g["average_grown"].prod() - 1}).reset_index(drop=True)


def hit_rates(rows: pd.DataFrame) -> dict:
    """Over both legs and each: the share of rows (trades or stock-days) that made money (a long whose stock rose, a
    short whose stock fell; `hit`), and the share that beat the average (a long above it, a short below it;
    `hit_vs_average`). A return of exactly 0 is not a hit."""
    long = (rows["leg"] == 1).to_numpy()
    ret, average = rows["ret"].to_numpy(), rows["average"].to_numpy()
    hit, beat = np.where(long, ret > 0, ret < 0), np.where(long, ret > average, ret < average)
    out = {}
    for name, mask in (("", np.ones(len(rows), bool)), ("_long", long), ("_short", ~long)):
        out[f"hit{name}"], out[f"hit_vs_average{name}"] = float(hit[mask].mean()), float(beat[mask].mean())
    return out


def summary(days: pd.DataFrame, trades: pd.DataFrame, held: pd.DataFrame) -> dict:
    """The long-short's mean daily return (bp), its Newey-West t-statistic and share of positive days; annualized mean,
    volatility and Sharpe ratio (no risk-free rate: the $1 long pays for the $1 short); the largest fall of the
    cumulative return (summed: the book is $1 a side every day, so returns are not reinvested); each leg against the
    average and its turnover; the number of trades and their mean length; and the hit rates of the trades and of the
    stock-days held (`hit_rates`)."""
    ls = days["long_short"].dropna()
    t = ic_summary(ls)
    cum = ls.cumsum()
    return {"mean_bp": 1e4 * float(ls.mean()), "t": t["ic_t"], "positive_days": t["ic_positive"],
            "annual_mean_pct": 100 * DAYS_A_YEAR * float(ls.mean()),
            "annual_vol_pct": 100 * np.sqrt(DAYS_A_YEAR) * float(ls.std()),
            "sharpe": float(np.sqrt(DAYS_A_YEAR) * ls.mean() / ls.std()),
            "max_drawdown_pct": 100 * float((np.maximum(cum.cummax(), 0) - cum).max()),
            "long_vs_average_bp": 1e4 * float(days["long_vs_average"].mean()),
            "short_vs_average_bp": 1e4 * float(days["short_vs_average"].mean()),
            "turnover_long": float(days["turnover_long"].mean()), "turnover_short": float(days["turnover_short"].mean()),
            "trades": int(len(trades)), "holding_days": float(trades["days"].mean()),
            **{f"trade_{k}": v for k, v in hit_rates(trades).items()},
            **{f"day_{k}": v for k, v in hit_rates(held).items()}}


def yearly(days: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """By calendar year: the long-short's mean daily return (bp), annualized Sharpe ratio and share of positive days,
    each leg against the average, and the hit rate of the trades entered that year."""
    g = days.groupby(days.index.year)
    ls = g["long_short"]
    out = pd.DataFrame({"mean_bp": 1e4 * ls.mean(), "sharpe": np.sqrt(DAYS_A_YEAR) * ls.mean() / ls.std(),
                        "positive_days": ls.apply(lambda s: float((s.dropna() > 0).mean())),
                        "long_vs_average_bp": 1e4 * g["long_vs_average"].mean(),
                        "short_vs_average_bp": 1e4 * g["short_vs_average"].mean()})
    out["trade_hit"] = trades.groupby(trades["entry"].dt.year).apply(lambda t: hit_rates(t)["hit"])
    out.index.name = "year"
    return out
