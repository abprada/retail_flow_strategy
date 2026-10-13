"""Signal evaluation (explanations/features/features_nn_linear_tree.md, validation measures): daily rank IC, its
summary, R² against zero, mean squared error, and the average return by prediction decile; `measures` and
`summarize` put them together for a set of scored predictions. Callers pass the stocks scored (>= $1B)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

NW_LAGS = 5  # Newey-West lags for the IC's t-statistic: a trading week


def _group_rank(codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The average rank (1 to n) of each value within its group code, ties sharing their average rank, like pandas'
    groupby rank. Sorts with numpy: about twice as fast as pandas on millions of rows."""
    if not len(values):
        return np.empty(0)
    order = np.argsort(values, kind="stable")
    order = order[np.argsort(codes[order], kind="stable")]  # by group, then by value within it
    g, v = codes[order], values[order]
    first = np.r_[True, g[1:] != g[:-1]]
    pos = np.arange(len(v)) - np.maximum.accumulate(np.where(first, np.arange(len(v)), 0)) + 1.0
    run = np.cumsum(first | np.r_[True, v[1:] != v[:-1]]) - 1  # runs of tied values within a group
    ranks = np.empty(len(v))
    ranks[order] = (np.bincount(run, weights=pos) / np.bincount(run))[run]
    return ranks


def _by_day(day, pred, ret):
    """The rows with both a prediction and a return: each one's day code, the days, each day's number of rows, the
    predictions' ranks within their day, and the returns."""
    df = pd.DataFrame({"day": np.asarray(day), "pred": np.asarray(pred, dtype=float),
                       "ret": np.asarray(ret, dtype=float)}).dropna()
    codes, days = pd.factorize(df["day"], sort=True)
    n = np.bincount(codes).astype(float)
    return codes, pd.DatetimeIndex(days, name="day"), n, _group_rank(codes, df["pred"].to_numpy()), df["ret"].to_numpy()


def _ic(codes, days, n, pred_rank, ret):
    """Each day's Pearson correlation of the prediction ranks with the return ranks (the Spearman correlation), from
    per-day sums. A day the prediction is constant (a model penalized to all-zero coefficients) has no ranking skill:
    its IC is 0, not undefined."""
    q = _group_rank(codes, ret)
    s = lambda w: np.bincount(codes, weights=w, minlength=len(n))
    sp, sq = s(pred_rank), s(q)
    cov = s(pred_rank * q) - sp * sq / n
    var = (s(pred_rank ** 2) - sp ** 2 / n) * (s(q ** 2) - sq ** 2 / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.Series(np.where(var > 0, cov / np.sqrt(var), 0.0), index=days, name="ic")


def _deciles_by_day(codes, n, pred_rank, ret):
    """Each day's average return in each prediction decile (days x 10, decile 1 lowest). A decile left empty on a day
    by tied predictions (all of them, for a model penalized to a constant) gets that day's average return: the
    prediction says nothing about which stocks belong there."""
    key = codes * 10 + np.clip(np.ceil(pred_rank / n[codes] * 10), 1, 10).astype(int) - 1
    count = np.bincount(key, minlength=10 * len(n)).reshape(-1, 10)
    total = np.bincount(key, weights=ret, minlength=10 * len(n)).reshape(-1, 10)
    day_mean = (np.bincount(codes, weights=ret, minlength=len(n)) / n)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, day_mean)


def daily_rank_ic(day: pd.Series, pred: pd.Series, ret: pd.Series) -> pd.Series:
    """Each day's Spearman correlation between prediction and return, indexed by day; 0 on a day the prediction is
    constant."""
    codes, days, n, pred_rank, r = _by_day(day, pred, ret)
    return _ic(codes, days, n, pred_rank, r)


def daily_ic_and_deciles(day: pd.Series, pred: pd.Series, ret: pd.Series) -> tuple[pd.Series, pd.Series]:
    """daily_rank_ic and decile_means from one ranking of the predictions, for large sets of rows."""
    codes, days, n, pred_rank, r = _by_day(day, pred, ret)
    deciles = pd.Series(_deciles_by_day(codes, n, pred_rank, r).mean(axis=0), index=pd.Index(range(1, 11), name="decile"))
    return _ic(codes, days, n, pred_rank, r), deciles


def ic_summary(ic: pd.Series) -> dict:
    """Mean daily IC, its Newey-West t-statistic, the share of days above 0, and the number of days."""
    ic = ic.dropna()
    if ic.std() == 0:  # no variation (e.g. every day 0): no t-statistic
        return {"ic_mean": float(ic.mean()), "ic_t": 0.0, "ic_positive": float((ic > 0).mean()), "days": int(len(ic))}
    fit = sm.OLS(ic.to_numpy(), np.ones(len(ic))).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return {"ic_mean": float(ic.mean()), "ic_t": float(fit.tvalues[0]), "ic_positive": float((ic > 0).mean()),
            "days": int(len(ic))}


def r2_zero(pred: np.ndarray, target: np.ndarray) -> float:
    """1 - sum of squared errors / sum of squared targets: R² against a prediction of 0 (the targets are centred)."""
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    ok = ~(np.isnan(pred) | np.isnan(target))
    return float(1 - ((target[ok] - pred[ok]) ** 2).sum() / (target[ok] ** 2).sum())


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error, the loss every model trains on."""
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    ok = ~(np.isnan(pred) | np.isnan(target))
    return float(((target[ok] - pred[ok]) ** 2).mean())


def decile_means(day: pd.Series, pred: pd.Series, ret: pd.Series) -> pd.Series:
    """The average return of each prediction decile (1 lowest to 10 highest): within each day, then over days. An
    empty decile on a day (tied predictions) gets that day's average return."""
    codes, days, n, pred_rank, r = _by_day(day, pred, ret)
    return pd.Series(_deciles_by_day(codes, n, pred_rank, r).mean(axis=0), index=pd.Index(range(1, 11), name="decile"))


def daily_decile_spread(day: pd.Series, pred: pd.Series, ret: pd.Series) -> pd.Series:
    """Each day's average return of the top prediction decile minus the bottom one: the daily return of an
    equal-weighted long-short on deciles 10 and 1, before costs; indexed by day. Empty deciles (tied predictions) get
    the day's average return, so a constant prediction has a spread of 0."""
    codes, days, n, pred_rank, r = _by_day(day, pred, ret)
    by_day = _deciles_by_day(codes, n, pred_rank, r)
    return pd.Series(by_day[:, 9] - by_day[:, 0], index=days, name="spread")


def measures(scored: pd.DataFrame, target: str) -> tuple[dict, pd.Series]:
    """IC, R², MSE and the decile profile of one set of scored predictions (day, ret_next, target, pred); also
    returns the daily ICs."""
    ic_days, deciles = daily_ic_and_deciles(scored["day"], scored["pred"], scored["ret_next"])
    vs_average = 1e4 * (deciles - deciles.mean())  # bp per day, so the market's drift drops out
    return {
        "ic_mean": float(ic_days.mean()),
        "r2_pct": 100 * r2_zero(scored["pred"], scored[target]),  # in percent: raw values are ~0.0001
        "mse": mse(scored["pred"], scored[target]),
        "decile_spread_bp": 1e4 * float(deciles.loc[10] - deciles.loc[1]),
        "long_leg_bp": float(vs_average.loc[10]),
        "short_leg_bp": -float(vs_average.loc[1]),
        **{f"decile_{d}_vs_avg_bp": float(v) for d, v in vs_average.items()},
    }, ic_days


def summarize(scored: pd.DataFrame, target: str) -> dict:
    """The out-of-sample summary over every fold's predictions: the measures, plus the Newey-West t-statistic and
    share of positive days of the daily IC and of the daily decile long-short, and both by year."""
    summary, ic_days = measures(scored, target)
    spread_days = daily_decile_spread(scored["day"], scored["pred"], scored["ret_next"])
    ic, spread = ic_summary(ic_days), ic_summary(spread_days)
    summary.update({"ic_t": ic["ic_t"], "ic_positive": ic["ic_positive"], "days": ic["days"],
                    "decile_spread": spread["ic_mean"],  # in return units
                    "decile_spread_t": spread["ic_t"], "decile_spread_positive": spread["ic_positive"]})
    for year, value in ic_days.groupby(ic_days.index.year).mean().items():
        summary[f"ic_{year}"] = float(value)
    for year, value in spread_days.groupby(spread_days.index.year).mean().items():
        summary[f"spread_{year}_bp"] = 1e4 * float(value)
    return summary
