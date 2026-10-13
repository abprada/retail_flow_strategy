"""The walk-forward folds (explanations/models/training.md): an expanding window from the first day, refit every year,
quarter (the sweeps) or month (the walk-forward with the chosen settings), predicting the next period out of
sample."""

from dataclasses import dataclass

import pandas as pd

MIN_TRAIN_YEARS = 5
KEYS = ("day", "permno", "ret_next", "cap_prev", "large_cap")  # the panel's columns that are not features
MID, BJZZ = ("imb_mid_trades",), ("imb_bjzz_trades",)  # the retail imbalance by each signing rule


@dataclass(frozen=True)
class Fold:
    name: str
    train_days: pd.DatetimeIndex
    predict_days: pd.DatetimeIndex


def folds(days, first: str, every: str) -> list[Fold]:
    """One fold per year ("year"), quarter ("quarter") or month ("month") from `first` on: train on every day before
    the period, predict the period. Raises if the first fold has less than MIN_TRAIN_YEARS of training data, counted
    in whole years from the start of the first day's year (the sample starts on its first trading day, not on 1
    January)."""
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days))))
    if pd.Timestamp(first) < pd.Timestamp(days.min().year, 1, 1) + pd.DateOffset(years=MIN_TRAIN_YEARS):
        raise ValueError(f"fewer than {MIN_TRAIN_YEARS} years of training days before {first}")
    later = days[days >= pd.Timestamp(first)]
    periods = later.to_period({"year": "Y", "quarter": "Q", "month": "M"}[every])
    return [Fold(str(p), days[days < p.start_time], later[periods == p]) for p in periods.unique()]


def last_months(days: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    """The days in the last `months` months of a training window: the early-stopping hold-out."""
    return days[days > days.max() - pd.DateOffset(months=months)]


def feature_columns(columns, bjzz: bool = False) -> list[str]:
    """The panel's feature columns in order: everything but the keys and targets, with the midpoint imbalance
    (default) or the BJZZ one (the benchmark) in its place."""
    drop = set(KEYS) | set(MID if bjzz else BJZZ)
    return [c for c in columns if c not in drop and not c.startswith("target_")]
