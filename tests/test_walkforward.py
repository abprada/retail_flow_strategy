import numpy as np
import pandas as pd
import pytest

from dsr.walkforward import feature_columns, fold_rows, folds, last_months


def test_yearly_folds_train_on_every_earlier_day_and_predict_the_year():
    days = pd.bdate_range("2014-01-01", "2023-12-31")
    f = folds(days, first="2019-01-01", every="year")
    assert [x.name for x in f] == ["2019", "2020", "2021", "2022", "2023"]
    assert f[0].train_days.min() == days.min() and f[0].train_days.max() < pd.Timestamp("2019-01-01")
    assert f[0].predict_days.min() >= pd.Timestamp("2019-01-01") and f[0].predict_days.max() < pd.Timestamp("2020-01-01")
    assert f[-1].train_days.max() < pd.Timestamp("2023-01-01")  # expanding: everything before the year
    assert sum(len(x.predict_days) for x in f) == ((days >= "2019-01-01")).sum()


def test_quarterly_folds_and_the_five_year_minimum():
    days = pd.bdate_range("2014-01-01", "2023-12-31")
    f = folds(days, first="2019-01-01", every="quarter")
    assert len(f) == 20 and f[0].name == "2019Q1" and f[1].train_days.max() < pd.Timestamp("2019-04-01")
    with pytest.raises(ValueError):
        folds(days, first="2018-01-01", every="year")  # only 4 years before it
    # the sample starts on its first trading day, 2 January: still 5 full years before 2019
    assert len(folds(pd.bdate_range("2014-01-02", "2023-12-31"), first="2019-01-01", every="year")) == 5


def test_last_months_holds_out_the_end_of_the_training_window():
    days = pd.bdate_range("2014-01-01", "2018-12-31")
    held = last_months(days, 12)
    assert held.min() >= pd.Timestamp("2018-01-01") and held.max() == days.max()


def test_feature_columns_leave_out_keys_targets_and_one_imbalance_rule():
    cols = ["day", "permno", "ret_next", "cap_prev", "large_cap", "spread_20d", "target_demeaned", "target_rank", "target_class",
            "ret_prev", "imb_mid_trades", "imb_bjzz_trades", "log_cap"]
    assert feature_columns(cols) == ["ret_prev", "imb_mid_trades", "log_cap"]
    assert feature_columns(cols, bjzz=True) == ["ret_prev", "imb_bjzz_trades", "log_cap"]


def test_monthly_folds_predict_each_calendar_month_from_every_earlier_day():
    days = pd.bdate_range("2014-01-01", "2023-12-31")
    f = folds(days, first="2019-01-01", every="month")
    assert len(f) == 60 and f[0].name == "2019-01" and f[-1].name == "2023-12"
    assert f[1].train_days.max() < pd.Timestamp("2019-02-01") <= f[1].predict_days.min()
    assert f[1].predict_days.max() < pd.Timestamp("2019-03-01")
    assert sum(len(x.predict_days) for x in f) == (days >= "2019-01-01").sum()


def test_fold_rows_train_and_hold_out_on_the_universe_only_when_asked():
    days = pd.bdate_range("2014-01-01", "2019-03-29")
    day = np.repeat(days.to_numpy(), 2)  # two stocks a day: the first in the universe, the second not
    universe = np.tile([True, False], len(days))
    y = np.ones(len(day))
    y[2] = np.nan  # a row without a target is never trained on
    fold = folds(days, first="2019-01-01", every="quarter")[0]
    window = day < np.datetime64("2019-01-01")
    late = day > np.datetime64("2017-12-31")  # the last 12 months of the window, the hold-out
    fit, hold, rows = fold_rows(day, y, universe, fold, 12, holds_out=True, train_rows="universe")
    assert not (fit | hold)[~universe].any() and (fit | hold).sum() == (window & universe).sum() - 1
    assert (hold == (window & late & universe)).all() and not (fit & hold).any()
    assert (rows == ~window).all()  # every row of the quarter is predicted, in the universe or not
    fit, hold, _ = fold_rows(day, y, universe, fold, 12, holds_out=True)  # all rows: the universe and the rest
    assert (fit | hold).sum() == window.sum() - 1 and hold[~universe].any()
    fit, hold, _ = fold_rows(day, y, universe, fold, 12, holds_out=False, train_rows="universe")  # the linear models
    assert (fit == (window & universe & ~np.isnan(y))).all() and (fit & hold).sum() == hold.sum()
    with pytest.raises(ValueError):
        fold_rows(day, y, universe, fold, 12, holds_out=True, train_rows="large")
