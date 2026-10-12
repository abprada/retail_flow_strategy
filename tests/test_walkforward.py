import pandas as pd
import pytest

from dsr.walkforward import feature_columns, folds, last_months


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
    cols = ["day", "permno", "ret_next", "cap_prev", "large_cap", "target_demeaned", "target_rank", "target_class",
            "ret_prev", "imb_mid_trades", "imb_bjzz_trades", "log_cap"]
    assert feature_columns(cols) == ["ret_prev", "imb_mid_trades", "log_cap"]
    assert feature_columns(cols, bjzz=True) == ["ret_prev", "imb_bjzz_trades", "log_cap"]
