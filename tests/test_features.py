import numpy as np
import pandas as pd

from dsr.features import winsorize_standardize


def test_each_day_is_clipped_at_1_and_99_percent_then_has_mean_0_and_sd_1():
    day = pd.Series(np.repeat(["d1", "d2"], 101))
    x = pd.DataFrame({"a": np.r_[np.arange(100.0), 1e6, np.arange(101.0) * 10]})
    z = winsorize_standardize(x, day)
    assert list(z.index) == list(x.index)
    for d in ["d1", "d2"]:
        s = z.loc[day == d, "a"]
        assert abs(s.mean()) < 1e-12 and abs(s.std() - 1) < 1e-12
    # the 1e6 outlier is clipped to the day's 99th percentile, so it lands next to the stocks below it
    assert z.loc[100, "a"] < 2


def test_missing_values_become_0_and_a_constant_day_is_all_0():
    day = pd.Series(["d1"] * 4 + ["d2"] * 3)
    x = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0, 7.0, 7.0, 7.0]})
    z = winsorize_standardize(x, day)
    assert z.loc[1, "a"] == 0
    assert (z.loc[day == "d2", "a"] == 0).all()
    assert z["a"].notna().all()
