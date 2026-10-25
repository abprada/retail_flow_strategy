import numpy as np
import pandas as pd

from dsr.factors import LOOKBACK, SKIP, momentum, universe_factors


def test_momentum_compounds_the_window_ending_skip_rows_back_and_not_before_a_full_history():
    n = LOOKBACK + 10
    days = pd.bdate_range("2020-01-01", periods=n)
    ret = np.full(n, 0.001)
    ret[n - 1] = 5.0  # the last day's own return: never in the window
    mom = momentum(days, np.zeros(n, dtype=int), ret)
    assert mom.iloc[:LOOKBACK + 1].isna().all()
    assert np.isclose(mom.iloc[n - 1], 1.001 ** (LOOKBACK - SKIP + 1) - 1)


def test_universe_factors_market_size_and_momentum():
    day = np.repeat(pd.bdate_range("2020-01-01", periods=2), 10)
    cap = np.tile(np.arange(1.0, 11.0), 2)
    ret = np.tile(np.arange(10.0), 2) / 100
    mom = np.tile(np.arange(10.0), 2)
    f = universe_factors(day, ret, cap, mom)
    assert np.allclose(f["market"], (ret[:10] * cap[:10]).sum() / cap[:10].sum())
    assert np.allclose(f["size"], 0.02 - 0.07)  # stocks 0-4 minus 5-9
    assert np.allclose(f["momentum"], np.mean([0.07, 0.08, 0.09]) - np.mean([0.0, 0.01, 0.02]))
