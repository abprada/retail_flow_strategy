import numpy as np
import pandas as pd
import pytest

from dsr.evaluation import (_group_rank, daily_decile_spread, daily_ic_and_deciles, daily_rank_ic, decile_means,
                            ic_summary, mse, r2_summary, r2_zero)


def _days(n_days=3, n=50, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n)
    ret = rng.normal(size=n_days * n)
    return pd.Series(day), pd.Series(ret)


def test_daily_rank_ic_is_the_spearman_correlation_of_each_day():
    day, ret = _days()
    ic = daily_rank_ic(day, ret, ret)
    assert len(ic) == 3 and np.allclose(ic, 1.0)
    assert np.allclose(daily_rank_ic(day, -ret ** 3, ret), -1.0)  # ranks only: any decreasing map gives -1


def test_ic_summary_mean_share_positive_and_t_statistic():
    ic = pd.Series([0.02, 0.04, -0.01, 0.03] * 50)
    s = ic_summary(ic)
    assert s["ic_mean"] == pytest.approx(0.02) and s["ic_positive"] == pytest.approx(0.75)
    assert s["ic_t"] > 5 and s["days"] == 200


def test_r2_against_zero_and_decile_means_rise_with_a_perfect_prediction():
    assert r2_zero(np.array([1.0, -1.0]), np.array([1.0, -1.0])) == 1.0
    assert r2_zero(np.array([0.0, 0.0]), np.array([1.0, -1.0])) == 0.0
    assert mse(np.array([0.0, 1.0, np.nan]), np.array([1.0, -1.0, 5.0])) == 2.5  # (1 + 4) / 2, the missing row skipped
    day, ret = _days(n_days=5, n=100)
    d = decile_means(day, ret, ret)
    assert list(d.index) == list(range(1, 11)) and d.is_monotonic_increasing


def test_a_constant_prediction_has_ic_0_instead_of_failing():
    day, ret = _days()
    ic = daily_rank_ic(day, pd.Series(0.0, index=ret.index), ret)
    assert len(ic) == 3 and (ic == 0).all()
    assert ic_summary(ic)["ic_mean"] == 0


def test_decile_means_with_a_constant_prediction_has_every_decile_and_no_spread():
    day, ret = _days()
    d = decile_means(day, pd.Series(0.0, index=ret.index), ret)
    assert list(d.index) == list(range(1, 11))
    assert d.loc[10] - d.loc[1] == 0


def test_daily_decile_spread_is_each_day_top_minus_bottom_decile():
    day, ret = _days(n_days=4, n=100)
    s = daily_decile_spread(day, ret, ret)
    assert len(s) == 4 and (s > 0).all()
    d = pd.DataFrame({"day": day, "ret": ret})
    first = d[d["day"] == d["day"].iloc[0]]["ret"].sort_values()
    assert s.iloc[0] == pytest.approx(first.iloc[-10:].mean() - first.iloc[:10].mean())
    assert (daily_decile_spread(day, pd.Series(0.0, index=ret.index), ret) == 0).all()  # constant: no spread


def test_ranks_and_ic_match_pandas_with_ties_and_shuffled_rows():
    day, ret = _days(n_days=5, n=200)
    rng = np.random.default_rng(1)
    pred = pd.Series(np.round(rng.normal(size=len(ret)), 1))  # rounded: many ties
    shuffle = rng.permutation(len(ret))
    day, ret, pred = day.iloc[shuffle].reset_index(drop=True), ret.iloc[shuffle].reset_index(drop=True), pred.iloc[shuffle].reset_index(drop=True)
    codes, _ = pd.factorize(day, sort=True)
    pandas_ranks = pd.DataFrame({"day": day, "p": pred}).groupby("day")["p"].rank().to_numpy()
    assert np.allclose(_group_rank(codes, pred.to_numpy()), pandas_ranks)
    spearman = [g["p"].corr(g["r"], method="spearman") for _, g in pd.DataFrame({"day": day, "p": pred, "r": ret}).groupby("day")]
    ic, deciles = daily_ic_and_deciles(day, pred, ret)
    assert np.allclose(ic.to_numpy(), spearman) and np.allclose(daily_rank_ic(day, pred, ret), ic)
    assert np.allclose(deciles, decile_means(day, pred, ret))


def test_r2_summary_pools_the_days_like_r2_zero():
    day, ret = _days(n_days=10)
    pred = 0.3 * ret + np.random.default_rng(1).normal(size=len(ret))
    s = r2_summary(day, pred, ret)
    assert s["r2_pct"] == pytest.approx(100 * r2_zero(pred, ret))
    gain = (ret ** 2 - (ret - pred) ** 2).groupby(day).sum()
    assert s["r2_t"] == pytest.approx(ic_summary(gain)["ic_t"])
