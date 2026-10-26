import numpy as np
import pandas as pd
import pytest

from dsr.backtest import (books, daily, drawdowns, positions, sharpe_t, stress, summary, trades, turnover, universe,
                          yearly)
from dsr.evaluation import ic_summary


def _panel(rets, preds=None):
    """One row per day and stock from a days x stocks array of returns; the prediction is the return unless given."""
    rets = np.asarray(rets, dtype=float)
    preds = rets if preds is None else np.asarray(preds, dtype=float)
    n_days, n = rets.shape
    day = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n)
    return pd.DataFrame({"day": day, "permno": np.tile(np.arange(n), n_days), "pred": preds.ravel(), "ret": rets.ravel()})


def test_positions_hold_the_top_and_bottom_bucket_of_each_day():
    p = _panel(np.tile(np.arange(10.0), (2, 1)))
    held, average = positions(p["day"], p["permno"], p["pred"], p["ret"], buckets=5)
    first = held[held["day"] == held["day"].iloc[0]]
    assert sorted(first.loc[first["leg"] == 1, "permno"]) == [8, 9]
    assert sorted(first.loc[first["leg"] == -1, "permno"]) == [0, 1]
    assert np.allclose(average, 4.5) and np.allclose(held["average"], 4.5)
    deciles, _ = positions(p["day"], p["permno"], p["pred"], p["ret"], buckets=10)
    assert len(deciles) == 4  # one stock a leg a day


def test_a_missing_return_is_ranked_counts_as_zero_and_a_day_without_returns_is_dropped():
    rets = np.tile(np.arange(10.0), (3, 1))
    rets[0, 9] = np.nan  # the top prediction's return is missing
    rets[2] = np.nan  # the sample's last day: no returns at all
    p = _panel(rets, preds=np.tile(np.arange(10.0), (3, 1)))
    held, average = positions(p["day"], p["permno"], p["pred"], p["ret"], buckets=10)
    assert len(average) == 2
    top = held[(held["day"] == average.index[0]) & (held["leg"] == 1)]
    assert list(top["permno"]) == [9] and top["ret"].iloc[0] == 0
    assert average.iloc[0] == pytest.approx(np.arange(9.0).sum() / 10)


def test_daily_returns_are_the_long_leg_minus_the_short_leg_and_each_against_the_average():
    p = _panel([[0.0, 0.01, 0.02, 0.03], [0.04, 0.0, 0.0, 0.0]], preds=[[0, 1, 2, 3], [0, 1, 2, 3]])
    held, average = positions(p["day"], p["permno"], p["pred"], p["ret"], buckets=2)
    d = daily(held, average)
    assert d["long"].iloc[0] == pytest.approx(0.025) and d["short"].iloc[0] == pytest.approx(0.005)
    assert d["long_short"].iloc[0] == pytest.approx(0.02)
    assert d["long_vs_average"].iloc[0] == pytest.approx(0.01) and d["short_vs_average"].iloc[0] == pytest.approx(0.01)
    assert d["long_short"].iloc[1] == pytest.approx(-0.02)
    b = books(d)
    assert b["short leg"][0].iloc[0] == pytest.approx(-0.005)
    assert (b["long leg − average"][0] + b["average − short leg"][0]).equals(b["long-short"][0])


def test_turnover_is_the_share_of_the_leg_replaced_after_the_returns_move_the_weights():
    days = pd.bdate_range("2020-01-01", periods=4)
    leg = pd.DataFrame({"day": np.repeat(days, 2), "permno": [1, 2, 1, 2, 1, 3, 4, 5],
                        "ret": [0.0, 0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0]})
    t = turnover(leg, days)
    assert np.isnan(t.iloc[0])
    assert t.iloc[1] == 0  # the same stocks, equal returns: nothing to trade
    assert t.iloc[2] == pytest.approx(0.5 * (abs(0.5 - 0.55) + 0.45 + 0.5))  # 1 drifted to 55%, 2 sold, 3 bought
    assert t.iloc[3] == 1  # every stock new


def test_a_trade_is_a_stay_in_one_leg_over_consecutive_days():
    days = pd.bdate_range("2020-01-01", periods=5)
    held = pd.DataFrame({"day": days[[0, 1, 2, 4, 0]], "permno": [7, 7, 7, 7, 8], "leg": [1, 1, 1, 1, -1],
                         "ret": [0.1, -0.1, 0.1, 0.0, 0.02], "average": [0.0, 0.0, 0.0, 0.0, 0.03]})
    t = trades(held, days)
    assert len(t) == 3  # stock 7's gap on day 4 breaks it in two
    first = t[(t["permno"] == 7) & (t["entry"] == days[0])].iloc[0]
    assert first["days"] == 3 and first["ret"] == pytest.approx(1.1 * 0.9 * 1.1 - 1)
    s = summary(daily(held, pd.Series(0.0, index=days)), t, held)
    assert s["trade_hit_long"] == 0.5  # +8.9% made money, 0% did not
    assert s["trade_hit_short"] == 0 and s["trade_hit_vs_average_short"] == 1  # rose 2%, but less than the average


def test_summary_of_a_known_series():
    days = pd.bdate_range("2020-01-01", periods=4)
    d = pd.DataFrame({"long_short": [0.01, -0.02, 0.01, 0.03], "long_vs_average": 0.0, "short_vs_average": 0.0,
                      "turnover_long": 0.5, "turnover_short": 0.5}, index=days)
    t = pd.DataFrame({"permno": [1, 2], "leg": [1, -1], "entry": days[:2], "days": [4, 3], "ret": [0.1, 0.1],
                      "average": [0.0, 0.0]})
    s = summary(d, t, t.rename(columns={"entry": "day"}))
    assert s["mean_bp"] == pytest.approx(75) and s["positive_days"] == 0.75
    assert s["trades"] == 2 and s["holding_days"] == 3.5 and s["trade_hit"] == 0.5
    assert s["sharpe"] == pytest.approx(np.sqrt(252) * 0.0075 / np.std([0.01, -0.02, 0.01, 0.03], ddof=1))
    assert s["max_drawdown_pct"] == pytest.approx(2.0)  # from +1% to -1%
    y = yearly(d, t)
    assert list(y.index) == [2020] and y.loc[2020, "mean_bp"] == pytest.approx(75)


def test_universe_returns_equal_and_value_weighted():
    day = np.repeat(pd.bdate_range("2020-01-01", periods=3), 2)
    ret = np.array([0.01, 0.03, np.nan, 0.02, np.nan, np.nan])  # a missing return; the last day has none
    cap = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    u = universe(day, ret, cap)
    assert len(u) == 2
    assert u["equal"].iloc[0] == pytest.approx(0.02) and u["value"].iloc[0] == pytest.approx(0.015)
    assert u["equal"].iloc[1] == pytest.approx(0.01)  # the missing return counts as 0, as in positions


def test_a_drawdown_is_a_run_of_days_below_the_running_peak():
    r = pd.Series([1.0, -2.0, 1.0, 2.0, -1.0, -1.0, 3.0, -1.0])  # summed: 1, -1, 0, 2, 1, 0, 3, 2
    assert list(drawdowns(r)) == [2, 2, 1]  # the last one has not recovered
    assert list(drawdowns(pd.Series([-1.0, 2.0]))) == [1]  # the peak starts at 0
    assert drawdowns(pd.Series([1.0, 1.0])).empty


def test_stress_of_a_known_series():
    days = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([0.04, -0.01, -0.02, -0.01, 0.01], index=days)
    market = pd.Series([0.02, 0.0, -0.01, 0.0, 0.01], index=days)
    s = stress(r, market, drop=0.4)
    assert s["mean_bp"] == pytest.approx(20) and s["annual_mean_pct"] == pytest.approx(50.4)
    sr = r.mean() / r.std()
    assert s["sharpe_t"] == pytest.approx(sharpe_t(r))
    assert s["sortino"] == pytest.approx(np.sqrt(252) * 0.002 / np.sqrt(0.0006 / 5))
    assert s["win_days"] == 0.4 and s["average_win_bp"] == pytest.approx(250)
    assert s["average_loss_bp"] == pytest.approx(-400 / 3)
    assert s["drawdowns"] == 1 and s["drawdown_days_mean"] == 4 and s["max_drawdown_pct"] == pytest.approx(4)
    assert s["corr_universe"] == pytest.approx(np.corrcoef(r, market)[0, 1])
    assert s["mean_bp_without_best_days"] == pytest.approx(-400 / 3)  # 40% of 5 days: the 4% and the 1% days
    assert s["mean_bp_without_best_wins"] == pytest.approx(-75)  # 40% of the 2 winning days, rounded: the 4% day


def test_sharpe_t_without_lags_is_mertens_with_the_sample_moments():
    r = pd.Series(0.001 + 0.01 * np.random.default_rng(0).standard_t(4, 300))
    x = r - r.mean()
    sd = np.sqrt((x ** 2).mean())
    sr, skew, kurt = r.mean() / sd, (x ** 3).mean() / sd ** 3, (x ** 4).mean() / sd ** 4
    assert sharpe_t(r, lags=0) == pytest.approx(sr / np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / len(r)))


def test_sharpe_t_of_returns_of_one_size_is_the_newey_west_t_of_their_mean_scaled():
    rng, up = np.random.default_rng(1), [True]
    for u in rng.random(399):
        up.append(u > 0.2 if up[-1] else u < 0.4)  # runs of up and down days: autocorrelated, more up than down
    r = pd.Series(np.where(up, 0.01, -0.01))  # r² is constant, so only the mean is uncertain
    assert sharpe_t(r) == pytest.approx(ic_summary(r)["ic_t"] * (1 - r.mean() ** 2 / (r ** 2).mean()))


def test_sharpe_t_is_lower_when_returns_run_on():
    e = np.random.default_rng(2).normal(0.001, 0.01, 2000)
    r = pd.Series(e + 0.5 * np.r_[0.0, e[:-1]])  # positively autocorrelated
    assert sharpe_t(r) < 0.85 * sharpe_t(r, lags=0)
