import numpy as np
import pandas as pd
import pytest

from dsr.backtest import daily, positions, summary, trades, turnover, universe, yearly


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
