import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dsr.panel import (
    BINARY_FEATURES, WINDOW_COLUMNS, crsp_windows, family_a, family_b, family_c, family_d, session_prices, spread_20d,
    targets, window_sums,
)


def test_targets_rank_within_day_and_size_group_and_leave_rows_without_a_return_empty():
    day = pd.Timestamp("2020-01-02")
    panel = pd.DataFrame({
        "day": [day] * 11,
        "cap_prev": [2_000_000.0] * 5 + [500_000.0] * 5 + [2_000_000.0],
        "ret_next": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, np.nan],
    })
    spread = [5.0] * 5 + [50.0] * 4 + [np.nan] + [5.0]
    t = targets(panel, spread)
    assert list(t.index) == list(panel.index)
    # each group of 5 is ranked on its own: the top stock of each group gets the same target
    assert t.loc[4, "target_rank"] == pytest.approx(t.loc[9, "target_rank"])
    assert t.loc[0, "target_rank"] == pytest.approx(-t.loc[4, "target_rank"])
    assert list(t["target_class"].iloc[:5]) == [-1, 0, 0, 0, 1]
    assert t["large_cap"].tolist() == [True] * 5 + [False] * 5 + [True]
    assert t["spread_20d"].to_numpy() == pytest.approx(np.array(spread), nan_ok=True)  # kept as it is, for costs
    # demeaned within the day and size group (each group winsorized on its own, symmetric here, so the means stay
    # 0.03 and 0.30)
    big, small = np.array([0.01, 0.02, 0.03, 0.04, 0.05]), np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    big, small = (np.clip(x, np.quantile(x, 0.01), np.quantile(x, 0.99)) for x in (big, small))
    assert t.loc[0, "target_demeaned"] == pytest.approx(big[0] - 0.03)
    assert t.loc[9, "target_demeaned"] == pytest.approx(small[-1] - 0.30)
    assert pd.isna(t.loc[10, "target_demeaned"])
    assert pd.isna(t.loc[10, "target_rank"]) and pd.isna(t.loc[10, "target_class"])


def test_crsp_windows_use_only_days_before_the_decision_day_and_never_mix_stocks():
    dates = pd.bdate_range("2020-01-01", periods=25)
    rets = np.linspace(-0.02, 0.03, 25)
    crsp = pd.concat([
        pd.DataFrame({"permno": 1, "date": dates, "ret": rets}),
        pd.DataFrame({"permno": 2, "date": dates, "ret": 0.5}),
    ])
    w = crsp_windows(crsp).set_index(["permno", "day"])
    t = 21  # the decision day dates[21]: its windows end at dates[20]
    row = w.loc[(1, dates[t])]
    assert row["ret_2_5"] == pytest.approx(np.prod(1 + rets[t - 5:t - 1]) - 1)
    assert row["ret_6_20"] == pytest.approx(np.prod(1 + rets[t - 20:t - 5]) - 1)
    assert row["vol_20"] == pytest.approx(np.std(np.log1p(rets[t - 20:t]), ddof=1))
    assert w.loc[(2, dates[t]), "ret_2_5"] == pytest.approx(1.5 ** 4 - 1)
    # too few days: the third decision day has 1 day in t-5 to t-2, fewer than the 3 of 4 needed
    assert pd.isna(w.loc[(1, dates[2]), "ret_2_5"]) and w.loc[(1, dates[4]), "ret_2_5"] == pytest.approx(
        np.prod(1 + rets[0:3]) - 1)


def _bars(path, rows):
    t = pd.DataFrame(rows, columns=["permno", "piece", "bar", "high", "low", "close"])
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), path)


def test_session_prices_take_the_last_trade_before_the_cutoff_and_before_the_last_hour(tmp_path):
    _bars(tmp_path / "bars.parquet", [
        (1, "pre_open", 0, 99.0, 1.0, 50.0),  # not the session: ignored
        (1, "session", 0, 11.0, 9.0, 10.0),
        (1, "session", 31, 12.0, 10.0, 11.0),  # 14:40-14:50, the last bar before the last hour
        (1, "session", 32, 13.0, 11.0, 12.0),
        (1, "session", 37, 14.0, 12.0, 13.0),
        (2, "session", 13, 5.0, 4.0, 4.5),  # on an early close, 11:40-11:50
        (2, "session", 19, 6.0, 4.0, 5.0),
    ])
    normal = session_prices(tmp_path / "bars.parquet", early_close=False).set_index("permno")
    assert normal.loc[1].tolist() == [13.0, 11.0, 14.0, 9.0]
    early = session_prices(tmp_path / "bars.parquet", early_close=True).set_index("permno")
    assert early.loc[2, ["price_cut", "price_1450"]].tolist() == [5.0, 4.5]
    assert list(normal.columns) == ["price_cut", "price_1450", "high", "low"]


def test_family_a_pieces_compound_to_ret_1550():
    day = pd.Timestamp("2020-01-02")
    panel = pd.DataFrame({"day": [day, day], "permno": [1, 2], "open_t": [10.0, 20.0],
                          "ret_1550": [0.05, -0.02], "ret_prev": [0.01, 0.0]}, index=[7, 8])
    prices = pd.DataFrame({"day": [day, day], "permno": [1, 2], "price_cut": [10.5, 19.0],
                           "price_1450": [10.2, 19.5], "high": [11.0, 21.0], "low": [9.5, 18.0]})
    windows = pd.DataFrame({"day": [day], "permno": [1], "ret_2_5": [0.1], "ret_6_20": [0.2], "vol_20": [0.03]})
    a = family_a(panel, prices, windows)
    assert list(a.index) == [7, 8]
    pieces = (1 + a[["ret_overnight", "ret_open_1450", "ret_last_hour"]]).prod(axis=1) - 1
    assert pieces.tolist() == pytest.approx(panel["ret_1550"].tolist())
    assert a.loc[7, "range_today"] == pytest.approx(np.log(11.0 / 9.5))
    assert pd.isna(a.loc[8, "ret_2_5"])


def test_window_sums_add_up_all_four_pieces(tmp_path):
    cols = ["permno", "piece", *WINDOW_COLUMNS]
    rows = [(1, piece, *[i + 1] * len(WINDOW_COLUMNS))
            for i, piece in enumerate(["prev_last10", "prev_evening", "pre_open", "session"])]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows, columns=cols), preserve_index=False), tmp_path / "b.parquet")
    w = window_sums(tmp_path / "b.parquet").set_index("permno")
    assert (w.loc[1, list(WINDOW_COLUMNS)] == 10).all()  # 1 + 2 + 3 + 4


def _retail(day, permno, buys, sells, dollars=1000.0):
    return {"day": day, "permno": permno, "mid_buy_trades": buys, "mid_sell_trades": sells,
            "mid_buy_dollars": buys * 10.0, "mid_sell_dollars": sells * 10.0, "bjzz_buy_trades": buys,
            "bjzz_sell_trades": sells, "bjzz_buy_dollars": buys * 10.0, "bjzz_sell_dollars": sells * 10.0,
            "dollars": dollars, "trades": 100, "off_trades": 10, "off_odd_trades": 4}


def test_family_b_imbalances_and_histories_use_only_earlier_days():
    days = pd.bdate_range("2020-01-01", periods=25)
    # stock 1: 8 buys and 2 sells every day, except 30 buys on the last day; stock 2: too few trades on its last day
    retail = pd.DataFrame([_retail(d, 1, 30 if i == 24 else 8, 2) for i, d in enumerate(days)]
                          + [_retail(d, 2, 1, 2) if i == 24 else _retail(d, 2, 5, 5) for i, d in enumerate(days)])
    panel = pd.DataFrame({"day": [days[24], days[24]], "permno": [1, 2]}, index=[5, 6])
    b = family_b(panel, retail, days)
    assert list(b.index) == [5, 6]
    assert b.loc[5, "imb_mid_trades"] == pytest.approx(28 / 32)
    assert b.loc[5, "imb_mid_5d"] == pytest.approx(0.6)  # the earlier days only, not today's 28/32
    assert b.loc[5, "retail_abnormal"] == pytest.approx(np.log(33 / 11))
    assert b.loc[5, "retail_trades"] == pytest.approx(np.log(33))
    assert b.loc[5, "retail_share"] == pytest.approx(320 / 1000)
    assert b.loc[5, "odd_lot_share"] == pytest.approx(0.4)
    assert pd.isna(b.loc[6, "imb_mid_trades"]) and b.loc[6, "imb_mid_5d"] == pytest.approx(0.0)


def test_family_b_history_needs_enough_days_in_the_universe():
    days = pd.bdate_range("2020-01-01", periods=25)
    kept = [i for i in range(25) if i not in (20, 21)]  # out of the universe on two of the last five days
    retail = pd.DataFrame([_retail(days[i], 1, 8, 2) for i in kept])
    b = family_b(pd.DataFrame({"day": [days[24]], "permno": [1]}), retail, days)
    assert pd.isna(b.loc[0, "imb_mid_5d"])  # 3 of the 5 days, fewer than 4
    assert b.loc[0, "retail_abnormal"] == pytest.approx(0.0)  # 18 of 20 days is enough


def test_family_c_activity_against_the_previous_20_days_and_the_session_spread():
    days = pd.bdate_range("2020-01-01", periods=25)
    sums = pd.DataFrame([_retail(d, 1, 8, 2, dollars=3000.0 if i == 24 else 1000.0) for i, d in enumerate(days)])
    panel = pd.DataFrame({"day": [days[24]], "permno": [1], "spread_bp": [9.0]})
    c = family_c(panel, sums, days)
    assert c.loc[0, "dollars"] == pytest.approx(np.log1p(3000.0))
    assert c.loc[0, "dollars_abnormal"] == pytest.approx(np.log(3001 / 1001))  # today is not in its own average
    assert c.loc[0, "trades"] == pytest.approx(np.log1p(100))
    assert c.loc[0, "spread"] == pytest.approx(np.log1p(9.0))


def test_family_d_logs_size_and_price_and_keeps_the_flags_as_0_or_1():
    panel = pd.DataFrame({"cap_prev": [1e6, 2e5, 3e5, 4e5], "price_prev": [50.0, -4.0, 10.0, 20.0],
                          "security_type": ["common stock", "foreign common stock", "ADR", "REIT"],
                          "early_close": [True, True, True, True]}, index=[3, 4, 5, 6])
    d = family_d(panel)
    assert list(d.index) == [3, 4, 5, 6]
    assert d.loc[3, "log_cap"] == pytest.approx(np.log(1e6))
    assert d.loc[4, "log_price"] == pytest.approx(np.log(4.0))  # a negative CRSP price is a bid/ask midpoint
    assert d[["is_foreign", "is_adr", "is_reit"]].to_numpy().tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert (d["early_close"] == 1).all()
    assert set(BINARY_FEATURES) == {"is_foreign", "is_adr", "is_reit", "early_close"}


def test_target_demeaned_is_winsorized_at_1_and_99_percent_within_the_day_and_group_before_demeaning():
    day = pd.Timestamp("2020-01-02")
    rets = np.r_[np.linspace(-0.05, 0.05, 199), 5.0]  # one +500% stock among 200 large caps
    t = targets(pd.DataFrame({"day": day, "cap_prev": 2_000_000.0, "ret_next": rets}), np.full(200, 5.0))
    clipped = np.clip(rets, np.quantile(rets, 0.01), np.quantile(rets, 0.99))
    assert t["target_demeaned"].to_numpy() == pytest.approx(clipped - clipped.mean())
    assert t["target_demeaned"].max() < 0.06


def test_spread_20d_is_the_median_over_the_last_20_days_including_the_decision_day():
    calendar = pd.bdate_range("2020-01-01", periods=30)
    # stock 1 every day, its spread rising by 1 bp a day; stock 2 on the last 15 days only, with one day out
    one = pd.DataFrame({"day": calendar, "permno": 1, "spread_bp": np.arange(30.0)})
    two = pd.DataFrame({"day": calendar[15:].delete(3), "permno": 2, "spread_bp": 10.0})
    spreads = pd.concat([one, two], ignore_index=True)
    panel = pd.DataFrame({"day": [calendar[25], calendar[14], calendar[13], calendar[29]], "permno": [1, 1, 1, 2]})
    s = spread_20d(panel, spreads, calendar)
    assert s[0] == np.median(np.arange(6.0, 26.0))  # days 6 to 25: the decision day in, no later day
    assert s[1] == np.median(np.arange(0.0, 15.0))  # 15 days of history are enough
    assert np.isnan(s[2]) and np.isnan(s[3])  # 14 are not, whether the window is young or has a day out
