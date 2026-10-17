import pandas as pd

from dsr.taq import check_day, check_prices, compare_prices_with_bars


def _day():
    """One symbol: in the regular session's first bar, two off-exchange BJZZ buys (z code 9) at a one-cent spread,
    signed buys by the midpoint method too (q code 12), odd lots of 30 shares at $10.008, then one exchange trade of
    100 shares at the ask, $10.01, the bar's only round lot."""
    cells = pd.DataFrame([
        {"symbol": "X", "bucket": 1, "venue": 1, "z": 9, "q": 12, "spread": 1, "size": 1, "trades": 2, "shares": 60, "dollars": 600.48},
        {"symbol": "X", "bucket": 1, "venue": 0, "z": 0, "q": 14, "spread": 1, "size": 2, "trades": 1, "shares": 100, "dollars": 1001.0},
    ])
    zero = {f"{r}_{v}": 0 for r in ["bjzz_sell", "mid_sell"] for v in ["trades", "shares", "dollars"]}
    bars = pd.DataFrame([{
        "symbol": "X", "bucket": 1, "bar": 0, "open": 10.008, "high": 10.01, "low": 10.008, "close": 10.01,
        "rl_open": 10.01, "rl_high": 10.01, "rl_low": 10.01, "rl_close": 10.01, "rl_trades": 1,
        "trades": 3, "shares": 160, "dollars": 1601.48,
        "bjzz_buy_trades": 2, "bjzz_buy_shares": 60, "bjzz_buy_dollars": 600.48,
        "mid_buy_trades": 2, "mid_buy_shares": 60, "mid_buy_dollars": 600.48, **zero,
        "off_trades": 2, "off_shares": 60, "off_dollars": 600.48,
    }])
    daily = pd.DataFrame({"symbol": ["X"], "spread_bp": [9.99]})
    audit = pd.DataFrame([
        {"section": "step", "item": "raw", "venue": 0, "kept": True, "trades": 2, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "raw", "venue": 1, "kept": True, "trades": 3, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "sale_condition", "venue": 0, "kept": False, "trades": 1, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "records", "venue": 1, "kept": False, "trades": 1, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "kept", "venue": 0, "kept": True, "trades": 1, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "kept", "venue": 1, "kept": True, "trades": 2, "shares": 0, "dollars": 0.0},
    ])
    return daily, cells, bars, audit


def test_check_day_passes_on_consistent_tables():
    checks = check_day(*_day(), regular_bars=38)
    assert checks["passed"].all(), checks[~checks["passed"]]


def test_check_day_catches_a_retail_column_that_does_not_match_the_cells():
    daily, cells, bars, audit = _day()
    bars.loc[0, "bjzz_buy_trades"] = 3
    checks = check_day(daily, cells, bars, audit, regular_bars=38).set_index("check")["passed"]
    assert not checks["bars' bjzz_buy columns equal the matching cells"]
    assert checks.drop("bars' bjzz_buy columns equal the matching cells").all()


def test_check_day_catches_dollars_off_by_more_than_a_cent_and_bars_out_of_range():
    daily, cells, bars, audit = _day()
    bars.loc[0, "dollars"] += 0.02
    bars.loc[0, "bar"] = 20
    checks = check_day(daily, cells, bars, audit, regular_bars=20).set_index("check")["passed"]
    assert not checks["cells and bars have the same trades, shares and dollars per symbol and bucket"]
    assert not checks["regular bars are numbered 0 to 19, other buckets 0"]


def test_check_day_passes_on_a_bar_of_odd_lots_only_and_on_a_pull_without_round_lot_columns():
    daily, cells, bars, audit = _day()
    bars.loc[0, ["rl_open", "rl_high", "rl_low", "rl_close", "rl_trades"]] = [None, None, None, None, 0]
    cells = cells.assign(size=1)  # no trade of 100 shares or more left either
    assert check_day(daily, cells, bars, audit, regular_bars=38)["passed"].all()
    old = _day()
    old_bars = old[2].drop(columns=["rl_open", "rl_high", "rl_low", "rl_close", "rl_trades"])
    assert check_day(old[0], old[1], old_bars, old[3], regular_bars=38)["passed"].all()


def test_check_day_catches_round_lot_prices_missing_or_outside_the_bar():
    daily, cells, bars, audit = _day()
    bars.loc[0, "rl_close"] = None
    checks = check_day(daily, cells, bars, audit, regular_bars=38).set_index("check")["passed"]
    assert not checks["bars' round-lot prices are there exactly when rl_trades > 0, and rl_trades <= trades"]
    daily, cells, bars, audit = _day()
    bars.loc[0, ["rl_high", "rl_close"]] = 10.02  # above the bar's high
    checks = check_day(daily, cells, bars, audit, regular_bars=38).set_index("check")["passed"]
    assert not checks["bars' round-lot prices lie within the bar's low and high, and their open and close within their own"]
    assert checks["bars' round-lot prices are there exactly when rl_trades > 0, and rl_trades <= trades"]


def test_check_day_catches_a_trade_of_100_shares_or_more_left_out_of_the_round_lots():
    daily, cells, bars, audit = _day()
    bars.loc[0, ["rl_open", "rl_high", "rl_low", "rl_close", "rl_trades"]] = [None, None, None, None, 0]
    checks = check_day(daily, cells, bars, audit, regular_bars=38).set_index("check")["passed"]
    assert not checks["bars' rl_trades count at least every trade of 100 shares or more under $100,000"]
    assert checks["bars' round-lot prices are there exactly when rl_trades > 0, and rl_trades <= trades"]


def test_check_day_catches_audit_steps_that_do_not_add_up():
    daily, cells, bars, audit = _day()
    audit.loc[audit["item"].eq("records"), "trades"] = 2
    checks = check_day(daily, cells, bars, audit, regular_bars=38).set_index("check")["passed"]
    assert not checks["audit: raw minus the removed steps equals kept"]
    assert checks["audit: kept equals the trades in cells"]


def _prices():
    """Two bars of one symbol from the price-only pull: the first regular bar and the last ten minutes."""
    return pd.DataFrame({"symbol": ["X", "X"], "bucket": [1, 2], "bar": [0, 0], "rl_open": [10.0, 10.2],
                         "rl_high": [10.3, 10.2], "rl_low": [9.9, 10.1], "rl_close": [10.1, 10.1], "rl_trades": [5, 2],
                         "rl100_open": [10.0, None], "rl100_high": [10.2, None], "rl100_low": [10.0, None],
                         "rl100_close": [10.1, None], "rl100_trades": [3, 0]})


def test_check_prices_passes_on_consistent_prices_and_catches_a_close_above_the_high():
    assert check_prices(_prices(), regular_bars=38)["passed"].all()
    prices = _prices()
    prices.loc[0, "rl_close"] = 10.4
    checks = check_prices(prices, regular_bars=38).set_index("check")["passed"]
    assert not checks["prices: open and close lie within low and high"]
    assert checks.drop("prices: open and close lie within low and high").all()


def test_check_prices_catches_a_bar_without_a_round_lot_or_outside_the_session():
    prices = _prices()
    prices.loc[1, ["bucket", "rl_trades"]] = [3, 0]
    checks = check_prices(prices, regular_bars=38).set_index("check")["passed"]
    assert not checks["prices: buckets 1 and 2 only, regular bars numbered 0 to 37"]
    assert not checks["prices: every row has a round lot and all four prices"]


def test_prices_against_the_full_pull_catch_a_round_lot_high_above_the_bar():
    bars = _prices()[["symbol", "bucket", "bar"]].assign(high=[10.3, 10.2], low=[9.8, 10.1], trades=[9, 2])
    assert compare_prices_with_bars(_prices(), bars)["passed"].all()
    bars.loc[0, "high"] = 10.25
    result = compare_prices_with_bars(_prices(), bars).set_index("check")["passed"]
    assert not result["round-lot high and low within the bar's high and low"]
    assert result["every price row has a bar in the full pull"]


def test_check_prices_catches_rl100_prices_outside_the_rl_range_or_without_their_trades():
    prices = _prices()
    prices.loc[0, "rl100_high"] = 10.35  # above the high of all round lots
    checks = check_prices(prices, regular_bars=38).set_index("check")["passed"]
    assert not checks["prices: rl100_ prices lie within the rl_ low and high, open and close within their own"]
    prices = _prices()
    prices.loc[1, "rl100_close"] = 10.1  # a price without a trade of 100 shares or more
    checks = check_prices(prices, regular_bars=38).set_index("check")["passed"]
    assert not checks["prices: rl100_ prices are there exactly when rl100_trades > 0, and rl100_trades <= rl_trades"]
