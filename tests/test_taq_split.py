import pandas as pd

from dsr.taq_split import (CODES, chart_prices, decision_day_prices, decision_day_tables, stack_panels, stock_days,
                            write_decision_day_prices, write_decision_days)

D0, D1, D2 = "20231228", "20231229", "20240102"  # three trading days; the test period starts at D2


def _links(permno=1, primary=True, symbol="A", pilot=None):
    """One WCT day's link of a symbol."""
    return pd.DataFrame({"symbol": [symbol], "permno": pd.array([permno], dtype="Int64"), "primary": [primary],
                         "pilot_group": [pilot]})


def _tables(buckets, symbol="A", close=10.0, dollars=100_000.0):
    """cells and bars with one row per bucket given (counts stored as doubles, as the pull does); daily one row.

    Each bar trades `dollars`, and its last trade is at `close`."""
    bars = pd.DataFrame({"symbol": symbol, "bucket": buckets, "bar": 0, "trades": [10 * (b + 1) for b in buckets],
                         "shares": [100.0 * (b + 1) for b in buckets], "bjzz_buy_trades": 1.0,
                         "close": close, "rl_close": close, "rl_trades": 1.0, "dollars": dollars})
    cells = bars[["symbol", "bucket", "trades", "shares"]].assign(venue=0, z=0, q=0, spread=0, size=0)
    return {"daily": pd.DataFrame({"symbol": [symbol], "spread_bp": [5.0]}), "cells": cells, "bars": bars}


def _universe(permnos=(1,), day=D1, facprc=1.0, div=0.0):
    return pd.DataFrame({"decision_day": pd.Timestamp(day), "permno": pd.array(list(permnos), dtype="Int64"),
                         "security_type": "common stock", "facprc_t": facprc, "div_t": div, "ret_next": 0.01})


def _day(prev=None, own=None, prev_links=None, own_links=None, universe=None):
    return decision_day_tables(prev if prev is not None else _tables([0, 1, 2, 3]),
                               own if own is not None else _tables([0, 1, 2, 3]),
                               prev_links if prev_links is not None else _links(),
                               own_links if own_links is not None else _links(),
                               D0, D1, universe if universe is not None else _universe())


def test_a_day_holds_the_previous_days_last_bar_and_evening_then_its_own_morning_and_session():
    bars = _day()["bars"]
    assert bars["piece"].tolist() == ["prev_last10", "prev_evening", "pre_open", "session"]
    assert bars["trades"].tolist() == [30, 40, 10, 20]
    assert not {"close", "dollars"} - set(bars.columns)  # trade rows are copied unchanged


def test_rows_lose_the_columns_the_filter_makes_redundant_and_counts_are_integers():
    bars = _day()["bars"]
    assert not {"symbol", "primary", "traded_day", "bucket"} & set(bars.columns)
    assert bars["permno"].tolist() == [1, 1, 1, 1]
    assert str(bars["shares"].dtype) == "int64" and str(bars["bjzz_buy_trades"].dtype) == "int64"


def test_round_lot_columns_are_carried_and_their_count_is_an_integer():
    bars = _day()["bars"]
    assert bars["rl_close"].tolist() == [10.0] * 4
    assert str(bars["rl_trades"].dtype) == "int64"


def test_only_the_universes_primary_symbols_are_kept():
    prev, own = _tables([0, 1, 2, 3]), _tables([0, 1, 2, 3])
    for t in (prev, own):  # a second symbol B
        for name in ["bars", "cells"]:
            t[name] = pd.concat([t[name], t[name].assign(symbol="B")], ignore_index=True)
    other_stock = pd.concat([_links(), _links(2, symbol="B")], ignore_index=True)  # stock 2 not in the universe
    out = _day(prev, own, other_stock, other_stock)
    assert set(out["bars"]["permno"]) == {1} and set(out["cells"]["permno"]) == {1}
    twin = pd.concat([_links(), _links(1, primary=False, symbol="B")], ignore_index=True)  # B: not the primary symbol
    assert len(_day(prev, own, twin, twin)["bars"]) == 4


def test_each_row_takes_the_permno_of_the_day_it_traded():
    """Stocks 7 and 8 swap symbols A and B overnight; both in the universe. B's bars have 1,000 more trades."""
    def two(buckets):
        a, b = _tables(buckets), _tables(buckets, symbol="B")
        b["bars"]["trades"] += 1000
        return {name: pd.concat([a[name], b[name]], ignore_index=True) for name in a}
    yesterday = pd.concat([_links(7), _links(8, symbol="B")], ignore_index=True)
    today = pd.concat([_links(8), _links(7, symbol="B")], ignore_index=True)
    bars = _day(two([0, 1, 2, 3]), two([0, 1, 2, 3]), yesterday, today, _universe((7, 8)))["bars"]
    stock_7 = bars[bars["permno"] == 7].set_index("piece")["trades"].to_dict()
    assert stock_7 == {"prev_last10": 30, "prev_evening": 40, "pre_open": 1010, "session": 1020}  # A, then B


def test_daily_has_one_row_per_universe_stock_that_traded_enough_before_both_cutoffs():
    daily = _day(universe=_universe((1, 5)))["daily"].set_index("permno")  # stock 5 did not trade
    assert list(daily.index) == [1]
    assert daily.loc[1, "symbol"] == "A" and daily.loc[1, "spread_bp"] == 5.0 and daily.loc[1, "ret_next"] == 0.01


def test_rule_2_needs_50_thousand_dollars_before_the_cutoff_today_and_the_day_before():
    """Buckets 0 and 1 count (pre-open and session); a day's last 10 minutes and evening do not."""
    thin = _tables([0, 1, 2, 3], dollars=24_000.0)  # 48,000 before the cutoff
    assert _day(own=thin)["daily"].empty
    assert _day(prev=thin)["daily"].empty
    assert len(_day(prev=_tables([0, 1, 2, 3], dollars=25_000.0))["daily"]) == 1  # exactly 50,000
    assert _day(prev=_tables([2, 3], dollars=1e9))["daily"].empty  # traded only after yesterday's cutoff
    assert _day(own=thin)["bars"].empty and _day(own=thin)["cells"].empty


def test_ret_1550_from_the_last_trades_before_both_cutoffs_with_the_split_factor_and_dividend():
    """Yesterday's last trade before the cutoff at 40; today, after a 2-for-1 split and a 0.10 dividend, at 20.5."""
    prev = _tables([0, 1, 2, 3], close=40.0)
    prev["bars"].loc[prev["bars"]["bucket"].isin([2, 3]), "close"] = 99.0  # after the cutoff: not used
    own = _tables([0, 1, 2, 3], close=20.5)
    daily = _day(prev, own, universe=_universe(facprc=2.0, div=0.1))["daily"]
    assert abs(daily["ret_1550"].iloc[0] - ((20.5 * 2.0 + 0.1) / 40.0 - 1)) < 1e-12
    assert "div_t" not in daily and "traded_before_cutoff" not in daily


def test_first_day_without_a_previous_day_has_no_stocks():
    """No price before the previous cutoff, so no return to 15:50 (the real first day always has one)."""
    out = decision_day_tables(None, _tables([0, 1, 2, 3]), None, _links(), None, D1, _universe())
    assert out["daily"].empty and out["bars"].empty


def test_every_code_has_a_meaning():
    assert CODES.set_index(["column", "code"]).index.is_unique
    for column, n in {"piece": 4, "venue": 2, "z": 12, "q": 16, "spread": 5, "size": 5}.items():
        assert len(CODES[CODES["column"] == column]) == n


def _crsp(rows):
    """CRSP daily rows: permno, date, price, cap (thousands), dollar_volume, ret, ret_span, open, facprc."""
    out = pd.DataFrame(rows, columns=["permno", "date", "price", "cap", "dollar_volume", "ret", "ret_span", "open",
                                      "facprc"])
    return out.assign(date=pd.to_datetime(out["date"]))


NAMES = pd.DataFrame({
    "permno": [1, 2, 3, 4, 5], "namedt": pd.to_datetime(["2010-01-01"] * 5),
    "nameenddt": pd.to_datetime(["2030-01-01"] * 5), "ticker": ["A", "B", "C", "D", "E"],
    "securitytype": ["EQTY", "EQTY", "FUND", "FUND", "EQTY"], "securitysubtype": ["COM", "COM", "CEF", "ETF", "COM"],
    "sharetype": ["NS"] * 5, "usincflg": ["Y"] * 5, "issuertype": ["CORP", "CORP", "FUND", "FUND", "CORP"],
    "primaryexch": ["N"] * 5,
})


def test_universe_from_the_previous_days_price_and_cap_and_the_security_type():
    crsp = _crsp([
        (1, "2023-12-28", 10.0, 160_000, 1e6, 0.01, "D1", 9.9, 1.0),  # in: $10, $160M, common stock
        (2, "2023-12-28", 0.5, 160_000, 1e6, 0.01, "D1", 0.5, 1.0),    # out: under $1
        (5, "2023-12-28", 10.0, 160_000, 99_999, 0.01, "D1", 10.0, 1.0),  # out: under $100,000 traded (rule 1)
        (3, "2023-12-28", 10.0, 160_000, 1e6, 0.01, "D1", 10.0, 1.0),  # out: closed-end fund
        (4, "2023-12-28", 10.0, 160_000, 1e6, 0.01, "D1", 10.0, 1.0),  # out: ETF
        (1, "2023-12-29", 11.0, 161_000, 1e6, 0.02, "D1", 10.5, 1.0),
        (1, "2024-01-02", 12.0, 162_000, 1e6, 0.03, "D2", 11.5, 2.0),
    ])
    out = stock_days(crsp, NAMES, [D0, D1, D2], D2, early_closes=set())
    day1 = out[out["decision_day"] == pd.Timestamp(D1)].set_index("permno")
    assert list(day1.index) == [1]
    assert day1.loc[1, "price_prev"] == 10.0 and day1.loc[1, "cap_prev"] == 160_000 and day1.loc[1, "open_t"] == 10.5
    assert pd.isna(day1.loc[1, "ret_next"])  # 2023-12-29's target ends in the test period: the embargo
    day2 = out[out["decision_day"] == pd.Timestamp(D2)].set_index("permno")
    assert day2.loc[1, "facprc_t"] == 2.0 and pd.isna(day2.loc[1, "ret_next"])  # no next day in the sample


def test_target_kept_only_over_one_trading_day():
    crsp = _crsp([
        (1, "2023-12-28", 10.0, 160_000, 1e6, 0.01, "D1", 10.0, 1.0),
        (1, "2023-12-29", 11.0, 161_000, 1e6, 0.02, "D1", 10.5, 1.0),
        (1, "2024-01-02", 12.0, 162_000, 1e6, 0.03, "D4", 11.5, 1.0),
    ])
    out = stock_days(crsp, NAMES, [D0, D1, D2], "20240103", early_closes=set()).set_index("decision_day")
    assert out.loc[pd.Timestamp(D1), "ret_next"] == 0.03 and not out.loc[pd.Timestamp(D1), "ret_next_is_delisting"]


def test_dividend_on_its_ex_date():
    crsp = _crsp([
        (1, "2023-12-28", 10.0, 160_000, 1e6, 0.01, "D1", 10.0, 1.0),
        (1, "2023-12-29", 11.0, 161_000, 1e6, 0.02, "D1", 10.5, 1.0),
    ])
    dividends = pd.DataFrame({"permno": [1, 1], "date": pd.to_datetime(["2023-12-29", "2023-12-29"]), "div": [0.1, 0.05]})
    out = stock_days(crsp, NAMES, [D0, D1, D2], D2, early_closes=set(), dividends=dividends).set_index("decision_day")
    assert abs(out.loc[pd.Timestamp(D1), "div_t"] - 0.15) < 1e-12
    out = stock_days(crsp, NAMES, [D0, D1, D2], D2, early_closes=set())
    assert out["div_t"].eq(0).all()


def test_a_distribution_with_its_own_share_factor_is_left_to_the_split_factor():
    """A stock distribution (shares, not cash) on a day CRSP adjusts the price by: counting its amount too would count
    it twice. Its amount counts on a day without a price factor, as cash does."""
    crsp = _crsp([
        (1, "2023-12-28", 100.0, 160_000, 1e6, 0.01, "D1", 100.0, 1.0),
        (1, "2023-12-29", 10.0, 161_000, 1e6, 0.02, "D1", 10.5, 10.0),  # a price factor of 10 that day
        (2, "2023-12-28", 10.0, 160_000, 1e6, 0.01, "D1", 10.0, 1.0),
        (2, "2023-12-29", 10.0, 160_000, 1e6, 0.01, "D1", 10.0, 1.0),
    ])
    dividends = pd.DataFrame({"permno": [1, 1, 2], "date": pd.to_datetime(["2023-12-29"] * 3),
                              "div": [90.0, 0.5, 2.0], "facpr": [9.0, 0.0, 1.0]})
    out = stock_days(crsp, NAMES, [D0, D1, D2], D2, early_closes=set(), dividends=dividends)
    out = out[out["decision_day"] == pd.Timestamp(D1)].set_index("permno")
    assert out.loc[1, "div_t"] == 0.5 and out.loc[2, "div_t"] == 2.0


def test_cap_floor_is_100_million_in_crsps_thousands():
    crsp = _crsp([(1, "2023-12-28", 10.0, 99_999, 1e6, 0.0, "D1", 10.0, 1.0)])
    assert len(stock_days(crsp, NAMES, [D0, D1, D2], D2, early_closes=set())) == 0


def test_writer_puts_each_day_in_its_period_and_resumes(tmp_path):
    pull, out = tmp_path / "pull", tmp_path / "out"
    for day in [D0, D1, D2]:
        (pull / day).mkdir(parents=True)
        for name, table in _tables([0, 1, 2, 3]).items():
            table.to_parquet(pull / day / f"{name}.parquet")
    universe = {D1: _universe(day=D1), D2: _universe(day=D2)}
    first = write_decision_days(pull, out, [D0, D1, D2], D2, lambda day: _links(), universe.get)
    assert first[["decision_day", "period"]].values.tolist() == [[D1, "train"], [D2, "test"]]
    assert (out / "test" / D2 / "daily.parquet").exists()
    assert write_decision_days(pull, out, [D0, D1, D2], D2, lambda day: _links(), universe.get).equals(first)
    stack_panels(out)
    panel = pd.read_parquet(out / "panel_train.parquet")
    assert panel["day"].tolist() == [pd.Timestamp(D1)] and panel.columns[:2].tolist() == ["day", "permno"]
    assert len(pd.read_parquet(out / "panel_test.parquet")) == 1
    assert not list(out.glob("*/*/daily.parquet")) and (out / "train" / D1 / "bars.parquet").exists()
    assert write_decision_days(pull, out, [D0, D1, D2], D2, lambda day: _links(), universe.get)[
        ["decision_day", "period", "bars_rows"]].equals(first[["decision_day", "period", "bars_rows"]])  # resumes


def test_daily_columns_keep_their_type_on_days_without_values():
    """Outside the Tick Size Pilot every pilot_group is empty; it must still be text, so days can be read together."""
    import pyarrow as pa
    daily = _day()["daily"]
    kind = pa.Table.from_pandas(daily).schema.field("pilot_group").type
    assert pa.types.is_string(kind) or pa.types.is_large_string(kind)


def test_stack_panels_skips_a_period_not_written(tmp_path):
    day = tmp_path / "train" / D1
    day.mkdir(parents=True)
    _day()["daily"].to_parquet(day / "daily.parquet")
    stack_panels(tmp_path)
    assert (tmp_path / "panel_train.parquet").exists() and not (tmp_path / "panel_test.parquet").exists()


def _prices(symbol="A", rl=10.0, rl100=10.5):
    """The price-only pull of one WCT day: a session bar (bucket 1) and the last ten minutes (bucket 2)."""
    row = {f"rl_{p}": rl for p in ["open", "high", "low", "close"]} | {f"rl100_{p}": rl100 for p in ["open", "high", "low", "close"]}
    return pd.DataFrame([{"symbol": symbol, "bucket": b, "bar": 0, **row, "rl_trades": 4, "rl100_trades": 2} for b in [1, 2]])


def test_a_days_prices_hold_the_previous_last_ten_minutes_then_the_session_with_the_permno():
    out = decision_day_prices(_prices(rl=9.0), _prices(), _links(), _links(), D0, D1, {1}, set())
    assert out["piece"].tolist() == ["prev_last10", "session"] and out["permno"].tolist() == [1, 1]
    assert out["rl_close"].tolist() == [9.0, 10.0] and out["px_close"].tolist() == [9.0, 10.0]  # the old rule: rl_
    assert str(out["px_trades"].dtype) == "int64" and "symbol" not in out
    assert decision_day_prices(_prices(), _prices(), _links(), _links(), D0, D1, {2}, set()).empty  # not in the panel


def test_from_2025_11_03_chart_prices_keep_the_100_share_round_lot_except_for_stocks_already_below_it():
    prices = _prices().assign(permno=[1, 2])
    before = chart_prices(prices, "20251031", small_lot={2})
    after = chart_prices(prices, "20251103", small_lot={2})
    assert before["px_close"].tolist() == [10.0, 10.0]
    assert after["px_close"].tolist() == [10.5, 10.0] and after["px_trades"].tolist() == [2, 4]


def test_price_writer_puts_each_day_next_to_its_bars_and_resumes(tmp_path):
    prices_dir, out = tmp_path / "prices", tmp_path / "out"
    for day in [D0, D1, D2]:
        (prices_dir / day).mkdir(parents=True)
        _prices().to_parquet(prices_dir / day / "prices.parquet")
    for period, day in [("train", D1), ("test", D2)]:
        (out / period / day).mkdir(parents=True)
    first = write_decision_day_prices(prices_dir, out, [D0, D1, D2], D2, lambda day: _links(), lambda day: {1}, set())
    assert first[["decision_day", "period", "prices_rows"]].values.tolist() == [[D1, "train", 2], [D2, "test", 2]]
    assert (out / "test" / D2 / "prices.parquet").exists() and not list(out.glob("*/*/*.part"))
    again = write_decision_day_prices(prices_dir, out, [D0, D1, D2], D2, lambda day: _links(), lambda day: {1}, set())
    assert again.equals(first)  # resumes: days written are skipped
