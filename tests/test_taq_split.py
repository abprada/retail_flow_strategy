import pandas as pd

from dsr.taq_split import CODES, decision_day_tables, stock_days, write_decision_days

D0, D1, D2 = "20231228", "20231229", "20240102"  # three trading days; the test period starts at D2


def _links(permno=1, primary=True, symbol="A", pilot=None):
    """One WCT day's link of a symbol."""
    return pd.DataFrame({"symbol": [symbol], "permno": pd.array([permno], dtype="Int64"), "primary": [primary],
                         "pilot_group": [pilot]})


def _tables(buckets, symbol="A"):
    """cells and bars with one row per bucket given (counts stored as doubles, as the pull does); daily one row."""
    bars = pd.DataFrame({"symbol": symbol, "bucket": buckets, "bar": 0, "trades": [10 * (b + 1) for b in buckets],
                         "shares": [100.0 * (b + 1) for b in buckets], "bjzz_buy_trades": 1.0})
    cells = bars[["symbol", "bucket", "trades", "shares"]].assign(venue=0, z=0, q=0, spread=0, size=0)
    return {"daily": pd.DataFrame({"symbol": [symbol], "spread_bp": [5.0]}), "cells": cells, "bars": bars}


def _universe(permnos=(1,), day=D1):
    return pd.DataFrame({"decision_day": pd.Timestamp(day), "permno": pd.array(list(permnos), dtype="Int64"),
                         "security_type": "common stock", "ret_next": 0.01})


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


def test_rows_lose_the_columns_the_filter_makes_redundant_and_counts_are_integers():
    bars = _day()["bars"]
    assert not {"symbol", "primary", "traded_day", "bucket"} & set(bars.columns)
    assert bars["permno"].tolist() == [1, 1, 1, 1]
    assert str(bars["shares"].dtype) == "int64" and str(bars["bjzz_buy_trades"].dtype) == "int64"


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
    """Symbol A belonged to stock 7 yesterday and to stock 8 today; both in the universe."""
    out = _day(prev_links=_links(7), own_links=_links(8), universe=_universe((7, 8)))
    assert out["bars"].set_index("piece")["permno"].to_dict() == {
        "prev_last10": 7, "prev_evening": 7, "pre_open": 8, "session": 8}


def test_daily_has_one_row_per_universe_stock_even_without_trades():
    daily = _day(universe=_universe((1, 5)))["daily"].set_index("permno")  # stock 5 did not trade
    assert list(daily.index) == [1, 5]
    assert daily.loc[1, "symbol"] == "A" and daily.loc[1, "spread_bp"] == 5.0 and daily.loc[1, "traded_before_cutoff"]
    assert pd.isna(daily.loc[5, "spread_bp"]) and not daily.loc[5, "traded_before_cutoff"]
    assert daily.loc[5, "ret_next"] == 0.01


def test_first_day_without_a_previous_day():
    out = decision_day_tables(None, _tables([0, 1, 2, 3]), None, _links(), None, D1, _universe())
    assert out["bars"]["piece"].tolist() == ["pre_open", "session"]


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
    "permno": [1, 2, 3], "namedt": pd.to_datetime(["2010-01-01"] * 3), "nameenddt": pd.to_datetime(["2030-01-01"] * 3),
    "ticker": ["A", "B", "C"], "securitytype": ["EQTY", "EQTY", "FUND"], "securitysubtype": ["COM", "COM", "CEF"],
    "sharetype": ["NS", "NS", "NS"], "usincflg": ["Y", "Y", "Y"], "issuertype": ["CORP", "CORP", "FUND"],
    "primaryexch": ["N", "N", "N"],
})


def test_universe_from_the_previous_days_price_and_cap_and_the_security_type():
    crsp = _crsp([
        (1, "2023-12-28", 10.0, 60_000, 1e6, 0.01, "D1", 9.9, 1.0),   # in: $10, $60M, common stock
        (2, "2023-12-28", 0.5, 60_000, 1e6, 0.01, "D1", 0.5, 1.0),    # out: under $1
        (3, "2023-12-28", 10.0, 60_000, 1e6, 0.01, "D1", 10.0, 1.0),  # out: closed-end fund
        (1, "2023-12-29", 11.0, 61_000, 1e6, 0.02, "D1", 10.5, 1.0),
        (1, "2024-01-02", 12.0, 62_000, 1e6, 0.03, "D2", 11.5, 2.0),
    ])
    out = stock_days(crsp, NAMES, [D0, D1, D2], D2, early_closes=set())
    day1 = out[out["decision_day"] == pd.Timestamp(D1)].set_index("permno")
    assert list(day1.index) == [1]
    assert day1.loc[1, "price_prev"] == 10.0 and day1.loc[1, "cap_prev"] == 60_000 and day1.loc[1, "open_t"] == 10.5
    assert pd.isna(day1.loc[1, "ret_next"])  # 2023-12-29's target ends in the test period: the embargo
    day2 = out[out["decision_day"] == pd.Timestamp(D2)].set_index("permno")
    assert day2.loc[1, "facprc_t"] == 2.0 and pd.isna(day2.loc[1, "ret_next"])  # no next day in the sample


def test_target_kept_only_over_one_trading_day():
    crsp = _crsp([
        (1, "2023-12-28", 10.0, 60_000, 1e6, 0.01, "D1", 10.0, 1.0),
        (1, "2023-12-29", 11.0, 61_000, 1e6, 0.02, "D1", 10.5, 1.0),
        (1, "2024-01-02", 12.0, 62_000, 1e6, 0.03, "D4", 11.5, 1.0),
    ])
    out = stock_days(crsp, NAMES, [D0, D1, D2], "20240103", early_closes=set()).set_index("decision_day")
    assert out.loc[pd.Timestamp(D1), "ret_next"] == 0.03 and not out.loc[pd.Timestamp(D1), "ret_next_is_delisting"]


def test_cap_floor_is_50_million_in_crsps_thousands():
    crsp = _crsp([(1, "2023-12-28", 10.0, 49_999, 1e6, 0.0, "D1", 10.0, 1.0)])
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


def test_daily_columns_keep_their_type_on_days_without_values():
    """Outside the Tick Size Pilot every pilot_group is empty; it must still be text, so days can be read together."""
    import pyarrow as pa
    daily = _day()["daily"]
    kind = pa.Table.from_pandas(daily).schema.field("pilot_group").type
    assert pa.types.is_string(kind) or pa.types.is_large_string(kind)
