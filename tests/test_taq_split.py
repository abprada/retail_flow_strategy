import pandas as pd

from dsr.taq_split import CODES, decision_day_tables, write_decision_days


def _links(permno, primary=True):
    """One WCT day's link of symbol A."""
    return pd.DataFrame({"symbol": ["A"], "permno": pd.array([permno], dtype="Int64"), "primary": [primary]})


def _tables(rows):
    """cells and bars with a bucket column, one row per bucket given; daily one row."""
    bars = pd.DataFrame({"symbol": "A", "bucket": rows, "bar": 0, "trades": [10 * (b + 1) for b in rows]})
    cells = bars[["symbol", "bucket", "trades"]].assign(z=0)
    return {"daily": pd.DataFrame({"symbol": ["A"], "spread_bp": [5.0]}), "cells": cells, "bars": bars}


def test_a_decision_day_holds_its_own_morning_and_session_and_the_previous_days_last_bar_and_evening():
    prev, own = _tables([0, 1, 2, 3]), _tables([0, 1, 2, 3])
    out = decision_day_tables(prev, own, _links(1), _links(1), "20190603", "20190604")
    assert sorted(out["bars"]["bucket"]) == [0, 1, 2, 3]
    assert out["bars"].set_index("bucket")["trades"].to_dict() == {0: 10, 1: 20, 2: 30, 3: 40}
    assert len(out["cells"]) == 4
    assert out["daily"][["symbol", "spread_bp"]].equals(own["daily"])  # the session's spread is the day's own


def test_rows_are_copied_unchanged_and_none_twice():
    prev, own = _tables([0, 1, 2, 3]), _tables([0, 1, 2, 3])
    prev["bars"]["trades"] += 1000  # tell the two days apart
    out = decision_day_tables(prev, own, _links(1), _links(1), "20190603", "20190604")["bars"].set_index("bucket")["trades"]
    assert out.to_dict() == {0: 10, 1: 20, 2: 1030, 3: 1040}


def test_first_decision_day_without_a_previous_day():
    out = decision_day_tables(None, _tables([0, 1, 2, 3]), None, _links(1), None, "20190604")
    assert sorted(out["bars"]["bucket"]) == [0, 1]


def test_writer_puts_each_decision_day_in_its_period_and_resumes(tmp_path):
    pull, out = tmp_path / "pull", tmp_path / "out"
    for i, day in enumerate(["20231228", "20231229", "20240102"]):
        (pull / day).mkdir(parents=True)
        for name, table in _tables([0, 1, 2, 3]).items():
            if "trades" in table:
                table = table.assign(trades=table["trades"] + 100 * i)
            table.to_parquet(pull / day / f"{name}.parquet")
    days = ["20231228", "20231229", "20240102"]
    links = lambda day: _links(1)
    first = write_decision_days(pull, out, days, "20240102", links)
    assert first[["decision_day", "period"]].values.tolist() == [["20231229", "train"], ["20240102", "test"]]
    bars = pd.read_parquet(out / "test" / "20240102" / "bars.parquet").set_index("bucket")["trades"]
    assert bars.to_dict() == {2: 130, 3: 140, 0: 210, 1: 220}  # 2023-12-29's evening, 2024-01-02's own day
    again = write_decision_days(pull, out, days, "20240102", links)  # everything written: skipped, same rows
    assert again.equals(first)


def test_rows_say_their_piece_and_run_in_time_order():
    prev, own = _tables([0, 1, 2, 3]), _tables([0, 1, 2, 3])
    bars = decision_day_tables(prev, own, _links(1), _links(1), "20190603", "20190604")["bars"]
    assert bars["piece"].tolist() == ["prev_last10", "prev_evening", "pre_open", "session"]
    assert bars["bucket"].tolist() == [2, 3, 0, 1]


def test_every_code_has_a_meaning():
    codes = CODES.set_index(["column", "code"])
    assert codes.index.is_unique
    for column, n in {"bucket": 4, "venue": 2, "z": 12, "q": 16, "spread": 5, "size": 5}.items():
        assert sorted(CODES.loc[CODES["column"] == column, "code"]) == list(range(n))


def test_each_row_takes_the_permno_of_the_day_it_traded():
    """A's ticker moved to another stock overnight: yesterday's pieces keep yesterday's PERMNO."""
    out = decision_day_tables(_tables([0, 1, 2, 3]), _tables([0, 1, 2, 3]), _links(7), _links(8, primary=False),
                              "20190603", "20190604")
    bars = out["bars"].set_index("piece")
    assert bars["permno"].to_dict() == {"prev_last10": 7, "prev_evening": 7, "pre_open": 8, "session": 8}
    assert bars["primary"].to_dict() == {"prev_last10": True, "prev_evening": True, "pre_open": False, "session": False}
    assert out["daily"]["permno"].tolist() == [8]


def test_a_symbol_without_a_link_has_no_permno_and_is_not_primary():
    none = pd.DataFrame({"symbol": [], "permno": pd.array([], dtype="Int64"), "primary": pd.array([], dtype=bool)})
    out = decision_day_tables(None, _tables([0, 1]), None, none, None, "20190604")
    assert out["bars"]["permno"].isna().all() and not out["bars"]["primary"].any()


def test_each_row_says_the_day_it_traded():
    out = decision_day_tables(_tables([0, 1, 2, 3]), _tables([0, 1, 2, 3]), _links(1), _links(1), "20190603", "20190604")
    days = out["bars"].set_index("piece")["traded_day"].dt.strftime("%Y-%m-%d").to_dict()
    assert days == {"prev_last10": "2019-06-03", "prev_evening": "2019-06-03", "pre_open": "2019-06-04",
                    "session": "2019-06-04"}
    assert out["daily"]["traded_day"].dt.strftime("%Y-%m-%d").tolist() == ["2019-06-04"]
