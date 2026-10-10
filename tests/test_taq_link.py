import pandas as pd

from dsr.taq_link import link_permnos, mark_primary, pilot_groups


def _names(*rows):
    """CRSP name records: permno, namedt, nameenddt, cusip9, securitytype, securitysubtype, sharetype, ticker."""
    cols = ["permno", "namedt", "nameenddt", "cusip9", "securitytype", "securitysubtype", "sharetype", "ticker"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def _masters(*rows):
    """Master rows: day, symbol, cusip."""
    out = pd.DataFrame(rows, columns=["day", "symbol", "cusip"])
    return out.assign(day=pd.to_datetime(out["day"]))


def _link(symbols, masters, names):
    symbols = pd.DataFrame(symbols, columns=["day", "symbol"]).assign(day=lambda d: pd.to_datetime(d["day"]))
    return link_permnos(symbols, masters, names).set_index(["day", "symbol"])


NAMES = _names(
    ("1", "2010-01-01", "2015-06-30", "AAA111111", "EQTY", "COM", "NS", "OLD"),   # CUSIP reused by another
    ("2", "2015-07-01", "2025-12-31", "AAA111111", "EQTY", "COM", "NS", "NEW"),   # PERMNO from 2015-07
    ("3", "2016-01-11", "2025-12-31", "BBB222222", "FUND", "ETF", "NS", "ETF"),   # CRSP starts a few days late
    ("4", "2010-01-01", "2012-12-31", "CCC333333", "EQTY", "COM", "NS", "C1"),    # two PERMNOs, neither dated
    ("5", "2018-01-01", "2025-12-31", "CCC333333", "EQTY", "COM", "NS", "C2"),
    ("6", "2010-01-01", "2025-12-31", "DDD444444", "EQTY", "COM", "AD", "ADR"),
    ("7", "2010-01-01", "2025-12-31", "DDD444444", "EQTY", "COM", "AD", "ADR2"),   # two PERMNOs, both dated
    ("8", "2010-01-01", "2025-12-31", "", "EQTY", "COM", "NS", "NOCUSIP"),       # blank CUSIP never matches
)


def test_dated_record_gives_the_permno_in_force_that_day():
    masters = _masters(("2015-06-01", "A", "AAA111111"), ("2015-08-03", "A", "AAA111111"))
    out = _link([("2015-06-01", "A"), ("2015-08-03", "A")], masters, NAMES)
    assert out.loc[(pd.Timestamp("2015-06-01"), "A"), "permno"] == 1
    assert out.loc[(pd.Timestamp("2015-08-03"), "A"), "permno"] == 2
    assert (out["rule"] == "dated").all()


def test_cusip_of_a_single_permno_links_outside_its_dates():
    out = _link([("2016-01-05", "B")], _masters(("2016-01-05", "B", "BBB222222")), NAMES)
    row = out.loc[(pd.Timestamp("2016-01-05"), "B")]
    assert row["permno"] == 3 and row["rule"] == "undated"


def test_cusip_of_several_permnos_none_dated_is_not_linked():
    out = _link([("2015-03-02", "C")], _masters(("2015-03-02", "C", "CCC333333")), NAMES)
    row = out.loc[(pd.Timestamp("2015-03-02"), "C")]
    assert pd.isna(row["permno"]) and row["rule"] == "no CRSP match"


def test_cusip_of_two_dated_permnos_is_ambiguous():
    out = _link([("2019-06-04", "D")], _masters(("2019-06-04", "D", "DDD444444")), NAMES)
    row = out.loc[(pd.Timestamp("2019-06-04"), "D")]
    assert pd.isna(row["permno"]) and row["rule"] == "ambiguous"


def test_blank_or_missing_cusip_is_not_linked():
    masters = _masters(("2019-06-04", "T", ""), ("2019-06-04", "X", "ZZZ999999"))
    out = _link([("2019-06-04", "T"), ("2019-06-04", "X")], masters, NAMES)
    assert out.loc[(pd.Timestamp("2019-06-04"), "T"), "rule"] == "no CUSIP"
    assert out.loc[(pd.Timestamp("2019-06-04"), "X"), "rule"] == "no CRSP match"
    assert out["permno"].isna().all()


def test_symbol_missing_from_the_days_master_takes_the_latest_earlier_one():
    masters = _masters(("2015-08-03", "A", "AAA111111"), ("2015-08-04", "Z", "ZZZ999999"))
    out = _link([("2015-08-05", "A"), ("2015-08-05", "Q")], masters, NAMES)
    row = out.loc[(pd.Timestamp("2015-08-05"), "A")]
    assert row["permno"] == 2 and row["master_day"] == pd.Timestamp("2015-08-03")
    assert out.loc[(pd.Timestamp("2015-08-05"), "Q"), "rule"] == "not in a master"


def test_symbol_listed_twice_with_different_cusips_is_not_linked():
    masters = _masters(("2015-06-01", "A", "AAA111111"), ("2015-06-01", "A", "DDD444444"))
    out = _link([("2015-06-01", "A")], masters, NAMES)
    row = out.loc[(pd.Timestamp("2015-06-01"), "A")]
    assert pd.isna(row["permno"]) and row["rule"] == "two CUSIPs in the master"


def test_one_row_per_symbol_asked_for():
    masters = _masters(("2015-06-01", "A", "AAA111111"), ("2015-06-01", "A", "AAA111111"))
    out = _link([("2015-06-01", "A"), ("2015-06-01", "B")], masters, NAMES)
    assert len(out) == 2 and out.index.is_unique


def test_match_ignores_the_check_digit():
    """CRSP writes X for the check digit of some CUSIPs (BLX: P1699413X, the master's P16994132)."""
    names = _names(("9", "2010-01-01", "2025-12-31", "P1699413X", "EQTY", "COM", "NS", "BLX"))
    out = _link([("2019-06-04", "BLX")], _masters(("2019-06-04", "BLX", "P16994132")), names)
    assert out.loc[(pd.Timestamp("2019-06-04"), "BLX"), "permno"] == 9


def test_symbol_written_without_its_space_in_the_master():
    """Until 2016-10 the master writes Nasdaq class shares joined (LVNTA) where the trades split them (LVNT A)."""
    masters = _masters(("2015-06-01", "AAA11", "AAA111111"))
    out = _link([("2015-06-01", "AAA1 1")], masters, NAMES)
    row = out.loc[(pd.Timestamp("2015-06-01"), "AAA1 1")]
    assert row["permno"] == 1 and row["master_symbol"] == "AAA11" and row["master_day"] == pd.Timestamp("2015-06-01")


def test_earlier_master_at_most_five_trading_days_back():
    days = ["2015-08-03", "2015-08-04", "2015-08-05", "2015-08-06", "2015-08-07", "2015-08-10", "2015-08-11"]
    masters = _masters(("2015-08-03", "A", "AAA111111"), ("2015-08-04", "B", "AAA111111"),
                       *[(d, "Z", "ZZZ999999") for d in days])
    out = _link([("2015-08-10", "A"), ("2015-08-11", "A"), ("2015-08-11", "B")], masters, NAMES)
    assert out.loc[(pd.Timestamp("2015-08-10"), "A"), "permno"] == 2  # 5 trading days back
    assert out.loc[(pd.Timestamp("2015-08-11"), "A"), "rule"] == "not in a master"  # 6 back
    assert out.loc[(pd.Timestamp("2015-08-11"), "B"), "permno"] == 2  # 5 back


def test_primary_symbol_per_permno_and_day():
    links = pd.DataFrame({
        "day": pd.to_datetime(["2016-01-14"] * 8),
        "symbol": ["KMI", "KMI WI", "ACE", "CB", "MKC V", "TBPH V", "TBPH", "ZZZ"],
        "permno": pd.array([1, 1, 2, 2, 3, 4, 4, pd.NA], dtype="Int64"),
        "ticker": ["KMI", "KMI", "ACE", "ACE", "MKC", "TBPH", "TBPH", None],
    })
    out = mark_primary(links)
    assert out["symbols_on_permno"].tolist() == [2, 2, 2, 2, 1, 2, 2, 0]
    # the regular symbol over its when-issued twin; CRSP's ticker between two regular ones; a symbol alone on its
    # PERMNO whatever its suffix (MKC V is McCormick's voting class, not when issued); never without a PERMNO
    assert out["primary"].tolist() == [True, False, True, False, True, False, True, False]


def test_class_share_symbol_matches_its_crsp_ticker_without_the_space():
    links = pd.DataFrame({"day": pd.to_datetime(["2016-01-14"] * 2), "symbol": ["BRK B", "BRKX"],
                          "permno": pd.array([5, 5], dtype="Int64"), "ticker": ["BRKB", "BRKB"]})
    assert mark_primary(links)["primary"].tolist() == [True, False]


def test_pilot_group_inside_the_pilot_from_the_settled_assignment():
    masters = pd.DataFrame({
        "day": pd.to_datetime(["2016-09-30", "2016-10-05", "2016-10-24", "2016-10-31", "2017-06-01", "2018-09-28",
                               "2018-10-01"]),
        "symbol": ["ABC"] * 7,
        "cusip": ["AAA111111"] * 7,
        "tick_pilot_indicator": [None, "", "C", "3", "3", "3", "C"],
    })
    out = pilot_groups(masters)
    # none before or after the pilot; early October takes the group the CUSIP shows on 2016-10-31 (G3 stocks still
    # show C until they switch)
    assert out["pilot_group"].tolist() == [None, "3", "3", "3", "3", "3", None]
