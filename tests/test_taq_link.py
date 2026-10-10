import pandas as pd

from dsr.taq_link import link_permnos


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
    assert out.loc[(pd.Timestamp("2015-08-03"), "A"), "ticker"] == "NEW"


def test_cusip_of_a_single_permno_links_outside_its_dates():
    out = _link([("2016-01-05", "B")], _masters(("2016-01-05", "B", "BBB222222")), NAMES)
    row = out.loc[(pd.Timestamp("2016-01-05"), "B")]
    assert row["permno"] == 3 and row["rule"] == "undated"
    assert row["securitysubtype"] == "ETF"  # classification from the nearest name record


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
