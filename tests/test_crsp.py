import pandas as pd

from dsr.crsp import ONE_TRADING_DAY_SPANS, class_on, is_common_stock, security_type


def _records(*rows):
    cols = ["permno", "namedt", "nameenddt", "securitytype", "securitysubtype", "sharetype", "usincflg", "issuertype"]
    out = pd.DataFrame([dict(zip(cols, r)) for r in rows])
    return out.assign(namedt=pd.to_datetime(out["namedt"]), nameenddt=pd.to_datetime(out["nameenddt"]))


def test_class_on_takes_the_record_in_force_and_leaves_gaps_missing():
    records = _records(
        (1, "2010-01-01", "2015-12-31", "EQTY", "COM", "NS", "Y", "CORP"),
        (1, "2016-01-04", "2025-12-31", "EQTY", "COM", "NS", "N", "CORP"),  # reincorporated abroad
        (2, "2018-01-01", "2025-12-31", "FUND", "ETF", "NS", "Y", "FUND"),
    )
    df = pd.DataFrame({"permno": [1, 1, 1, 2], "date": pd.to_datetime(["2015-06-01", "2016-01-02", "2017-03-01", "2017-03-01"])},
                      index=[10, 11, 12, 13])
    cls = class_on(df, records)
    assert list(cls.index) == [10, 11, 12, 13]
    assert cls.loc[10, "usincflg"] == "Y" and cls.loc[12, "usincflg"] == "N"
    assert cls.loc[[11, 13], "securitytype"].isna().all()  # between records; before the ETF's first record


def test_common_stock_is_share_code_10_or_11():
    cls = pd.DataFrame({
        "securitytype": ["EQTY", "EQTY", "EQTY", "EQTY", "FUND", None],
        "securitysubtype": ["COM", "COM", "COM", "COM", "ETF", None],
        "sharetype": ["NS", "NS", "AD", "NS", "NS", None],
        "usincflg": ["Y", "N", "N", "Y", "Y", None],
        "issuertype": ["CORP", "CORP", "CORP", "REIT", "FUND", None],
    })
    assert is_common_stock(cls).tolist() == [True, False, False, False, False, False]


def test_security_type_names_each_class():
    cls = pd.DataFrame({
        "securitytype": ["EQTY", "EQTY", "EQTY", "EQTY", "EQTY", "FUND", "FUND", None, "EQTY"],
        "securitysubtype": ["COM", "COM", "COM", "COM", "COM", "ETF", "CEF", None, "PFD"],
        "sharetype": ["NS", "NS", "AD", "NS", "UG", "NS", "NS", None, "NS"],
        "usincflg": ["Y", "N", "N", "Y", "Y", "Y", "Y", None, "Y"],
        "issuertype": ["CORP", "CORP", "CORP", "REIT", "CORP", "FUND", "FUND", None, "CORP"],
    })
    assert security_type(cls).tolist() == ["common stock", "foreign common stock", "ADR", "REIT", "unit", "ETF",
                                           "closed-end fund", "unclassified", "other"]


def test_one_trading_day_spans():
    spans = pd.Series(["D1", "D3", "D4", "DU", "DD", "P2", "MR", None])
    assert spans.str.fullmatch(ONE_TRADING_DAY_SPANS).fillna(False).tolist() == [True] * 5 + [False] * 3


def test_class_on_accepts_dates_of_any_resolution():
    records = _records((1, "2010-01-01", "2025-12-31", "EQTY", "COM", "NS", "Y", "CORP"))
    df = pd.DataFrame({"permno": [1], "date": pd.to_datetime(["2015-06-01"]).astype("datetime64[us]")})
    assert class_on(df, records.astype({"namedt": "datetime64[ns]", "nameenddt": "datetime64[ns]"})).loc[0, "usincflg"] == "Y"
