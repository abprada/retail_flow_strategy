"""CRSP definitions shared by both models: the daily file's columns, the return to predict, and each security's class.

The Robintrack panels (scripts/robintrack/01_clean_explore_robintrack.py) and the TAQ panels both use these, so "common stock",
the previous day's price and the next-day return mean the same in Model 1 and Model 2. Columns joined from the daily
file are named by when they are known (explanations/robintrack/robintrack.md, "Daily count and change"): `*_prev` for the previous
trading day (known before the 15:50 cutoff), `*_t` for day t (known only after its close), and `ret_next` for the
return from day t's close to the next trading day's close, the return to predict.
"""

import numpy as np
import pandas as pd

# crsp.stkdlysecuritydata columns and their short names
DAILY_COLUMNS = {
    "dlyret": "ret", "dlyretx": "retx", "dlyretdurflg": "ret_span", "dlyprc": "price", "dlyprcflg": "price_flag",
    "dlyopen": "open", "dlyhigh": "high", "dlylow": "low", "dlyclose": "close", "dlyvol": "volume",
    "dlyprcvol": "dollar_volume", "dlycap": "cap", "dlydelflg": "delisted",
}

# CRSP return spans of one trading day: D1 to D4 by calendar days, DU for long closures, DD for a delisting return
ONE_TRADING_DAY_SPANS = r"D[1-4UD]"

# the stocknames_v2 columns that classify a security
NAME_RECORD_COLUMNS = ["permno", "namedt", "nameenddt", "ticker", "securitytype", "securitysubtype", "sharetype",
                       "usincflg", "issuertype", "primaryexch"]


def read_name_records(path) -> pd.DataFrame:
    """CRSP's name records (stocknames_v2): one per PERMNO and period of unchanged name data, sorted by start date."""
    return pd.read_csv(path, usecols=NAME_RECORD_COLUMNS, parse_dates=["namedt", "nameenddt"]).sort_values("namedt")


def class_on(df: pd.DataFrame, name_records: pd.DataFrame) -> pd.DataFrame:
    """The name record in force on each row's date (df has permno and date), aligned to df's index; missing when no
    record covers the date. A security is classified with what CRSP says about it that day."""
    unit = df["date"].dtype  # merge_asof needs both sides at the same resolution
    cls = pd.merge_asof(
        df[["permno", "date"]].rename_axis("_row").reset_index().sort_values("date"),
        name_records.astype({"namedt": unit, "nameenddt": unit}).sort_values("namedt"),
        left_on="date", right_on="namedt", by="permno",
    )
    return cls[cls["date"] <= cls["nameenddt"]].set_index("_row").reindex(df.index)


def is_common_stock(cls: pd.DataFrame) -> pd.Series:
    """Common stock, CRSP share code 10 or 11: equity, common, ordinary shares, US-incorporated, issued by a corporation."""
    return (
        cls["securitytype"].eq("EQTY") & cls["securitysubtype"].eq("COM") & cls["sharetype"].eq("NS")
        & cls["usincflg"].eq("Y") & cls["issuertype"].eq("CORP")
    ).fillna(False).astype(bool)


def security_type(cls: pd.DataFrame) -> pd.Series:
    """The security's class from its name record: common stock, foreign common stock, ADR, REIT, unit, ETF,
    closed-end fund, unclassified (no record in force) or other."""
    is_com = cls["securitytype"].eq("EQTY") & cls["securitysubtype"].eq("COM")
    return pd.Series(np.select(
        [
            is_com & cls["issuertype"].eq("REIT"),
            is_com & cls["sharetype"].eq("UG"),
            is_com & cls["sharetype"].eq("AD") & cls["issuertype"].eq("CORP"),
            is_com & cls["sharetype"].eq("NS") & cls["issuertype"].eq("CORP") & cls["usincflg"].eq("Y"),
            is_com & cls["sharetype"].eq("NS") & cls["issuertype"].eq("CORP") & cls["usincflg"].eq("N"),
            cls["securitytype"].eq("FUND") & cls["securitysubtype"].eq("ETF"),
            cls["securitytype"].eq("FUND") & cls["securitysubtype"].eq("CEF"),
            cls["securitytype"].isna(),
        ],
        ["REIT", "unit", "ADR", "common stock", "foreign common stock", "ETF", "closed-end fund", "unclassified"],
        default="other",
    ), index=cls.index)
