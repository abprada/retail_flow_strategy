"""TAQ symbols to CRSP PERMNOs, by date (explanations/taq_pull.md, section 11).

A TAQ symbol is never matched by ticker. On each day the symbol's CUSIP comes from that day's master file (or, if that
day's master does not list it, from the latest earlier master that does), and its PERMNO from CRSP's name history: the
PERMNO whose CUSIP (`cusip9`) was in force that day. CUSIPs are compared on their first 8 characters: the 9th is a
check digit computed from them, which CRSP writes as X for a few securities (BLX: P1699413X in CRSP, P16994132 in the
master). A CUSIP with no name record in force that day but used by a single PERMNO anywhere in CRSP links to that
PERMNO: CRSP sometimes starts a new CUSIP a few days after TAQ shows it. The match fixes how the data was recorded and selects nothing by value, so it runs on the full sample.
"""

import pandas as pd

CLASS_COLUMNS = ["securitytype", "securitysubtype", "sharetype", "ticker"]


def link_permnos(symbols: pd.DataFrame, masters: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, symbol) of `symbols`, with its CUSIP and PERMNO.

    symbols: day (datetime), symbol. masters: day, symbol, cusip for every row of the master files. names: CRSP's
    stocknames_v2 with permno, namedt, nameenddt, cusip9 and CLASS_COLUMNS, as text.

    Returns day, symbol, master_day (the master the CUSIP comes from), cusip, permno (Int64, missing when not linked),
    rule, and CLASS_COLUMNS from the name record used. rule is one of:
      dated                     one PERMNO had this CUSIP in force that day
      undated                   none had, and the CUSIP belongs to a single PERMNO in all of CRSP (classified by its
                                name record nearest the day)
      ambiguous                 several PERMNOs had the CUSIP in force that day
      no CRSP match             the CUSIP is not in CRSP, or only under several PERMNOs, none of them in force
      no CUSIP                  the master's CUSIP is blank
      two CUSIPs in the master  the master lists the symbol twice with different CUSIPs
      not in a master           no master on or before the day lists the symbol
    """
    ns = "datetime64[ns]"
    out = symbols[["day", "symbol"]].drop_duplicates().astype({"day": ns}).reset_index(drop=True)

    # the symbol's CUSIP in each master; listed twice with different CUSIPs gives none
    m = masters[["day", "symbol", "cusip"]].astype({"day": ns}).assign(cusip=lambda d: d["cusip"].fillna("").str.strip())
    m = m.drop_duplicates().groupby(["day", "symbol"], as_index=False).agg(cusip=("cusip", "first"),
                                                                         cusips=("cusip", "size"))
    m = m.rename(columns={"day": "master_day"})
    out = out.merge(m, left_on=["day", "symbol"], right_on=["master_day", "symbol"], how="left")
    missing = out["master_day"].isna()
    if missing.any():  # the latest earlier master that lists the symbol
        earlier = pd.merge_asof(out.loc[missing, ["day", "symbol"]].reset_index().sort_values("day"),
                                m.sort_values("master_day"), left_on="day", right_on="master_day", by="symbol",
                                allow_exact_matches=False).set_index("index")
        out.loc[missing, ["master_day", "cusip", "cusips"]] = earlier[["master_day", "cusip", "cusips"]]

    nm = names[names["cusip9"].fillna("").str.strip() != ""]
    nm = nm.assign(permno=nm["permno"].astype("int64"), namedt=pd.to_datetime(nm["namedt"]).astype(ns),
                   nameenddt=pd.to_datetime(nm["nameenddt"]).astype(ns),
                   cusip8=nm["cusip9"].str.strip().str[:8]).reset_index(drop=True)
    out["cusip8"] = out["cusip"].str[:8]
    usable = out["cusip"].fillna("").ne("") & out["cusips"].eq(1)

    # dated: the name records in force that day
    cand = out.loc[usable, ["day", "cusip8"]].reset_index().merge(
        nm[["cusip8", "permno", "namedt", "nameenddt"]].reset_index(names="record"), on="cusip8")
    cand = cand[(cand["namedt"] <= cand["day"]) & (cand["day"] <= cand["nameenddt"])]
    permnos = cand.groupby("index")["permno"].nunique()
    dated = cand[cand["index"].isin(permnos[permnos == 1].index)].groupby("index")["record"].first()
    ambiguous = permnos[permnos > 1].index

    # undated: a CUSIP of a single PERMNO in all of CRSP, classified by its name record nearest the day
    single = nm.groupby("cusip8")["permno"].nunique()
    single = single[single == 1].index
    rest = out.index[usable & ~out.index.isin(permnos.index) & out["cusip8"].isin(single)]
    near = out.loc[rest, ["day", "cusip8"]].reset_index().merge(
        nm[["cusip8", "namedt", "nameenddt"]].reset_index(names="record"), on="cusip8")
    near["gap"] = (near["namedt"] - near["day"]).clip(lower=pd.Timedelta(0)) + (
        near["day"] - near["nameenddt"]).clip(lower=pd.Timedelta(0))
    undated = near.sort_values(["index", "gap", "record"]).groupby("index")["record"].first()

    record = pd.concat([dated, undated])
    out["permno"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[record.index, "permno"] = nm.loc[record.to_numpy(), "permno"].to_numpy()
    for col in CLASS_COLUMNS:
        out[col] = pd.Series(nm.loc[record.to_numpy(), col].to_numpy(), index=record.index).reindex(out.index)

    out["rule"] = "no CRSP match"
    out.loc[out["cusip"].fillna("").eq(""), "rule"] = "no CUSIP"
    out.loc[out["cusips"].gt(1), "rule"] = "two CUSIPs in the master"
    out.loc[out["master_day"].isna(), "rule"] = "not in a master"
    out.loc[ambiguous, "rule"] = "ambiguous"
    out.loc[dated.index, "rule"] = "dated"
    out.loc[undated.index, "rule"] = "undated"
    return out[["day", "symbol", "master_day", "cusip", "permno", "rule", *CLASS_COLUMNS]]
