"""TAQ symbols to CRSP PERMNOs, by date (explanations/taq/taq_pull.md, section 11).

A TAQ symbol is never matched by ticker. On each day the symbol's CUSIP comes from that day's master file: under the
symbol itself or, failing that, written without its space (until October 2016 the master writes Nasdaq class shares
joined, LVNTA, where the trades split them, LVNT A); if that day's master lists neither, from the latest master that
lists the symbol in the max_days_back trading days before (a stock's last trading day is often missing from that
morning's master; a cap keeps a reused ticker from taking up an old CUSIP). and its PERMNO from CRSP's name history: the
PERMNO whose CUSIP (`cusip9`) was in force that day. CUSIPs are compared on their first 8 characters: the 9th is a
check digit computed from them, which CRSP writes as X for a few securities (BLX: P1699413X in CRSP, P16994132 in the
master). A CUSIP with no name record in force that day but used by a single PERMNO anywhere in CRSP links to that
PERMNO: CRSP sometimes starts a new CUSIP a few days after TAQ shows it. The match fixes how the data was recorded and selects nothing by value, so it runs on the full sample. Classifying the linked security is dsr.crsp.class_on's
job, the same as for the Robintrack panels.
"""

import pandas as pd


def link_permnos(symbols: pd.DataFrame, masters: pd.DataFrame, names: pd.DataFrame, max_days_back: int = 5
                  ) -> pd.DataFrame:
    """One row per (day, symbol) of `symbols`, with its CUSIP and PERMNO.

    symbols: day (datetime), symbol. masters: day, symbol, cusip for every row of the master files. names: CRSP's
    stocknames_v2 with permno, namedt, nameenddt and cusip9, as text. Trading days are counted in master files, one
    per trading day.

    Returns day, symbol, master_day and master_symbol (the master and its symbol the CUSIP comes from), cusip, permno
    (Int64, missing when not linked) and rule, one of:
      dated                     one PERMNO had this CUSIP in force that day
      undated                   none had, and the CUSIP belongs to a single PERMNO in all of CRSP
      ambiguous                 several PERMNOs had the CUSIP in force that day
      no CRSP match             the CUSIP is not in CRSP, or only under several PERMNOs, none of them in force
      no CUSIP                  the master's CUSIP is blank
      two CUSIPs in the master  the master lists the symbol twice with different CUSIPs
      not in a master           neither that day's master nor one in the max_days_back trading days before lists it
    """
    ns = "datetime64[ns]"
    out = symbols[["day", "symbol"]].drop_duplicates().astype({"day": ns}).reset_index(drop=True)

    # the symbol's CUSIP in each master; listed twice with different CUSIPs gives none
    m = masters[["day", "symbol", "cusip"]].astype({"day": ns}).assign(cusip=lambda d: d["cusip"].fillna("").str.strip())
    m = m.drop_duplicates().groupby(["day", "symbol"], as_index=False).agg(cusip=("cusip", "first"),
                                                                         cusips=("cusip", "size"))
    m = m.rename(columns={"day": "master_day"})
    out = out.merge(m, left_on=["day", "symbol"], right_on=["master_day", "symbol"], how="left")
    out["master_symbol"] = out["symbol"].where(out["master_day"].notna())
    missing = out["master_day"].isna()
    if missing.any():  # the same day's master, the symbol written without its space (one master symbol only)
        joined = m.assign(key=m["symbol"].str.replace(" ", "")).drop_duplicates(["master_day", "key"], keep=False)
        same = out.loc[missing, ["day", "symbol"]].reset_index().assign(key=lambda d: d["symbol"].str.replace(" ", ""))
        same = same.merge(joined.rename(columns={"symbol": "master_symbol"}), left_on=["day", "key"],
                          right_on=["master_day", "key"]).set_index("index")
        out.loc[same.index, ["master_day", "master_symbol", "cusip", "cusips"]] = same[
            ["master_day", "master_symbol", "cusip", "cusips"]]
    missing = out["master_day"].isna()
    if missing.any():  # the latest earlier master that lists the symbol
        earlier = pd.merge_asof(out.loc[missing, ["day", "symbol"]].reset_index().sort_values("day"),
                                m.sort_values("master_day"), left_on="day", right_on="master_day", by="symbol",
                                allow_exact_matches=False).set_index("index")
        master_days = pd.DatetimeIndex(m["master_day"].unique()).sort_values()
        back = master_days.searchsorted(earlier["day"]) - master_days.searchsorted(earlier["master_day"])
        earlier.loc[back > max_days_back, ["master_day", "cusip", "cusips"]] = pd.NA
        out.loc[missing, ["master_day", "cusip", "cusips"]] = earlier[["master_day", "cusip", "cusips"]]
        out.loc[missing, "master_symbol"] = out.loc[missing, "symbol"].where(out.loc[missing, "master_day"].notna())

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
    dated = cand[cand["index"].isin(permnos[permnos == 1].index)].groupby("index")["permno"].first()
    ambiguous = permnos[permnos > 1].index

    # undated: a CUSIP of a single PERMNO in all of CRSP
    single = nm.groupby("cusip8")["permno"].agg(["nunique", "first"])
    single = single.loc[single["nunique"] == 1, "first"]
    rest = out.index[usable & ~out.index.isin(permnos.index) & out["cusip8"].isin(single.index)]
    undated = out.loc[rest, "cusip8"].map(single)

    out["permno"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[dated.index, "permno"] = dated.to_numpy()
    out.loc[undated.index, "permno"] = undated.to_numpy()

    out["rule"] = "no CRSP match"
    out.loc[out["cusip"].fillna("").eq(""), "rule"] = "no CUSIP"
    out.loc[out["cusips"].gt(1), "rule"] = "two CUSIPs in the master"
    out.loc[out["master_day"].isna(), "rule"] = "not in a master"
    out.loc[ambiguous, "rule"] = "ambiguous"
    out.loc[dated.index, "rule"] = "dated"
    out.loc[undated.index, "rule"] = "undated"
    return out[["day", "symbol", "master_day", "master_symbol", "cusip", "permno", "rule"]]


WHEN_ISSUED_SUFFIXES = ("WI", "WD", "V")  # when issued, when distributed; V is also Nasdaq's "when issued"


def mark_primary(links: pd.DataFrame) -> pd.DataFrame:
    """Which symbol carries each PERMNO on each day, so a stock is never counted twice.

    links: day, symbol, permno (missing when not linked) and ticker (CRSP's ticker for that PERMNO that day). Returns
    links with symbols_on_permno (how many symbols map to the row's PERMNO that day; 0 without a PERMNO) and primary:
      - a symbol alone on its PERMNO that day is primary, whatever its suffix (MKC V is McCormick's voting class);
      - among several, those without a when-issued suffix (WI, WD, V: shares of a split, spin-off or new issue trading
        before they exist, at their own price) are preferred, and if several remain (a merger's last day, when the
        master already gives the target's ticker the acquirer's CUSIP), the one matching CRSP's ticker, compared
        without spaces;
      - a symbol without a PERMNO is never primary.
    """
    out = links.copy()
    linked = out["permno"].notna()
    key = [out["day"], out["permno"]]
    out["symbols_on_permno"] = linked.groupby(key).transform("size").where(linked, 0).astype(int)
    twin = out["symbol"].str.split(" ").str[1].isin(WHEN_ISSUED_SUFFIXES).fillna(False)
    regular = linked & ~twin
    n_regular = regular.groupby(key).transform("sum")
    is_ticker = out["symbol"].str.replace(" ", "").eq(out["ticker"].fillna("").str.replace(" ", ""))
    n_ticker = (regular & is_ticker).groupby(key).transform("sum")
    primary = out["symbols_on_permno"].eq(1) | (regular & (n_regular.eq(1) | (is_ticker & n_ticker.eq(1))))
    out["primary"] = (linked & primary).fillna(False).astype(bool)
    return out


# SEC Tick Size Pilot: first and last day, and the first day the master shows every stock's assigned group
PILOT_START, PILOT_END, PILOT_SETTLED = pd.Timestamp("2016-10-03"), pd.Timestamp("2018-09-28"), pd.Timestamp("2016-10-31")


def pilot_groups(masters: pd.DataFrame) -> pd.DataFrame:
    """Each master row's Tick Size Pilot group (C control, 1, 2, 3), or None outside the pilot.

    masters: day, symbol, cusip and tick_pilot_indicator. The master leaves the group blank from 2016-10-03 to 10-21 and
    shows group 3 stocks as C until each was phased in (late October), so before PILOT_SETTLED a stock takes the group
    its CUSIP shows on that day; FINRA published the assignments before the pilot began, so this uses nothing unknown
    at the time. After the pilot the master keeps writing C for former pilot stocks, which means nothing.
    """
    out = masters[["day", "symbol", "cusip"]].copy()
    group = masters["tick_pilot_indicator"].fillna("").str.strip().replace("", None)
    settled = masters.loc[masters["day"].eq(PILOT_SETTLED), ["cusip"]].assign(g=group).dropna()
    settled = settled.drop_duplicates("cusip", keep=False).set_index("cusip")["g"]
    early = masters["day"].between(PILOT_START, PILOT_SETTLED - pd.Timedelta(days=1))
    group = group.where(~early, masters["cusip"].map(settled))
    out["pilot_group"] = group.where(masters["day"].between(PILOT_START, PILOT_END)).astype(object).where(lambda g: g.notna(), None)
    return out
