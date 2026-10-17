"""Check the planned symbol -> PERMNO matching (TAQ master CUSIP to CRSP's CUSIP) against WRDS's TAQ-CRSP link.

    WRDS_USERNAME=<you> uv run python scripts/taq/checks/TAQ_permno_link.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push. WRDS is only contacted when the link sample
is not downloaded yet.

The pull keys every stock by its TAQ symbol, which is matched to a CRSP PERMNO locally: the day's master file
(data/raw/taq/master/) gives the symbol's CUSIP, and CRSP's name history (data/raw/crsp/stocknames_v2.csv) gives the
PERMNO whose CUSIP (cusip9) was in force that day. WRDS publishes its own daily link
(wrdsapps_link_crsp_taqm.taqmclink); it is built on CRSP's old format and ends on 2024-12-31, so it cannot cover 2025.
This compares the two on the first trading day of 2014, the first trading day of June of each year 2014 to 2024, and
2024-12-31: for every TAQ symbol either side links, whether both give the same PERMNO.

Outputs (gitignored), in data/raw/taq/coverage/link/:
    taqmclink_sample.csv   WRDS's link rows on those days, as Postgres exports them
    comparison.csv         per day: symbols by outcome
    disagreements.csv      every symbol-day where the two differ, with both sides' CUSIP and PERMNO
"""

import os
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "raw" / "taq" / "coverage" / "link"
MASTER = ROOT / "data" / "raw" / "taq" / "master"
NAMES = ROOT / "data" / "raw" / "crsp" / "stocknames_v2.csv"
LINK = OUT / "taqmclink_sample.csv"


def sample_days() -> list[str]:
    sessions = pd.Series(xcals.get_calendar("XNYS", start="2014-01-01", end="2024-12-31").sessions)
    june = sessions[sessions.dt.month == 6].groupby(sessions.dt.year).min()
    return sorted({"2014-01-02", "2024-12-31", *june.dt.strftime("%Y-%m-%d")})


def ours(day: str, names: pd.DataFrame) -> pd.DataFrame:
    """The day's master symbols with the PERMNO(s) whose CRSP CUSIP was in force that day (none if no match)."""
    master = pd.read_parquet(MASTER / f"mastm_{day.replace('-', '')}.parquet", columns=["symbol_15", "cusip"])
    master = master.rename(columns={"symbol_15": "symbol", "cusip": "cusip_master"})
    live = names[(names["namedt"] <= day) & (names["nameenddt"] >= day)][["permno", "cusip9"]].drop_duplicates()
    matched = master.merge(live, left_on="cusip_master", right_on="cusip9", how="left")
    return (matched.groupby("symbol", as_index=False)
            .agg(cusip_master=("cusip_master", "first"), permno_ours=("permno", lambda s: " ".join(sorted(s.dropna())))))


def main() -> None:
    days = sample_days()
    OUT.mkdir(parents=True, exist_ok=True)
    if not LINK.exists():
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            copy_query(conn, "SELECT * FROM wrdsapps_link_crsp_taqm.taqmclink WHERE date IN ("
                       + ", ".join(f"'{d}'" for d in days) + ") ORDER BY date, sym_root, sym_suffix", LINK)

    names = pd.read_csv(NAMES, usecols=["permno", "namedt", "nameenddt", "cusip9"], dtype=str, keep_default_na=False)
    link = pd.read_csv(LINK, dtype=str, keep_default_na=False)
    link["symbol"] = link["sym_root"] + link["sym_suffix"].map(lambda s: f" {s}" if s else "")
    summary, disagreements = [], []
    for day in days:
        both = ours(day, names).merge(
            link.loc[link["date"] == day, ["symbol", "permno", "cusip", "match_lvl"]].rename(
                columns={"permno": "permno_link", "cusip": "cusip_link"}),
            on="symbol", how="outer")
        has_ours, has_link = both["permno_ours"].fillna("").ne(""), both["permno_link"].notna()
        outcome = pd.Series("neither", index=both.index)
        outcome[has_ours & has_link & both["permno_ours"].eq(both["permno_link"])] = "same PERMNO"
        outcome[has_ours & has_link & both["permno_ours"].ne(both["permno_link"])] = "different PERMNO"
        outcome[~has_ours & has_link & both["cusip_master"].isna()] = "WRDS only: symbol not in master"
        outcome[~has_ours & has_link & both["cusip_master"].notna()] = "WRDS only: master CUSIP not in CRSP"
        outcome[has_ours & ~has_link] = "ours only"
        both["outcome"] = outcome
        counts = both["outcome"].value_counts()
        summary.append({"day": day, "linked by WRDS": int(has_link.sum()), "linked by us": int(has_ours.sum()), **counts})
        disagreements.append(both[~both["outcome"].isin(["same PERMNO", "neither"])].assign(day=day))

    summary = pd.DataFrame(summary).fillna(0).set_index("day")
    summary = summary.astype(int)
    summary.to_csv(OUT / "comparison.csv")
    pd.concat(disagreements).to_csv(OUT / "disagreements.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string())
    total = summary.sum()
    print(f"\nall {len(days)} days: {total['same PERMNO']:,} of {total['linked by WRDS']:,} WRDS links "
          f"({total['same PERMNO'] / total['linked by WRDS']:.2%}) have the same PERMNO from the master CUSIP")


if __name__ == "__main__":
    main()
