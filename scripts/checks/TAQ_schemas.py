"""Check that the inputs of the TAQ cleaning look the same in every year from 2014 as on the day it was built on.

    WRDS_USERNAME=<you> uv run python scripts/checks/TAQ_schemas.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push.

The cleaning (scripts/01_clean_and_explore_TAQ_raw_trades.py) was written on 2019-06-04 only. For the first trading day of 2014
and the first trading day in June of each year 2014 to 2025, this saves:
    columns.csv        the columns of wct, mastm and ctm
    conditions.csv     wct trades by sale condition (tr_scond) and venue (ex)
    master_codes.csv   master-file symbols by listing exchange, Tick Size Pilot group and security type
    opening.csv        opening prints (sale condition O) on each stock's listing exchange before 15:50, by listing
                       exchange: stocks, stocks with more than one print, stocks with more than one price, most prints
    corrections.csv    ctm rows by correction code (tr_corr other than 00)
Each query is saved as it finishes and skipped when its file exists, so a stopped run resumes where it stopped.
At the end it prints what differs from 2019-06-03.

Outputs (gitignored): data/raw/taq/coverage/schemas/<YYYYMMDD>/
"""

import os
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "taq" / "coverage" / "schemas"
REFERENCE = "20190603"  # the June sample day closest to the day the cleaning was built on
TABLES = ["wct", "mastm", "ctm"]
MASTER_CODES = ["listed_exchange", "tick_pilot_indicator", "sec_type"]
SYMBOL = "w.sym_root || CASE WHEN coalesce(w.sym_suffix, '') = '' THEN '' ELSE ' ' || w.sym_suffix END"


def sample_days() -> list[str]:
    sessions = xcals.get_calendar("XNYS", start="2014-01-01", end="2025-12-31").sessions.strftime("%Y%m%d")
    june = [next(d for d in sessions if d[:4] == str(y) and d[4:6] == "06") for y in range(2014, 2026)]
    return sorted({sessions[0], *june})


def fetch_day(conn, ymd: str) -> None:
    out = OUT / ymd
    copy_query(conn, "SELECT table_name, column_name, data_type FROM information_schema.columns "
                     f"WHERE table_schema = 'taqmsec' AND table_name IN ({', '.join(f"'{t}_{ymd}'" for t in TABLES)}) "
                     "ORDER BY table_name, ordinal_position", out / "columns.csv")
    master_columns = set(pd.read_csv(out / "columns.csv").query(f"table_name == 'mastm_{ymd}'")["column_name"])

    print(f"{ymd}: sale conditions", flush=True)
    copy_query(conn, f"SELECT tr_scond, ex, count(*) AS trades FROM taqmsec.wct_{ymd} GROUP BY 1, 2 ORDER BY 3 DESC",
               out / "conditions.csv")

    codes = [c for c in MASTER_CODES if c in master_columns]
    if codes:
        copy_query(conn, f"SELECT {', '.join(codes)}, count(*) AS symbols FROM taqmsec.mastm_{ymd} "
                         f"GROUP BY {', '.join(codes)} ORDER BY symbols DESC", out / "master_codes.csv")

    if {"symbol_15", "listed_exchange"} <= master_columns:
        print(f"{ymd}: opening prints", flush=True)
        copy_query(conn, f"""
            WITH per_stock AS (
                SELECT m.listed_exchange, {SYMBOL} AS symbol, count(*) AS prints, count(DISTINCT w.price) AS prices
                FROM taqmsec.wct_{ymd} w JOIN taqmsec.mastm_{ymd} m ON m.symbol_15 = {SYMBOL}
                WHERE w.tr_scond LIKE '%O%' AND w.ex = m.listed_exchange AND w.time_m < '15:50'
                GROUP BY 1, 2
            )
            SELECT listed_exchange, count(*) AS stocks, sum((prints > 1)::int) AS stocks_several_prints,
                   sum((prices > 1)::int) AS stocks_several_prices, max(prints) AS most_prints
            FROM per_stock GROUP BY 1 ORDER BY 2 DESC""", out / "opening.csv")

    print(f"{ymd}: corrections", flush=True)
    copy_query(conn, f"SELECT tr_corr, count(*) AS n_rows FROM taqmsec.ctm_{ymd} WHERE tr_corr <> '00' GROUP BY 1 ORDER BY 1",
               out / "corrections.csv")


def read(ymd: str, name: str) -> pd.DataFrame:
    path = OUT / ymd / name
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


def report(days: list[str]) -> None:
    ref_cols = read(REFERENCE, "columns.csv")
    ref_cols = {t: set(ref_cols.loc[ref_cols["table_name"] == f"{t}_{REFERENCE}", "column_name"]) for t in TABLES}
    ref_chars = set("".join(read(REFERENCE, "conditions.csv")["tr_scond"]))
    for ymd in days:
        cols = read(ymd, "columns.csv")
        print(f"\n== {ymd}")
        for t in TABLES:
            have = set(cols.loc[cols["table_name"] == f"{t}_{ymd}", "column_name"])
            missing, extra = sorted(ref_cols[t] - have), sorted(have - ref_cols[t])
            if missing or extra:
                print(f"  {t}: missing {missing or '-'}; extra {extra or '-'}")
        cond = read(ymd, "conditions.csv")
        cond["trades"] = cond["trades"].astype(int)
        new_chars = {c: int(cond.loc[cond["tr_scond"].str.contains(c, regex=False), "trades"].sum())
                     for c in set("".join(cond["tr_scond"])) - ref_chars}
        print(f"  trades {cond['trades'].sum():,}; venues {sorted(set(cond['ex']))}")
        print(f"  condition codes not seen on {REFERENCE}: {new_chars or 'none'}")
        codes = read(ymd, "master_codes.csv")
        for c in MASTER_CODES:
            if c in codes:
                print(f"  master {c}: {sorted(set(codes[c]))}")
        opening = read(ymd, "opening.csv")
        if not opening.empty:
            print("  opening prints on the listing exchange:")
            print("    " + opening.to_string(index=False).replace("\n", "\n    "))
        corr = read(ymd, "corrections.csv")
        print(f"  corrections: {dict(zip(corr.get('tr_corr', []), corr.get('n_rows', [])))}")


def main() -> None:
    days = sample_days()
    todo = [d for d in days if not (OUT / d / "corrections.csv").exists()]
    if todo:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            for ymd in todo:
                fetch_day(conn, ymd)
    report(days)


if __name__ == "__main__":
    main()
