"""Test run: download one minute of millisecond Daily TAQ (WRDS schema taqmsec) for one day, unchanged.

    WRDS_USERNAME=<you> uv run python scripts/00_fetch_TAQ.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push. Files already downloaded are skipped.

Daily TAQ keeps one table per day and file type, e.g. taqmsec.ctm_20190604. Times (`time_m`) are New York time.
The minute is the one just before the 15:50 market-on-close cutoff, on a day inside the Robintrack training period.

Outputs (gitignored), in data/raw/taq/<date>/:
    tables.csv                    every taqmsec table for that date
    <table>_columns.csv           column names and types of ctm, complete_nbbo and mastm, as WRDS lists them
    ctm_<hhmm>.csv                trades (consolidated trade file) with time_m in the minute, all columns
    complete_nbbo_<hhmm>.csv      NBBO changes with time_m in the minute, all columns
    mastm.csv                     the day's master file: one row per symbol, with its CUSIP (for matching to PERMNOs)
"""

import csv
import os
from pathlib import Path

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[1]
DATE = "2019-06-04"
MINUTE_START, MINUTE_END = "15:49:00", "15:50:00"  # start included, end excluded
OUT = ROOT / "data" / "raw" / "taq" / DATE
YMD = DATE.replace("-", "")
HHMM = MINUTE_START[:5].replace(":", "")


def columns_of(conn, table: str) -> list[str]:
    path = OUT / f"{table}_columns.csv"
    copy_query(
        conn,
        "SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema = 'taqmsec' AND table_name = '{table}_{YMD}' ORDER BY ordinal_position",
        path,
    )
    with open(path) as f:
        names = [row["column_name"] for row in csv.DictReader(f)]
    if not names:
        path.unlink()
        raise SystemExit(f"taqmsec.{table}_{YMD} not found; see {(OUT / 'tables.csv').relative_to(ROOT)}")
    return names


def main() -> None:
    username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
    print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
    with wrds_connect(username) as conn:
        copy_query(
            conn,
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = 'taqmsec' AND table_name LIKE '%\\_{YMD}' ORDER BY table_name",
            OUT / "tables.csv",
        )
        for table in ["ctm", "complete_nbbo"]:
            names = columns_of(conn, table)
            if "time_m" not in names:
                raise SystemExit(f"taqmsec.{table}_{YMD} has no time_m column; its columns are: {names}")
            print(f"downloading taqmsec.{table}_{YMD}, {MINUTE_START} to {MINUTE_END}", flush=True)
            copy_query(
                conn,
                f"SELECT * FROM taqmsec.{table}_{YMD} "
                f"WHERE time_m >= '{MINUTE_START}' AND time_m < '{MINUTE_END}'",
                OUT / f"{table}_{HHMM}.csv",
            )
        columns_of(conn, "mastm")
        print(f"downloading taqmsec.mastm_{YMD}", flush=True)
        copy_query(conn, f"SELECT * FROM taqmsec.mastm_{YMD}", OUT / "mastm.csv")
    for path in sorted(OUT.glob("*.csv")):
        with open(path) as f:
            n = sum(1 for _ in f) - 1
        print(f"{path.relative_to(ROOT)}: {n:,} rows, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
