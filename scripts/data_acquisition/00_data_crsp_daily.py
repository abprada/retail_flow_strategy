"""Download CRSP's daily stock data (CIZ format, crsp.stkdlysecuritydata) from 2013 onwards from WRDS, unchanged.

    WRDS_USERNAME=<you> uv run python scripts/data_acquisition/00_data_crsp_daily.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push. Files already downloaded are skipped.

Every row dated 2013-01-01 or later is kept, through the latest day CRSP has (2025-12-31 in the annual release). The TAQ
sample's first decision day is 2014-01-02, so 2013 gives a year of history before it for trailing windows. Later steps
read only the dates they need: the Robintrack notebook keeps its own period.

Outputs (gitignored):
    data/raw/crsp/stkdlysecuritydata_columns.csv          the table's column names and types, as WRDS lists them
    data/raw/crsp/stkdlysecuritydata_from_2013.csv        every row dated 2013-01-01 or later, all columns
"""

import csv
import os
from pathlib import Path

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
CRSP = ROOT / "data" / "raw" / "crsp"
COLUMNS = CRSP / "stkdlysecuritydata_columns.csv"
DAILY = CRSP / "stkdlysecuritydata_from_2013.csv"
START = "2013-01-01"


def main() -> None:
    if not DAILY.exists():
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            copy_query(
                conn,
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'crsp' AND table_name = 'stkdlysecuritydata' ORDER BY ordinal_position",
                COLUMNS,
            )
            with open(COLUMNS) as f:
                names = [row["column_name"] for row in csv.DictReader(f)]
            if "dlycaldt" not in names:
                COLUMNS.unlink()
                raise SystemExit(f"crsp.stkdlysecuritydata has no dlycaldt column; its columns are: {names}")
            print(f"connected; downloading crsp.stkdlysecuritydata from {START} (several GB, takes a while)", flush=True)
            copy_query(conn, f"SELECT * FROM crsp.stkdlysecuritydata WHERE dlycaldt >= '{START}'", DAILY)
    with open(DAILY) as f:
        header = f.readline().strip()
    print(f"{DAILY.relative_to(ROOT)}: {DAILY.stat().st_size / 1e9:.1f} GB; columns: {header}")


if __name__ == "__main__":
    main()
