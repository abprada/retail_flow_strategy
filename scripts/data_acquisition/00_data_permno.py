"""Download CRSP's ticker and name history (CIZ format, crsp.stocknames_v2) from WRDS, unchanged.

    WRDS_USERNAME=<you> uv run python scripts/data_acquisition/00_data_permno.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push.

Output (gitignored):
    data/raw/crsp/stocknames_v2.csv   crsp.stocknames_v2 as Postgres exports it, all columns and rows
                                      (matching to Robintrack happens later)
"""

import os
from pathlib import Path

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
STOCKNAMES = ROOT / "data" / "raw" / "crsp" / "stocknames_v2.csv"


def main() -> None:
    if STOCKNAMES.exists():
        print(f"{STOCKNAMES.relative_to(ROOT)} already downloaded")
    else:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            print("connected; downloading crsp.stocknames_v2", flush=True)
            copy_query(conn, "SELECT * FROM crsp.stocknames_v2", STOCKNAMES)
    with open(STOCKNAMES) as f:
        header = f.readline().strip()
        n_rows = sum(1 for _ in f)
    print(f"{STOCKNAMES.relative_to(ROOT)}: {n_rows:,} rows; columns: {header}")


if __name__ == "__main__":
    main()
