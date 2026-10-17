"""Download CRSP's corporate-event records (CIZ format) from WRDS, unchanged.

    WRDS_USERNAME=<you> uv run python scripts/data_acquisition/00_data_corporate_events.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push. Files already downloaded are skipped.

Outputs (gitignored), each a whole table as Postgres exports it, all columns and rows:
    data/raw/crsp/stkdistributions.csv   crsp.stkdistributions: dividends, splits, reverse splits, spin-offs
    data/raw/crsp/stkdelists.csv         crsp.stkdelists: delistings with their reason (merger, exchange, ...)
    data/raw/crsp/metaflaginfo.csv       crsp.metaflaginfo: what each code in the two tables above means
"""

import os
from pathlib import Path

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
CRSP = ROOT / "data" / "raw" / "crsp"
TABLES = ["stkdistributions", "stkdelists", "metaflaginfo"]


def main() -> None:
    todo = [t for t in TABLES if not (CRSP / f"{t}.csv").exists()]
    if todo:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            for table in todo:
                print(f"connected; downloading crsp.{table}", flush=True)
                copy_query(conn, f"SELECT * FROM crsp.{table}", CRSP / f"{table}.csv")
    for table in TABLES:
        path = CRSP / f"{table}.csv"
        with open(path) as f:
            header = f.readline().strip()
            n_rows = sum(1 for _ in f)
        print(f"{path.relative_to(ROOT)}: {n_rows:,} rows; columns: {header}")


if __name__ == "__main__":
    main()
