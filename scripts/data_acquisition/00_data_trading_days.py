"""Download CRSP's daily market index file (crsp.dsi) from WRDS, unchanged. Its dates are the trading days.

    WRDS_USERNAME=<you> uv run python scripts/data_acquisition/00_data_trading_days.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push.

Output (gitignored):
    data/raw/crsp/dsi.csv   crsp.dsi as Postgres exports it, all columns and rows: one row per trading day with
                            CRSP's market returns. This is the legacy file format, updated through December 2024.
"""

import os
from pathlib import Path

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
DSI = ROOT / "data" / "raw" / "crsp" / "dsi.csv"


def main() -> None:
    if DSI.exists():
        print(f"{DSI.relative_to(ROOT)} already downloaded")
    else:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            print("connected; downloading crsp.dsi", flush=True)
            copy_query(conn, "SELECT * FROM crsp.dsi", DSI)
    with open(DSI) as f:
        header = f.readline().strip()
        n_rows = sum(1 for _ in f)
    print(f"{DSI.relative_to(ROOT)}: {n_rows:,} rows; columns: {header}")


if __name__ == "__main__":
    main()
