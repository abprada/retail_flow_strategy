"""Download the TAQ master file (taqmsec.mastm_YYYYMMDD) of every day of the TAQ pull from WRDS, unchanged.

    WRDS_USERNAME=<you> uv run python scripts/data_acquisition/00_data_taq_master.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push. Days already downloaded are skipped, so a
stopped run (a dropped connection, for example) resumes where it stopped.

The days are those of the TAQ pull (scripts/data_acquisition/wrds_cloud/days.csv: every NYSE session from 2013-12-31 to 2025-12-31).
A day's master file lists the securities of that day: symbol, CUSIP, security type, listing exchange, round lot,
shares outstanding and Tick Size Pilot group. It links the pull's symbols to CRSP PERMNOs (by CUSIP and date) locally.
Its layout changes over the years (2014 and 2015 have listed_market instead of listed_exchange and no
tick_pilot_indicator; see scripts/taq/checks/TAQ_master.py), so every column is kept and each day keeps its own layout.

Output (gitignored):
    data/raw/taq/master/mastm_<YYYYMMDD>.parquet   every row and column of the day's master file, as text
"""

import csv
import os
import time
from pathlib import Path

import pyarrow.parquet as pq

from dsr.data import query_to_parquet, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
DAYS = ROOT / "scripts" / "wrds_cloud" / "days.csv"
OUT = ROOT / "data" / "raw" / "taq" / "master"


def main() -> None:
    with open(DAYS) as f:
        days = [row["date"].replace("-", "") for row in csv.DictReader(f)]
    todo = [d for d in days if not (OUT / f"mastm_{d}.parquet").exists()]
    print(f"{len(days) - len(todo):,} of {len(days):,} days already here", flush=True)
    if todo:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = 'taqmsec' AND table_name ~ '^mastm_[0-9]{8}$'")
                on_wrds = {row[0] for row in cur.fetchall()}
            missing = [d for d in todo if f"mastm_{d}" not in on_wrds]
            if missing:
                raise SystemExit(f"no master file on WRDS for {len(missing)} days: {missing[:10]}")
            print(f"connected; downloading {len(todo):,} master files", flush=True)
            start = time.perf_counter()
            for i, ymd in enumerate(todo, start=1):
                query_to_parquet(conn, f"SELECT * FROM taqmsec.mastm_{ymd}", None, OUT / f"mastm_{ymd}.parquet")
                if i % 100 == 0 or i == len(todo):
                    left = (time.perf_counter() - start) / i * (len(todo) - i)
                    print(f"{i:,} of {len(todo):,} days (last {ymd}); about {left / 60:.0f} minutes left", flush=True)

    files = sorted(OUT.glob("mastm_*.parquet"))
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    size = sum(p.stat().st_size for p in files) / 1e9
    print(f"{OUT.relative_to(ROOT)}: {len(files):,} of {len(days):,} days, {rows:,} rows, {size:.2f} GB")


if __name__ == "__main__":
    main()
