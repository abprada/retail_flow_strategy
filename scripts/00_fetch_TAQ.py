"""Download one full day of millisecond Daily TAQ (WRDS schema taqmsec): WRDS Consolidated Trades and the master file.

    WRDS_USERNAME=<you> uv run python scripts/00_fetch_TAQ.py [YYYY-MM-DD]

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push. Files already downloaded are skipped.

The date defaults to 2019-06-04, inside the Robintrack training period. Every row and column of the day is kept
(all hours). WCT (wct_YYYYMMDD) is a table WRDS builds: every trade with the NBBO prevailing just before it, which
would replace the separate trade (ctm) and NBBO (complete_nbbo) tables. The script first counts each table's rows
and asks before downloading. Values are kept as the text Postgres exports: each table is copied to CSV, converted to parquet with
every column as a string, and the CSV deleted.

Outputs (gitignored), in data/raw/taq/<date>/:
    tables.csv                every taqmsec table for that date
    <table>_columns.csv       column names and types of wct and mastm, as WRDS lists them
    row_counts.csv            rows per table
    wct.parquet               trades with their prevailing NBBO, all columns
    mastm.parquet             the master file, all columns
"""

import csv
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[1]
DATE = sys.argv[1] if len(sys.argv) > 1 else "2019-06-04"
OUT = ROOT / "data" / "raw" / "taq" / DATE
YMD = DATE.replace("-", "")

TABLES = ["wct", "mastm"]


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


def query_to_parquet(conn, query: str, columns: list[str], dest: Path) -> Path:
    """Copy a query's result to dest as parquet, every column as a string. Skipped if dest exists."""
    if dest.exists():
        return dest
    tmp_csv = dest.with_suffix(".csv")
    copy_query(conn, query, tmp_csv)
    part = dest.with_name(dest.name + ".part")
    reader = pacsv.open_csv(
        tmp_csv,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in columns}),
    )
    with pq.ParquetWriter(part, reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
    part.rename(dest)
    tmp_csv.unlink()
    return dest


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
        selected = {table: columns_of(conn, table) for table in TABLES}

        print("counting rows", flush=True)
        copy_query(
            conn,
            " UNION ALL ".join(f"SELECT '{t}' AS table_name, count(*) AS n_rows FROM taqmsec.{t}_{YMD}" for t in TABLES),
            OUT / "row_counts.csv",
        )
        with open(OUT / "row_counts.csv") as f:
            counts = {row["table_name"]: int(row["n_rows"]) for row in csv.DictReader(f)}
        for table, n in counts.items():
            print(f"  taqmsec.{table}_{YMD}: {n:,} rows, {len(selected[table])} columns")
        if input("download? [y/N] ").strip().lower() != "y":
            raise SystemExit("not downloaded")

        for table, columns in selected.items():
            print(f"downloading taqmsec.{table}_{YMD}", flush=True)
            query_to_parquet(conn, f"SELECT {', '.join(columns)} FROM taqmsec.{table}_{YMD}", columns,
                             OUT / f"{table}.parquet")

    for table in TABLES:
        path = OUT / f"{table}.parquet"
        print(f"{path.relative_to(ROOT)}: {pq.ParquetFile(path).metadata.num_rows:,} rows, "
              f"{path.stat().st_size / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()
