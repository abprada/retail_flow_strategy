"""Rehearse the golden test on a local Postgres: run the pull's Postgres SQL on 2019-06-04 and compare with the reference.

    uv run --with pgserver python scripts/taq/checks/TAQ_query_postgres_local.py [--read-only]

With --read-only, the session is set to read-only transactions, as on WRDS, and the query is the version without
temporary tables (the one WRDS runs).

Starts a local Postgres server (the `pgserver` package), loads the downloaded 2019-06-04 files into
`taqmsec.wct_20190604` and `taqmsec.ctm_20190604` with the column types WRDS lists for them (only the flagged `ctm`
rows exist locally, which is all the query reads), then runs exactly the statements the WRDS run will send
(src/dsr/taq_sql.py, Postgres dialect) through the same code path (dsr.taq.run_day, COPY to CSV), runs the
conservation checks and compares every table with the Python reference (scripts/taq/checks/TAQ_reference_day.py).
Catches Postgres-only errors (types, functions, syntax) without a WRDS connection.

Outputs (gitignored): data/raw/taq/checks/postgres_local/20190604/ (the four tables as CSV, timing, checks,
comparison); the server's data directory in data/raw/taq/checks/postgres_local/pgdata/ (loaded once, reused).
"""

import csv
import sys
import time
from pathlib import Path

import pandas as pd
import pgserver
import psycopg
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from dsr.taq import KEYS, check_day, compare_tables, read_pulled_csv, run_day
from dsr.taq_sql import Dialect, day_parameters, day_sql

ROOT = Path(__file__).resolve().parents[3]
DATE = "2019-06-04"
YMD = DATE.replace("-", "")
RAW = ROOT / "data" / "raw" / "taq" / DATE
BASE = ROOT / "data" / "raw" / "taq" / "checks" / "postgres_local"
OUT = BASE / YMD
REFERENCE = ROOT / "data" / "raw" / "taq" / "checks" / "reference" / YMD


def load(conn, table: str, parquet: Path, columns_csv: Path) -> None:
    """Create taqmsec.<table> with WRDS's column types and copy the parquet file's text values into it."""
    with open(columns_csv) as f:
        cols = [(r["column_name"], r["data_type"]) for r in csv.DictReader(f)]
    data = pq.read_table(parquet)
    cols = [(c, t) for c, t in cols if c in data.column_names]
    tmp = BASE / f"{table}.csv"
    # empty text becomes an unquoted empty field, which COPY reads as NULL
    data = data.select([c for c, _ in cols])
    data = pa.table({c: pa.compute.if_else(pa.compute.equal(data[c], ""), None, data[c]) for c, _ in cols})
    pacsv.write_csv(data, tmp, pacsv.WriteOptions(include_header=False, quoting_style="none"))
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS taqmsec.{table}")
        cur.execute(f"CREATE TABLE taqmsec.{table} ({', '.join(f'{c} {t}' for c, t in cols)})")
        with open(tmp, "rb") as f, cur.copy(f"COPY taqmsec.{table} FROM STDIN WITH (FORMAT csv)") as copy:
            while chunk := f.read(1 << 24):
                copy.write(chunk)
        cur.execute(f"ANALYZE taqmsec.{table}")
    conn.commit()
    tmp.unlink()


def main() -> None:
    t0 = time.time()
    BASE.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(BASE / "pgdata", cleanup_mode=None)
    with psycopg.connect(server.get_uri()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS taqmsec")
            cur.execute(f"SELECT to_regclass('taqmsec.wct_{YMD}') IS NOT NULL")
            loaded = cur.fetchone()[0]
        conn.commit()
        if not loaded:
            print("loading wct and ctm into the local Postgres (a few minutes, once)", flush=True)
            load(conn, f"ctm_{YMD}", RAW / "ctm_corrections.parquet", RAW / "ctm_columns.csv")
            load(conn, f"wct_{YMD}", RAW / "wct.parquet", RAW / "wct_columns.csv")
            print(f"loaded ({time.time() - t0:.0f} s)", flush=True)

        day = day_parameters(DATE, has_nano=True)
        read_only = "--read-only" in sys.argv
        if read_only:
            with conn.cursor() as cur:
                cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            conn.commit()
        sql = day_sql(Dialect("postgres"), day, temp_tables=not read_only)
        timing = run_day(conn, sql, OUT)
    timing.to_csv(OUT / "timing.csv", index=False)
    print(timing.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    tables = {name: read_pulled_csv(OUT / f"{name}.csv") for name in KEYS}
    checks = check_day(tables["daily"], tables["cells"], tables["bars"], tables["audit"],
                       (day.cutoff_ns - day.open_ns) // (600 * 10**9))
    checks.to_csv(OUT / "checks.csv", index=False)
    print(checks[~checks["passed"]].to_string(index=False) if not checks["passed"].all() else "conservation checks pass")
    comparison = pd.concat([compare_tables(name, pd.read_parquet(REFERENCE / f"{name}.parquet"), tables[name])
                            .assign(table=name) for name in KEYS])
    comparison.to_csv(OUT / "comparison.csv", index=False)
    differing = comparison[comparison["rows_differing"] > 0]
    print("matches the reference in every table" if differing.empty else differing.to_string(index=False))
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
