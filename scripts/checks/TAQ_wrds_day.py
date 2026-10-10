"""Run the pull's query on WRDS for one or more days: the golden test (2019-06-04) and the pilot days.

    WRDS_USERNAME=<you> uv run python scripts/checks/TAQ_wrds_day.py               # the golden test, 2019-06-04
    WRDS_USERNAME=<you> uv run python scripts/checks/TAQ_wrds_day.py 2014-01-02 2019-11-29 ...

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS sends one Duo push for the connection, used for every day given.

For each day: reads from information_schema whether that day's wct and ctm have nanoseconds and trade IDs, generates
the Postgres SQL (src/dsr/taq_sql.py, explanations/taq_pull.md), runs it on WRDS (temporary tables, then one COPY per
output table), and saves the four tables, the SQL, the time and size of every statement, and the conservation checks.
For 2019-06-04 it also compares every table row by row with the Python reference
(scripts/checks/TAQ_reference_day.py): the golden test passes only if nothing differs.

Outputs (gitignored), in data/raw/taq/checks/wrds_test/<YYYYMMDD>/:
    daily.csv, cells.csv, bars.csv, audit.csv   the tables as WRDS returns them
    query.sql                                   the statements sent, with the query version (a hash of them)
    timing.csv, checks.csv, comparison.csv      seconds and bytes per statement; checks; the golden comparison
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import pandas as pd

from dsr.data import wrds_connect
from dsr.taq import KEYS, check_day, compare_tables, read_pulled_csv, run_day
from dsr.taq_sql import Dialect, day_parameters, day_sql

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "taq" / "checks" / "wrds_test"
REFERENCE = ROOT / "data" / "raw" / "taq" / "checks" / "reference"


def has_column(conn, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_schema = 'taqmsec' "
                    "AND table_name = %s AND column_name = %s", (table, column))
        return cur.fetchone()[0] > 0


def read_only(conn) -> bool:
    """WRDS allows only read-only transactions, where not even a temporary table can be created."""
    with conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        return cur.fetchone()[0] == "on"


def run(conn, date: str) -> pd.DataFrame:
    ymd = date.replace("-", "")
    out = OUT / ymd
    out.mkdir(parents=True, exist_ok=True)
    day = day_parameters(date, has_nano=has_column(conn, f"wct_{ymd}", "time_m_nano"))
    temp_tables = not read_only(conn)
    conn.rollback()
    sql = day_sql(Dialect("postgres"), day, temp_tables=temp_tables)
    text = ";\n\n".join([*sql["setup"], *(sql[name] for name in KEYS)]) + ";\n"
    version = hashlib.sha1(text.encode()).hexdigest()[:12]
    (out / "query.sql").write_text(f"-- query version {version}\n-- {day}\n{text}")
    print(f"{date}: close {day.close_ns // 3_600_000_000_000}h, nanoseconds {day.has_nano}, "
          f"version {version}, {'temporary tables' if temp_tables else 'read-only: one statement per table'}", flush=True)

    start = time.perf_counter()
    try:
        timing = run_day(conn, sql, out)
    finally:
        conn.rollback()  # drops the day's temporary tables
    timing.to_csv(out / "timing.csv", index=False)
    print(timing.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"{date}: {time.perf_counter() - start:.0f} s on WRDS, {timing['bytes'].sum() / 1e6:.1f} MB of CSV", flush=True)

    tables = {name: read_pulled_csv(out / f"{name}.csv") for name in KEYS}
    checks = check_day(tables["daily"], tables["cells"], tables["bars"], tables["audit"],
                       (day.cutoff_ns - day.open_ns) // (600 * 10**9))
    checks.to_csv(out / "checks.csv", index=False)
    print("conservation checks pass" if checks["passed"].all() else checks[~checks["passed"]].to_string(index=False))
    for name in KEYS:
        print(f"  {name}: {len(tables[name]):,} rows")

    if (REFERENCE / ymd / "bars.parquet").exists():
        comparison = pd.concat([compare_tables(name, pd.read_parquet(REFERENCE / ymd / f"{name}.parquet"), tables[name])
                                .assign(table=name) for name in KEYS])
        comparison.to_csv(out / "comparison.csv", index=False)
        differing = comparison[comparison["rows_differing"] > 0]
        print("GOLDEN TEST PASSED: matches the reference in every table" if differing.empty
              else "GOLDEN TEST FAILED:\n" + differing.to_string(index=False))
    return timing.assign(day=date)


def main() -> None:
    dates = sys.argv[1:] or ["2019-06-04"]
    username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
    print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
    with wrds_connect(username) as conn:
        for date in dates:
            run(conn, date)


if __name__ == "__main__":
    main()
