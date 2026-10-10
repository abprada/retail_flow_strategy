"""Run the pull's per-day SQL locally with DuckDB on the downloaded 2019-06-04 files, and compare it with the reference.

    uv run --with duckdb python scripts/checks/TAQ_query_local.py [--cte]

With --cte, the query is the read-only version WRDS needs (no temporary tables; every output table in one statement).

Generates the DuckDB dialect of the query (src/dsr/taq_sql.py) for 2019-06-04, runs it on data/raw/taq/2019-06-04/
(wct.parquet, ctm_corrections.parquet), writes the four tables, runs the conservation checks, and compares every table
row by row with the Python reference (scripts/checks/TAQ_reference_day.py), if that has been built.

Outputs (gitignored), in data/intermediate/taq/duckdb/20190604/:
    daily.parquet, cells.parquet, bars.parquet, audit.parquet, checks.csv, comparison.csv
"""

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

from dsr.taq import KEYS, check_day, compare_tables
from dsr.taq_sql import Dialect, day_parameters, day_sql

ROOT = Path(__file__).resolve().parents[2]
DATE = "2019-06-04"
RAW = ROOT / "data" / "raw" / "taq" / DATE
REFERENCE = ROOT / "data" / "intermediate" / "taq" / "reference" / DATE.replace("-", "")
OUT = ROOT / "data" / "intermediate" / "taq" / "duckdb" / DATE.replace("-", "")
NS = 10**9


def main() -> None:
    t0 = time.time()
    day = day_parameters(DATE, has_nano=True)
    sql = day_sql(Dialect("duckdb", local_dir=str(RAW)), day, temp_tables="--cte" not in sys.argv)
    con = duckdb.connect()
    if "--limited" in sys.argv:  # the WRDS Cloud job's limits
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '16GB'")
    for statement in sql["setup"]:
        con.execute(statement)
    print(f"setup done ({time.time() - t0:.0f} s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in KEYS:
        start = time.time()
        tables[name] = con.execute(sql[name]).df()
        tables[name].to_parquet(OUT / f"{name}.parquet", compression="zstd", index=False)
        print(f"{name}: {len(tables[name]):,} rows in {time.time() - start:.1f} s", flush=True)

    regular_bars = (day.cutoff_ns - day.open_ns) // (600 * NS)
    checks = check_day(tables["daily"], tables["cells"], tables["bars"], tables["audit"], regular_bars)
    checks.to_csv(OUT / "checks.csv", index=False)
    print(checks.to_string(index=False))

    if (REFERENCE / "bars.parquet").exists():
        comparison = pd.concat([compare_tables(name, pd.read_parquet(REFERENCE / f"{name}.parquet"), tables[name])
                                .assign(table=name) for name in KEYS])
        comparison.to_csv(OUT / "comparison.csv", index=False)
        differing = comparison[comparison["rows_differing"] > 0]
        print("matches the reference in every table" if differing.empty else differing.to_string(index=False))
    else:
        print(f"no reference yet in {REFERENCE.relative_to(ROOT)}")
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
