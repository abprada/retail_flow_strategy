"""Run the TAQ pull on WRDS Cloud for a list of days (explanations/taq_pull.md).

On WRDS Cloud, in a folder holding this script, days.csv and the package folder dsr/ (taq.py, taq_sql.py):

    python3 taq_cloud_day.py --work <scratch>/work --out <scratch>/out 2019-06-04 [more days]
    python3 taq_cloud_day.py --work <scratch>/work --out <scratch>/out --from 2014-01-02 --to 2014-12-31

For each day: pulls the day's raw rows from the WRDS database (the 11 columns of `wct` the query uses, and the rows of
`ctm` with tr_corr other than 00) as CSV into the work folder; computes the four tables with DuckDB, running the query
validated against the reference (src/dsr/taq_sql.py, DuckDB dialect, with temporary tables, which DuckDB allows);
writes them as zstd parquet into <out>/<YYYYMMDD>/ (built under a temporary name, renamed when complete); runs the
conservation checks; appends a line to <out>/manifest.csv; and deletes the day's raw rows. A day already in the
manifest with the same code version and passing checks is skipped, so an interrupted run resumes where it stopped.

The database connection uses psycopg2 (installed on WRDS Cloud) or psycopg; the password comes from ~/.pgpass.
"""

import argparse
import csv
import getpass
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

# WRDS Cloud allows 100 processes and threads per user; numpy's math library would start one thread per core (64)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))  # when run from the repository

import duckdb
import pandas as pd

from dsr.taq import KEYS, check_day
from dsr.taq_sql import Day, Dialect, day_sql

NS = 10**9
WCT_COLUMNS = ["sym_root", "sym_suffix", "time_m", "time_m_nano", "ex", "tr_scond", "price", "size", "tr_seqnum",
               "nbb", "nbo"]
CTM_COLUMNS = ["sym_root", "sym_suffix", "time_m", "time_m_nano", "ex", "tr_scond", "price", "size", "tr_seqnum",
               "tr_corr", "tr_id"]
MANIFEST_FIELDS = ["day", "version", "close", "has_nano", "has_tr_id", "wct_rows_mb", "ctm_rows_mb", "pull_seconds",
                   "compute_seconds", "daily_rows", "cells_rows", "bars_rows", "audit_rows", "out_mb", "checks_passed"]


def code_version() -> str:
    """A hash of the code that defines the output: the query template, the checks and this script."""
    h = hashlib.sha1()
    for path in [HERE / "dsr" / "taq_sql.py", HERE / "dsr" / "taq.py", Path(__file__)]:
        if not path.exists():  # run from the repository
            path = HERE.parents[1] / "src" / "dsr" / path.name if path.name != Path(__file__).name else path
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def connect(dsn: str | None, user: str):
    """psycopg2 if installed (WRDS Cloud), else psycopg; returns (connection, copy function)."""
    kwargs = {"host": "wrds-pgdata.wharton.upenn.edu", "port": 9737, "dbname": "wrds", "user": user, "sslmode": "require"}
    if not (Path.home() / ".pgpass").exists() and not dsn and "PGPASSWORD" not in os.environ:
        kwargs["password"] = getpass.getpass(f"WRDS password for {user}: ")
    try:
        import psycopg2

        conn = psycopg2.connect(dsn) if dsn else psycopg2.connect(**kwargs)

        def copy(sql: str, path: Path) -> None:
            with conn.cursor() as cur, open(path, "wb") as f:
                cur.copy_expert(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER)", f)
    except ImportError:
        import psycopg

        conn = psycopg.connect(dsn) if dsn else psycopg.connect(**kwargs)

        def copy(sql: str, path: Path) -> None:
            with conn.cursor() as cur, cur.copy(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER)") as cp, \
                    open(path, "wb") as f:
                f.writelines(cp)
    return conn, copy


def columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'taqmsec' "
                    "AND table_name = %s", (table,))
        return [row[0] for row in cur.fetchall()]


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def run_day(conn, copy, date: str, close: str, work: Path, out: Path, version: str, threads: int, memory: str) -> dict:
    ymd = date.replace("-", "")
    wct_cols, ctm_cols = columns(conn, f"wct_{ymd}"), columns(conn, f"ctm_{ymd}")
    if not wct_cols or not ctm_cols:
        raise RuntimeError(f"taqmsec.wct_{ymd} or ctm_{ymd} not found")
    conn.rollback()
    raw = work / ymd
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)

    start = time.perf_counter()
    print(f"{date}: pulling the day's trades from the database", flush=True)
    copy(f"SELECT {', '.join(c for c in WCT_COLUMNS if c in wct_cols)} FROM taqmsec.wct_{ymd}", raw / "wct.csv")
    copy(f"SELECT {', '.join(c for c in CTM_COLUMNS if c in ctm_cols)} FROM taqmsec.ctm_{ymd} WHERE tr_corr <> '00'",
         raw / "ctm.csv")
    conn.rollback()
    pull_seconds = time.perf_counter() - start
    print(f"{date}: pulled {(raw / 'wct.csv').stat().st_size / 1e6:,.0f} MB in {pull_seconds:.0f} s; computing", flush=True)

    hours, minutes = map(int, close.split(":"))
    close_ns = (hours * 3600 + minutes * 60) * NS
    day = Day(ymd=ymd, open_ns=(9 * 3600 + 30 * 60) * NS, cutoff_ns=close_ns - 600 * NS, close_ns=close_ns,
              has_nano="time_m_nano" in wct_cols, has_tr_id="tr_id" in ctm_cols)
    sql = day_sql(Dialect("duckdb", local_dir=str(raw), local_format="csv"), day, temp_tables=True)

    part = out / f"{ymd}.part"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    start = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{work / 'duckdb_tmp'}'")
    con.execute(f"SET threads = {threads}")
    con.execute(f"SET memory_limit = '{memory}'")
    for statement in sql["setup"]:
        con.execute(statement)
    for name in KEYS:
        step = time.perf_counter()
        con.execute(f"COPY ({sql[name]}) TO '{part / name}.parquet' (FORMAT parquet, COMPRESSION zstd)")
        print(f"{date}:   {name} in {time.perf_counter() - step:.0f} s", flush=True)
    con.close()
    compute_seconds = time.perf_counter() - start

    tables = {name: pd.read_parquet(part / f"{name}.parquet") for name in KEYS}
    checks = check_day(tables["daily"], tables["cells"], tables["bars"], tables["audit"],
                       (day.cutoff_ns - day.open_ns) // (600 * NS))
    checks.to_csv(part / "checks.csv", index=False)
    final = out / ymd
    shutil.rmtree(final, ignore_errors=True)
    part.rename(final)
    row = {
        "day": date, "version": version, "close": close, "has_nano": day.has_nano, "has_tr_id": day.has_tr_id,
        "wct_rows_mb": round((raw / "wct.csv").stat().st_size / 1e6, 1),
        "ctm_rows_mb": round((raw / "ctm.csv").stat().st_size / 1e6, 2),
        "pull_seconds": round(pull_seconds, 1), "compute_seconds": round(compute_seconds, 1),
        **{f"{name}_rows": len(tables[name]) for name in KEYS},
        "out_mb": round(sum(p.stat().st_size for p in final.glob("*.parquet")) / 1e6, 1),
        "checks_passed": bool(checks["passed"].all()),
    }
    shutil.rmtree(raw)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("days", nargs="*", help="YYYY-MM-DD")
    parser.add_argument("--from", dest="start")
    parser.add_argument("--to", dest="end")
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="16GB")
    parser.add_argument("--user", default=os.environ.get("WRDS_USERNAME") or os.environ.get("USER"))
    parser.add_argument("--dsn", help="a connection string instead of WRDS (for a local rehearsal)")
    args = parser.parse_args()

    with open(HERE / "days.csv") as f:
        closes = {r["date"]: r["close"] for r in csv.DictReader(f)}
    days = list(args.days) + [d for d in closes if args.start and args.end and args.start <= d <= args.end]
    unknown = [d for d in days if d not in closes]
    if unknown:
        raise SystemExit(f"not NYSE trading days in days.csv: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    version = code_version()
    done = {r["day"] for r in read_manifest(manifest) if r["version"] == version and r["checks_passed"] == "True"}
    todo = [d for d in days if d not in done]
    print(f"code version {version}: {len(days)} days asked, {len(days) - len(todo)} already done", flush=True)
    if not todo:
        return
    conn, copy = connect(args.dsn, args.user)
    for date in todo:
        start = time.perf_counter()
        try:
            row = run_day(conn, copy, date, closes[date], args.work, args.out, version, args.threads, args.memory)
        except Exception as e:  # noqa: BLE001 (one bad day must not stop the run; it is retried next time)
            conn.rollback()
            print(f"{date}: FAILED: {e!r}", flush=True)
            continue
        new = not manifest.exists()
        with open(manifest, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"{date}: pull {row['pull_seconds']:.0f} s ({row['wct_rows_mb']:.0f} MB), compute {row['compute_seconds']:.0f} s, "
              f"{row['out_mb']:.1f} MB out, checks {'pass' if row['checks_passed'] else 'FAIL'} "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
