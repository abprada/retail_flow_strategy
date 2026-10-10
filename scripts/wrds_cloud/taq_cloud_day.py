"""Run the TAQ pull on WRDS Cloud for a list of days (explanations/taq_pull.md, explanations/TAQ/wrds_cloud.md).

On WRDS Cloud, in a folder holding this script, days.csv and the package folder dsr/ (taq.py, taq_sql.py):

    python3 taq_cloud_day.py --work <scratch>/work --out <scratch>/out 2019-06-04 [more days]
    python3 taq_cloud_day.py --work <scratch>/work --out <scratch>/out --days-file days_1.txt

Each day goes through two stages, each in its own process:
  pull:    copies the day's raw rows out of the WRDS database (the 11 columns of `wct` the query uses, and the rows of
           `ctm` with tr_corr other than 00) as CSV into a raw folder (written as <day>.pulling, renamed when complete);
  compute: computes the four tables with DuckDB (the query validated against the reference: src/dsr/taq_sql.py,
           DuckDB dialect), writes them as zstd parquet into <out>/<YYYYMMDD>/ (built as .part, renamed when
           complete), runs the conservation checks, appends a line to <out>/manifest_<job>.csv, and deletes the raw
           rows.
The two stages overlap: while one day computes, the next day is pulled. They use different resources (the database
and the network against the CPU), so a job takes about the time of its slower stage instead of their sum.

A day already in a manifest with the same code version and passing checks is skipped, so an interrupted job resumes
where it stopped. A day that fails (or whose process is killed, for example for memory) is reported and retried on
the next run; the job goes on with the next day. Raw rows and DuckDB's spill files go to the node's local disk
($TMPDIR) when it has room, else to the work folder on scratch.

The database connection uses psycopg2 (installed on WRDS Cloud) or psycopg; the password comes from ~/.pgpass.
"""

import argparse
import csv
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# WRDS Cloud allows 100 processes and threads per user; numpy's math library would start one thread per core (64)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))  # when run from the repository

NS = 10**9
WCT_COLUMNS = ["sym_root", "sym_suffix", "time_m", "time_m_nano", "ex", "tr_scond", "price", "size", "tr_seqnum",
               "nbb", "nbo"]
CTM_COLUMNS = ["sym_root", "sym_suffix", "time_m", "time_m_nano", "ex", "tr_scond", "price", "size", "tr_seqnum",
               "tr_corr"]
MANIFEST_FIELDS = ["day", "version", "job", "close", "has_nano", "has_tr_id", "wct_rows_mb", "ctm_rows_mb",
                   "pull_seconds", "compute_seconds", "daily_rows", "cells_rows", "bars_rows", "audit_rows", "out_mb",
                   "checks_passed"]
LOCAL_RAW_GB, LOCAL_SPILL_GB = 60, 30  # free space needed on the node's local disk to use it

# Bumped by hand when this script changes what a day's output contains (the columns pulled, the parameters); the
# query and the checks are hashed directly. Changes that only affect how the job runs leave finished days valid.
OUTPUT_VERSION = "1"


def code_version() -> str:
    """A hash of what defines a day's output: the query template, the checks and OUTPUT_VERSION."""
    h = hashlib.sha1(OUTPUT_VERSION.encode())
    for name in ("taq_sql.py", "taq.py"):
        path = HERE / "dsr" / name
        if not path.exists():  # run from the repository
            path = HERE.parents[1] / "src" / "dsr" / name
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


def job_memory_limit() -> int | None:
    """The memory this process may use, in bytes, from its cgroup (where SGE enforces a job's memory); None if unknown.

    Reads the cgroup path from /proc/self/cgroup (v1: the `memory` line; v2: the `0::` line) and takes the smallest
    limit along that path and its parents."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    candidates = []
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2
            base, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):  # cgroup v1
            base, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        parts = [p for p in path.split("/") if p]
        for depth in range(len(parts), -1, -1):
            try:
                value = (base.joinpath(*parts[:depth]) / name).read_text().strip()
            except OSError:
                continue
            if value.isdigit() and int(value) < 1 << 50:  # "max" or a huge number means no limit
                candidates.append(int(value))
    return min(candidates) if candidates else None


def duckdb_memory(setting: str) -> str:
    """DuckDB's memory cap: half the job's limit (the rest for Python, reading the CSV, the next day's pull and the
    checks), so that DuckDB spills to disk before the job is killed; 8GB if the limit cannot be read; or the value."""
    if setting != "auto":
        return setting
    limit = job_memory_limit()
    return f"{max(2, int(limit * 0.5 / 2**30))}GB" if limit else "8GB"


def local_disk(min_free_gb: int) -> Path | None:
    """The node's local disk for this job ($TMPDIR under SGE), if it has min_free_gb free."""
    tmp = os.environ.get("TMPDIR")
    if tmp and Path(tmp).is_dir() and shutil.disk_usage(tmp).free >= min_free_gb * 2**30:
        return Path(tmp)
    return None


def finished_days(out: Path, version: str) -> set[str]:
    done = set()
    for path in out.glob("manifest*.csv"):
        with open(path) as f:
            done |= {r["day"] for r in csv.DictReader(f) if r["version"] == version and r["checks_passed"] == "True"}
    return done


def pull(args, date: str) -> None:
    """Stage 1: the day's raw rows as CSV in <raw>/<YYYYMMDD>/, with meta.json."""
    ymd = date.replace("-", "")
    final, part = args.raw / ymd, args.raw / f"{ymd}.pulling"
    shutil.rmtree(final, ignore_errors=True)
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    conn, copy = connect(args.dsn, args.user)
    try:
        wct_cols, ctm_cols = columns(conn, f"wct_{ymd}"), columns(conn, f"ctm_{ymd}")
        if not wct_cols or not ctm_cols:
            raise RuntimeError(f"taqmsec.wct_{ymd} or ctm_{ymd} not found")
        conn.rollback()
        start = time.perf_counter()
        copy(f"SELECT {', '.join(c for c in WCT_COLUMNS if c in wct_cols)} FROM taqmsec.wct_{ymd}", part / "wct.csv")
        copy(f"SELECT {', '.join(c for c in CTM_COLUMNS if c in ctm_cols)} FROM taqmsec.ctm_{ymd} WHERE tr_corr <> '00'",
             part / "ctm.csv")
        seconds = time.perf_counter() - start
    finally:
        conn.close()
    meta = {"has_nano": "time_m_nano" in wct_cols, "has_tr_id": "tr_id" in ctm_cols, "pull_seconds": round(seconds, 1),
            "wct_rows_mb": round((part / "wct.csv").stat().st_size / 1e6, 1),
            "ctm_rows_mb": round((part / "ctm.csv").stat().st_size / 1e6, 2)}
    (part / "meta.json").write_text(json.dumps(meta))
    part.rename(final)
    print(f"{date}: pulled {meta['wct_rows_mb']:,.0f} MB in {seconds:.0f} s", flush=True)


def compute(args, date: str, close: str, version: str) -> bool:
    """Stage 2: the four tables from the day's raw rows; returns whether the checks pass."""
    import duckdb
    import pandas as pd

    from dsr.taq import KEYS, check_day
    from dsr.taq_sql import Day, Dialect, day_sql

    ymd = date.replace("-", "")
    raw = args.raw / ymd
    meta = json.loads((raw / "meta.json").read_text())
    hours, minutes = map(int, close.split(":"))
    close_ns = (hours * 3600 + minutes * 60) * NS
    day = Day(ymd=ymd, open_ns=(9 * 3600 + 30 * 60) * NS, cutoff_ns=close_ns - 600 * NS, close_ns=close_ns,
              has_nano=meta["has_nano"])
    sql = day_sql(Dialect("duckdb", local_dir=str(raw), local_format="csv"), day, temp_tables=True)

    part = args.out / f"{ymd}.part"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    spill = args.spill / f"duckdb_{ymd}"
    start = time.perf_counter()
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{spill}'")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{duckdb_memory(args.memory)}'")
    con.execute("SET preserve_insertion_order = false")  # lets DuckDB write large results without holding them
    for statement in sql["setup"]:
        con.execute(statement)
    for name in KEYS:
        step = time.perf_counter()
        con.execute(f"COPY ({sql[name]}) TO '{part / name}.parquet' (FORMAT parquet, COMPRESSION zstd)")
        print(f"{date}:   {name} in {time.perf_counter() - step:.0f} s", flush=True)
    con.close()
    shutil.rmtree(spill, ignore_errors=True)
    compute_seconds = time.perf_counter() - start

    tables = {name: pd.read_parquet(part / f"{name}.parquet") for name in KEYS}
    checks = check_day(tables["daily"], tables["cells"], tables["bars"], tables["audit"],
                       (day.cutoff_ns - day.open_ns) // (600 * NS))
    checks.to_csv(part / "checks.csv", index=False)
    final = args.out / ymd
    shutil.rmtree(final, ignore_errors=True)
    part.rename(final)
    passed = bool(checks["passed"].all())
    row = {"day": date, "version": version, "job": args.job, "close": close, "has_nano": day.has_nano,
           "has_tr_id": meta["has_tr_id"], "wct_rows_mb": meta["wct_rows_mb"], "ctm_rows_mb": meta["ctm_rows_mb"],
           "pull_seconds": meta["pull_seconds"], "compute_seconds": round(compute_seconds, 1),
           **{f"{name}_rows": len(tables[name]) for name in KEYS},
           "out_mb": round(sum(p.stat().st_size for p in final.glob("*.parquet")) / 1e6, 1), "checks_passed": passed}
    manifest = args.out / f"manifest_{args.job}.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    shutil.rmtree(raw)
    print(f"{date}: pull {meta['pull_seconds']:.0f} s ({meta['wct_rows_mb']:.0f} MB), compute {compute_seconds:.0f} s, "
          f"{row['out_mb']:.1f} MB out, checks {'pass' if passed else 'FAIL'}", flush=True)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("days", nargs="*", help="YYYY-MM-DD")
    parser.add_argument("--from", dest="start")
    parser.add_argument("--to", dest="end")
    parser.add_argument("--days-file", type=Path, help="one YYYY-MM-DD per line, run in that order")
    parser.add_argument("--work", required=True, type=Path, help="a folder on scratch for raw rows and spill files")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory", default="auto", help="DuckDB's memory cap, or auto: half the job's limit")
    parser.add_argument("--user", default=os.environ.get("WRDS_USERNAME") or os.environ.get("USER"))
    parser.add_argument("--dsn", help="a connection string instead of WRDS (for a local rehearsal)")
    parser.add_argument("--stage", choices=["pull", "compute"], help=argparse.SUPPRESS)  # a child process
    parser.add_argument("--raw", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--spill", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--job", default=os.environ.get("JOB_ID", "local"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    with open(HERE / "days.csv") as f:
        closes = {r["date"]: r["close"] for r in csv.DictReader(f)}
    days = list(args.days)
    if args.days_file:
        days += [line.strip() for line in args.days_file.read_text().splitlines() if line.strip()]
    days += [d for d in closes if args.start and args.end and args.start <= d <= args.end]
    unknown = [d for d in days if d not in closes]
    if unknown:
        raise SystemExit(f"not NYSE trading days in days.csv: {unknown}")
    version = code_version()

    if args.stage == "pull":
        pull(args, days[0])
        return
    if args.stage == "compute":
        sys.exit(0 if compute(args, days[0], closes[days[0]], version) else 1)

    # the parent: pulls day i+1 while day i computes, each stage in its own process
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.work / f"job_{args.job}"
    raw = (local_disk(LOCAL_RAW_GB) or work) / "raw"
    spill = (local_disk(LOCAL_SPILL_GB) or work) / "spill"
    raw.mkdir(parents=True, exist_ok=True)
    spill.mkdir(parents=True, exist_ok=True)
    done = finished_days(args.out, version)
    todo = [d for d in dict.fromkeys(days) if d not in done]
    limit = job_memory_limit()
    print(f"code version {version}, job {args.job}: {len(days)} days asked, {len(days) - len(todo)} already done", flush=True)
    print(f"job memory limit {limit / 2**30:.1f} GB" if limit else "job memory limit unknown",
          f"; DuckDB capped at {duckdb_memory(args.memory)}, {args.threads} threads; raw rows in {raw}; spill in {spill}",
          flush=True)
    if not todo:
        return
    child = [sys.executable, str(Path(__file__).resolve()), "--work", str(args.work), "--out", str(args.out),
             "--threads", str(args.threads), "--memory", args.memory, "--user", args.user or "", "--raw", str(raw),
             "--spill", str(spill), "--job", args.job] + (["--dsn", args.dsn] if args.dsn else [])

    def start(stage: str, date: str) -> subprocess.Popen:
        return subprocess.Popen([*child, "--stage", stage, date])

    def report(date: str, stage: str, code: int) -> None:
        reason = "killed, probably out of memory" if code in (-9, 137) else f"exit code {code}"
        print(f"{date}: FAILED in {stage} ({reason}); it is retried on the next run", flush=True)
        failed.append(date)

    failed: list[str] = []
    started = time.perf_counter()
    pulling = start("pull", todo[0])
    for i, date in enumerate(todo):
        pulled = pulling.wait() == 0
        if not pulled:
            report(date, "pull", pulling.returncode)
        computing = start("compute", date) if pulled else None
        pulling = start("pull", todo[i + 1]) if i + 1 < len(todo) else None
        if computing is not None and computing.wait() != 0:
            report(date, "compute", computing.returncode)
        if pulling is None:
            break
    if pulling is not None:
        pulling.wait()
    shutil.rmtree(work, ignore_errors=True)
    print(f"done: {len(todo) - len(failed)} days written, {len(failed)} failed{': ' + ' '.join(failed) if failed else ''} "
          f"({(time.perf_counter() - started) / 3600:.1f} hours)", flush=True)


if __name__ == "__main__":
    main()
