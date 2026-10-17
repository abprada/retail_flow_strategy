"""Check how far back WRDS Consolidated Trades (taqmsec.wct_YYYYMMDD) go, and what scanning one day costs.

    WRDS_USERNAME=<you> uv run python scripts/taq/checks/TAQ_coverage.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push. Nothing is downloaded but table metadata
and a few numbers per sample day.

1. Lists every daily table in taqmsec and counts days per year for wct, ctm, mastm and complete_nbbo.
2. For one sample day per year from 2014 (the first wct day in June, plus the latest wct day), on WRDS:
   - checks that WCT's columns are the same as on 2019-06-04;
   - times a row count (a full scan of the day);
   - times a per-symbol aggregation, as the feature SQL would run: off-exchange trades by sub-penny side;
   - reads the table's size on disk, when WRDS allows it.

Outputs (gitignored), in data/raw/taq/coverage/:
    tables.csv        every taqmsec table name ending in a date
    days_per_year.csv days with a table, per table kind and year
    sample_days.csv   per sample day: rows, size, seconds for the count and the aggregation, column differences
"""

import csv
import os
import re
import time
from collections import Counter
from pathlib import Path

import psycopg

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "raw" / "taq" / "coverage"
KINDS = ["wct", "ctm", "mastm", "complete_nbbo"]
REFERENCE_DAY = "20190604"
SAMPLE_MONTH = "06"
FIRST_SAMPLE_YEAR = "2014"  # the sample starts in 2014 (odd lots on the tape from 2013-12-09)

# what a day's feature query will do at least: one scan, a filter on venue, a sub-penny split and a group by symbol
AGGREGATION = """
SELECT count(*) FROM (
    SELECT sym_root, sym_suffix,
           count(*) AS trades,
           sum(CASE WHEN ex = 'D' AND (price * 100 - floor(price * 100)) > 0.6 THEN 1 ELSE 0 END) AS subpenny_buys,
           sum(CASE WHEN ex = 'D' AND (price * 100 - floor(price * 100)) > 0 AND (price * 100 - floor(price * 100)) < 0.4 THEN 1 ELSE 0 END)
               AS subpenny_sells
    FROM taqmsec.wct_{ymd}
    GROUP BY sym_root, sym_suffix
) per_symbol
"""


def timed(conn, query: str) -> tuple[int, float]:
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(query)
        value = cur.fetchone()[0]
    return value, time.perf_counter() - start


def columns(conn, ymd: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'taqmsec' AND table_name = %s ORDER BY ordinal_position", (f"wct_{ymd}",))
        return [row[0] for row in cur.fetchall()]


def size_bytes(conn, ymd: str) -> int | None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_total_relation_size(%s::regclass)", (f"taqmsec.wct_{ymd}",))
            return cur.fetchone()[0]
    except psycopg.Error:  # WRDS may not grant the size function
        conn.rollback()
        return None


def main() -> None:
    username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
    print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
    with wrds_connect(username) as conn:
        copy_query(conn, "SELECT table_name FROM information_schema.tables "
                         "WHERE table_schema = 'taqmsec' AND table_name ~ '_[0-9]{8}$' ORDER BY table_name",
                   OUT / "tables.csv")
        with open(OUT / "tables.csv") as f:
            names = [row["table_name"] for row in csv.DictReader(f)]
        days = {kind: sorted(m.group(1) for n in names if (m := re.fullmatch(f"{kind}_([0-9]{{8}})", n))) for kind in KINDS}

        per_year = {kind: Counter(d[:4] for d in ds) for kind, ds in days.items()}
        years = sorted(set().union(*per_year.values()))
        with open(OUT / "days_per_year.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", *KINDS])
            w.writerows([y, *(per_year[k][y] for k in KINDS)] for y in years)
        print(f"{'year':6}" + "".join(f"{k:>15}" for k in KINDS))
        for y in years:
            print(f"{y:6}" + "".join(f"{per_year[k][y]:>15}" for k in KINDS))
        wct = days["wct"]
        if not wct:
            raise SystemExit("no wct tables in taqmsec")
        print(f"wct: {len(wct):,} days, {wct[0]} to {wct[-1]}", flush=True)

        reference = columns(conn, REFERENCE_DAY)
        samples = sorted({next((d for d in wct if d[:4] == y and d[4:6] >= SAMPLE_MONTH), None) for y in
                          sorted({d[:4] for d in wct if d[:4] >= FIRST_SAMPLE_YEAR})} - {None} | {wct[-1]})
        rows = []
        for ymd in samples:
            print(f"wct_{ymd}: counting", flush=True)
            cols = columns(conn, ymd)
            n_rows, count_s = timed(conn, f"SELECT count(*) FROM taqmsec.wct_{ymd}")
            print(f"wct_{ymd}: {n_rows:,} rows in {count_s:.0f} s; aggregating", flush=True)
            n_symbols, agg_s = timed(conn, AGGREGATION.format(ymd=ymd))
            size = size_bytes(conn, ymd)
            rows.append({
                "day": ymd, "rows": n_rows, "symbols": n_symbols,
                "size_gb": None if size is None else round(size / 1e9, 2),
                "count_seconds": round(count_s, 1), "aggregation_seconds": round(agg_s, 1),
                "columns_missing": " ".join(c for c in reference if c not in cols),
                "columns_extra": " ".join(c for c in cols if c not in reference),
            })
            print(f"wct_{ymd}: {n_symbols:,} symbols aggregated in {agg_s:.0f} s", flush=True)
            with open(OUT / "sample_days.csv", "w", newline="") as f:  # rewritten each day, so an interrupted run keeps what it has
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    for r in rows:
        print(r)
    print(f"written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
