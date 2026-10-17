"""Check, day by day, how the trade file's flagged trades pair with their records: on trade ID, as the pull did until
code version 75c9d5e71451, and on sequence number, as it does now (src/dsr/taq_sql.py).

Runs on WRDS Cloud, where the database is close and ~/.pgpass holds the password. Copy it to ~/diag and submit a job
from there (qsub fails on absolute paths):

    python3 TAQ_ctm_pairing.py --days-file days.csv --out ctm_pairing.csv [--threads 4]

Each flagged trade (08 cancelled, 07 erroneous, 01 corrected) has a record published later (10, 11, 12). Trade IDs
are missing before 2016 and empty on most flagged rows until 2017-11-03, so the pull paired those days' trades with no
record; they are pulled again. A later day is kept as it is if both keys give every flagged trade the same record. The
old query joined on symbol, venue and trade ID, whatever the record's code; the new one joins on symbol, venue,
sequence number and the matching code, and takes the first record published if there are several.

For every day the flagged rows (tr_corr <> '00') are read and one line is appended to --out: flagged trades, records,
flagged trades with an empty trade ID, those paired by trade ID (and with more than one record), those paired by
sequence number (and with more than one record), and how many get a different record (or none) under the two keys.
A day already in --out is skipped, so an interrupted run resumes.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import psycopg2

RECORD = {"08": "10", "07": "11", "01": "12"}  # flagged trade -> its record
KEY = ["sym_root", "sym_suffix", "ex"]
FIELDS = ["day", "flagged", "records", "tr_id_empty", "paired_by_tr_id", "several_by_tr_id", "paired_by_seqnum",
          "several_by_seqnum", "different"]


def pairing(rows: pd.DataFrame) -> dict:
    """The day's counts from its flagged rows."""
    rows = rows.assign(sym_suffix=rows["sym_suffix"].fillna(""))
    flagged = rows[rows["tr_corr"].isin(RECORD)].reset_index(drop=True).rename_axis("o").reset_index()
    flagged = flagged.assign(code=flagged["tr_corr"].map(RECORD))
    records = rows[rows["tr_corr"].isin(RECORD.values())].reset_index(drop=True).rename_axis("r").reset_index()
    records = records.rename(columns={"tr_corr": "code"})

    # as the old query: SQL never matches a null trade ID (pandas would), and the record's code was not checked
    old = flagged[flagged["tr_id"].notna()].merge(records[records["tr_id"].notna()], on=[*KEY, "tr_id"])
    new = flagged.merge(records, on=[*KEY, "code", "tr_seqnum"], suffixes=("", "_r"))
    n_old, n_new = old.groupby("o")["r"].nunique(), new.groupby("o")["r"].nunique()
    first = new.sort_values(["o", "time_m_r", "r"]).groupby("o")["r"].first()
    single_old = old[old["o"].isin(n_old[n_old == 1].index)].set_index("o")["r"]
    same = (single_old.reindex(flagged["o"]) == first.reindex(flagged["o"])).sum()
    empty = flagged["tr_id"].isna() | (flagged["tr_id"] == "")
    return {"flagged": len(flagged), "records": len(records), "tr_id_empty": int(empty.sum()),
            "paired_by_tr_id": len(n_old), "several_by_tr_id": int((n_old > 1).sum()), "paired_by_seqnum": len(n_new),
            "several_by_seqnum": int((n_new > 1).sum()), "different": int(len(flagged) - same)}


def run(days: list[str], out: Path, user: str, lock) -> None:
    conn = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user=user, sslmode="require")
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        for day in days:
            ymd = day.replace("-", "")
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'taqmsec' "
                        "AND table_name = %s", (f"ctm_{ymd}",))
            tr_id = "tr_id" if "tr_id" in {r[0] for r in cur.fetchall()} else "NULL"
            cur.execute(f"SELECT sym_root, sym_suffix, ex, CAST(tr_seqnum AS BIGINT), tr_corr, CAST({tr_id} AS TEXT), "
                        f"time_m FROM taqmsec.ctm_{ymd} WHERE tr_corr <> '00'")
            rows = pd.DataFrame(cur.fetchall(), columns=["sym_root", "sym_suffix", "ex", "tr_seqnum", "tr_corr", "tr_id",
                                                         "time_m"])
            line = {"day": day, **pairing(rows)}
            with lock, open(out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(line)
    conn.close()


def main() -> None:
    import threading

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days-file", required=True, type=Path, help="a CSV with a date column (days.csv)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--user", default=os.environ.get("USER"))
    args = parser.parse_args()

    days = list(pd.read_csv(args.days_file, dtype=str)["date"])
    if not args.out.exists():
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    done = set(pd.read_csv(args.out, dtype=str)["day"])
    todo = [d for d in days if d not in done]
    print(f"{len(days):,} days, {len(todo):,} to do", flush=True)
    start, lock = time.perf_counter(), threading.Lock()
    with ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(lambda k: run(todo[k::args.threads], args.out, args.user, lock), range(args.threads)))
    print(f"done in {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
