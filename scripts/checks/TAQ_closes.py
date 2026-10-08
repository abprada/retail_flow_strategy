"""Check NYSE's close times (exchange_calendars, XNYS) against the closing auctions on the tape, 2014 to 2025.

    WRDS_USERNAME=<you> uv run python scripts/checks/TAQ_closes.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push.

Every bucket and bar of a day hangs on its close (16:00, or 13:00 on an early close), and the calendar's early closes
should be checked against the data before the pull uses them. For each trading day, the script reads the time of the
first closing print (sale condition 6) of three stocks listed on different exchanges: IBM (NYSE), AAPL (Nasdaq),
SPY (NYSE Arca). It does the calendar's 26 early closes first, then every other day, and saves after each day, so a
run can be stopped and resumed. At the end it lists the days where the tape and the calendar disagree.

Output (gitignored):
    data/raw/taq/coverage/closing_prints.csv   per day and symbol: the first closing print's time and the number of
                                               closing prints
"""

import csv
import os
import time
from pathlib import Path

import exchange_calendars as xcals

from dsr.data import wrds_connect

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "taq" / "coverage" / "closing_prints.csv"
START, END = "2014-01-01", "2025-12-31"
SYMBOLS = ["IBM", "AAPL", "SPY"]
FIELDS = ["day", "symbol", "first_close_print", "close_prints"]

QUERY = """
SELECT sym_root, min(time_m)::text, count(*)
FROM taqmsec.wct_{ymd}
WHERE sym_root IN ({symbols}) AND coalesce(sym_suffix, '') = '' AND tr_scond LIKE '%6%'
GROUP BY sym_root
"""


def calendar_closes() -> dict[str, str]:
    schedule = xcals.get_calendar("XNYS", start=START, end=END).schedule
    closes = schedule["close"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    return dict(zip(schedule.index.strftime("%Y%m%d"), closes))


def main() -> None:
    closes = calendar_closes()
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {row["day"] for row in csv.DictReader(f)}
    days = sorted(closes, key=lambda d: (closes[d] == "16:00", d))  # early closes first
    todo = [d for d in days if d not in done]
    print(f"{len(closes):,} trading days, {sum(c != '16:00' for c in closes.values())} early closes; "
          f"{len(todo):,} still to check", flush=True)

    if todo:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        new_file = not OUT.exists()
        with wrds_connect(username) as conn, open(OUT, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            symbols = ", ".join(f"'{s}'" for s in SYMBOLS)
            for i, ymd in enumerate(todo, 1):
                start = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(QUERY.format(ymd=ymd, symbols=symbols))
                    found = {sym: (first, n) for sym, first, n in cur.fetchall()}
                for sym in SYMBOLS:
                    first, n = found.get(sym, ("", 0))
                    writer.writerow({"day": ymd, "symbol": sym, "first_close_print": first, "close_prints": n})
                f.flush()
                if i <= 30 or i % 100 == 0:
                    shown = "  ".join(f"{s} {found.get(s, ('none',))[0][:8]}" for s in SYMBOLS)
                    print(f"[{i:,}/{len(todo):,}] {ymd} calendar {closes[ymd]}  tape: {shown}  "
                          f"({time.perf_counter() - start:.1f} s)", flush=True)

    with open(OUT) as f:
        rows = list(csv.DictReader(f))
    disagree = []
    for ymd, close in sorted(closes.items()):
        hours = {r["first_close_print"][:2] for r in rows if r["day"] == ymd and r["first_close_print"]}
        if hours != {close[:2]}:
            disagree.append((ymd, close, sorted(hours)))
    print(f"days where the tape's closing prints are not all in the calendar's close hour: {len(disagree)}")
    for ymd, close, hours in disagree:
        print(f"  {ymd}: calendar {close}, closing prints in hour(s) {hours or 'none'}")


if __name__ == "__main__":
    main()
