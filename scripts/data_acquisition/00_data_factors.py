"""Download Fama-French's daily factors from WRDS (the ff schema's five-factor daily table and ff.factors_daily), unchanged: market (mktrf), size (smb), value (hml),
profitability (rmw), investment/CMA (cma), momentum (umd) and the risk-free rate (rf).

    WRDS_USERNAME=<you> uv run python scripts/data_acquisition/00_data_factors.py

The password is read from ~/.pgpass, from PGPASSWORD, or asked for interactively, as in 00_data_crsp_daily.py. A file
already downloaded is skipped.

Every row dated 2013-01-01 or later is kept, as for CRSP. The factors are close-to-close returns in decimals (0.01 = 1%)
on the date they are earned: to regress a prediction made on day t, whose return is day t's close to day t+1's close,
use the factors dated t+1 (the next trading day).

Outputs (gitignored):
    data/raw/factors/factors_daily_columns.csv   the ff schema's tables, column names and types, as WRDS lists them
    data/raw/factors/factors_daily_from_2013.csv every row dated 2013-01-01 or later, all columns
"""

import csv
import os
from pathlib import Path

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
FACTORS = ROOT / "data" / "raw" / "factors"
COLUMNS = FACTORS / "factors_daily_columns.csv"
DAILY = FACTORS / "factors_daily_from_2013.csv"
START = "2013-01-01"
FIVE = ("date", "mktrf", "smb", "hml", "rmw", "cma", "rf")  # the five-factor table; momentum is in factors_daily


def main() -> None:
    if not DAILY.exists():
        FACTORS.mkdir(parents=True, exist_ok=True)
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            copy_query(
                conn,
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'ff' ORDER BY table_name, ordinal_position",
                COLUMNS,
            )
            with open(COLUMNS) as f:
                tables: dict[str, list[str]] = {}
                for row in csv.DictReader(f):
                    tables.setdefault(row["table_name"], []).append(row["column_name"])
            five = next((t for t, cols in tables.items() if t.endswith("_daily") and set(FIVE) <= set(cols)), None)
            if five is None or "ff.factors_daily" not in {f"ff.{t}" for t in tables}:
                COLUMNS.unlink()
                raise SystemExit(f"no daily table in schema ff with {FIVE}; the ff tables and columns are: {tables}")
            print(f"five factors from ff.{five}, momentum from ff.factors_daily", flush=True)
            copy_query(
                conn,
                f"SELECT f.date, f.mktrf, f.smb, f.hml, f.rmw, f.cma, m.umd, f.rf FROM ff.{five} f "
                f"LEFT JOIN ff.factors_daily m ON m.date = f.date WHERE f.date >= '{START}' ORDER BY f.date",
                DAILY,
            )
    with open(DAILY) as f:
        rows = list(csv.reader(f))
    print(f"{DAILY.relative_to(ROOT)}: {len(rows) - 1:,} days, {rows[1][0]} to {rows[-1][0]}; columns: {', '.join(rows[0])}")


if __name__ == "__main__":
    main()
