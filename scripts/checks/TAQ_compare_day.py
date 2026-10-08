"""Compare a pulled day's four tables (parquet, as the WRDS Cloud job writes them) with the 2019-06-04 reference.

    uv run python scripts/checks/TAQ_compare_day.py <folder with daily/cells/bars/audit.parquet>

Prints GOLDEN TEST PASSED if every table matches the reference (scripts/checks/TAQ_reference_day.py) row by row, or
the columns that differ.
"""

import sys
from pathlib import Path

import pandas as pd

from dsr.taq import KEYS, compare_tables

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data" / "intermediate" / "taq" / "reference" / "20190604"


def main() -> None:
    folder = Path(sys.argv[1])
    comparison = pd.concat([compare_tables(name, pd.read_parquet(REFERENCE / f"{name}.parquet"),
                                           pd.read_parquet(folder / f"{name}.parquet")).assign(table=name)
                            for name in KEYS])
    comparison.to_csv(folder / "comparison.csv", index=False)
    differing = comparison[comparison["rows_differing"] > 0]
    print("GOLDEN TEST PASSED: matches the reference in every table" if differing.empty
          else "GOLDEN TEST FAILED:\n" + differing.to_string(index=False))


if __name__ == "__main__":
    main()
