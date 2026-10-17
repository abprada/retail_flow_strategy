"""Check the master file's older layout (2014 to 2016) before the pull relies on it.

    WRDS_USERNAME=<you> uv run python scripts/taq/checks/TAQ_master.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push.

scripts/taq/checks/TAQ_schemas.py found that the master file (mastm) has no listed_exchange or tick_pilot_indicator in
2014 and 2015 (it has listed_market instead), and that on 2016-06-01 both exist but the new ones are empty. This saves:
    fill.csv          for the first trading day of each month from 2016-06 to 2017-06: master rows, and how many have
                      listed_exchange, listed_market and tick_pilot_indicator filled, by Tick Size Pilot group
    listed_market.csv on 2014-01-02, 2015-06-01 and 2016-06-01: master rows by listed_market
    known.csv         the same days: listed_market of stocks whose listing exchange is known
    opening.csv       the same days: opening prints (sale condition O) on the listing exchange given by listed_market,
                      before 15:50, as in scripts/taq/checks/TAQ_schemas.py
Each query is saved as it finishes and skipped when its file exists.

Outputs (gitignored): data/raw/taq/coverage/master/
"""

import os
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "raw" / "taq" / "coverage" / "master"
OLD_DAYS = ["20140102", "20150601", "20160601"]
KNOWN = {"IBM": "NYSE", "GE": "NYSE", "AAPL": "Nasdaq", "MSFT": "Nasdaq", "SPY": "NYSE Arca", "GLD": "NYSE Arca",
         "BATS": "Cboe BZX", "SIRI": "Nasdaq"}
COLUMNS = ["listed_exchange", "listed_market", "tick_pilot_indicator"]
SYMBOL = "w.sym_root || CASE WHEN coalesce(w.sym_suffix, '') = '' THEN '' ELSE ' ' || w.sym_suffix END"


def first_of_months() -> list[str]:
    sessions = xcals.get_calendar("XNYS", start="2016-06-01", end="2017-06-30").sessions
    return sorted(pd.Series(sessions).groupby(sessions.to_period("M")).min().dt.strftime("%Y%m%d"))


def master_columns(conn, ymd: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'taqmsec' "
                    "AND table_name = %s", (f"mastm_{ymd}",))
        return {row[0] for row in cur.fetchall()}


def main() -> None:
    username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
    print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
    with wrds_connect(username) as conn:
        parts = []
        for ymd in first_of_months():
            have = master_columns(conn, ymd)
            filled = ", ".join(f"sum((coalesce({c}, '') <> '')::int) AS {c}_filled" if c in have else f"NULL AS {c}_filled"
                               for c in COLUMNS)
            group = "coalesce(tick_pilot_indicator, '')" if "tick_pilot_indicator" in have else "''"
            path = OUT / "fill" / f"{ymd}.csv"
            copy_query(conn, f"SELECT '{ymd}' AS day, {group} AS pilot_group, count(*) AS n_rows, {filled} "
                             f"FROM taqmsec.mastm_{ymd} GROUP BY 2 ORDER BY 2", path)
            parts.append(pd.read_csv(path, dtype=str, keep_default_na=False))
            print(f"{ymd}: master columns read", flush=True)
        pd.concat(parts).to_csv(OUT / "fill.csv", index=False)

        known = ", ".join(f"'{s}'" for s in KNOWN)
        for ymd in OLD_DAYS:
            copy_query(conn, f"SELECT listed_market, count(*) AS symbols FROM taqmsec.mastm_{ymd} GROUP BY 1 ORDER BY 2 DESC",
                       OUT / f"listed_market_{ymd}.csv")
            copy_query(conn, f"SELECT symbol_15, listed_market FROM taqmsec.mastm_{ymd} WHERE symbol_15 IN ({known})",
                       OUT / f"known_{ymd}.csv")
            print(f"{ymd}: opening prints", flush=True)
            copy_query(conn, f"""
                WITH per_stock AS (
                    SELECT m.listed_market, {SYMBOL} AS symbol, count(*) AS prints, count(DISTINCT w.price) AS prices
                    FROM taqmsec.wct_{ymd} w JOIN taqmsec.mastm_{ymd} m ON m.symbol_15 = {SYMBOL}
                    WHERE w.tr_scond LIKE '%O%' AND w.ex = m.listed_market AND w.time_m < '15:50'
                    GROUP BY 1, 2
                )
                SELECT listed_market, count(*) AS stocks, sum((prints > 1)::int) AS stocks_several_prints,
                       sum((prices > 1)::int) AS stocks_several_prices, max(prints) AS most_prints
                FROM per_stock GROUP BY 1 ORDER BY 2 DESC""", OUT / f"opening_{ymd}.csv")

    print("\nmaster rows with each column filled, first trading day of each month:")
    print(pd.read_csv(OUT / "fill.csv", dtype=str, keep_default_na=False).to_string(index=False))
    for ymd in OLD_DAYS:
        print(f"\n== {ymd}")
        print("listed_market codes:", pd.read_csv(OUT / f"listed_market_{ymd}.csv", dtype=str, keep_default_na=False)
              .to_string(index=False, header=False).replace("\n", "; "))
        k = pd.read_csv(OUT / f"known_{ymd}.csv", dtype=str, keep_default_na=False)
        print("known stocks:", "; ".join(f"{s} ({KNOWN[s]}) -> {m!r}" for s, m in zip(k["symbol_15"], k["listed_market"])))
        print("opening prints on the listing exchange given by listed_market:")
        print(pd.read_csv(OUT / f"opening_{ymd}.csv", dtype=str).to_string(index=False))


if __name__ == "__main__":
    main()
