"""Check how trades outside the regular session reach the tape, 2019 against 2023 to 2025 (overnight trading).

    WRDS_USERNAME=<you> uv run python scripts/checks/TAQ_overnight.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push.

Overnight venues (Blue Ocean ATS, used by Robinhood's 24 Hour Market from May 2023) match orders roughly from 20:00 to
04:00. TAQ stamps a trade when the tape published it, so such trades may appear at any hour, or the next morning as late
reports, possibly with sale conditions the cleaning's whitelist drops (it keeps blank, @, I, F, and T or U outside the
session). This counts, for a Monday and a Wednesday in 2019, 2023, 2024 and 2025, every wct trade outside 09:30 to 16:00
by 15-minute slot, venue (off-exchange or exchange) and full sale-condition string, and every trade by hour.

Outputs (gitignored), in data/raw/taq/coverage/overnight/:
    <day>_by_hour.csv        trades and shares by hour and venue, all day
    <day>_outside.csv        trades and shares outside 09:30-16:00 by 15-minute slot, venue and sale condition
    summary.csv              per day: trades outside the session by window, and the share the whitelist would drop
"""

import os
import re
from pathlib import Path

import pandas as pd

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "taq" / "coverage" / "overnight"
DAYS = ["20190603", "20190605", "20230911", "20230913", "20240603", "20240605", "20250602", "20250604"]  # Mon, Wed
KEPT_OUTSIDE = re.compile(r"[ @IFTU]*")  # the whitelist outside the session


def fetch_day(conn, ymd: str) -> None:
    copy_query(conn, f"""
        SELECT extract(hour FROM time_m)::int AS hour, (ex = 'D') AS off_exchange, count(*) AS trades, sum(size) AS shares
        FROM taqmsec.wct_{ymd} GROUP BY 1, 2 ORDER BY 1, 2""", OUT / f"{ymd}_by_hour.csv")
    copy_query(conn, f"""
        SELECT to_char(time_m, 'HH24') || ':' || lpad((15 * floor(extract(minute FROM time_m) / 15))::int::text, 2, '0') AS slot,
               (ex = 'D') AS off_exchange, coalesce(tr_scond, '') AS tr_scond, count(*) AS trades, sum(size) AS shares
        FROM taqmsec.wct_{ymd}
        WHERE time_m < '09:30' OR time_m >= '16:00'
        GROUP BY 1, 2, 3 ORDER BY 1, 2, 4 DESC""", OUT / f"{ymd}_outside.csv")


def window(slot: str) -> str:
    if slot < "04:00":
        return "00:00-04:00"
    if slot < "08:00":
        return "04:00-08:00"
    if slot < "09:30":
        return "08:00-09:30"
    if slot < "20:00":
        return "16:00-20:00"
    return "20:00-24:00"


def summarize(ymd: str) -> list[dict]:
    d = pd.read_csv(OUT / f"{ymd}_outside.csv", dtype={"slot": str, "tr_scond": str}, keep_default_na=False)
    d["window"] = d["slot"].map(window)
    d["kept"] = d["tr_scond"].map(lambda c: bool(KEPT_OUTSIDE.fullmatch(c)))
    rows = []
    for (w, off), g in d.groupby(["window", "off_exchange"]):
        dropped = g[~g["kept"]]
        rows.append({
            "day": ymd, "weekday": pd.Timestamp(ymd).day_name(), "window": w, "venue": "off" if off in (True, "t") else "exch",
            "trades": int(g["trades"].sum()), "shares": int(g["shares"].sum()),
            "share_of_trades_whitelist_drops": dropped["trades"].sum() / g["trades"].sum(),
            "top_dropped_conditions": ", ".join(f"'{c}' {n:,}" for c, n in
                                                dropped.groupby("tr_scond")["trades"].sum().nlargest(4).items()),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [d for d in DAYS if not (OUT / f"{d}_outside.csv").exists()]
    if todo:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            for ymd in todo:
                print(f"{ymd}: counting", flush=True)
                fetch_day(conn, ymd)
    summary = pd.DataFrame([r for d in DAYS for r in summarize(d)])
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 200)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
