"""Check that WCT's prevailing NBBO is the quote standing strictly before each trade in 2014 to 2017, as it is in 2019.

    WRDS_USERNAME=<you> uv run python scripts/checks/TAQ_nbbo_match.py

The password is read from ~/.pgpass (line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<you>:<password>`, chmod 600),
from PGPASSWORD, or asked for interactively. WRDS may also send a Duo push.

WCT gives every trade the national best bid and offer standing just before it (explanations/TAQ/wct.md), checked only
on 2019-06-04 (99.89% equal to the last complete_nbbo row strictly before the trade). Barber et al. (2024) write that
WRDS matches the NBBO in the nanosecond before the trade only since 2017, and wct has no nanosecond fields before 2016.
If earlier days took the quote at the trade's own timestamp, possibly the quote the trade itself caused, the position of
a trade in the spread would mean something else in 2014 to 2016.

For one day a year (2014 to 2017, and 2019 as the known case), the script downloads, unchanged, the wct trades and the
complete_nbbo rows of 40 symbols (10 large stocks and ETFs, and 30 others chosen by a hash of the symbol among those with
2,000 to 100,000 trades that day), trades from 10:00 to 11:00 and quotes from 09:30 to 11:00. Then, locally, it compares
each trade's nbb and nbo with two as-of matches in complete_nbbo: the last quote strictly before the trade, and the last
quote at or before it. Where the two differ (a quote change at the trade's own timestamp), it counts which one WCT took.

Outputs (gitignored), in data/raw/taq/coverage/nbbo_match/:
    <day>/wct.csv, <day>/complete_nbbo.csv, <day>/columns.csv   the downloaded rows and the tables' columns, unchanged
    summary.csv                                                 the comparison, one row per day
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from dsr.data import copy_query, wrds_connect

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "taq" / "coverage" / "nbbo_match"
DAYS = ["20140602", "20150601", "20160601", "20170601", "20190603"]  # the first trading day of June
FIXED = ["AAPL", "MSFT", "IBM", "GE", "XOM", "JPM", "BAC", "F", "SPY", "QQQ"]
N_HASHED = 30
TRADES_FROM, QUOTES_FROM, UNTIL = "10:00", "09:30", "11:00"
SYMBOL = "sym_root || CASE WHEN coalesce(sym_suffix, '') = '' THEN '' ELSE ' ' || sym_suffix END"


def columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'taqmsec' "
                    "AND table_name = %s ORDER BY ordinal_position", (table,))
        return [row[0] for row in cur.fetchall()]


def fetch_day(conn, ymd: str) -> None:
    out = OUT / ymd
    if (out / "complete_nbbo.csv").exists():
        return
    wct_cols, nbbo_cols = columns(conn, f"wct_{ymd}"), columns(conn, f"complete_nbbo_{ymd}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"table": ["wct"] * len(wct_cols) + ["complete_nbbo"] * len(nbbo_cols),
                  "column": wct_cols + nbbo_cols}).to_csv(out / "columns.csv", index=False)

    print(f"{ymd}: choosing symbols", flush=True)
    fixed = ", ".join(f"'{s}'" for s in FIXED)
    with conn.cursor() as cur:
        cur.execute(f"""
            WITH n AS (SELECT {SYMBOL} AS symbol, count(*) AS trades FROM taqmsec.wct_{ymd} GROUP BY 1)
            (SELECT symbol FROM n WHERE symbol IN ({fixed}))
            UNION
            (SELECT symbol FROM n WHERE trades BETWEEN 2000 AND 100000 AND symbol NOT IN ({fixed})
             ORDER BY md5(symbol) LIMIT {N_HASHED})""")
        symbols = sorted(row[0] for row in cur.fetchall())
    chosen = ", ".join("'" + s.replace("'", "''") + "'" for s in symbols)

    keep = lambda have, want: ", ".join(c for c in want if c in have)
    print(f"{ymd}: downloading wct and complete_nbbo rows of {len(symbols)} symbols", flush=True)
    copy_query(conn, f"""
        SELECT {keep(wct_cols, ['sym_root', 'sym_suffix', 'time_m', 'time_m_nano', 'tr_seqnum', 'ex', 'tr_scond',
                                'price', 'size', 'qtime', 'qtime_nano', 'nbb', 'nbo'])}
        FROM taqmsec.wct_{ymd}
        WHERE {SYMBOL} IN ({chosen}) AND time_m >= '{TRADES_FROM}' AND time_m < '{UNTIL}'""", out / "wct.csv")
    copy_query(conn, f"""
        SELECT {keep(nbbo_cols, ['sym_root', 'sym_suffix', 'time_m', 'time_m_nano', 'best_bid', 'best_ask',
                                 'natbbo_ind', 'qu_cond'])}
        FROM taqmsec.complete_nbbo_{ymd}
        WHERE {SYMBOL} IN ({chosen}) AND time_m >= '{QUOTES_FROM}' AND time_m < '{UNTIL}'""", out / "complete_nbbo.csv")


def nanoseconds(df: pd.DataFrame, time_col: str, nano_col: str) -> np.ndarray:
    t = pd.to_timedelta(df[time_col]).to_numpy().astype("timedelta64[ns]").astype("int64")
    if nano_col in df:
        t = t + pd.to_numeric(df[nano_col]).fillna(0).astype("int64").to_numpy()
    return t


def compare(ymd: str) -> dict:
    out = OUT / ymd
    w = pd.read_csv(out / "wct.csv", dtype=str, keep_default_na=False)
    q = pd.read_csv(out / "complete_nbbo.csv", dtype=str, keep_default_na=False)
    for df in (w, q):
        df["symbol"] = np.where(df["sym_suffix"].eq(""), df["sym_root"], df["sym_root"] + " " + df["sym_suffix"])
    w["t"] = nanoseconds(w, "time_m", "time_m_nano")
    w["qt"] = nanoseconds(w.assign(qtime=w["qtime"].replace("", "00:00:00")), "qtime", "qtime_nano")
    q["t"] = nanoseconds(q, "time_m", "time_m_nano")
    for c in ("nbb", "nbo"):
        w[c] = pd.to_numeric(w[c].replace("", np.nan))
    for c in ("best_bid", "best_ask"):
        q[c] = pd.to_numeric(q[c].replace("", np.nan))
    w = w[w["nbb"].notna() & w["nbo"].notna() & w["qtime"].ne("")].sort_values("t", kind="stable")
    q = q.reset_index(names="row").sort_values(["t", "row"], kind="stable")  # ties keep the file's order

    quotes = q[["symbol", "t", "best_bid", "best_ask"]]
    before = pd.merge_asof(w, quotes, on="t", by="symbol", allow_exact_matches=False)
    at = pd.merge_asof(w, quotes, on="t", by="symbol", allow_exact_matches=True)
    eq = lambda m: np.isclose(m["nbb"], m["best_bid"]) & np.isclose(m["nbo"], m["best_ask"])
    eq_before, eq_at = eq(before), eq(at)
    differ = (before["best_bid"].notna() & ~(np.isclose(before["best_bid"], at["best_bid"])
                                               & np.isclose(before["best_ask"], at["best_ask"]))).to_numpy()
    tied = np.isin(w["t"].to_numpy(), q["t"].to_numpy())  # some quote shares the trade's timestamp (any symbol)
    qt, t = w["qt"].to_numpy(), w["t"].to_numpy()
    return {
        "day": ymd,
        "trades": len(w),
        "symbols": w["symbol"].nunique(),
        "time_has_nanoseconds": "time_m_nano" in w,
        "share_equal_to_quote_strictly_before": eq_before.mean(),
        "share_equal_to_quote_at_or_before": eq_at.mean(),
        "trades_with_a_quote_change_at_their_own_time": int(differ.sum()),
        "of_those_wct_took_quote_before": eq_before[differ].mean() if differ.any() else np.nan,
        "of_those_wct_took_quote_at_trade_time": eq_at[differ].mean() if differ.any() else np.nan,
        "share_trade_time_shared_with_some_quote": tied.mean(),
        "share_qtime_before_trade": (qt < t).mean(),
        "share_qtime_equal_trade": (qt == t).mean(),
        "share_qtime_after_trade": (qt > t).mean(),
    }


def main() -> None:
    todo = [d for d in DAYS if not (OUT / d / "complete_nbbo.csv").exists()]
    if todo:
        username = os.environ.get("WRDS_USERNAME") or input("WRDS username: ")
        print("connecting to WRDS: approve the Duo push on your phone (gives up after 2 minutes)", flush=True)
        with wrds_connect(username) as conn:
            for ymd in todo:
                fetch_day(conn, ymd)
    summary = pd.DataFrame([compare(d) for d in DAYS])
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.set_index("day").T.to_string(float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
