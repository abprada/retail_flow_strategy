"""The TAQ pull's per-day SQL (explanations/taq_pull.md), generated for DuckDB (local development) or Postgres (WRDS).

One template, two dialects. The few differences are switches here: the source tables and their column types,
integer division, regular expressions and time arithmetic. Integer division is only ever applied to positive numbers,
where DuckDB's `//` and Postgres's `/` on integers agree.

`day_sql` returns the statements to run in order: `setup`, then one SELECT per output table. With temporary tables
(possible on a local server), `setup` builds the corrections, their pairs and the per-trade table once and the four
SELECTs read them. WRDS only allows read-only transactions, where no table can be created, not even a temporary one:
then `setup` is empty and every SELECT carries the same steps as common table expressions (WITH), so each output
table is computed in one statement from the day's raw tables.
"""

from dataclasses import dataclass

TEN_MINUTES_NS = 600 * 10**9


@dataclass(frozen=True)
class Day:
    """A WCT day's parameters (taq_pull.md section 2), times in nanoseconds after midnight."""

    ymd: str  # YYYYMMDD
    open_ns: int
    cutoff_ns: int
    close_ns: int
    has_nano: bool  # wct and ctm have time_m_nano (from 2016)


class Dialect:
    def __init__(self, name: str, local_dir: str | None = None, local_format: str = "parquet"):
        """local_dir holds the day's raw rows for DuckDB: as text parquet (wct.parquet, ctm_corrections.parquet, the
        downloaded day) or as Postgres CSV (wct.csv, ctm.csv, pulled on WRDS Cloud)."""
        if name not in ("duckdb", "postgres") or local_format not in ("parquet", "csv"):
            raise ValueError((name, local_format))
        self.name, self.local_dir, self.local_format = name, local_dir, local_format

    def idiv(self, a: str, b: str) -> str:
        """Integer division of positive integers."""
        return f"(({a}) // ({b}))" if self.name == "duckdb" else f"(({a}) / ({b}))"

    def full_match(self, col: str, pattern: str) -> str:
        return f"regexp_full_match({col}, '{pattern}')" if self.name == "duckdb" else f"({col} ~ '^{pattern}$')"

    def t_ns(self, time_col: str, nano_col: str | None) -> str:
        if self.name == "duckdb":
            nano = f" + COALESCE(CAST(NULLIF({nano_col}, '') AS BIGINT), 0)" if nano_col else ""
            t = f"CAST({time_col} AS TIME)"
            return (f"((CAST(hour({t}) AS BIGINT) * 3600 + minute({t}) * 60) * 1000000 + microsecond({t})) * 1000{nano}")
        nano = f" + COALESCE({nano_col}, 0)" if nano_col else ""
        return f"CAST(EXTRACT(EPOCH FROM {time_col}) * 1000000 AS BIGINT) * 1000{nano}"

    def source(self, table: str, ymd: str) -> str:
        if self.name == "duckdb" and self.local_format == "csv":
            return f"read_csv('{self.local_dir}/{table}.csv', header = true, all_varchar = true)"
        if self.name == "duckdb":
            files = {"wct": "wct.parquet", "ctm": "ctm_corrections.parquet"}
            return f"read_parquet('{self.local_dir}/{files[table]}')"
        return f"taqmsec.{table}_{ymd}" if table == "wct" else f"(SELECT * FROM taqmsec.ctm_{ymd} WHERE tr_corr <> '00') AS ctm_flagged"

    def num(self, col: str) -> str:
        """A price column as an exact decimal (the local parquet files hold text; empty means missing)."""
        if self.name == "duckdb":
            return f"CAST(NULLIF({col}, '') AS DECIMAL(18, 6))"
        return col

    def float_(self, expr: str) -> str:
        """An exact decimal sent as a double: Postgres would print numeric results with 15 to 20 useless digits."""
        return f"CAST({expr} AS {'DOUBLE' if self.name == 'duckdb' else 'DOUBLE PRECISION'})"

    def int_(self, col: str) -> str:
        if self.name == "duckdb":
            return f"CAST(NULLIF({col}, '') AS BIGINT)"
        return f"CAST({col} AS BIGINT)"


def _typed(d: Dialect, table: str, day: Day, extra: str = "") -> str:
    """Rows of wct or ctm with the columns every step uses, typed."""
    nano = "time_m_nano" if day.has_nano else None
    return f"""
        SELECT sym_root, COALESCE(sym_suffix, '') AS sym_suffix,
               sym_root || CASE WHEN COALESCE(sym_suffix, '') = '' THEN '' ELSE ' ' || sym_suffix END AS symbol,
               ex, COALESCE(tr_scond, '') AS cond, {d.t_ns('time_m', nano)} AS t, {d.int_('tr_seqnum')} AS seq,
               {d.num('price')} AS price, {d.int_('size')} AS size{extra}
        FROM {d.source(table, day.ymd)}"""


def day_sql(d: Dialect, day: Day, temp_tables: bool = True) -> dict[str, list[str] | str]:
    key = ["sym_root", "sym_suffix", "ex", "seq", "t"]
    on = lambda a, b: " AND ".join(f"{a}.{k} = {b}.{k}" for k in key)
    cut, close, open_ = day.cutoff_ns, day.close_ns, day.open_ns

    ctm = _typed(d, 'ctm', day, ', tr_corr')
    # Each flagged trade is paired with its record (08 with 10, 07 with 11, 01 with 12) on symbol, venue and sequence
    # number, which the record repeats in every year. Trade IDs cannot be used: ctm has none before 2016, and they are
    # empty on most flagged rows until 2017-11-03. Should a key have several records, the first published counts. A
    # trade without a record is left as wct has it.
    pairs = f"""
        SELECT o.*, r.t AS t_record, r.price AS price_r, r.size AS size_r, r.cond AS cond_r, r.t IS NOT NULL AS paired,
               r.t IS NOT NULL AND (o.t >= {cut} OR r.t < {cut}) AS record_counts
        FROM ctm_f o LEFT JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY sym_root, sym_suffix, ex, seq, tr_corr ORDER BY t) AS nth
            FROM ctm_f WHERE tr_corr IN ('10', '11', '12')
        ) r ON r.nth = 1 AND r.tr_corr = CASE o.tr_corr WHEN '08' THEN '10' WHEN '07' THEN '11' ELSE '12' END
               AND o.sym_root = r.sym_root AND o.sym_suffix = r.sym_suffix AND o.ex = r.ex AND o.seq = r.seq
        WHERE o.tr_corr IN ('01', '07', '08')"""

    keys = ", ".join(key)
    normal, extended = d.full_match("cond", "[ @IF]*"), d.full_match("cond", "[ @IFTU]*")
    s1 = f"""
        WITH w AS ({_typed(d, 'wct', day, f", {d.num('nbb')} AS nbb, {d.num('nbo')} AS nbo")}),
        actions AS (
            SELECT {keys}, 'records' AS action, CAST(NULL AS DECIMAL(18, 6)) AS price_r, CAST(NULL AS BIGINT) AS size_r,
                   CAST(NULL AS VARCHAR) AS cond_r
            FROM ctm_f WHERE tr_corr IN ('10', '11', '12')
            UNION ALL
            SELECT {keys}, 'cancelled_or_erroneous', NULL, NULL, NULL FROM pairs
            WHERE tr_corr IN ('07', '08') AND record_counts
            UNION ALL
            SELECT {keys}, 'restore', price_r, size_r, cond_r FROM pairs
            WHERE tr_corr = '01' AND paired AND NOT record_counts
        ),
        r AS (
            SELECT w.symbol, w.sym_root, w.sym_suffix, w.ex, w.t, w.seq, w.nbb, w.nbo,
                   w.price AS price0, w.size AS size0, a.action,
                   CASE WHEN a.action = 'restore' THEN a.price_r ELSE w.price END AS price,
                   CASE WHEN a.action = 'restore' THEN a.size_r ELSE w.size END AS size,
                   CASE WHEN a.action = 'restore' THEN a.cond_r ELSE w.cond END AS cond
            FROM w LEFT JOIN actions a ON {on('w', 'a')}
        ),
        b AS (
            SELECT r.*,
                   CASE WHEN t < {open_} AND ({normal} OR {extended}) THEN 0
                        WHEN t >= {open_} AND t < {cut} AND {normal} THEN 1
                        WHEN t >= {cut} AND t < {close} AND {normal} THEN 2
                        WHEN t >= {close} AND ({normal} OR {extended}) THEN 3 END AS bucket,
                   CAST(ROUND(price * 1000000) AS BIGINT) AS p,
                   CASE WHEN nbb > 0 AND nbo > 0 THEN CAST(ROUND(nbb * 1000000) AS BIGINT) END AS bid,
                   CASE WHEN nbb > 0 AND nbo > 0 THEN CAST(ROUND(nbo * 1000000) AS BIGINT) END AS ask
            FROM r
        ),
        c AS (
            SELECT b.*,
                   CASE WHEN action IN ('records', 'cancelled_or_erroneous') THEN action
                        WHEN bucket IS NULL THEN 'sale_condition'
                        WHEN price < 1 OR size <= 0 THEN 'price_or_size'
                        ELSE 'kept' END AS fate,
                   CASE WHEN ex = 'D' THEN 1 ELSE 0 END AS venue,
                   COALESCE(bid < ask, FALSE) AS valid,
                   COALESCE(bid = ask, FALSE) AS locked,
                   p - bid AS d, ask - bid AS w,
                   p % 10000 AS zt,
                   p * size AS dm
            FROM b
        )
        SELECT c.*,
               CASE WHEN zt = 0 THEN 0 WHEN zt = 5000 THEN 6
                    WHEN zt < 5000 THEN 1 + LEAST({d.idiv('zt', '1000')}, 4)
                    ELSE 6 + LEAST({d.idiv('zt + 999', '1000')} - 5, 5) END AS z,
               CASE WHEN NOT valid THEN 0 WHEN d < 0 THEN 1 WHEN d = 0 THEN 2 WHEN d = w THEN 14 WHEN d > w THEN 15
                    WHEN 2 * d = w THEN 8
                    WHEN 10 * d < 5 * w THEN 3 + {d.idiv('10 * d', 'w')}
                    ELSE 8 + {d.idiv('10 * d + w - 1', 'w')} - 5 END AS q,
               CASE WHEN NOT valid THEN 0 WHEN w <= 10000 THEN 1 WHEN w <= 20000 THEN 2 WHEN w <= 50000 THEN 3 ELSE 4 END
                   AS spread,
               CASE WHEN dm >= 100000000000 THEN 4 WHEN size = 1 THEN 0 WHEN size <= 99 THEN 1 WHEN size <= 499 THEN 2
                    ELSE 3 END AS size_code,
               CASE WHEN bucket = 1 THEN {d.idiv(f't - {open_}', str(TEN_MINUTES_NS))} ELSE 0 END AS bar,
               venue = 1 AND dm < 100000000000 AS retail
        FROM c"""

    kept = "SELECT * FROM s1 WHERE fate = 'kept'"
    usd = lambda expr: d.float_(f"{expr} / 1000000.0")
    spread_bp = d.float_("AVG((ask - bid) * 10000.0 / ((ask + bid) / 2.0))")
    daily = f"""
        SELECT symbol, {spread_bp} AS spread_bp
        FROM s1 WHERE fate = 'kept' AND bucket = 1 AND (valid OR locked)
        GROUP BY symbol"""
    cells = f"""
        SELECT symbol, bucket, venue, z, q, spread, size_code AS size,
               COUNT(*) AS trades, SUM(size) AS shares, {usd('SUM(dm)')} AS dollars
        FROM ({kept}) k
        GROUP BY symbol, bucket, venue, z, q, spread, size_code"""

    flags = """
               retail AND zt > 6000 AS bjzz_buy, retail AND zt > 0 AND zt < 4000 AS bjzz_sell,
               retail AND zt <> 0 AND valid AND 10 * d > 6 * w AS mid_buy,
               retail AND zt <> 0 AND valid AND 10 * d < 4 * w AS mid_sell,
               valid AND 2 * p > ask + bid AS qr_buy, valid AND 2 * p < ask + bid AS qr_sell"""
    part = "PARTITION BY symbol, bucket, bar"
    s = lambda f, x: f"SUM(CASE WHEN {f} THEN {x} ELSE 0 END)"
    retail_cols = ",\n".join(
        f"{s(r, '1')} AS {r}_trades, {s(r, 'size')} AS {r}_shares, {usd(s(r, 'dm'))} AS {r}_dollars"
        for r in ["bjzz_buy", "bjzz_sell", "mid_buy", "mid_sell"])
    bars_f = f"""
            SELECT symbol, bucket, bar, t, seq, price, size, dm, venue, p, bid, ask, valid,{flags},
                   ROW_NUMBER() OVER ({part} ORDER BY t, seq) AS rn_first,
                   ROW_NUMBER() OVER ({part} ORDER BY t DESC, seq DESC) AS rn_last,
                   ROW_NUMBER() OVER ({part}, valid ORDER BY t DESC, seq DESC) AS rn_last_valid
            FROM ({kept}) k"""
    bars = f"""
        SELECT symbol, bucket, bar,
               MAX(CASE WHEN rn_first = 1 THEN price END) AS open, MAX(price) AS high, MIN(price) AS low,
               MAX(CASE WHEN rn_last = 1 THEN price END) AS close,
               {d.float_('MAX(CASE WHEN valid AND rn_last_valid = 1 THEN (ask + bid) / 2000000.0 END)')} AS last_mid,
               MAX(CASE WHEN rn_last = 1 THEN t END) AS last_trade_time,
               COUNT(*) AS trades, SUM(size) AS shares, {usd('SUM(dm)')} AS dollars,
               {retail_cols},
               {s('venue = 1', '1')} AS off_trades, {s('venue = 1', 'size')} AS off_shares,
               {usd(s('venue = 1', 'dm'))} AS off_dollars,
               {s('venue = 1 AND size <= 99', '1')} AS off_odd_trades, {s('venue = 1 AND size <= 99', 'size')} AS off_odd_shares,
               {s('venue = 1 AND size = 1', '1')} AS off_one_trades,
               {s('qr_buy', 'size')} AS qr_buy_shares, {s('qr_sell', 'size')} AS qr_sell_shares,
               {s('qr_buy AND venue = 0', 'size')} AS exch_qr_buy_shares, {s('qr_sell AND venue = 0', 'size')} AS exch_qr_sell_shares,
               {s('(bjzz_buy OR bjzz_sell) AND (mid_buy OR mid_sell)', '1')} AS both_signed_trades,
               {s('(bjzz_buy AND mid_buy) OR (bjzz_sell AND mid_sell)', '1')} AS same_sign_trades
        FROM f GROUP BY symbol, bucket, bar"""

    raw_dollars = d.float_("SUM(price0 * size0)")
    sum_dollars = d.float_("SUM(price * size)")
    sum_c_dollars = d.float_("SUM(c.price * c.size)")
    after1 = "fate NOT IN ('records', 'cancelled_or_erroneous')"
    audit = f"""
        SELECT 'step' AS section, 'raw' AS item, venue, TRUE AS kept, COUNT(*) AS trades, SUM(size0) AS shares,
               {raw_dollars} AS dollars FROM s1 GROUP BY venue
        UNION ALL
        SELECT 'step', fate, venue, FALSE, COUNT(*), SUM(size0), {raw_dollars} FROM s1
        WHERE fate IN ('records', 'cancelled_or_erroneous') GROUP BY fate, venue
        UNION ALL
        SELECT 'step', 'restored_original', venue, TRUE, COUNT(*), SUM(size0), {raw_dollars} FROM s1
        WHERE action = 'restore' GROUP BY venue
        UNION ALL
        SELECT 'step', fate, venue, fate = 'kept', COUNT(*), SUM(size), {sum_dollars} FROM s1
        WHERE fate IN ('sale_condition', 'price_or_size', 'kept') GROUP BY fate, venue
        UNION ALL
        SELECT 'condition', cond, venue, bucket IS NOT NULL, COUNT(*), SUM(size), {sum_dollars} FROM s1
        WHERE {after1} GROUP BY cond, venue, bucket IS NOT NULL
        UNION ALL
        SELECT 'venue_code', ex, venue, fate = 'kept', COUNT(*), SUM(size), {sum_dollars} FROM s1
        WHERE {after1} GROUP BY ex, venue, fate = 'kept'
        UNION ALL
        SELECT 'ctm', tr_corr, CASE WHEN ex = 'D' THEN 1 ELSE 0 END, FALSE, COUNT(*), SUM(size), {sum_dollars}
        FROM ctm_f GROUP BY tr_corr, CASE WHEN ex = 'D' THEN 1 ELSE 0 END
        UNION ALL
        SELECT 'ctm', 'unmatched', CASE WHEN c.ex = 'D' THEN 1 ELSE 0 END, FALSE, COUNT(*), SUM(c.size),
               {sum_c_dollars}
        FROM ctm_f c WHERE NOT EXISTS (SELECT 1 FROM s1 WHERE {on('s1', 'c')})
        GROUP BY CASE WHEN c.ex = 'D' THEN 1 ELSE 0 END
        UNION ALL
        SELECT 'ctm', 'record_after_cutoff', CASE WHEN ex = 'D' THEN 1 ELSE 0 END, FALSE, COUNT(*), SUM(size),
               {sum_dollars}
        FROM pairs WHERE paired AND NOT record_counts GROUP BY CASE WHEN ex = 'D' THEN 1 ELSE 0 END
        UNION ALL
        SELECT 'ctm', 'unpaired', CASE WHEN ex = 'D' THEN 1 ELSE 0 END, FALSE, COUNT(*), SUM(size), {sum_dollars}
        FROM pairs WHERE NOT paired GROUP BY CASE WHEN ex = 'D' THEN 1 ELSE 0 END"""

    if d.name == "duckdb":
        # DuckDB takes each bar's first and last trade with arg_min and arg_max: one pass, no sorting, unlike the three
        # window functions (Postgres has no arg_min). Trades are ordered by time, then sequence number: one key
        order = "CAST(t AS HUGEINT) * 10000000000000000000 + seq"
        bars_f = f"""
            SELECT symbol, bucket, bar, t, seq, price, size, dm, venue, p, bid, ask, valid,{flags},
                   {order} AS ord
            FROM ({kept}) k"""
        bars = bars.replace("MAX(CASE WHEN rn_first = 1 THEN price END)", "ARG_MIN(price, ord)")
        bars = bars.replace("MAX(CASE WHEN rn_last = 1 THEN price END)", "ARG_MAX(price, ord)")
        bars = bars.replace("MAX(CASE WHEN valid AND rn_last_valid = 1 THEN (ask + bid) / 2000000.0 END)",
                            "ARG_MAX((ask + bid) / 2000000.0, ord) FILTER (WHERE valid)")
        bars = bars.replace("MAX(CASE WHEN rn_last = 1 THEN t END)", "MAX(t)")
        assert "rn_" not in bars

    if temp_tables:
        setup = [f"CREATE TEMP TABLE ctm_f AS {ctm}", f"CREATE TEMP TABLE pairs AS {pairs}", f"CREATE TEMP TABLE s1 AS {s1}"]
        return {"setup": setup, "daily": daily, "cells": cells, "bars": f"WITH f AS ({bars_f}) {bars}", "audit": audit}
    prefix = f"WITH ctm_f AS ({ctm}), pairs AS ({pairs}), s1 AS ({s1})"
    return {"setup": [], "daily": f"{prefix} {daily}", "cells": f"{prefix} {cells}",
            "bars": f"{prefix}, f AS ({bars_f}) {bars}", "audit": f"{prefix} {audit}"}


def day_parameters(date: str, has_nano: bool, cutoff_offset_minutes: int = 10) -> Day:
    """A day's parameters from NYSE's calendar: the open at 09:30, the close (16:00 or 13:00) and the cutoff before it."""
    import exchange_calendars as xcals

    ns = 10**9
    close = xcals.get_calendar("XNYS").session_close(date).tz_convert("America/New_York")
    close_ns = int((close - close.normalize()).total_seconds()) * ns
    return Day(ymd=date.replace("-", ""), open_ns=(9 * 3600 + 30 * 60) * ns, cutoff_ns=close_ns - cutoff_offset_minutes * 60 * ns,
               close_ns=close_ns, has_nano=has_nano)
