import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Robintrack: explore, then clean

    The raw files in `data/raw/robintrack/popularity_export/` are read as they are; nothing here writes to them. Cleaned output goes to `data/intermediate/robintrack/`.
    """)
    return


@app.cell
def _():
    from itertools import groupby

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from dsr.crsp import DAILY_COLUMNS, ONE_TRADING_DAY_SPANS, class_on, is_common_stock, read_name_records, security_type

    return (
        DAILY_COLUMNS,
        ONE_TRADING_DAY_SPANS,
        class_on,
        groupby,
        is_common_stock,
        mo,
        np,
        pd,
        plt,
        read_name_records,
        security_type,
    )


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parents[1] / "data"
    RAW_DIR = DATA_DIR / "raw" / "robintrack" / "popularity_export"
    CRSP_NAMES_PATH = DATA_DIR / "raw" / "crsp" / "stocknames_v2.csv"
    INTERMEDIATE_DIR = DATA_DIR / "intermediate"
    return CRSP_NAMES_PATH, DATA_DIR, INTERMEDIATE_DIR, RAW_DIR


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: inventory

    One row per raw file: size, header, row count, and the first and last timestamp. Timestamps are UTC (the scraper saved `utcnow()`).
    """)
    return


@app.cell(disabled=True)
def _(RAW_DIR, pd):
    def _scan_file(path):
        """Size, header, row count and first/last timestamp of one raw CSV, read as bytes."""
        data = path.read_bytes()
        header, _, body = data.partition(b"\n")
        rows = body.rstrip(b"\n")
        first = rows.split(b"\n", 1)[0] if rows else b""
        last = rows.rsplit(b"\n", 1)[-1] if rows else b""
        return {
            "file": path.name,
            "bytes": len(data),
            "header": header.decode(),
            "n_rows": rows.count(b"\n") + 1 if rows else 0,
            "first_ts": first.split(b",")[0].strip(b'"').decode() or None,
            "last_ts": last.split(b",")[0].strip(b'"').decode() or None,
        }


    inventory = pd.DataFrame([_scan_file(p) for p in sorted(RAW_DIR.iterdir())])
    inventory["first_ts"] = pd.to_datetime(inventory["first_ts"], utc=True)
    inventory["last_ts"] = pd.to_datetime(inventory["last_ts"], utc=True)
    return (inventory,)


@app.cell
def _(inventory, mo):
    _start, _end = inventory["first_ts"].min(), inventory["last_ts"].max()
    mo.md(f"""
    - **Files:** {len(inventory):,}, {inventory["bytes"].sum() / 1e9:.2f} GB
    - **Headers:** {", ".join(f"`{h}` ({n:,} files)" for h, n in inventory["header"].value_counts().items())}
    - **Rows:** {inventory["n_rows"].sum():,} in total; per file median {inventory["n_rows"].median():,.0f}, min {inventory["n_rows"].min():,}, max {inventory["n_rows"].max():,}
    - **Span:** first snapshot {_start}, last snapshot {_end}
    - **Present on the first day ({_start.date()}):** {(inventory["first_ts"].dt.date == _start.date()).sum():,} files
    - **Present on the last day ({_end.date()}):** {(inventory["last_ts"].dt.date == _end.date()).sum():,} files
    """)
    return


@app.cell
def _(inventory, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3.5))
    _axes[0].hist(inventory["n_rows"], bins=80)
    _axes[0].set_title("rows per file")
    _axes[1].hist(inventory["first_ts"].dt.tz_convert(None), bins=80)
    _axes[1].set_title("first snapshot per file")
    _axes[2].hist(inventory["last_ts"].dt.tz_convert(None), bins=80)
    _axes[2].set_title("last snapshot per file")
    for _ax in _axes[1:]:
        _ax.tick_params(axis="x", labelrotation=30)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(inventory, mo):
    mo.ui.table(inventory)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: one series per PERMNO

    Each raw file follows one Robinhood instrument and is named after its ticker at export (August 2020). Every snapshot gets the PERMNO of the security it belongs to:

    1. **Ticker** from the file name (`_OUT.csv` is `OUT`); `BASE.C` means share class C.
    2. **CRSP candidates** from `stocknames_v2`: the ticker with that share class, or the joined form (`MOG.B` is `MOGB` in CRSP since 2014). An undotted ticker that CRSP lists under several classes takes the class without a dotted file of its own (`BH.A` exists, so `BH` is class B).
    3. **Date match:** a snapshot belongs to the latest session that has opened, its New York date or the day before if taken before 09:30 ET, because CRSP ticker changes take effect at the open.
    4. **Two classes in one file:** until the dotted files start on 2020-01-16, 15 base files also hold their sibling class. In a sweep of two rows with different values the higher is the base class; any other row goes to the class whose last value is closer. Those rows are matched as the sibling class.
    5. **One instrument per file:** an instrument can keep its id through a ticker change, so the file name can be a ticker it only got later (`CVLG` was `CVTI` until July 2020; `GOLD` was Barrick's `ABX` in 2018, when `GOLD` was Randgold). Rows up to a file's last date match take the PERMNO most of its matched rows point to, while that security exists in CRSP. Rows after it ends are dropped (`CZR`'s last days after the Caesars merger); rows before it began keep their date match (funds moved into a new trust get a new PERMNO). Rows after the last date match are not filled: an instrument still reporting under its old ticker after a rename is stale (`PAH` after it became `ESI`), and the new ticker's file holds the security.
    6. **Overlap at a rename:** if two files hold the same PERMNO at the same time, the date-matched rows win over rows that rule 5 filled or moved to another PERMNO (the new instrument's first rows can start minutes before the old one's last).
    7. A snapshot left without a PERMNO is dropped and counted in the match report.
    8. **No double counting:** within a PERMNO, rows from different files (or from the split) must not overlap in time.
    """)
    return


@app.cell(disabled=True)
def _(CRSP_NAMES_PATH, pd):
    crsp_names = pd.read_csv(
        CRSP_NAMES_PATH,
        dtype={"ticker": str, "shareclass": str},
        parse_dates=["namedt", "nameenddt", "securitybegdt", "securityenddt"],
    )
    crsp_by_ticker = dict(tuple(crsp_names.groupby("ticker")))
    crsp_life = crsp_names.groupby("permno").agg(begin=("securitybegdt", "min"), end=("securityenddt", "max"))
    return crsp_by_ticker, crsp_life, crsp_names


@app.cell
def _(inventory):
    _reserved = {"CON", "AUX", "LST", "PRN", "NUL", "EOF", "INP", "OUT"}  # the exporter prefixed these with "_"
    rt_keys = inventory[["file"]].copy()
    rt_keys["ticker"] = rt_keys["file"].str.removesuffix(".csv")
    _prefixed = rt_keys["ticker"].str.startswith("_") & rt_keys["ticker"].str[1:].isin(_reserved)
    rt_keys.loc[_prefixed, "ticker"] = rt_keys.loc[_prefixed, "ticker"].str[1:]
    _parts = rt_keys["ticker"].str.split(".", n=1, expand=True)
    rt_keys["base"], rt_keys["cls"] = _parts[0], _parts[1]
    dotted_classes = rt_keys.dropna(subset=["cls"]).groupby("base")["cls"].agg(set).to_dict()
    return dotted_classes, rt_keys


@app.cell
def _(crsp_by_ticker, crsp_life, crsp_names, dotted_classes, groupby, np, pd):
    TWO_CLASS_CUTOFF = pd.Timestamp("2020-01-16")  # the dotted share-class files start here


    def crsp_spells(base, cls):
        """CRSP name spells a Robintrack ticker can refer to."""
        spells = crsp_by_ticker.get(base, crsp_names.iloc[:0])
        if cls is not None:  # CRSP writes a class as ticker + shareclass (BRK, B) or joined (MOGB)
            return pd.concat([spells[spells["shareclass"] == cls], crsp_by_ticker.get(base + cls, crsp_names.iloc[:0])])
        return spells[~spells["shareclass"].isin(dotted_classes.get(base, set()))]


    def session_date(ts):
        """Date of the latest session that has opened at each UTC time: the New York date, or the day before before 09:30."""
        ny = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        before_open = ny.dt.hour * 60 + ny.dt.minute < 9 * 60 + 30
        return ny.dt.tz_localize(None).dt.normalize() - pd.to_timedelta(before_open.astype(int), unit="D")


    def permno_on(dates, spells):
        """PERMNO whose spell contains each date (0 if none or several), and the number of PERMNOs that matched."""
        d = dates.to_numpy()
        permno = np.zeros(len(d), dtype=np.int64)
        n_hits = np.zeros(len(d), dtype=np.int64)
        for p, s in spells.groupby("permno"):
            inside = np.zeros(len(d), dtype=bool)
            for start, end in zip(s["namedt"].to_numpy(), s["nameenddt"].to_numpy()):
                inside |= (d >= start) & (d <= end)
            permno[inside] = p
            n_hits += inside
        permno[n_hits != 1] = 0
        return permno, n_hits


    def sweep_ids(ts):
        """Number the sweeps: rows less than 5 minutes apart belong to one sweep."""
        return (ts.diff().dt.total_seconds() >= 300).cumsum().to_numpy()


    def first_two_class_sweep(ts, values):
        """Start of the first sweep with exactly two rows more than 10% apart, or None."""
        g = pd.DataFrame({"sweep": sweep_ids(ts), "ts": ts, "v": values}).groupby("sweep").agg(
            n=("v", "size"), lo=("v", "min"), hi=("v", "max"), t0=("ts", "min"))
        hit = g.loc[(g["n"] == 2) & (g["hi"] - g["lo"] > 0.1 * g["hi"]), "t0"]
        return hit.min() if len(hit) else None


    def sibling_rows(ts, values, start, end):
        """True for rows of the sibling class in a file that holds two share classes between start and end.

        In a sweep of two rows with different values the higher is the base class. Any other row in the window goes to
        the class whose last value it is closer to.
        """
        v = values.to_numpy()
        sibling = np.zeros(len(v), dtype=bool)
        sweep = sweep_ids(ts)
        window = np.flatnonzero(((ts >= start) & (ts < end)).to_numpy())
        base_last = sib_last = None
        for _, grp in groupby(window, key=lambda i: sweep[i]):
            grp = list(grp)
            if len(grp) == 2 and v[grp[0]] != v[grp[1]]:
                lo, hi = sorted(grp, key=lambda i: v[i])
                sibling[lo] = True
                base_last, sib_last = v[hi], v[lo]
                continue
            for i in grp:
                if sib_last is not None and abs(v[i] - sib_last) < abs(v[i] - base_last):
                    sibling[i], sib_last = True, v[i]
                else:
                    base_last = v[i]
        return sibling


    def by_identity(dates, permno):
        """Fill rows up to the last date-matched one with the PERMNO most matched rows have, while it exists in CRSP.

        Rows after that security ends are dropped (0). Rows after the last date match are not filled: an instrument that
        keeps reporting under its old ticker after a rename is stale, and the new ticker's file holds the security.
        """
        hit = np.flatnonzero(permno != 0)
        if len(hit) == 0:
            return permno
        identity = pd.Series(permno[hit]).mode().iloc[0]
        begin, end = crsp_life.loc[identity, ["begin", "end"]]
        d = dates.to_numpy()
        out = permno.copy()
        out[(np.arange(len(d)) <= hit[-1]) & (d >= begin) & (d <= end)] = identity
        out[d > end] = 0
        return out

    return (
        TWO_CLASS_CUTOFF,
        by_identity,
        crsp_spells,
        first_two_class_sweep,
        permno_on,
        session_date,
        sibling_rows,
    )


@app.cell
def _(
    RAW_DIR,
    TWO_CLASS_CUTOFF,
    by_identity,
    crsp_spells,
    dotted_classes,
    first_two_class_sweep,
    np,
    pd,
    permno_on,
    rt_keys,
    session_date,
    sibling_rows,
):
    _rows, _report = [], []
    for _k in rt_keys.itertuples(index=False):
        _df = pd.read_csv(RAW_DIR / _k.file)
        _ts = pd.to_datetime(_df["timestamp"])
        _v = _df["users_holding"]
        assert (_v == _v.round()).all(), _k.file  # ACB and AHT write some counts as 1385.0
        _v = _v.astype("int64")
        _cls = None if pd.isna(_k.cls) else _k.cls

        # base file with a dotted sibling: split off the sibling's rows before the dotted files start
        _sib = np.zeros(len(_df), dtype=bool)
        _sib_cls = dotted_classes.get(_k.base, set()) if _cls is None else set()
        if _sib_cls:
            _pre = _ts < TWO_CLASS_CUTOFF
            _start = first_two_class_sweep(_ts[_pre], _v[_pre])
            if _start is not None:
                _sib = sibling_rows(_ts, _v, _start, TWO_CLASS_CUTOFF)

        _dates = session_date(_ts)
        _permno = np.zeros(len(_df), dtype=np.int64)
        _hits = np.zeros(len(_df), dtype=np.int64)
        _permno[~_sib], _hits[~_sib] = permno_on(_dates[~_sib], crsp_spells(_k.base, _cls))
        if _sib.any() and len(_sib_cls) == 1:
            _permno[_sib], _hits[_sib] = permno_on(_dates[_sib], crsp_spells(_k.base, next(iter(_sib_cls))))
        _date_matched = _permno != 0
        _matched_to = _permno.copy()  # the date match alone, before rule 5
        for _part in (~_sib, _sib):
            if _part.any():
                _permno[_part] = by_identity(_dates[_part], _permno[_part])

        _ok = _permno != 0
        _filled = _permno != _matched_to  # added or reassigned by rule 5
        _report.append({
            "file": _k.file,
            "ticker": _k.ticker,
            "n_rows": len(_df),
            "n_sibling": int(_sib.sum()),
            "n_date_matched": int(_date_matched.sum()),
            "n_no_permno": int((_hits == 0).sum()),
            "n_ambiguous": int((_hits > 1).sum()),
            "n_added_by_identity": int((_ok & ~_date_matched).sum()),
            "n_reassigned_by_identity": int((_ok & _date_matched & _filled).sum()),
            "n_dropped_after_end": int((~_ok & _date_matched).sum()),
            "n_matched": int(_ok.sum()),
            "permnos": " ".join(str(p) for p in sorted(set(_permno[_ok]))),
        })
        _rows.append(pd.DataFrame({
            "permno": _permno[_ok],
            "timestamp": _df["timestamp"][_ok].to_numpy(),
            "users_holding": _v[_ok].to_numpy(),
            "source_file": _k.file,
            "sibling_split": _sib[_ok],
            "filled": _filled[_ok],
        }))

    rt_by_permno = pd.concat(_rows, ignore_index=True)

    # where two files hold one PERMNO at the same time, date-matched rows win over rows filled or reassigned by file identity
    # (on the day of a rename the new instrument's first rows overlap the old one's last rows by minutes)
    _spans = (
        rt_by_permno[~rt_by_permno["filled"]].groupby(["permno", "source_file"])["timestamp"].agg(["min", "max"]).reset_index()
    )
    _f = rt_by_permno[rt_by_permno["filled"]].reset_index().merge(_spans, on="permno", suffixes=("", "_other"))
    _clash = _f[(_f["source_file"] != _f["source_file_other"]) & _f["timestamp"].between(_f["min"], _f["max"])]
    _clash = _clash.drop_duplicates("index")
    rt_by_permno = rt_by_permno.drop(index=_clash["index"]).reset_index(drop=True)

    match_report = pd.DataFrame(_report)
    match_report["n_dropped_overlap"] = match_report["file"].map(_clash.groupby("source_file").size()).fillna(0).astype(int)
    match_report["n_matched"] -= match_report["n_dropped_overlap"]
    return match_report, rt_by_permno


@app.cell
def _(match_report, mo, rt_by_permno):
    _r = match_report
    _files_per_permno = rt_by_permno.groupby("permno")["source_file"].nunique()
    mo.md(f"""
    - **Rows kept:** {_r["n_matched"].sum():,} of {_r["n_rows"].sum():,} ({_r["n_matched"].sum() / _r["n_rows"].sum():.1%}); {_r["n_sibling"].sum():,} of them split off to a sibling class
    - **Date match:** {_r["n_date_matched"].sum():,} rows; {_r["n_no_permno"].sum():,} had no PERMNO on their date and {_r["n_ambiguous"].sum():,} had several ({", ".join(_r.loc[_r["n_ambiguous"] > 0, "ticker"])})
    - **File identity:** {_r["n_added_by_identity"].sum():,} rows added (history under an earlier ticker), {_r["n_reassigned_by_identity"].sum():,} moved from their date match to the file's own security (`GOLD`'s 2018 rows from Randgold to Barrick), {_r["n_dropped_after_end"].sum():,} dropped after their security ended, {_r["n_dropped_overlap"].sum():,} dropped where another file's date-matched rows hold the same PERMNO
    - **Files with nothing kept:** {(_r["n_matched"] == 0).sum():,} (securities CRSP does not cover, e.g. OTC ADRs and ETNs)
    - **PERMNOs:** {rt_by_permno["permno"].nunique():,}; {(_files_per_permno > 1).sum():,} are fed by more than one file (ticker changes and the class splits)
    """)
    return


@app.cell
def _(mo, rt_by_permno):
    _seg = (
        rt_by_permno.groupby(["permno", "source_file", "sibling_split"])["timestamp"]
        .agg(first="min", last="max", n_rows="size")
        .reset_index()
        .sort_values(["permno", "first"])
    )
    _seg["prev_last"] = _seg.groupby("permno")["last"].transform(lambda s: s.cummax().shift())
    permno_segments = _seg.drop(columns="prev_last").assign(overlaps_earlier=_seg["first"] <= _seg["prev_last"])
    _multi = permno_segments[permno_segments.groupby("permno")["permno"].transform("size") > 1]
    mo.vstack([
        mo.md(f"**Double counting:** {permno_segments['overlaps_earlier'].sum()} segments overlap an earlier segment of the same PERMNO. PERMNOs built from several segments:"),
        mo.ui.table(_multi),
    ])
    return (permno_segments,)


@app.cell
def _(match_report, mo):
    mo.vstack([
        mo.md("**Files with rows left out** (no PERMNO or ambiguous on their date):"),
        mo.ui.table(match_report[match_report["n_matched"] < match_report["n_rows"]].sort_values("n_rows", ascending=False)),
    ])
    return


@app.cell
def _(mo):
    write_button = mo.ui.run_button(label="Write data/intermediate/robintrack/by_permno/")
    write_button
    return (write_button,)


@app.cell
def _(
    INTERMEDIATE_DIR,
    match_report,
    mo,
    permno_segments,
    rt_by_permno,
    write_button,
):
    mo.stop(not write_button.value, mo.md("Press the button above to write one CSV per PERMNO."))
    mo.stop(permno_segments["overlaps_earlier"].any(), mo.md("**Not written:** some segments overlap, see the double-counting table."))
    _out = INTERMEDIATE_DIR / "robintrack" / "by_permno"
    _out.mkdir(parents=True, exist_ok=True)
    for _old in _out.glob("*.csv"):
        _old.unlink()
    for _p, _g in rt_by_permno.groupby("permno"):
        _g.sort_values("timestamp")[["timestamp", "users_holding", "source_file"]].to_csv(_out / f"{_p}.csv", index=False)
    match_report.to_csv(INTERMEDIATE_DIR / "robintrack" / "match_report.csv", index=False)
    mo.md(f"Wrote {rt_by_permno['permno'].nunique():,} files to `data/intermediate/robintrack/by_permno/` and the match report to `data/intermediate/robintrack/match_report.csv`.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # PERMNO view

    Everything below reads only `data/intermediate/robintrack/by_permno/`, the per-PERMNO files written in Step 2, and never uses the cells above. Those are disabled (Step 1's raw scan and the CRSP load, with everything that depends on them) so they don't re-run; enable them again to rebuild the files.
    """)
    return


@app.cell
def _(INTERMEDIATE_DIR):
    PERMNO_DIR = INTERMEDIATE_DIR / "robintrack" / "by_permno"
    return (PERMNO_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    ## P1: inventory by PERMNO

    One row per PERMNO file: size, row count, first and last snapshot (UTC), the raw files it was built from, and CRSP's name, ticker and security type on the last name record up to the end of Robintrack (2020-08-13).
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, PERMNO_DIR, pd):
    def _scan_permno_file(path):
        """Rows, first/last timestamp and source files of one per-PERMNO CSV (rows are in time order)."""
        df = pd.read_csv(path, usecols=["timestamp", "source_file"], dtype=str, engine="pyarrow")
        return {
            "permno": int(path.stem),
            "bytes": path.stat().st_size,
            "n_rows": len(df),
            "first_ts": df["timestamp"].iloc[0],
            "last_ts": df["timestamp"].iloc[-1],
            "n_source_files": df["source_file"].nunique(),
            "source_files": " ".join(df["source_file"].unique()),
        }


    _names = pd.read_csv(
        CRSP_NAMES_PATH,
        dtype=str,
        usecols=["permno", "namedt", "ticker", "issuernm", "securitytype", "sharetype", "primaryexch"],
    )
    _names = (
        _names[_names["namedt"] <= "2020-08-13"]
        .sort_values(["permno", "namedt"])
        .drop_duplicates("permno", keep="last")
        .astype({"permno": int})
        .drop(columns="namedt")
    )
    permno_inventory = pd.DataFrame([_scan_permno_file(p) for p in sorted(PERMNO_DIR.glob("*.csv"))])
    permno_inventory["first_ts"] = pd.to_datetime(permno_inventory["first_ts"], utc=True)
    permno_inventory["last_ts"] = pd.to_datetime(permno_inventory["last_ts"], utc=True)
    permno_inventory = permno_inventory.merge(_names, on="permno", how="left")
    return (permno_inventory,)


@app.cell
def _(mo, permno_inventory):
    _inv = permno_inventory
    _start, _end = _inv["first_ts"].min(), _inv["last_ts"].max()
    _types = ", ".join(f"{k} {v:,}" for k, v in _inv["securitytype"].value_counts(dropna=False).items())
    _shares = ", ".join(f"{k} {v:,}" for k, v in _inv["sharetype"].value_counts(dropna=False).items())
    mo.md(f"""
    - **PERMNOs:** {len(_inv):,} files, {_inv["bytes"].sum() / 1e9:.2f} GB
    - **Rows:** {_inv["n_rows"].sum():,} in total; per PERMNO median {_inv["n_rows"].median():,.0f}, min {_inv["n_rows"].min():,}, max {_inv["n_rows"].max():,}
    - **Span:** first snapshot {_start}, last snapshot {_end}
    - **Present on the first day ({_start.date()}):** {(_inv["first_ts"].dt.date == _start.date()).sum():,}; **on the last day ({_end.date()}):** {(_inv["last_ts"].dt.date == _end.date()).sum():,}
    - **Built from more than one raw file:** {(_inv["n_source_files"] > 1).sum():,} PERMNOs
    - **CRSP security type:** {_types}; **share type:** {_shares}
    """)
    return


@app.cell
def _(permno_inventory, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3.5))
    _axes[0].hist(permno_inventory["n_rows"], bins=80)
    _axes[0].set_title("rows per PERMNO")
    _axes[1].hist(permno_inventory["first_ts"].dt.tz_convert(None), bins=80)
    _axes[1].set_title("first snapshot per PERMNO")
    _axes[2].hist(permno_inventory["last_ts"].dt.tz_convert(None), bins=80)
    _axes[2].set_title("last snapshot per PERMNO")
    for _ax in _axes[1:]:
        _ax.tick_params(axis="x", labelrotation=30)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, permno_inventory):
    mo.ui.table(permno_inventory)
    return


@app.cell
def _(mo, permno_inventory):
    def _fences(s):
        """Tukey fences: 1.5 interquartile ranges beyond the first and third quartiles."""
        q1, q3 = s.quantile([0.25, 0.75])
        return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


    _inv = permno_inventory
    _rows_lo, _rows_hi = _fences(_inv["n_rows"])
    _first_lo, _first_hi = _fences(_inv["first_ts"])
    _last_lo, _last_hi = _fences(_inv["last_ts"])
    _flagged = _inv.assign(
        rows_outlier=~_inv["n_rows"].between(_rows_lo, _rows_hi),
        first_snapshot_outlier=~_inv["first_ts"].between(_first_lo, _first_hi),
        last_snapshot_outlier=~_inv["last_ts"].between(_last_lo, _last_hi),
    )
    _flags = ["rows_outlier", "first_snapshot_outlier", "last_snapshot_outlier"]
    permno_outliers = _flagged.loc[
        _flagged[_flags].any(axis=1),
        ["permno", "ticker", "issuernm", "securitytype", "sharetype", "n_rows", "first_ts", "last_ts", "n_source_files", *_flags],
    ].sort_values(["first_ts", "permno"])
    _combos = (
        permno_outliers[_flags]
        .apply(lambda r: " + ".join(n for n, v in zip(["rows", "first", "last"], r) if v), axis=1)
        .value_counts()
    )
    mo.vstack([
        mo.md(f"""
    **Outliers by Tukey's rule** (more than 1.5 interquartile ranges beyond the quartiles):

    - **Rows:** fences {_rows_lo:,.0f} to {_rows_hi:,.0f}; {(_flagged["n_rows"] < _rows_lo).sum():,} PERMNOs below, {(_flagged["n_rows"] > _rows_hi).sum():,} above
    - **First snapshot:** fences {_first_lo:%Y-%m-%d %H:%M} to {_first_hi:%Y-%m-%d %H:%M}; {(_flagged["first_ts"] < _first_lo).sum():,} before, {(_flagged["first_ts"] > _first_hi).sum():,} after
    - **Last snapshot:** fences {_last_lo:%Y-%m-%d %H:%M} to {_last_hi:%Y-%m-%d %H:%M}; {(_flagged["last_ts"] < _last_lo).sum():,} before, {(_flagged["last_ts"] > _last_hi).sum():,} after
    - **Flagged:** {len(permno_outliers):,} PERMNOs; by combination: {"; ".join(f"{k} {v:,}" for k, v in _combos.items())}
    """),
        mo.ui.table(permno_outliers),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## P2: how much each PERMNO is held

    Per PERMNO, the median and the maximum of `users_holding` over all its snapshots, and how many PERMNOs reach 100, 1,000 and 10,000 holders, for stocks (CRSP `EQTY`) and funds (`FUND`).
    """)
    return


@app.cell
def _(PERMNO_DIR, pd, permno_inventory):
    def _holder_stats(path):
        """Median and maximum holders of one per-PERMNO CSV."""
        v = pd.read_csv(path, usecols=["users_holding"], engine="pyarrow")["users_holding"]
        return {"permno": int(path.stem), "median_holders": v.median(), "max_holders": v.max()}


    permno_holdings = pd.DataFrame([_holder_stats(p) for p in sorted(PERMNO_DIR.glob("*.csv"))]).merge(
        permno_inventory[["permno", "ticker", "issuernm", "securitytype", "sharetype"]], on="permno", how="left"
    )
    return (permno_holdings,)


@app.cell
def _(mo, pd, permno_holdings):
    _h = permno_holdings
    _counts = pd.DataFrame(
        {
            f"{_label} with {_stat.split('_')[0]} holders >= level": [
                (_h.loc[_h["securitytype"] == _type, _stat] >= _t).sum() for _t in (100, 1_000, 10_000)
            ]
            for _stat in ("median_holders", "max_holders")
            for _type, _label in (("EQTY", "stocks"), ("FUND", "funds"))
        },
        index=pd.Index([100, 1_000, 10_000], name="level (holders)"),
    )
    _med = _h.groupby("securitytype")["median_holders"].median()
    mo.vstack([
        mo.md(f"""
    - **PERMNOs:** {(_h["securitytype"] == "EQTY").sum():,} stocks, {(_h["securitytype"] == "FUND").sum():,} funds
    - **Median holders of the typical PERMNO:** stocks {_med["EQTY"]:,.0f}, funds {_med["FUND"]:,.0f}
    - **Never held by anyone (max 0):** {(_h["max_holders"] == 0).sum():,} PERMNOs

    Number of PERMNOs whose median (or maximum) holder count over the sample is at least each level. Every cell counts PERMNOs; it is not a number of holders.
    """),
        _counts.reset_index(),
    ])
    return


@app.cell
def _(np, permno_holdings, plt):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _bins = np.linspace(0, np.log10(permno_holdings["max_holders"].max() + 1), 60)
    for _type in ("EQTY", "FUND"):
        _ax.hist(np.log10(permno_holdings.loc[permno_holdings["securitytype"] == _type, "median_holders"] + 1),
                 bins=_bins, alpha=0.6, label=_type)
    _ax.set_xlabel("log10(median holders + 1)")
    _ax.set_ylabel("PERMNOs")
    _ax.set_title("median Robinhood holders per PERMNO")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, permno_holdings):
    mo.ui.table(permno_holdings.sort_values("median_holders", ascending=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Initial cleaning

    Mechanical fixes to how Robintrack recorded the data, applied to the per-PERMNO series from Step 2 (`data/intermediate/robintrack/by_permno/`). No rule here selects rows by comparing their values with a threshold chosen from the data; filters such as a holdings floor wait for the train/test split. Each step flags rows; nothing is written yet.
    """)
    return


@app.cell
def _(PERMNO_DIR, pd):
    def _read_permno_file(path):
        """One per-PERMNO CSV with its PERMNO as the first column (rows are in time order)."""
        df = pd.read_csv(path, engine="pyarrow", dtype={"timestamp": str, "users_holding": "int64", "source_file": str})
        df.insert(0, "permno", int(path.stem))
        return df


    rt_permno = pd.concat([_read_permno_file(p) for p in sorted(PERMNO_DIR.glob("*.csv"))], ignore_index=True)
    rt_permno["source_file"] = rt_permno["source_file"].astype("category")
    return (rt_permno,)


@app.cell
def _(mo):
    mo.md(r"""
    ## C1: placeholder zeros

    Rows with 0 holders before a PERMNO's first nonzero snapshot are not observations. Most are placeholders for an instrument that has not started trading yet (half end within a day). The long runs include instruments Robintrack read as 0 for a year until 2020-01-16, the day the share-class files start: Alcoa (`AA`) goes from 0 to 2,201 holders that morning, VBI Vaccines (`VBIV`) from 0 to 7,789. These rows are flagged to be dropped; a PERMNO that is 0 throughout has no observation and drops out.
    """)
    return


@app.cell
def _(pd, permno_inventory, rt_permno):
    _nonzero = rt_permno["users_holding"].ne(0)
    c1_placeholder = ~_nonzero.groupby(rt_permno["permno"]).cummax()

    _lead = rt_permno[c1_placeholder].groupby("permno").agg(
        n_leading_zero=("timestamp", "size"), first_ts=("timestamp", "first")
    )
    _first_nonzero = rt_permno[_nonzero].groupby("permno")["timestamp"].first().rename("first_nonzero_ts")
    c1_summary = _lead.join(_first_nonzero, how="left").reset_index()
    c1_summary["first_ts"] = pd.to_datetime(c1_summary["first_ts"], utc=True)
    c1_summary["first_nonzero_ts"] = pd.to_datetime(c1_summary["first_nonzero_ts"], utc=True)
    c1_summary["all_zero"] = c1_summary["first_nonzero_ts"].isna()
    c1_summary["zero_days"] = (c1_summary["first_nonzero_ts"] - c1_summary["first_ts"]).dt.total_seconds() / 86400
    c1_summary = c1_summary.merge(permno_inventory[["permno", "ticker", "issuernm", "securitytype"]], on="permno", how="left")
    return c1_placeholder, c1_summary


@app.cell
def _(c1_placeholder, c1_summary, mo, rt_permno):
    _s = c1_summary
    _d = _s["zero_days"]
    _jan16 = (_s["first_nonzero_ts"].dt.strftime("%Y-%m-%d") == "2020-01-16") & (_d > 30)
    mo.vstack([
        mo.md(f"""
    - **Rows flagged:** {c1_placeholder.sum():,} of {len(rt_permno):,} ({c1_placeholder.mean():.2%})
    - **PERMNOs with leading zeros:** {len(_s):,}; {_s["all_zero"].sum():,} are 0 throughout and drop out entirely
    - **How long the zeros last** (first snapshot to first nonzero): under a day {(_d < 1).sum():,}, 1 to 30 days {((_d >= 1) & (_d <= 30)).sum():,}, over 30 days {(_d > 30).sum():,}
    - **Long zero runs that end on 2020-01-16:** {_jan16.sum():,} PERMNOs
    """),
        mo.ui.table(_s.sort_values("n_leading_zero", ascending=False)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C2: restarts at renames

    Some PERMNOs are built from two raw files. Where a series moves from one file to the next, it moves to another Robinhood instrument, often a new one created for a ticker change that starts at 0 and climbs (UTX ends at 7,301 holders; RTX restarts at 0 on 2020-04-03). Two rules:

    1. Zeros at the start of each later file's stretch are placeholders, as in C1, and are flagged to be dropped.
    2. Every switch is a **break**: a daily change must not be taken across it. The two levels are not stitched together, because a new instrument's climb mixes positions moved over from the old one with new buyers. The breaks are listed in `c2_breaks`.

    For the share-class splits the level carries on (`LEN.csv` to `LEN.B.csv`: 51 to 53), but those switches fall on 2020-01-16, right after the January 7 to 15 data gap, so they are breaks either way.
    """)
    return


@app.cell
def _(c1_placeholder, np, pd, permno_inventory, rt_permno):
    _permno, _file, _v, _ts = (rt_permno[c] for c in ("permno", "source_file", "users_holding", "timestamp"))
    _new_seg = _file.ne(_file.shift()) | _permno.ne(_permno.shift())  # a stretch of rows from one raw file
    _seg = _new_seg.cumsum()
    _later = _new_seg.groupby(_permno).cumsum() > 1  # rows in a PERMNO's second or later stretch
    _nonzero = _v.ne(0)
    c2_placeholder = ~_nonzero.groupby(_seg).cummax() & _later & ~c1_placeholder  # rows C1 already flags are left to C1

    _starts = np.flatnonzero((_new_seg & _later).to_numpy())
    _seg_ids = _seg.to_numpy()[_starts]
    _first_nonzero = _v[_nonzero & _later].groupby(_seg[_nonzero & _later]).first()
    _n_zeros = c2_placeholder.groupby(_seg).sum()
    c2_breaks = pd.DataFrame({
        "permno": _permno.to_numpy()[_starts],
        "from_file": _file.to_numpy()[_starts - 1],
        "to_file": _file.to_numpy()[_starts],
        "last_ts_before": pd.to_datetime(_ts.to_numpy()[_starts - 1], utc=True),
        "first_ts_after": pd.to_datetime(_ts.to_numpy()[_starts], utc=True),
        "holders_before": _v.to_numpy()[_starts - 1],
        "holders_after": _v.to_numpy()[_starts],
        "first_nonzero_after": pd.Series(_seg_ids).map(_first_nonzero).to_numpy(),
        "n_placeholder_zeros": pd.Series(_seg_ids).map(_n_zeros).to_numpy(),
    })
    c2_breaks["gap_hours"] = (c2_breaks["first_ts_after"] - c2_breaks["last_ts_before"]).dt.total_seconds() / 3600
    c2_breaks["after_vs_before"] = c2_breaks["first_nonzero_after"] / c2_breaks["holders_before"].where(c2_breaks["holders_before"] > 0)
    c2_breaks = c2_breaks.merge(permno_inventory[["permno", "ticker", "issuernm", "securitytype"]], on="permno", how="left")
    return c2_breaks, c2_placeholder


@app.cell
def _(c2_breaks, c2_placeholder, mo):
    _b = c2_breaks
    _r = _b["after_vs_before"]
    mo.vstack([
        mo.md(f"""
    - **Breaks (switches to another raw file):** {len(_b):,} in {_b["permno"].nunique():,} PERMNOs
    - **Rows flagged:** {c2_placeholder.sum():,} placeholder zeros at the start of a later stretch, in {(_b["n_placeholder_zeros"] > 0).sum():,} breaks
    - **New stretch starts at 0:** {(_b["holders_after"] == 0).sum():,} breaks
    - **First nonzero after the switch relative to the last value before** (shown, not used as a rule): below half {(_r < 0.5).sum():,}, within 10% {((_r - 1).abs() <= 0.1).sum():,}, other {(((_r >= 0.5) & ((_r - 1).abs() > 0.1)) | _r.isna()).sum():,}
    """),
        mo.ui.table(_b.sort_values("first_ts_after")),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C3: repeated readings

    A full sweep asks Robinhood for about 8,600 instruments, 20 per request with at least a second between requests, so it takes at least 7 minutes and reads each instrument once. Two readings of one PERMNO less than 5 minutes apart therefore come from overlapping or restarted sweeps. Within a stretch from one raw file, every reading followed by another within 5 minutes is flagged, so each cluster keeps only its latest reading. Readings chain, so during the repeated sweeps a cluster can run well past 5 minutes; every flagged row is compared with that last reading. Rows already flagged by C1 or C2 are left to those steps.
    """)
    return


@app.cell
def _(pd, rt_permno):
    rt_ts = pd.to_datetime(rt_permno["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return (rt_ts,)


@app.cell
def _(c1_placeholder, c2_placeholder, np, pd, rt_permno, rt_ts):
    _next_same_stretch = rt_permno["permno"].eq(rt_permno["permno"].shift(-1)) & rt_permno["source_file"].eq(
        rt_permno["source_file"].shift(-1)
    )
    _gap_to_next = (rt_ts.shift(-1) - rt_ts).dt.total_seconds()
    c3_repeat = _next_same_stretch & (_gap_to_next < 300) & ~c1_placeholder & ~c2_placeholder

    # each flagged row gives way to the last reading of its cluster: the next row C3 does not flag (always in the same stretch)
    _flagged = c3_repeat.to_numpy()
    _kept = pd.Series(np.arange(len(rt_permno))).where(~_flagged).bfill().to_numpy()[_flagged].astype("int64")
    _v = rt_permno["users_holding"].to_numpy()
    _ts = rt_ts.to_numpy()
    _gap_to_kept = (_ts[_kept] - _ts[_flagged]) / np.timedelta64(1, "s")
    c3_summary = pd.DataFrame({
        "rows_flagged": [int(_flagged.sum())],
        "permnos_affected": [rt_permno.loc[c3_repeat, "permno"].nunique()],
        "gap_under_10s": [int((_gap_to_kept < 10).sum())],
        "gap_10s_to_1min": [int(((_gap_to_kept >= 10) & (_gap_to_kept < 60)).sum())],
        "gap_1_to_5min": [int(((_gap_to_kept >= 60) & (_gap_to_kept < 300)).sum())],
        "gap_5min_or_more": [int((_gap_to_kept >= 300).sum())],
        "longest_gap_min": [round(float(_gap_to_kept.max()) / 60, 1)],
        "same_value_as_kept": [int((_v[_kept] == _v[_flagged]).sum())],
    })
    return c3_repeat, c3_summary


@app.cell
def _(c3_repeat, c3_summary, mo, rt_permno):
    _c = c3_summary.iloc[0]
    _flagged = rt_permno.loc[c3_repeat, ["permno", "timestamp", "users_holding", "source_file"]]
    mo.vstack([
        mo.md(f"""
    - **Rows flagged:** {_c["rows_flagged"]:,} of {len(rt_permno):,} ({_c["rows_flagged"] / len(rt_permno):.2%}), in {_c["permnos_affected"]:,} PERMNOs
    - **Gap to the reading kept (the last of its cluster):** under 10 seconds {_c["gap_under_10s"]:,}; 10 seconds to 1 minute {_c["gap_10s_to_1min"]:,}; 1 to 5 minutes {_c["gap_1_to_5min"]:,}; 5 minutes or more {_c["gap_5min_or_more"]:,} (longest {_c["longest_gap_min"]} minutes)
    - **Same value as the reading kept:** {_c["same_value_as_kept"]:,} ({_c["same_value_as_kept"] / max(_c["rows_flagged"], 1):.1%})
    """),
        mo.ui.table(_flagged.head(1000)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C4: outages

    Robintrack swept every hour around the clock, so more than 1.5 hours without a reading for any PERMNO means at least one sweep was missed. Each outage is listed with the weekday counts it wipes out: no reading in the 15:40 to 15:50 New York window, which is the reading that becomes the day's count. Holidays are not separated out until there is a trading calendar.

    Why that window: it is the last reading retrieved before the 15:50 market-on-close cutoff, so the day's count is known at the decision. Robintrack's counts lag the time they were read by about 25 to 40 minutes (readings taken as late as 9:55 do not yet show the 9:30 open, and readings at 19:40 show no after-hours trading, which ended at 18:00), so the 15:4x reading shows holdings at about 15:00 to 15:25. Sweeps run between :40 and :59 past the hour; readings retrieved at 15:50 or later are after the cutoff and are not used.

    Nothing is dropped here. Outages are handled in the target: a daily change is taken only between two consecutive trading days that both have a count, so the first day after an outage has a count but no change, and changes resume the day after. The same rule covers a single PERMNO missing a count while the rest are fine. A percentage change whose previous count is 0 is undefined and left missing; that comes from real zero holders, not from outages.
    """)
    return


@app.cell
def _(np, pd, rt_ts):
    _t = np.unique(rt_ts.to_numpy())  # every moment any PERMNO was read
    _big = np.flatnonzero(np.diff(_t) > np.timedelta64(90, "m"))
    c4_outages = pd.DataFrame({
        "last_reading_before": pd.to_datetime(_t[_big]).tz_localize("UTC"),
        "first_reading_after": pd.to_datetime(_t[_big + 1]).tz_localize("UTC"),
    })
    c4_outages["hours"] = (c4_outages["first_reading_after"] - c4_outages["last_reading_before"]).dt.total_seconds() / 3600

    COUNT_WINDOW = (pd.Timedelta(hours=15, minutes=40), pd.Timedelta(hours=15, minutes=50))  # New York time; the reading that becomes the day's count


    def _counts_lost(before, after):
        """Weekdays whose whole count window falls inside the outage."""
        before_ny, after_ny = before.tz_convert("America/New_York"), after.tz_convert("America/New_York")
        days = pd.date_range(before_ny.normalize(), after_ny.normalize(), freq="B")
        return [
            d.strftime("%Y-%m-%d") for d in days
            if before_ny < d + COUNT_WINDOW[0] and after_ny >= d + COUNT_WINDOW[1]
        ]


    c4_outages["weekday_counts_lost"] = [
        _counts_lost(b, a) for b, a in zip(c4_outages["last_reading_before"], c4_outages["first_reading_after"])
    ]
    c4_outages["n_counts_lost"] = c4_outages["weekday_counts_lost"].str.len()

    # Direct check, independent of the outage list: weekdays with no reading at all inside the count window
    _ny = pd.DatetimeIndex(_t).tz_localize("UTC").tz_convert("America/New_York")
    _clock = _ny - _ny.normalize()
    _read = set(_ny[(_clock >= COUNT_WINDOW[0]) & (_clock < COUNT_WINDOW[1])].strftime("%Y-%m-%d"))
    c4_no_count_days = [d for d in pd.bdate_range("2018-05-02", "2020-08-13").strftime("%Y-%m-%d") if d not in _read]
    return COUNT_WINDOW, c4_no_count_days, c4_outages


@app.cell
def _(c4_no_count_days, c4_outages, mo):
    _o = c4_outages
    _in_outages = {d for _days in _o["weekday_counts_lost"] for d in _days}
    _elsewhere = sorted(set(c4_no_count_days) - _in_outages)
    mo.vstack([
        mo.md(f"""
    - **Outages (more than 1.5 hours without any reading):** {len(_o):,}, {_o["hours"].sum():,.0f} hours in total; longest {_o["hours"].max():,.1f} hours
    - **Weekday counts lost:** {_o["n_counts_lost"].sum():,}, in {(_o["n_counts_lost"] > 0).sum():,} outages
    - **Direct check, weekdays with no reading at 15:40 to 15:50:** {len(c4_no_count_days):,}; {"all inside the outages below" if not _elsewhere else "outside any outage: " + ", ".join(_elsewhere)}
    """),
        mo.ui.table(_o.assign(weekday_counts_lost=_o["weekday_counts_lost"].str.join(", ")).sort_values("last_reading_before")),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Corporate events (noted, nothing dropped)

    CRSP's records of events that change a stock's shares, matched to our PERMNOs inside the Robintrack period (2018-05-02 to 2020-08-13). Such events can move Robinhood holder counts without anyone trading: a reverse split or a new CUSIP can make Robinhood switch to a new instrument, and a stock merger turns the target's holders into the acquirer's. They are candidates to exclude from Model 1's target later. Nothing is dropped here.

    Sources, all raw:

    - `stkdistributions`: splits, reverse splits, stock dividends, spin-offs, merger and liquidation payments, rights. Left out: cash dividends, capital gains, return of capital and special cash dividends (cash only, nobody's shares change), and CRSP's share-count observations.
    - `stkdelists`: delistings (merger, exchange, liquidation, dropped by the exchange).
    - `stocknames_v2`: ticker, company name and CUSIP changes, i.e. a new CRSP name record whose field differs from the one before.

    Codes are decoded with CRSP's own `metaflaginfo`. A stock is listed as a **counterparty** when its shares were handed to another stock's holders (the acquirer in a stock merger, the new company in a spin-off), since its holder count can jump that day too.

    Point in time: `declared_date` is the announcement date. CRSP has it only for distributions, and even there not for mergers. Delistings and ticker, name and CUSIP changes carry only the date they took effect.
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, pd, permno_inventory):
    _crsp = CRSP_NAMES_PATH.parent
    _start, _end = "2018-05-02", "2020-08-13"
    _ours = set(permno_inventory["permno"])
    _flags = pd.read_csv(_crsp / "metaflaginfo.csv", dtype=str)
    _desc = dict(zip(_flags["flagtype"] + ":" + _flags["flagvalue"].fillna(""), _flags["flagdesc"]))

    # Distributions: everything except cash payouts (dividends, capital gains, return of capital) and share-count observations
    _DIS_KIND = {
        "STKSPL": "split", "STKDIV": "stock dividend", "SECSO": "spin-off",
        "SECMRG": "merger", "SECMWM": "merger", "SECUM": "merger", "SECSIM": "merger", "UMRGR": "merger",
        "CPM": "merger", "CPSCM": "merger", "CPSCMO": "merger",
        "SECEX": "exchange or reorganization", "SECRSD": "exchange or reorganization",
        "CPEX": "exchange or reorganization", "CPEXCSH": "exchange or reorganization", "CPRCSH": "exchange or reorganization",
        "CPFL": "liquidation", "CPPL": "liquidation", "CPSIL": "liquidation", "SECFL": "liquidation",
        "SECPL": "liquidation", "SECSTL": "liquidation", "ULIQ": "liquidation",
        "SECRD": "rights distribution", "URTSD": "rights distribution",
    }
    _d = pd.read_csv(_crsp / "stkdistributions.csv", dtype=str)
    _d = _d[
        _d["permno"].astype(int).isin(_ours)
        & _d["disexdt"].between(_start, _end)
        & ~_d["distype"].isin(["CD", "CG", "ROC", "SD", "TSOO"])
    ]
    _facshr = _d["disfacshr"].astype(float)
    _dis = pd.DataFrame({
        "permno": _d["permno"].astype(int),
        "event_date": _d["disexdt"],
        "declared_date": _d["disdeclaredt"],
        "kind": _d["disdetailtype"].map(_DIS_KIND).fillna("other security distribution"),
        "detail": ("AD:" + _d["disdetailtype"]).map(_desc),
        "other_permno": _d["dispermno"].astype(float),
        "source": "stkdistributions",
    })
    _is_split = _d["distype"].eq("FRS")
    _dis.loc[_is_split, "detail"] += ", shares x" + (1 + _facshr[_is_split]).round(4).astype(str)
    _dis.loc[_d["disdetailtype"].eq("STKSPL") & _facshr.lt(0), "kind"] = "reverse split"

    # Delistings: CRSP gives the effective date only
    _DEL_KIND = {"MER": "merger", "GEX": "exchange or reorganization", "GLI": "liquidation", "GDR": "delisted (dropped)"}
    _x = pd.read_csv(_crsp / "stkdelists.csv", dtype=str)
    _x = _x[_x["permno"].astype(int).isin(_ours) & _x["delistingdt"].between(_start, _end)]
    _del = pd.DataFrame({
        "permno": _x["permno"].astype(int),
        "event_date": _x["delistingdt"],
        "declared_date": None,
        "kind": _x["delactiontype"].map(_DEL_KIND).fillna("delisted (other)"),
        "detail": "delisted: " + ("CD:" + _x["delactiontype"]).map(_desc)
        + "; reason: " + ("DR:" + _x["delreasontype"].fillna("")).map(_desc)
        + "; paid in: " + ("PT:" + _x["delpaymenttype"].fillna("")).map(_desc).fillna("n/a"),
        "other_permno": _x["delpermno"].astype(float),
        "source": "stkdelists",
    })

    # Ticker, name and CUSIP changes: a new CRSP name record whose field differs from the one before; effective date only
    _n = pd.read_csv(CRSP_NAMES_PATH, dtype=str, usecols=["permno", "namedt", "ticker", "issuernm", "cusip"])
    _n["permno"] = _n["permno"].astype(int)
    _n = _n[_n["permno"].isin(_ours)].sort_values(["permno", "namedt"])
    _prev = _n.groupby("permno")[["ticker", "issuernm", "cusip"]].shift()
    _new = (_n.groupby("permno").cumcount() > 0) & _n["namedt"].between(_start, _end)
    _changes = []
    for _col, _kind in [("ticker", "ticker change"), ("issuernm", "name change"), ("cusip", "CUSIP change")]:
        _m = _new & _n[_col].fillna("").ne(_prev[_col].fillna(""))
        _changes.append(pd.DataFrame({
            "permno": _n.loc[_m, "permno"], "event_date": _n.loc[_m, "namedt"], "declared_date": None, "kind": _kind,
            "detail": _prev.loc[_m, _col].fillna("") + " -> " + _n.loc[_m, _col].fillna(""),
            "other_permno": float("nan"), "source": "stocknames_v2",
        }))

    _own = pd.concat([_dis, _del, *_changes], ignore_index=True)
    _own["affected_as"] = "own event"
    _own["other_permno"] = _own["other_permno"].where(_own["other_permno"].gt(0) & _own["other_permno"].ne(_own["permno"]))

    # Counterparties: a stock whose shares were handed to another stock's holders (acquirer in a stock merger, new spin-off)
    _cp = _own[_own["other_permno"].isin(_ours)].copy()
    _cp["permno"], _cp["other_permno"] = _cp["other_permno"].astype(int), _cp["permno"].astype(float)
    _cp["affected_as"] = "counterparty"

    _ev = pd.concat([_own, _cp], ignore_index=True)
    _ev["event_date"] = pd.to_datetime(_ev["event_date"])
    _ev["declared_date"] = pd.to_datetime(_ev["declared_date"])
    _ev["other_permno"] = _ev["other_permno"].astype("Int64")
    _ev = pd.merge_asof(  # CRSP ticker on the event date
        _ev.sort_values("event_date"),
        _n.assign(namedt=pd.to_datetime(_n["namedt"]))[["permno", "namedt", "ticker"]].sort_values("namedt"),
        left_on="event_date", right_on="namedt", by="permno",
    ).drop(columns="namedt")
    _ev["ticker"] = _ev["ticker"].fillna(_ev["permno"].map(_n.groupby("permno")["ticker"].first()))  # event before the first record
    corporate_events = _ev[
        ["permno", "ticker", "event_date", "declared_date", "kind", "affected_as", "detail", "other_permno", "source"]
    ].sort_values(["permno", "event_date", "kind"]).reset_index(drop=True)
    corporate_event_permnos = (
        corporate_events.groupby("permno")
        .agg(
            ticker=("ticker", "last"),
            n_events=("kind", "size"),
            kinds=("kind", lambda s: ", ".join(sorted(s.unique()))),
            first_event=("event_date", "min"),
            last_event=("event_date", "max"),
        )
        .reset_index()
    )
    return corporate_event_permnos, corporate_events


@app.cell
def _(corporate_event_permnos, corporate_events, mo, permno_inventory):
    _e = corporate_events
    _by_kind = (
        _e.groupby(["kind", "affected_as"])
        .agg(
            events=("permno", "size"),
            permnos=("permno", "nunique"),
            share_with_declared_date=("declared_date", lambda s: round(s.notna().mean(), 2)),
        )
        .reset_index()
        .sort_values("permnos", ascending=False)
    )
    mo.vstack([
        mo.md(f"""
    - **Events:** {len(_e):,} on {_e["permno"].nunique():,} of {len(permno_inventory):,} PERMNOs ({_e["permno"].nunique() / len(permno_inventory):.0%})
    - **With an announcement date:** {_e["declared_date"].notna().sum():,} of {len(_e):,}
    """),
        mo.md("**By kind**"),
        mo.ui.table(_by_kind),
        mo.md("**PERMNOs with events**"),
        mo.ui.table(corporate_event_permnos),
        mo.md("**All events**"),
        mo.ui.table(corporate_events),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C5: drops to zero

    C1 and C2 handle zeros at the start of a series. Here the count falls to 0 later, after real readings. If a reading of 0 comes straight after one with 10 or more holders, that zero and the zeros right after it are flagged to be dropped, and the end of the zeros is a **break** as in C2: no daily change is taken across it.

    Ten or more holders do not all close their positions in the hour between two sweeps, so these zeros are a recording failure and the true count during them is unknown, not 0. Most start the day a corporate event replaced the instrument: Eldorado Gold (`EGO`) reads 7,035 holders, does a reverse split on 2018-12-31, then reads 0 every hour until Robintrack refreshed its instrument list on 2020-01-16. The rest start on dates shared by several stocks, a fault on Robintrack's or Robinhood's side.

    Drops from fewer holders are kept: a drop from 1 to 0 is the last holder selling, and the drops from 2 to 9 are thinly held funds with no corporate event. Any cut-off from 5 to 13 holders flags the same runs.
    """)
    return


@app.cell
def _(
    c1_placeholder,
    c2_placeholder,
    c3_repeat,
    corporate_events,
    np,
    pd,
    permno_inventory,
    rt_permno,
    rt_ts,
):
    ZERO_DROP_MIN_HOLDERS = 10

    _rows = np.flatnonzero(~(c1_placeholder | c2_placeholder | c3_repeat).to_numpy())  # rows still kept after C1 to C3
    _p = rt_permno["permno"].to_numpy()[_rows]
    _v = rt_permno["users_holding"].to_numpy()[_rows]
    _ts = rt_ts.to_numpy()[_rows]
    _file = rt_permno["source_file"].to_numpy()[_rows]
    _same = np.r_[False, _p[1:] == _p[:-1]]
    _prev = np.r_[0, _v[:-1]]
    _start = _same & (_v == 0) & (_prev >= ZERO_DROP_MIN_HOLDERS)

    # a run is its starting zero plus the zeros right after it in the same PERMNO
    _new_group = ~_same | (_v != 0) | _start
    _group_first = np.flatnonzero(_new_group)
    _in_run = _start[_group_first][np.cumsum(_new_group) - 1]
    _flag = np.zeros(len(rt_permno), dtype=bool)
    _flag[_rows[_in_run]] = True
    c5_zero_run = pd.Series(_flag, index=rt_permno.index)

    _starts = np.flatnonzero(_start)
    _group_last = np.flatnonzero(np.r_[_new_group[1:], True])
    _ends = _group_last[np.searchsorted(_group_last, _starts)]
    _after = np.minimum(_ends + 1, len(_p) - 1)
    _recovers = (_ends + 1 < len(_p)) & (_p[_after] == _p[_starts])
    c5_runs = pd.DataFrame({
        "permno": _p[_starts],
        "source_file": _file[_starts],
        "holders_before": _prev[_starts],
        "first_zero_ts": pd.to_datetime(_ts[_starts]).tz_localize("UTC"),
        "last_zero_ts": pd.to_datetime(_ts[_ends]).tz_localize("UTC"),
        "n_zeros": _ends - _starts + 1,
        "holders_after": pd.Series(_v[_after], dtype="Int64").where(_recovers),
        "first_ts_after": pd.to_datetime(np.where(_recovers, _ts[_after], np.datetime64("NaT", "us"))).tz_localize("UTC"),
    })

    # shown, not used as a rule: CRSP corporate events of the same PERMNO within 7 days of the first zero
    _day = c5_runs["first_zero_ts"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    _near = c5_runs[["permno"]].assign(day=_day).reset_index().merge(corporate_events[["permno", "event_date", "kind"]], on="permno")
    _near = _near[(_near["event_date"] - _near["day"]).abs() <= pd.Timedelta(days=7)]
    c5_runs["crsp_events_within_7_days"] = c5_runs.index.map(_near.groupby("index")["kind"].agg(lambda s: ", ".join(sorted(set(s))))).fillna("")
    c5_runs["first_zero_day_ny"] = _day.dt.strftime("%Y-%m-%d")
    c5_runs = c5_runs.merge(permno_inventory[["permno", "ticker", "issuernm", "securitytype"]], on="permno", how="left")

    # every drop to 0 by the level it falls from, flagged or kept
    _drop = _same & (_v == 0) & (_prev > 0)
    c5_drops_by_level = pd.cut(_prev[_drop], [0, 1, ZERO_DROP_MIN_HOLDERS - 1, np.inf], labels=["1", f"2 to {ZERO_DROP_MIN_HOLDERS - 1}", f"{ZERO_DROP_MIN_HOLDERS} or more"]).value_counts().sort_index()
    return c5_drops_by_level, c5_runs, c5_zero_run


@app.cell
def _(c5_drops_by_level, c5_runs, c5_zero_run, mo):
    _r = c5_runs
    _matched = _r["crsp_events_within_7_days"] != ""
    _shared = _r.loc[~_matched, "first_zero_day_ny"].value_counts()
    _d = c5_drops_by_level
    mo.vstack([
        mo.md(f"""
    - **Runs flagged:** {len(_r):,} in {_r["permno"].nunique():,} PERMNOs, {c5_zero_run.sum():,} rows
    - **Come back afterwards:** {_r["holders_after"].notna().sum():,}; stay at 0 until the series ends: {_r["holders_after"].isna().sum():,}
    - **Within 7 days of a CRSP corporate event of the same PERMNO** (shown, not used as a rule): {_matched.sum():,} of {len(_r):,}; the others start on {", ".join(f"{d} ({n})" for d, n in _shared.items())}
    - **All drops to 0 by the level they fall from:** {", ".join(f"from {k}: {v:,}" for k, v in _d.items())}; only the last group is flagged
    """),
        mo.ui.table(_r.sort_values("holders_before", ascending=False)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Checks on the cleaned hourly rows

    Run on the rows T1 and T2 use: everything not flagged by C1, C2, C3 or C5. The hourly rows are not written themselves, but they feed both panels, so the write at the end refuses to run unless these checks pass.
    """)
    return


@app.cell
def _(c1_placeholder, c2_placeholder, c3_repeat, c5_zero_run):
    # every hourly row flagged by C1, C2, C3 or C5; T1 and T2 use only the rest
    rt_dropped = (c1_placeholder | c2_placeholder | c3_repeat | c5_zero_run).to_numpy()
    return (rt_dropped,)


@app.cell
def _(mo, np, pd, rt_dropped, rt_permno, rt_ts):
    _keep = ~rt_dropped
    _perm = rt_permno["permno"].to_numpy()[_keep]
    _ts = rt_ts.to_numpy()[_keep]
    _holders = rt_permno["users_holding"].to_numpy()[_keep]
    _n_missing = int(rt_permno.isna().to_numpy()[_keep].sum())
    _n_nat = int(np.isnat(_ts).sum())
    _start, _end = np.datetime64("2018-05-02"), np.datetime64("2020-08-14")
    _outside = int(((_ts < _start) | (_ts >= _end)).sum())
    _same = _perm[1:] == _perm[:-1]
    _not_increasing = int((_ts[1:][_same] <= _ts[:-1][_same]).sum())
    _is_int = np.issubdtype(_holders.dtype, np.integer)
    _n_negative = int((_holders < 0).sum())
    hourly_checks = pd.DataFrame([
        {"check": "no missing values in any column", "passed": _n_missing == 0, "detail": f"{_n_missing:,} missing"},
        {"check": "every timestamp parses", "passed": _n_nat == 0, "detail": f"{_n_nat:,} unparsed"},
        {"check": "timestamps within 2018-05-02 to 2020-08-13", "passed": _outside == 0,
         "detail": f"{_outside:,} outside; first {_ts.min()}, last {_ts.max()}"},
        {"check": "timestamps strictly increase within each PERMNO", "passed": _not_increasing == 0,
         "detail": f"{_not_increasing:,} out of order or repeated"},
        {"check": "holders are whole numbers and none negative", "passed": _is_int and _n_negative == 0,
         "detail": f"dtype {_holders.dtype}, {_n_negative:,} negative, min {_holders.min():,}"},
    ])
    mo.vstack([
        mo.md(f"**{int(hourly_checks['passed'].sum())} of {len(hourly_checks)} checks pass** on {_keep.sum():,} rows to be written."),
        mo.ui.table(hourly_checks),
    ])
    return (hourly_checks,)


@app.cell
def _(mo):
    mo.md(r"""
    ## T1: daily counts

    One number per PERMNO per trading day: the first reading retrieved between 15:40 and 15:50 New York time (C4 explains the window), from the rows kept after C1 to C5. No reading in the window means no count that day; there is no fallback to an earlier sweep. Readings on weekends and holidays are not used; the trading days come from CRSP's daily index file (`dsi.csv`).

    Each count carries its **segment**. A new segment starts at every switch to another raw file (C2) and after every run of zeros (C5), and T2 never takes a change across two segments.
    """)
    return


@app.cell
def _(CRSP_NAMES_PATH, pd):
    ALL_TRADING_DAYS = pd.DatetimeIndex(
        pd.read_csv(CRSP_NAMES_PATH.parent / "dsi.csv", usecols=["date"], parse_dates=["date"])["date"]
    ).sort_values()
    TRADING_DAYS = ALL_TRADING_DAYS[(ALL_TRADING_DAYS >= "2018-05-02") & (ALL_TRADING_DAYS <= "2020-08-13")]
    return ALL_TRADING_DAYS, TRADING_DAYS


@app.cell
def _(
    COUNT_WINDOW,
    TRADING_DAYS,
    c5_zero_run,
    np,
    pd,
    rt_dropped,
    rt_permno,
    rt_ts,
):
    _drop = rt_dropped
    _permno = rt_permno["permno"].to_numpy()
    _file = rt_permno["source_file"].cat.codes.to_numpy()
    _c5 = c5_zero_run.to_numpy()

    # segments: a new one starts with each PERMNO, at each switch to another raw file (C2) and after each zero run (C5)
    _new_segment = np.r_[True, (_permno[1:] != _permno[:-1]) | (_file[1:] != _file[:-1]) | (_c5[:-1] & ~_c5[1:])]
    _segment = np.cumsum(_new_segment)

    # readings in the count window; prefilter on the UTC clock (New York is UTC-4 in summer, UTC-5 in winter)
    _sod = rt_ts.to_numpy().astype("datetime64[s]").astype(np.int64) % 86400
    _lo, _hi = (int(w.total_seconds()) for w in COUNT_WINDOW)
    _near = np.zeros(len(_sod), dtype=bool)
    for _h in (4, 5):
        _near |= (_sod - _h * 3600 >= _lo) & (_sod - _h * 3600 < _hi)
    _cand = np.flatnonzero(~_drop & _near)
    _ny = pd.DatetimeIndex(rt_ts.to_numpy()[_cand]).tz_localize("UTC").tz_convert("America/New_York")
    _clock = _ny - _ny.normalize()
    _in = (_clock >= COUNT_WINDOW[0]) & (_clock < COUNT_WINDOW[1])
    _idx = _cand[_in]
    _window = pd.DataFrame({
        "permno": _permno[_idx],
        "date": _ny[_in].tz_localize(None).normalize(),
        "count_ts": _ny[_in].tz_convert("UTC"),
        "holders": rt_permno["users_holding"].to_numpy()[_idx],
        "segment": _segment[_idx],
    })
    _trading = _window["date"].isin(TRADING_DAYS)
    # rows are in time order within each PERMNO, so the first row of a PERMNO-day is its first reading in the window
    t1_count = _window[_trading].drop_duplicates(["permno", "date"], keep="first").reset_index(drop=True)
    t1_summary = pd.DataFrame({
        "readings_in_window": [len(_window)],
        "readings_on_non_trading_days": [int((~_trading).sum())],
        "permno_days_with_count": [len(t1_count)],
        "permno_days_with_2plus_readings": [int(_window[_trading].groupby(["permno", "date"]).size().gt(1).sum())],
        "permnos_with_a_count": [t1_count["permno"].nunique()],
        "trading_days": [len(TRADING_DAYS)],
        "trading_days_without_any_count": [len(TRADING_DAYS.difference(pd.DatetimeIndex(t1_count["date"].unique())))],
    })
    return t1_count, t1_summary


@app.cell
def _(TRADING_DAYS, mo, plt, t1_count, t1_summary):
    _s = t1_summary.iloc[0]
    _per_day = t1_count.groupby("date").size().reindex(TRADING_DAYS, fill_value=0)
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.plot(_per_day.index, _per_day.to_numpy(), lw=0.8)
    _ax.set_ylabel("PERMNOs with a count")
    _ax.set_title("PERMNOs with a daily count, by trading day")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    - **Daily counts:** {_s["permno_days_with_count"]:,} PERMNO-days, {_s["permnos_with_a_count"]:,} PERMNOs, {_s["trading_days"]:,} trading days
    - **Trading days with no count for any PERMNO:** {_s["trading_days_without_any_count"]:,} (the outages in C4)
    - **PERMNO-days with two or more readings in the window:** {_s["permno_days_with_2plus_readings"]:,}; the first is used
    - **Window readings on weekends and holidays, not used:** {_s["readings_on_non_trading_days"]:,}
    """),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## T2a: delivery days (merger acquirers and spin-offs)

    Two corporate events move holder counts without anyone trading:

    - **Stock merger, the acquirer:** the target's holders are converted into the acquirer's. On train its holders rise roughly one for one with the target's last count over the following days.
    - **Spin-off, the new company:** its shares are handed to the parent's holders.

    These are the CRSP events where this stock is the **counterparty**: another stock's holders received its shares. Trading days 0 to 5 from the event date are marked as delivery days and get no daily change in T2, so they are in neither Model 1's target nor the holder features. The event date is known only afterwards. That is fine here, because the marking only removes rows from a label and its history; it is never an input. The spin-off parent is kept: its holders rising with its price drop is buying.
    """)
    return


@app.cell
def _(TRADING_DAYS, corporate_events, np, pd):
    DELIVERY_DAYS = 5  # trading days after the event, inclusive of day 0

    _ev = corporate_events[corporate_events["kind"].isin(["merger", "spin-off"]) & corporate_events["affected_as"].eq("counterparty")]
    _start = TRADING_DAYS.searchsorted(_ev["event_date"])  # first trading day on or after the event
    delivery_days = pd.DataFrame({
        "permno": np.repeat(_ev["permno"].to_numpy(), DELIVERY_DAYS + 1),
        "pos": (_start[:, None] + np.arange(DELIVERY_DAYS + 1)).ravel(),
        "kind": np.repeat(_ev["kind"].to_numpy(), DELIVERY_DAYS + 1),
    })
    delivery_days = delivery_days[delivery_days["pos"] < len(TRADING_DAYS)]
    delivery_days = (
        delivery_days.assign(date=TRADING_DAYS[delivery_days["pos"].to_numpy()])
        .drop(columns="pos").drop_duplicates(["permno", "date"]).reset_index(drop=True)
    )
    return DELIVERY_DAYS, delivery_days


@app.cell
def _(mo):
    mo.md(r"""
    ## T2: daily changes

    The change in holders from the previous trading day's count to today's. It is taken only when both counts exist, they are in the same segment, the previous count is above 0 and the day is not a delivery day (T2a); otherwise it is missing and `status` says why.

    Everything here is mechanical. The choices made by looking at the data (the target's units, holder floors, winsorizing, which stocks to demean across, and the corporate-event exclusions) wait for the train/test split. The RTX restart shows why the corporate-event exclusion is needed: after the break on 2020-04-03 the new instrument climbs 2,323, 4,561, 6,575, 8,201 holders, positions moved over from UTX rather than buying.
    """)
    return


@app.cell
def _(TRADING_DAYS, delivery_days, np, pd, t1_count):
    _c = t1_count.sort_values(["permno", "date"]).reset_index(drop=True)
    _delivery = pd.MultiIndex.from_frame(_c[["permno", "date"]]).isin(pd.MultiIndex.from_frame(delivery_days[["permno", "date"]]))
    _day_no = pd.Series(TRADING_DAYS.get_indexer(_c["date"]))
    _same_permno = _c["permno"].eq(_c["permno"].shift())
    _prev = _c["holders"].shift()
    _status = np.select(
        [
            ~_same_permno,
            _day_no.diff().ne(1),
            _c["segment"].ne(_c["segment"].shift()),
            _prev.eq(0),
            _delivery,
        ],
        [
            "first count of the PERMNO",
            "previous trading day has no count",
            "break since the previous count",
            "previous count is 0",
            "delivery day (merger acquirer or spin-off)",
        ],
        default="ok",
    )
    _ok = _status == "ok"
    t2_change = _c.assign(
        holders_prev=_prev.where(_ok).astype("Int64"),
        change=(_c["holders"] - _prev).where(_ok).astype("Int64"),
        status=_status,
    )
    return (t2_change,)


@app.cell
def _(mo, t2_change):
    _st = t2_change["status"].value_counts().rename_axis("status").reset_index(name="permno_days")
    _st["share"] = (_st["permno_days"] / len(t2_change)).round(4)
    mo.vstack([
        mo.md(f"**Daily changes:** {int((t2_change['status'] == 'ok').sum()):,} of {len(t2_change):,} PERMNO-days with a count"),
        mo.ui.table(_st),
    ])
    return


@app.cell
def _(t2_change):
    # the % change in holders from the previous count to day t's, where the change is usable; the return panel's features use it
    t2_pct = (t2_change["change"] / t2_change["holders_prev"]).astype(float).where(t2_change["status"].eq("ok"))
    return (t2_pct,)


@app.cell
def _(mo):
    mo.md(r"""
    ## CRSP daily data and classification

    CRSP's daily stock data (`stkdlysecuritydata`) for the PERMNOs in T2, on the Robintrack days and the trading day on each side of them, and its name records (`stocknames_v2`), which classify each security on each day. Both panels join them on PERMNO and date, and name every column by when it is known:

    - **`*_prev`**: the previous trading day (return, price, open, high, low, close, volume, dollar volume, market cap). Known before the 15:50 cutoff on day t, so these can be inputs.
    - **`*_t`**: day t itself. Known only after day t's 16:00 close, so they describe the day and are never inputs to a decision at 15:50 on day t. What happens on day t up to 15:50 will come from TAQ.
    - **`ret_next`**: the total return from day t's close to the next trading day's close, which is what a market-on-close order sent before 15:50 earns. This is the return to predict. It is kept only when CRSP's return covers one trading day (`ret_next_span` D1 to D4, DU, or DD for a delisting return); it is left missing when the return spans several trading days (P) or is missing.
    - **Classification:** the CRSP name record in force on day t (`crsp_class_on`), so a security is classified with what CRSP says about it that day.

    Prices are as traded, not split-adjusted: CRSP's adjustment factors are computed backward from today and would carry later splits back in time. When a stock did not trade, CRSP's price is the bid-ask midpoint, marked `BA` in `price_flag_*`.
    """)
    return


@app.cell
def _(ALL_TRADING_DAYS, CRSP_NAMES_PATH, DAILY_COLUMNS, TRADING_DAYS, pd, t2_change):
    # the file runs from 2013 (for the TAQ sample); the panels only join the Robintrack period and the trading day on
    # each side of it (the previous-day and next-day columns)
    _first = ALL_TRADING_DAYS[ALL_TRADING_DAYS.get_loc(TRADING_DAYS[0]) - 1]
    _last = ALL_TRADING_DAYS[ALL_TRADING_DAYS.get_loc(TRADING_DAYS[-1]) + 1]
    crsp_daily = pd.read_csv(
        CRSP_NAMES_PATH.parent / "stkdlysecuritydata_from_2013.csv", usecols=["permno", "dlycaldt", *DAILY_COLUMNS],
        engine="pyarrow",
    ).rename(columns={"dlycaldt": "date", **DAILY_COLUMNS})
    crsp_daily["date"] = pd.to_datetime(crsp_daily["date"])
    crsp_daily = crsp_daily[crsp_daily["permno"].isin(set(t2_change["permno"])) & crsp_daily["date"].between(_first, _last)]
    return (crsp_daily,)


@app.cell
def _(CRSP_NAMES_PATH, class_on, read_name_records):
    crsp_name_records = read_name_records(CRSP_NAMES_PATH)


    def crsp_class_on(df):
        """The CRSP name record in force on each row's date, aligned to df's index; missing when no record covers the date."""
        return class_on(df, crsp_name_records)

    return (crsp_class_on,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Two panels

    Everything above is shared. Two panels come out of it, built separately below:

    - **Holder panel** (Model 1's target): one row per PERMNO-day with a usable change in holders. Its label is day t's change, so it needs day t's count.
    - **Return panel** (the return tests in 02): one row per stock and day t in the universe known at 15:50 on day t. The decision is made at 15:50, so which stocks are in it can only use what was known then; it uses day t-1's count.

    Both use the same thresholds, from the literature (see the holder panel's sample filters): a CRSP price of at least $1 (Eaton et al.) and at least 100 Robinhood holders (Barber et al.) at day t-1's count. Both use the same train/test split.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Train/test split

    From here on, everything chosen by looking at values (the target's units, holder floors, winsorizing, which stocks to demean across) is decided on the training period only, then applied unchanged to the test period.

    - **Train:** trading days 2018-05-02 to 2019-12-11.
    - **Test:** trading days 2019-12-12 to 2020-08-13. It starts the day Robinhood launched fractional shares, so it checks whether Model 1 survives that break; it also holds the January 2020 outage and the COVID period.

    Changes are assigned by their date: the change on 2019-12-12 (from the 2019-12-11 count) is a test row. The return panel also drops training rows whose next-day return ends on or after the split, so no training label reaches into the test period. Test is only counted. No cell looks at its values until the training-period choices are frozen.
    """)
    return


@app.cell
def _(pd):
    PRICE_MIN, HOLDERS_MIN = 1.0, 100  # at day t-1's close: a CRSP price of at least $1, at least 100 Robinhood holders
    SPLIT_DATE = pd.Timestamp("2019-12-12")  # first test day: Robinhood's fractional shares launch
    return HOLDERS_MIN, PRICE_MIN, SPLIT_DATE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Holder panel (Model 1's target)

    One row per PERMNO-day with a usable change in holders (T2), joined to CRSP, flagged by the sample filters, split into train and test, and given Model 1's target. Only the `in_target` rows are written, as `holders_train` and `holders_test`.

    ## CRSP join
    """)
    return


@app.cell
def _(ALL_TRADING_DAYS, ONE_TRADING_DAY_SPANS, crsp_daily, t2_change):
    _pos = ALL_TRADING_DAYS.get_indexer(t2_change["date"])
    holder_daily = (
        t2_change.assign(prev_day=ALL_TRADING_DAYS[_pos - 1], next_day=ALL_TRADING_DAYS[_pos + 1])
        .merge(crsp_daily.add_suffix("_prev").rename(columns={"permno_prev": "permno", "date_prev": "prev_day"}), on=["permno", "prev_day"], how="left")
        .merge(crsp_daily.add_suffix("_t").rename(columns={"permno_t": "permno", "date_t": "date"}), on=["permno", "date"], how="left")
        .merge(
            crsp_daily[["permno", "date", "ret", "ret_span"]].rename(columns={"date": "next_day", "ret": "ret_next", "ret_span": "ret_next_span"}),
            on=["permno", "next_day"], how="left",
        )
    )
    holder_daily["ret_next"] = holder_daily["ret_next"].where(holder_daily["ret_next_span"].str.fullmatch(ONE_TRADING_DAY_SPANS).fillna(False))
    return (holder_daily,)


@app.cell
def _(holder_daily, mo, pd):
    _n = len(holder_daily)
    _span = holder_daily["ret_next_span"]
    _rows = [
        ("PERMNO-days (all T2 rows)", _n),
        ("with a CRSP row on day t", int(holder_daily["delisted_t"].notna().sum())),
        ("with a CRSP row on the previous trading day", int(holder_daily["delisted_prev"].notna().sum())),
        ("ret_next available (one trading day)", int(holder_daily["ret_next"].notna().sum())),
        ("ret_next missing: no CRSP row the next day", int(_span.isna().sum())),
        ("ret_next missing: spans several trading days", int(_span.str.startswith("P").fillna(False).sum())),
        ("ret_next missing: CRSP has no return (MR)", int(_span.eq("MR").fillna(False).sum())),
        ("day t price is a bid-ask midpoint (BA)", int(holder_daily["price_flag_t"].eq("BA").fillna(False).sum())),
    ]
    mo.vstack([
        mo.md("**Coverage of the join** (counts only)"),
        mo.ui.table(pd.DataFrame(_rows, columns=["", "permno_days"]).assign(share=lambda d: (d["permno_days"] / _n).round(4))),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sample filters from the literature

    Each filter is applied row by row with what was known before the 15:50 cutoff on day t, so a stock is never dropped for its whole history because of something that happens later. Rows are flagged, not deleted; `in_sample` marks the holder-panel rows used from here on. The return panel applies the price and holder thresholds itself, from day t-1 information.

    | Column | Rule | From | Point-in-time form |
    |---|---|---|---|
    | `price_ok` | price of at least $1 | Eaton et al. (the price on day t-1) | CRSP price on the previous trading day |
    | `holders_ok` | at least 100 Robinhood holders | Barber et al. (`users_close(t-1) >= 100`); Eaton et al. use 50 | count on the previous trading day |
    | `is_common` | common stock, CRSP share code 10 or 11 | Ardia et al., Da, Fang & Lin, Eaton et al. | CRSP's classification on day t: equity, common, ordinary shares, US-incorporated, not a REIT |
    | `major_exchange` | NYSE, NYSE American or Nasdaq | Da, Fang & Lin (exchange codes 1 to 3) | CRSP's primary exchange on day t |

    `in_sample` applies the price filter only. The holder, common-stock and exchange columns are recorded but not applied: the holder floor is on hold, and funds stay in for now with `is_common` as the dummy.

    Not adopted, because they decide on a stock's whole history using later information: dropping stocks with a gap of more than 7 trading days or a series that never changes (Ardia et al.), dropping every stock that split (Da, Fang & Lin), and a price screen at one fixed date (Da, Fang & Lin, end of November 2019). Winsorizing (1% or 0.1% tails) waits for the target's units and is set on the training period.
    """)
    return


@app.cell
def _(HOLDERS_MIN, PRICE_MIN, crsp_class_on, holder_daily, is_common_stock):
    _cls = crsp_class_on(holder_daily)
    holder_panel = holder_daily.assign(
        price_ok=holder_daily["price_prev"].ge(PRICE_MIN),
        holders_ok=holder_daily["holders_prev"].ge(HOLDERS_MIN).fillna(False).astype(bool),
        is_common=is_common_stock(_cls),
        major_exchange=_cls["primaryexch"].isin(["N", "A", "Q"]),
    )
    holder_panel["in_sample"] = holder_panel["price_ok"]  # the holder floor is applied later, in Model 1's target
    return (holder_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Holder panel: train and test

    Rows are assigned by the date of the change.
    """)
    return


@app.cell
def _(SPLIT_DATE, holder_panel):
    holder_panel_train = holder_panel[holder_panel["date"] < SPLIT_DATE].reset_index(drop=True)
    holder_panel_test = holder_panel[holder_panel["date"] >= SPLIT_DATE].reset_index(drop=True)
    return holder_panel_test, holder_panel_train


@app.cell
def _(SPLIT_DATE, TRADING_DAYS, holder_panel_test, holder_panel_train, mo, pd):
    _tr_days, _te_days = TRADING_DAYS[TRADING_DAYS < SPLIT_DATE], TRADING_DAYS[TRADING_DAYS >= SPLIT_DATE]
    _sizes = pd.DataFrame([
        {"set": _set, "rows": len(_df), "permnos": _df["permno"].nunique(), "first_day": _df["date"].min().date(), "last_day": _df["date"].max().date()}
        for _set, _df in (("train", holder_panel_train), ("test", holder_panel_test))
    ])
    mo.vstack([
        mo.md(f"**Sizes only** (no values are summarized for test). Train has {len(_tr_days)} trading days, test {len(_te_days)}."),
        mo.ui.table(_sizes, selection=None),
    ])
    return


@app.cell
def _(holder_panel_train, mo, pd):
    _ok = holder_panel_train["status"].eq("ok")
    _in = _ok & holder_panel_train["in_sample"]
    _rows = [
        {"filter": _c, "usable_changes_passing": int((_ok & holder_panel_train[_c]).sum()), "share": round(float(holder_panel_train.loc[_ok, _c].mean()), 4)}
        for _c in ("price_ok", "holders_ok", "is_common", "major_exchange", "in_sample")
    ]
    mo.vstack([
        mo.md(f"""
    **Sample filters on train.** Usable changes: {int(_ok.sum()):,}. In sample (price filter): {int(_in.sum()):,} on {holder_panel_train.loc[_in, "permno"].nunique():,} PERMNOs, a median of {int(holder_panel_train[_in].groupby("date").size().median()):,} per trading day.
    """),
        mo.ui.table(pd.DataFrame(_rows)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Distributions on train

    In-sample training rows (CRSP price of at least $1 the day before). Two variables:

    - **Return:** `ret_next`, the total return from day t's close to the next trading day's close.
    - **Change in Robinhood holders:** `change / holders_prev`, the percentage change from the previous trading day's count to day t's, on rows whose change is usable (T2 status `ok`, so the previous count is above 0).

    Left: the middle 98% of each distribution. Right: the tails, as the share of observations at least as far out as each value, on log scales (a straight line means a power-law tail).
    """)
    return


@app.cell
def _(holder_panel_train, pd):
    _in = holder_panel_train["in_sample"]
    _pct = (holder_panel_train["change"] / holder_panel_train["holders_prev"]).astype(float).where(holder_panel_train["status"].eq("ok"))
    dist_train = pd.DataFrame({"return": holder_panel_train.loc[_in, "ret_next"], "holders_pct_change": _pct[_in]})


    def _stats(s):
        s = s.dropna()
        q = s.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
        return pd.Series({
            "count": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "skew": s.skew(),
            "excess_kurtosis": s.kurt(),
            "min": s.min(),
            **{f"p{100 * k:g}": v for k, v in q.items()},
            "max": s.max(),
            "share_exactly_0": (s == 0).mean(),
        })


    dist_summary = pd.DataFrame({c: _stats(dist_train[c]) for c in dist_train})
    return dist_summary, dist_train


@app.cell
def _(dist_train, np, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 7))
    _labels = {"return": "Return, close t to close t+1", "holders_pct_change": "Change in Robinhood holders"}
    for _row, (_col, _label) in enumerate(_labels.items()):
        _s = dist_train[_col].dropna()
        _lo, _hi = _s.quantile([0.01, 0.99])
        _ax = _axes[_row, 0]
        _ax.hist(_s[_s.between(_lo, _hi)], bins=200)
        _ax.set_title(f"{_label}: middle 98%")
        _ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        _ax.set_ylabel("PERMNO-days")
        _ax = _axes[_row, 1]
        for _side, _v in (("up", _s[_s > 0].to_numpy()), ("down", -_s[_s < 0].to_numpy())):
            _v = np.sort(_v)
            _keep = np.unique(np.geomspace(1, len(_v), 3000).astype(int)) - 1  # thin the points evenly on the log scale
            _ax.loglog(_v[_keep], (len(_v) - _keep) / len(_s), label=_side)
        _ax.set_title(f"{_label}: tails")
        _ax.set_xlabel("size of the move (1 = 100%)")
        _ax.set_ylabel("share at least this far out")
        _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(dist_summary, mo):
    mo.vstack([
        mo.md("**Summary statistics on train** (in-sample rows)"),
        mo.ui.table(dist_summary.reset_index(names="statistic").round(6), selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Holder floors and the target's units (train)

    In-sample training rows with a usable change, from the previous trading day's count $h_0$ to day t's $h_1$, in three units:

    - **% change:** $h_1 / h_0 - 1$
    - **log change:** $\log(h_1 / h_0)$; undefined when $h_1 = 0$, so those rows drop out of this unit
    - **change in log(1 + holders):** $\log(1 + h_1) - \log(1 + h_0)$; defined everywhere

    Each is shown under floors on the previous count ($h_0 \geq$ floor), the point-in-time form of Barber et al.'s `users_close(t-1) >= 100`. A floor of 1 is no floor: a change already needs a previous count above 0. Kurtosis is unstable with tails this heavy, so the table also gives it after clipping the 0.1% tails.
    """)
    return


@app.cell
def _(holder_panel_train, np, pd):
    HOLDER_FLOORS = [1, 10, 50, 100, 500, 1000]

    _t = holder_panel_train[holder_panel_train["in_sample"] & holder_panel_train["status"].eq("ok")]
    _h0 = _t["holders_prev"].astype(float)
    _h1 = _t["holders"].astype(float)
    units_train = pd.DataFrame({
        "date": _t["date"],
        "holders_prev": _h0,
        "pct_change": _h1 / _h0 - 1,
        "log_change": np.log(_h1.where(_h1 > 0) / _h0),
        "log1p_change": np.log1p(_h1) - np.log1p(_h0),
    })


    def _stats(s):
        s = s.dropna()
        lo, hi = s.quantile([0.001, 0.999])
        q = s.quantile([0.001, 0.01, 0.5, 0.99, 0.999])
        return {
            "count": len(s),
            "std": s.std(),
            "skew": s.skew(),
            "excess_kurtosis": s.kurt(),
            "excess_kurtosis_clip_0.1%": s.clip(lo, hi).kurt(),
            **{f"p{100 * k:g}": v for k, v in q.items()},
            "min": s.min(),
            "max": s.max(),
            "share_exactly_0": (s == 0).mean(),
        }


    floor_summary = pd.DataFrame([
        {
            "unit": _u,
            "floor": _f,
            "permnos_per_day": int(units_train[units_train["holders_prev"] >= _f].groupby("date").size().median()),
            **_stats(units_train.loc[units_train["holders_prev"] >= _f, _u]),
        }
        for _u in ("pct_change", "log_change", "log1p_change")
        for _f in HOLDER_FLOORS
    ])
    return floor_summary, units_train


@app.cell
def _(np, plt, units_train):
    _units = {"pct_change": "% change", "log_change": "log change", "log1p_change": "change in log(1 + holders)"}
    _show = [1, 50, 100, 1000]
    _fig, _axes = plt.subplots(3, 2, figsize=(12, 11))
    for _row, (_u, _label) in enumerate(_units.items()):
        _lo, _hi = units_train[_u].quantile([0.01, 0.99])
        _bins = np.linspace(_lo, _hi, 151)
        for _f in _show:
            _s = units_train.loc[units_train["holders_prev"] >= _f, _u].dropna()
            _axes[_row, 0].hist(_s[_s.between(_lo, _hi)], bins=_bins, density=True, histtype="step", log=True, label=f"floor {_f}")
            _x = np.sort(_s.abs().to_numpy())
            _x = _x[_x > 0]
            _keep = np.unique(np.geomspace(1, len(_x), 2000).astype(int)) - 1
            _axes[_row, 1].loglog(_x[_keep], (len(_x) - _keep) / len(_s), label=f"floor {_f}")
        _axes[_row, 0].set_title(f"{_label}: middle 98% without a floor (density, log scale)")
        _axes[_row, 1].set_title(f"{_label}: tails of the size of the move")
        _axes[_row, 1].set_ylabel("share at least this far out")
        for _ax in _axes[_row]:
            _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(floor_summary, mo):
    mo.vstack([
        mo.md("**By unit and floor** (train, in-sample rows with a usable change)"),
        mo.ui.table(floor_summary.round(4), selection=None, page_size=20),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model 1's target

    Chosen on train above, and applied with the same fixed rules to train and test separately; the test result is never summarized.

    1. **Rows:** a usable change (T2 status `ok`), a CRSP price of at least $1 the day before (`in_sample`), and at least 100 holders at the previous count (`holders_ok`, Barber et al.'s `users_close(t-1) >= 100`). These rows are `in_target`, and only they are kept in the holder files. The return panel defines its own universe from day t-1 information.
    2. **Unit:** `y_raw` $= \log(1 + h_t) - \log(1 + h_{t-1})$, the change in log holders from the previous trading day's count to day t's.
    3. **Clip within the day:** `y_clipped` is clipped at that day's 1st and 99th cross-sectional percentiles, so each day uses only its own cross-section.
    4. **Demean within the day:** `y` is `y_clipped` minus that day's cross-sectional mean.

    `y` is a label, never an input to Model 1: Model 1 nowcasts it from TAQ data before 15:50.
    """)
    return


@app.cell
def _(holder_panel_test, holder_panel_train, np):
    CLIP_QUANTILES = (0.01, 0.99)


    def _with_target(df):
        """Add Model 1's target; every step uses only the row's own day."""
        in_target = df["in_sample"] & df["holders_ok"] & df["status"].eq("ok")
        y_raw = (np.log1p(df["holders"].astype(float)) - np.log1p(df["holders_prev"].astype(float))).where(in_target)
        q = y_raw.groupby(df["date"]).quantile(list(CLIP_QUANTILES)).unstack()
        y_clipped = y_raw.clip(df["date"].map(q[CLIP_QUANTILES[0]]), df["date"].map(q[CLIP_QUANTILES[1]]))
        y = y_clipped - y_clipped.groupby(df["date"]).transform("mean")
        return df.assign(in_target=in_target, y_raw=y_raw, y_clipped=y_clipped, y=y)


    holders_train = _with_target(holder_panel_train).query("in_target").reset_index(drop=True)
    holders_test = _with_target(holder_panel_test).query("in_target").reset_index(drop=True)  # same fixed rules; not looked at
    return holders_test, holders_train


@app.cell
def _(holders_train, mo, plt):
    _t = holders_train
    _per_day = _t.groupby("date").size()
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.hist(_t["y"], bins=200)
    _ax.set_title("Model 1's target y (train)")
    _ax.set_ylabel("PERMNO-days")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    **Train only.**

    - **Rows in the target:** {len(_t):,} PERMNO-days, {_t["permno"].nunique():,} PERMNOs, {len(_per_day):,} days; a median of {int(_per_day.median()):,} PERMNOs per day (smallest day {_per_day.min():,} on {_per_day.idxmin():%Y-%m-%d})
    - **Clipped:** {int((_t["y_raw"] != _t["y_clipped"]).sum()):,} rows ({(_t["y_raw"] != _t["y_clipped"]).mean():.1%})
    - **y:** std {_t["y"].std():.4f}, excess kurtosis {_t["y"].kurt():.1f}, range {_t["y"].min():.3f} to {_t["y"].max():.3f}; largest absolute daily mean {_t.groupby("date")["y"].mean().abs().max():.1e}
    """),
        _fig,
    ])
    return


@app.cell
def _(
    SPLIT_DATE,
    TRADING_DAYS,
    holder_panel,
    holders_test,
    holders_train,
    mo,
    pd,
):
    _h = pd.concat([holders_train, holders_test])
    holder_checks = pd.DataFrame([
        ("holders: one row per PERMNO and date", not _h.duplicated(["permno", "date"]).any()),
        ("holders: every date is a trading day in the Robintrack period", bool(_h["date"].isin(TRADING_DAYS).all())),
        ("holders: train ends before the split, test starts on it",
         bool(holders_train["date"].max() < SPLIT_DATE <= holders_test["date"].min())),
        ("holders: exactly the in_target rows",
         len(_h) == int((holder_panel["in_sample"] & holder_panel["holders_ok"] & holder_panel["status"].eq("ok")).sum())),
        ("holders: never negative", bool((_h["holders"] >= 0).all())),
        ("holders: y on every row, with a daily mean of 0",
         bool(_h["y"].notna().all() and _h.groupby("date")["y"].mean().abs().max() < 1e-12)),
    ], columns=["check", "passed"])
    mo.vstack([mo.md("**Holder panel checks**"), mo.ui.table(holder_checks, selection=None)])
    return (holder_checks,)


@app.cell
def _(mo):
    mo.md(r"""
    # Return panel (the 15:50 universe)

    - **Rows:** every stock and trading day t in the **15:50 universe**: a CRSP price of at least $1 and at least 100 Robinhood holders at day t-1's count. A stock stays in even if day t's count is missing or a break follows.
    - **X, known by 15:50 on day t:** `rh_pct_lag1`, the % change in holders on day t-1 (from T2; missing if day t-1 had no usable change).
    - **Features of the holder-change history, known by 15:50 on day t** (built on each PERMNO's whole daily history, all ending at day t-1's change $g_{t-1}$): `rh_pct_accel_lag1` $= g_{t-1} - g_{t-2}$; `rh_pct_sq_lag1` $= g_{t-1}^2$; `rh_pct_vol_lag1`, the standard deviation of $g$ over the last 20 trading days (at least 10 changes); `rh_pct_mean_lag1`, the mean of $g$ over the same 20 days; `rh_pct_sqdev_lag1` $= (g_{t-1} - \bar g)^2$, with $\bar g$ the mean over the 20 trading days before (at least 10 changes); `rh_pct_dev_lag1` $= g_{t-1}$ minus its mean over the 60 trading days before (at least 30 changes). The volatility and the squared deviation are also built over a grid of windows, `rh_pct_vol{n}_lag1` and `rh_pct_sqdev{n}_lag1` for n of 5, 10, 20, 40 and 60 trading days (at least n/2 changes), so 02 can choose the window on train.
    - **Also known by 15:50 on day t, kept apart:** `rh_pct_t`, day t's own change (both counts are retrieved before the cutoff), missing when day t has no usable change. It is what Model 1 nowcasts, so the features above keep their one-day lag.
    - **y:** `ret_next`, CRSP's total return from day t's close to the next trading day's close. Kept when CRSP's return covers one trading day (D1 to D4, DU) or is a delisting payoff (DD). A delisting payoff can span more than a day, so those rows carry `ret_next_is_delisting`.
    - **Also:** `ret_prev`, `price_prev`, `cap_prev`, `dollar_volume_prev` (day t-1), and `is_common` and `security_type` (day t): CRSP's classification into common stock, foreign common stock, ADR, REIT, ETF, closed-end fund, unit or other.
    - **Not here:** demeaning, winsorizing and z-scores. They depend on which stocks are in the universe, so `02_robintrack_returns.py` does them each day within the universe it picks.
    """)
    return


@app.cell
def _(TRADING_DAYS, pd, t2_change, t2_pct):
    # Features of the daily % change in holders, built from each PERMNO's whole daily history on the trading-day grid
    # (not only universe rows), so a rolling window always spans consecutive trading days. A day without a usable change
    # is missing and does not count. Each row is keyed by the day of the change; the return panel joins it as day t-1,
    # so every feature uses counts up to day t-1 only.
    HOLDER_VOL_WINDOW, HOLDER_MEAN_WINDOW = 20, 60  # trading days, about one and three months
    _pct = t2_pct
    _wide = (
        pd.DataFrame({"permno": t2_change["permno"], "date": t2_change["date"], "pct": _pct})
        .pivot(index="date", columns="permno", values="pct")
        .reindex(TRADING_DAYS)
    )
    _features = {
        "rh_pct_accel_lag1": _wide - _wide.shift(1),  # acceleration: this change minus the previous day's
        "rh_pct_sq_lag1": _wide**2,  # squared change, a one-day variance
        "rh_pct_vol_lag1": _wide.rolling(HOLDER_VOL_WINDOW, min_periods=HOLDER_VOL_WINDOW // 2).std(),  # rolling volatility
        "rh_pct_mean_lag1": _wide.rolling(HOLDER_VOL_WINDOW, min_periods=HOLDER_VOL_WINDOW // 2).mean(),  # rolling mean, same window
        # squared deviation from the stock's own mean change over the previous 20 days, excluding the day itself
        "rh_pct_sqdev_lag1": (_wide - _wide.rolling(HOLDER_VOL_WINDOW, min_periods=HOLDER_VOL_WINDOW // 2).mean().shift(1)) ** 2,
        # deviation from the stock's own mean change over the previous 60 days, excluding the day itself
        "rh_pct_dev_lag1": _wide - _wide.rolling(HOLDER_MEAN_WINDOW, min_periods=HOLDER_MEAN_WINDOW // 2).mean().shift(1),
    }
    # vol and sqdev over a grid of rolling windows, for 02 to choose among on train; each needs at least half its window
    HOLDER_WINDOW_GRID = [5, 10, 20, 40, 60]
    for _n in HOLDER_WINDOW_GRID:
        _features[f"rh_pct_vol{_n}_lag1"] = _wide.rolling(_n, min_periods=_n // 2).std()
        _features[f"rh_pct_sqdev{_n}_lag1"] = (_wide - _wide.rolling(_n, min_periods=_n // 2).mean().shift(1)) ** 2
    holder_change_features = (
        pd.concat({_k: _v.stack() for _k, _v in _features.items()}, axis=1)
        .dropna(how="all")
        .rename_axis(["prev_day", "permno"])
        .reset_index()
    )
    HOLDER_CHANGE_FEATURES = list(_features)
    return HOLDER_CHANGE_FEATURES, holder_change_features


@app.cell
def _(
    ALL_TRADING_DAYS,
    HOLDERS_MIN,
    HOLDER_CHANGE_FEATURES,
    ONE_TRADING_DAY_SPANS,
    PRICE_MIN,
    TRADING_DAYS,
    crsp_class_on,
    crsp_daily,
    holder_change_features,
    is_common_stock,
    security_type,
    t1_count,
    t2_change,
    t2_pct,
):
    # One row per stock and day t in the universe known at 15:50 on day t: a CRSP price of at least $1 and at least
    # 100 holders at day t-1's count, whatever happens on day t. Nothing here needs day t's count.
    _count = t1_count[["permno", "date", "holders"]]
    _r = _count.rename(columns={"date": "prev_day", "holders": "holders_count_prev"}).assign(
        date=ALL_TRADING_DAYS[ALL_TRADING_DAYS.get_indexer(_count["date"]) + 1]
    )
    _r = _r[_r["date"].isin(TRADING_DAYS)]
    _r["next_day"] = ALL_TRADING_DAYS[ALL_TRADING_DAYS.get_indexer(_r["date"]) + 1]
    _pct = t2_change.assign(pct=t2_pct)[["permno", "date", "pct"]]
    _prev = crsp_daily[["permno", "date", "price", "ret", "cap", "dollar_volume"]].rename(
        columns={"date": "prev_day", "price": "price_prev", "ret": "ret_prev", "cap": "cap_prev", "dollar_volume": "dollar_volume_prev"}
    )
    _next = crsp_daily[["permno", "date", "ret", "ret_span"]].rename(columns={"date": "next_day", "ret": "ret_next", "ret_span": "ret_next_span"})
    _r = (
        _r.merge(_pct.rename(columns={"date": "prev_day", "pct": "rh_pct_lag1"}), on=["permno", "prev_day"], how="left")
        .merge(_pct.rename(columns={"pct": "rh_pct_t"}), on=["permno", "date"], how="left")
        .merge(holder_change_features, on=["permno", "prev_day"], how="left")
        .merge(_prev, on=["permno", "prev_day"], how="left")
        .merge(_next, on=["permno", "next_day"], how="left")
    )
    _r = _r[_r["price_prev"].ge(PRICE_MIN) & _r["holders_count_prev"].ge(HOLDERS_MIN)].copy()

    # the return to predict: one trading day (D1 to D4, DU), or a delisting payoff (DD), flagged because its span is not one day
    _r["ret_next"] = _r["ret_next"].where(_r["ret_next_span"].str.fullmatch(ONE_TRADING_DAY_SPANS).fillna(False))
    _r["ret_next_is_delisting"] = _r["ret_next_span"].eq("DD").fillna(False).astype(bool)

    # CRSP's classification on day t
    _cls = crsp_class_on(_r)
    _r["is_common"] = is_common_stock(_cls)
    # security type from the same classification, for choosing a universe later
    _r["security_type"] = security_type(_cls)

    returns_panel = _r.sort_values(["permno", "date"]).reset_index(drop=True)
    FEATURES_AT_1550 = ["rh_pct_lag1", *HOLDER_CHANGE_FEATURES]
    FEATURES_AFTER_CLOSE = ["rh_pct_t"]  # known by 15:50, but kept apart as Model 1's target
    return FEATURES_AFTER_CLOSE, FEATURES_AT_1550, returns_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Return panel: train and test

    A training row must have its return label end before the split: day t's return runs to the next trading day. Return rows dated 2019-12-11 whose return ends on 2019-12-12 are left out of both.
    """)
    return


@app.cell
def _(SPLIT_DATE, returns_panel):
    returns_train = returns_panel[returns_panel["next_day"] < SPLIT_DATE].reset_index(drop=True)
    returns_test = returns_panel[returns_panel["date"] >= SPLIT_DATE].reset_index(drop=True)
    return returns_test, returns_train


@app.cell
def _(mo, pd, returns_test, returns_train):
    _sizes = pd.DataFrame([
        {"set": _set, "rows": len(_df), "permnos": _df["permno"].nunique(), "first_day": _df["date"].min().date(), "last_day": _df["date"].max().date()}
        for _set, _df in (("train", returns_train), ("test", returns_test))
    ])
    mo.vstack([mo.md("**Sizes only** (no values are summarized for test)."), mo.ui.table(_sizes, selection=None)])
    return


@app.cell
def _(
    FEATURES_AFTER_CLOSE,
    FEATURES_AT_1550,
    SPLIT_DATE,
    mo,
    pd,
    returns_panel,
):
    _tr = returns_panel[returns_panel["date"] < SPLIT_DATE]
    _cov = pd.DataFrame({
        "column": FEATURES_AT_1550 + FEATURES_AFTER_CLOSE + ["ret_next"],
        "known": ["by 15:50 on day t"] * len(FEATURES_AT_1550) + ["by 15:50 on day t (Model 1's target)"] * len(FEATURES_AFTER_CLOSE) + ["after the next close"],
    })
    _cov["share_available_on_train"] = [round(float(_tr[c].notna().mean()), 3) for c in _cov["column"]]
    mo.vstack([
        mo.md(f"**Return panel on train:** {len(_tr):,} stock-days in the 15:50 universe; {int(_tr['ret_next_is_delisting'].sum()):,} delisting payoffs"),
        mo.ui.table(_cov, selection=None, page_size=20),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The train return panel before modelling

    What `returns_train` looks like, the rows `02_robintrack_returns.py` reads: every stock-day in the 15:50 universe on train. Nothing here changes the data.

    - **Holders:** `holders_count_prev`, day t-1's count.
    - **Holder change:** `rh_pct_lag1`, the % change into day t-1's count, plus its 20-day volatility and squared value.
    - **Market data at t-1:** price, market cap, dollar volume, turnover (dollar volume / market cap) and the day's return `ret_prev`.
    - **Target:** `ret_next`, the return from day t's close to the next.
    """)
    return


@app.cell
def _(pd, returns_train):
    OVERVIEW_COLUMNS = {
        "holders_count_prev": "holders at t-1",
        "rh_pct_lag1": "holder % change, t-1",
        "rh_pct_vol_lag1": "holder % change volatility, 20d",
        "rh_pct_sq_lag1": "holder % change squared",
        "rh_pct_accel_lag1": "holder % change acceleration",
        "rh_pct_dev_lag1": "holder % change minus its 60d mean",
        "price_prev": "price at t-1",
        "cap_prev": "market cap at t-1 ($ thousands)",
        "dollar_volume_prev": "dollar volume at t-1",
        "turnover_prev": "turnover at t-1",
        "ret_prev": "return on t-1",
        "ret_next": "return, close t to t+1",
    }
    # CRSP market cap is in $ thousands, dollar volume in $
    overview_train = returns_train.assign(turnover_prev=returns_train["dollar_volume_prev"] / (returns_train["cap_prev"] * 1e3))


    def _describe(s):
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        return {
            "missing": s.isna().mean(),
            "mean": s.mean(),
            "std": s.std(),
            "skew": s.skew(),
            "excess_kurtosis": s.kurt(),
            "min": s.min(),
            **{f"p{100 * k:g}": v for k, v in q.items()},
            "max": s.max(),
        }


    overview_stats = pd.DataFrame({_label: _describe(overview_train[_c]) for _c, _label in OVERVIEW_COLUMNS.items()}).T

    # one row per stock: how long it is in the universe and what it typically looks like
    overview_per_stock = overview_train.groupby("permno").agg(
        days=("date", "size"),
        first_day=("date", "min"),
        last_day=("date", "max"),
        median_holders=("holders_count_prev", "median"),
        median_cap=("cap_prev", "median"),
        median_dollar_volume=("dollar_volume_prev", "median"),
        median_price=("price_prev", "median"),
        mean_holder_pct_change=("rh_pct_lag1", "mean"),
        holder_pct_change_std=("rh_pct_lag1", "std"),
        mean_ret_next=("ret_next", "mean"),
        ret_next_std=("ret_next", "std"),
        share_common=("is_common", "mean"),
    )
    overview_per_day = overview_train.groupby("date").agg(
        stocks=("permno", "size"),
        median_holders=("holders_count_prev", "median"),
        total_holders=("holders_count_prev", "sum"),
        holder_pct_change_p10=("rh_pct_lag1", lambda s: s.quantile(0.1)),
        holder_pct_change_p50=("rh_pct_lag1", "median"),
        holder_pct_change_p90=("rh_pct_lag1", lambda s: s.quantile(0.9)),
        ret_next_mean=("ret_next", "mean"),
        ret_next_xs_std=("ret_next", "std"),
        holder_pct_change_xs_std=("rh_pct_lag1", "std"),
        share_common=("is_common", "mean"),
        delistings=("ret_next_is_delisting", "sum"),
    )
    return (
        OVERVIEW_COLUMNS,
        overview_per_day,
        overview_per_stock,
        overview_stats,
        overview_train,
    )


@app.cell
def _(
    OVERVIEW_COLUMNS,
    mo,
    overview_per_day,
    overview_per_stock,
    overview_stats,
    overview_train,
):
    _ps = overview_per_stock
    _days = overview_per_day["stocks"]
    _rank_cols = ["holders_count_prev", "cap_prev", "dollar_volume_prev", "turnover_prev", "price_prev",
                  "rh_pct_lag1", "rh_pct_vol_lag1", "ret_prev", "ret_next"]
    _spearman = overview_train[_rank_cols].sample(200_000, random_state=0).corr(method="spearman").rename(
        index=OVERVIEW_COLUMNS, columns=OVERVIEW_COLUMNS)
    mo.vstack([
        mo.md(f"""
    **Train return panel:** {len(overview_train):,} stock-days, {overview_train["permno"].nunique():,} PERMNOs, {len(_days):,} days ({overview_train["date"].min():%Y-%m-%d} to {overview_train["date"].max():%Y-%m-%d}).

    - **Stocks per day:** median {int(_days.median()):,}, from {_days.min():,} ({_days.idxmin():%Y-%m-%d}) to {_days.max():,} ({_days.idxmax():%Y-%m-%d}).
    - **Days per stock:** median {int(_ps["days"].median())} of {len(_days)}; {(_ps["days"] == len(_days)).mean():.0%} of PERMNOs are in every day, {(_ps["days"] < 20).mean():.0%} for fewer than 20.
    - **Common stock (share codes 10/11):** {overview_train["is_common"].mean():.1%} of rows. **Delisting payoffs:** {int(overview_train["ret_next_is_delisting"].sum()):,} rows.
    - **Holders are concentrated:** the top 1% of stocks by median holders hold {_ps["median_holders"].nlargest(int(len(_ps) * 0.01)).sum() / _ps["median_holders"].sum():.0%} of all holdings.
    """),
        mo.md("**Pooled over stock-days** (raw units: % changes and returns as fractions; missing is a share of rows)"),
        mo.ui.table(overview_stats.reset_index(names="variable").round(6), selection=None, page_size=20),
        mo.md("**Across stocks** (one row per PERMNO, then summarized)"),
        mo.ui.table(_ps.drop(columns=["first_day", "last_day"]).describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
                    .reset_index(names="per-stock statistic").round(6), selection=None, page_size=20),
        mo.md("**Rank correlations, pooled** (Spearman, 200,000 random stock-days)"),
        mo.ui.table(_spearman.round(2).reset_index(names="variable"), selection=None, page_size=20),
    ])
    return


@app.cell
def _(OVERVIEW_COLUMNS, np, overview_train, plt):
    _fig, _axes = plt.subplots(3, 4, figsize=(16, 10))
    _log_vars = ["holders_count_prev", "price_prev", "cap_prev", "dollar_volume_prev", "turnover_prev", "rh_pct_vol_lag1"]
    _mid_vars = ["rh_pct_lag1", "rh_pct_accel_lag1", "rh_pct_dev_lag1", "rh_pct_sq_lag1", "ret_prev", "ret_next"]
    for _ax, _c in zip(_axes.flat[:6], _log_vars):
        _s = overview_train[_c].dropna()
        _s = _s[_s > 0]
        _ax.hist(_s, bins=np.geomspace(_s.min(), _s.max(), 120))
        _ax.set_xscale("log")
        _ax.set_title(f"{OVERVIEW_COLUMNS[_c]} (log scale)")
    for _ax, _c in zip(_axes.flat[6:], _mid_vars):
        _s = overview_train[_c].dropna()
        _lo, _hi = _s.quantile([0.005, 0.995])
        _ax.hist(_s[_s.between(_lo, _hi)], bins=150)
        _ax.set_title(f"{OVERVIEW_COLUMNS[_c]}: middle 99%")
    for _ax in _axes.flat:
        _ax.set_ylabel("stock-days")
    _fig.suptitle("Train return panel: distributions")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(overview_per_day, plt):
    _d = overview_per_day
    _fig, _axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    _axes[0, 0].plot(_d.index, _d["stocks"])
    _axes[0, 0].set_title("Stocks in the 15:50 universe")
    _axes[0, 1].plot(_d.index, _d["total_holders"] / 1e6, label="total")
    _axes[0, 1].set_title("Robinhood holdings across the universe at t-1 (millions)")
    _axes[1, 0].fill_between(_d.index, _d["holder_pct_change_p10"], _d["holder_pct_change_p90"], alpha=0.3, label="p10 to p90")
    _axes[1, 0].plot(_d.index, _d["holder_pct_change_p50"], label="median")
    _axes[1, 0].set_title("Holder % change at t-1, across stocks")
    _axes[1, 0].legend()
    _axes[1, 1].plot(_d.index, _d["holder_pct_change_xs_std"])
    _axes[1, 1].set_yscale("log")
    _axes[1, 1].set_title("Cross-sectional std of the holder % change (log scale)")
    _axes[2, 0].bar(_d.index, _d["ret_next_mean"], width=1.5)
    _axes[2, 0].set_title("Universe mean of ret_next (what demeaning removes)")
    _axes[2, 1].plot(_d.index, _d["ret_next_xs_std"])
    _axes[2, 1].set_title("Cross-sectional std of ret_next")
    for _ax in _axes[1:, :].flat:
        _ax.axhline(0, color="grey", lw=0.5)
    _fig.suptitle("Train return panel over time (by day t)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, overview_train, plt):
    # who owns what: holders and the holder change against size, by market-cap decile within each day
    _t = overview_train[["date", "cap_prev", "holders_count_prev", "rh_pct_lag1", "rh_pct_vol_lag1", "ret_next", "turnover_prev"]].dropna(subset=["cap_prev"])
    _t = _t.assign(cap_decile=_t.groupby("date")["cap_prev"].rank(pct=True).mul(10).clip(upper=9.999).astype(int) + 1)
    overview_by_size = _t.groupby("cap_decile").agg(
        stock_days=("date", "size"),
        median_cap_musd=("cap_prev", lambda s: s.median() / 1e3),
        median_holders=("holders_count_prev", "median"),
        median_holder_pct_change=("rh_pct_lag1", "median"),
        holder_pct_change_abs_mean=("rh_pct_lag1", lambda s: s.abs().mean()),
        median_holder_vol=("rh_pct_vol_lag1", "median"),
        median_turnover=("turnover_prev", "median"),
        ret_next_std=("ret_next", "std"),
    )
    _sample = _t.sample(100_000, random_state=0)
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _hb = _ax.hexbin(_sample["cap_prev"] * 1e3, _sample["holders_count_prev"], xscale="log", yscale="log", gridsize=60, bins="log", mincnt=1)
    _ax.set_xlabel("market cap at t-1 ($)")
    _ax.set_ylabel("holders at t-1")
    _ax.set_title("Holders against size (100,000 random stock-days)")
    _fig.colorbar(_hb, label="stock-days (log)")
    _fig.tight_layout()
    mo.vstack([
        mo.md("**By market-cap decile within each day** (1 = smallest). Market cap in $ millions."),
        mo.ui.table(overview_by_size.reset_index().round(5), selection=None),
        _fig,
    ])
    return


@app.cell
def _(mo, returns_train):
    _mid = len(returns_train) // 2
    mo.vstack([
        mo.md(f"**The train return panel as written:** 100 rows from the middle, all {returns_train.shape[1]} columns"),
        mo.ui.table(returns_train.iloc[_mid - 50:_mid + 50], selection=None, page_size=25),
    ])
    return


@app.cell
def _(HOLDERS_MIN, PRICE_MIN, SPLIT_DATE, mo, pd, returns_test, returns_train):
    _r = pd.concat([returns_train, returns_test])
    return_checks = pd.DataFrame([
        ("returns: one row per PERMNO and date", not _r.duplicated(["permno", "date"]).any()),
        ("returns: every row in the 15:50 universe", bool((_r["price_prev"].ge(PRICE_MIN) & _r["holders_count_prev"].ge(HOLDERS_MIN)).all())),
        ("returns: train labels end before the split, test starts on it",
         bool(returns_train["next_day"].max() < SPLIT_DATE <= returns_test["date"].min())),
        ("returns: delisting payoffs flagged", bool((_r["ret_next_is_delisting"] == _r["ret_next_span"].eq("DD").fillna(False)).all())),
    ], columns=["check", "passed"])
    mo.vstack([mo.md("**Return panel checks**"), mo.ui.table(return_checks, selection=None)])
    return (return_checks,)


@app.cell
def _(mo):
    mo.md(r"""
    # Write the panels

    Four files: `holders_train` and `holders_test` (Model 1: `in_target` rows with the target) and `returns_train` and `returns_test` (the 15:50 universe, for the return tests). Train and test are separate files, so later work can read the training files without ever opening the test ones. Written only when every check passes (the hourly rows, the holder panel and the return panel) and the button is pressed.
    """)
    return


@app.cell
def _(holder_checks, hourly_checks, mo, pd, return_checks):
    panel_checks = pd.concat([
        pd.DataFrame([("hourly rows: every check on the cleaned hourly rows passes", bool(hourly_checks["passed"].all()))], columns=["check", "passed"]),
        holder_checks,
        return_checks,
    ], ignore_index=True)
    write_panel = mo.ui.run_button(label="Write the holder and return panels")
    mo.vstack([mo.ui.table(panel_checks, selection=None, page_size=20), write_panel])
    return panel_checks, write_panel


@app.cell
def _(
    DATA_DIR,
    INTERMEDIATE_DIR,
    holders_test,
    holders_train,
    mo,
    panel_checks,
    returns_test,
    returns_train,
    write_panel,
):
    mo.stop(not panel_checks["passed"].all(), mo.md("**Not written:** a check failed."))
    mo.stop(not write_panel.value, mo.md("Press the button to write."))
    _dir = INTERMEDIATE_DIR / "clean"
    _dir.mkdir(parents=True, exist_ok=True)
    _files = {"holders_train": holders_train, "holders_test": holders_test, "returns_train": returns_train, "returns_test": returns_test}
    for _name, _df in _files.items():
        _part = _dir / f"{_name}.parquet.part"
        _df.to_parquet(_part, index=False)
        _part.rename(_dir / f"{_name}.parquet")
    mo.md(f"Written to `{_dir.relative_to(DATA_DIR.parent)}`: " + ", ".join(f"`{k}.parquet` ({len(v):,} rows)" for k, v in _files.items()))
    return


if __name__ == "__main__":
    app.run()
