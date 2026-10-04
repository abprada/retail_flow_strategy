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

    return groupby, mo, np, pd, plt


@app.cell
def _(mo):
    DATA_DIR = mo.notebook_dir().parent / "data"
    RAW_DIR = DATA_DIR / "raw" / "robintrack" / "popularity_export"
    CRSP_NAMES_PATH = DATA_DIR / "raw" / "crsp" / "stocknames_v2.csv"
    INTERMEDIATE_DIR = DATA_DIR / "intermediate"
    return CRSP_NAMES_PATH, INTERMEDIATE_DIR, RAW_DIR


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
    6. **Overlap at a rename:** if two files hold the same PERMNO at the same time, the date-matched rows win over rows filled by rule 5 (the new instrument's first rows can start minutes before the old one's last).
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
        for _part in (~_sib, _sib):
            if _part.any():
                _permno[_part] = by_identity(_dates[_part], _permno[_part])

        _ok = _permno != 0
        _report.append({
            "file": _k.file,
            "ticker": _k.ticker,
            "n_rows": len(_df),
            "n_sibling": int(_sib.sum()),
            "n_date_matched": int(_date_matched.sum()),
            "n_no_permno": int((_hits == 0).sum()),
            "n_ambiguous": int((_hits > 1).sum()),
            "n_added_by_identity": int((_ok & ~_date_matched).sum()),
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
            "filled": (~_date_matched)[_ok],
        }))

    rt_by_permno = pd.concat(_rows, ignore_index=True)

    # where two files hold one PERMNO at the same time, date-matched rows win over rows filled by file identity
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
    - **File identity:** {_r["n_added_by_identity"].sum():,} rows added (history under an earlier ticker), {_r["n_dropped_after_end"].sum():,} dropped after their security ended, {_r["n_dropped_overlap"].sum():,} dropped where another file's date-matched rows hold the same PERMNO
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

    A full sweep asks Robinhood for about 8,600 instruments, 20 per request with at least a second between requests, so it takes at least 7 minutes and reads each instrument once. Two readings of one PERMNO less than 5 minutes apart therefore come from overlapping or restarted sweeps. Within a stretch from one raw file, every reading followed by another within 5 minutes is flagged, so each cluster keeps only its latest reading. Rows already flagged by C1 or C2 are left to those steps.
    """)
    return


@app.cell
def _(pd, rt_permno):
    rt_ts = pd.to_datetime(rt_permno["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return (rt_ts,)


@app.cell
def _(c1_placeholder, c2_placeholder, pd, rt_permno, rt_ts):
    _next_same_stretch = rt_permno["permno"].eq(rt_permno["permno"].shift(-1)) & rt_permno["source_file"].eq(
        rt_permno["source_file"].shift(-1)
    )
    _gap_to_next = (rt_ts.shift(-1) - rt_ts).dt.total_seconds()
    c3_repeat = _next_same_stretch & (_gap_to_next < 300) & ~c1_placeholder & ~c2_placeholder

    _v = rt_permno["users_holding"]
    _diff_to_kept = (_v.shift(-1) - _v)[c3_repeat]
    c3_summary = pd.DataFrame({
        "rows_flagged": [int(c3_repeat.sum())],
        "permnos_affected": [rt_permno.loc[c3_repeat, "permno"].nunique()],
        "gap_under_10s": [int((_gap_to_next[c3_repeat] < 10).sum())],
        "gap_10s_to_1min": [int(_gap_to_next[c3_repeat].between(10, 60, inclusive="left").sum())],
        "gap_1_to_5min": [int(_gap_to_next[c3_repeat].between(60, 300, inclusive="left").sum())],
        "same_value_as_kept": [int((_diff_to_kept == 0).sum())],
    })
    return c3_repeat, c3_summary


@app.cell
def _(c3_repeat, c3_summary, mo, rt_permno):
    _c = c3_summary.iloc[0]
    _flagged = rt_permno.loc[c3_repeat, ["permno", "timestamp", "users_holding", "source_file"]]
    mo.vstack([
        mo.md(f"""
    - **Rows flagged:** {_c["rows_flagged"]:,} of {len(rt_permno):,} ({_c["rows_flagged"] / len(rt_permno):.2%}), in {_c["permnos_affected"]:,} PERMNOs
    - **Gap to the kept reading:** under 10 seconds {_c["gap_under_10s"]:,}; 10 seconds to 1 minute {_c["gap_10s_to_1min"]:,}; 1 to 5 minutes {_c["gap_1_to_5min"]:,}
    - **Same value as the reading kept:** {_c["same_value_as_kept"]:,} ({_c["same_value_as_kept"] / max(_c["rows_flagged"], 1):.1%})
    """),
        mo.ui.table(_flagged.head(1000)),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C4: outages

    Robintrack swept every hour around the clock, so more than 1.5 hours without a reading for any PERMNO means at least one sweep was missed. Each outage is listed with the weekday closes it wipes out (the whole 14:30 to 16:00 New York window empty); holidays are not separated out until there is a trading calendar.

    Nothing is dropped here. Outages are handled in the target: a daily change is taken only between two consecutive trading days that both have a close value, so the first day after an outage has a close but no change, and changes resume the day after. The same rule covers a single PERMNO missing a close while the rest are fine. A percentage change whose previous close is 0 is undefined and left missing; that comes from real zero holders, not from outages.
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


    def _closes_lost(before, after):
        """Weekdays whose whole 14:30 to 16:00 New York window falls inside the outage."""
        before_ny, after_ny = before.tz_convert("America/New_York"), after.tz_convert("America/New_York")
        days = pd.date_range(before_ny.normalize(), after_ny.normalize(), freq="B")
        return [
            d.strftime("%Y-%m-%d") for d in days
            if before_ny < d + pd.Timedelta(hours=14, minutes=30) and after_ny > d + pd.Timedelta(hours=16)
        ]


    c4_outages["weekday_closes_lost"] = [
        _closes_lost(b, a) for b, a in zip(c4_outages["last_reading_before"], c4_outages["first_reading_after"])
    ]
    c4_outages["n_closes_lost"] = c4_outages["weekday_closes_lost"].str.len()
    return (c4_outages,)


@app.cell
def _(c4_outages, mo):
    _o = c4_outages
    mo.vstack([
        mo.md(f"""
    - **Outages (more than 1.5 hours without any reading):** {len(_o):,}, {_o["hours"].sum():,.0f} hours in total; longest {_o["hours"].max():,.1f} hours
    - **Weekday closes lost:** {_o["n_closes_lost"].sum():,}, in {(_o["n_closes_lost"] > 0).sum():,} outages
    """),
        mo.ui.table(_o.assign(weekday_closes_lost=_o["weekday_closes_lost"].str.join(", ")).sort_values("last_reading_before")),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Checks before writing

    Run on exactly the rows that will be written (everything not flagged by C1 to C3). The write step refuses to run unless every check passes.
    """)
    return


@app.cell
def _(c1_placeholder, c2_placeholder, c3_repeat, mo, np, pd, rt_permno, rt_ts):
    _keep = ~(c1_placeholder | c2_placeholder | c3_repeat).to_numpy()
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
    pre_write_checks = pd.DataFrame([
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
        mo.md(f"**{int(pre_write_checks['passed'].sum())} of {len(pre_write_checks)} checks pass** on {_keep.sum():,} rows to be written."),
        mo.ui.table(pre_write_checks),
    ])
    return


if __name__ == "__main__":
    app.run()
