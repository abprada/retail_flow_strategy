import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One stock-day as an image

    A first test of the chart images: one stock on one decision day, its session's ten-minute bars (09:30 to the 15:50 cutoff) drawn as a greyscale image in the style of Jiang, Kelly and Xiu (2023), *(Re-)Imag(in)ing Price Trends*:

    - **Each bar is 3 pixels wide:** the open as a tick on the left, the high-low range as the middle column, the close as a tick on the right.
    - **Price** fills the top 51 rows, scaled so the day's highest high is the top row and its lowest low the bottom one.
    - **Volume** (shares) fills the bottom 12 rows, scaled to the day's largest bar, with one black row between the two: 64 rows in all, the height of their 20-day image.
    - **Black and white only** (0 or 255, one channel), so the image packs into 1 bit a pixel. A bar without trades stays black.

    **Input:** the day's `bars.parquet` and `prices.parquet` in `data/intermediate/taq/train/<day>/` and the stock-day's row of `panel_train.parquet` (the official open and the quoted spread). A training day only.

    **Universe: stocks of $1B or more** (`cap_prev` at least `LARGE_CAP`, the previous trading day's market cap, known before the cutoff), the stocks the models are scored and traded on. Ten-minute bars of smaller stocks are too sparse to draw: a bar with a handful of trades has the bid and the offer as its high and low, and 8.4% of their session bars traded only odd lots and have no price, against 1.15% at $1B or more.

    **Prices from round lots:** the bars' own open, high, low and close come from every trade, odd lots included, and one stray odd lot sets the whole image's scale (`reports/bad_prints.md`). So the price bars are drawn from `prices.parquet` instead: its `px_` prices, from trades that are not odd lots under the round-lot rule before 2025-11-03, held fixed (`scripts/taq/01_clean_TAQ_prices.py`). A bar whose trades were all odd lots has no price, but keeps its volume. Volume and the color channels stay on every trade, from `bars`.

    **Bad prints left:** rule V2 cleans the round-lot prices as a backstop: a price at least 2% and at least 20 typical moves away from the prices around it is bad. A bad open or close is not drawn, and a bad low or high is pulled in to the bar's remaining open and close. The check at the end compares the bars as stored, the round-lot prices and the round-lot prices with V2 on 100 random training days.

    **Color channels:** retail trading and order flow are drawn as three more channels on the same pixels, each value standardized across the large caps in its bar (the last section).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from dsr.panel import LARGE_CAP, MIN_RETAIL_TRADES, SESSION_BARS

    return LARGE_CAP, SESSION_BARS, ds, mo, np, pd, plt, pq


@app.cell
def _(SESSION_BARS, mo, pd, pq):
    TAQ_DIR = mo.notebook_dir().parents[1] / "data" / "intermediate" / "taq"
    DAY, PERMNO = "20190612", 14593  # AAPL on a normal (not early-close) training day
    N_BARS = SESSION_BARS[False]  # 38: 09:30 to 15:50
    PRICE_ROWS, VOLUME_ROWS = 51, 12

    PRICES = ["open", "high", "low", "close"]

    _columns = [*PRICES, "last_mid", "shares"]
    _filters = [("permno", "==", PERMNO), ("piece", "==", "session")]
    stored = (pq.read_table(TAQ_DIR / "train" / DAY / "bars.parquet", columns=["permno", "piece", "bar", *_columns],
                            filters=_filters)
              .to_pandas().set_index("bar")[_columns]
              .astype(float).reindex(range(N_BARS)))  # a bar without trades is a row of NaN; prices from every trade
    _px = (pq.read_table(TAQ_DIR / "train" / DAY / "prices.parquet",
                         columns=["permno", "piece", "bar", *[f"px_{_p}" for _p in PRICES]], filters=_filters)
           .to_pandas().set_index("bar")[[f"px_{_p}" for _p in PRICES]].astype(float).reindex(range(N_BARS)))
    bars = stored.assign(**{_p: _px[f"px_{_p}"] for _p in PRICES})  # prices from round lots; NaN if only odd lots
    info = pq.read_table(TAQ_DIR / "panel_train.parquet", columns=["open_t", "spread_bp"],
                         filters=[("day", "==", pd.Timestamp(DAY).to_pydatetime()),
                                  ("permno", "==", PERMNO)]).to_pandas().iloc[0]
    bars
    return (
        DAY,
        PERMNO,
        PRICES,
        PRICE_ROWS,
        TAQ_DIR,
        VOLUME_ROWS,
        bars,
        info,
        stored,
    )


@app.cell
def _(np, pd):
    V2_MIN_DEV, V2_MIN_UNITS = 0.02, 20  # rule V2 of reports/bad_prints.md


    def clean_prints(s):
        """Session bars with trades of one or more stock-days, sorted by permno and bar, with columns permno, open,
        high, low, close, last_mid, open_t (the official open) and spread_bp (the day's average quoted spread to the
        cutoff). Returns them with bad prints cleaned and a bool column bad_<price> for each of the four prices.

        A price is bad when it is at least V2_MIN_DEV and at least V2_MIN_UNITS typical moves away from the prices
        around it: `before`, the previous bar's close (the official open for the first bar), and `after`, the median
        of the bar's close, its last midpoint and the next bar's open (the last midpoint for the last bar). A low or
        high counts by how far it lies beyond both, the open by its distance to the nearer of them, the close by its
        distance to the nearer of its last midpoint and the next bar's open. The typical move is the median change of
        `after` from bar to bar that day, at least one quoted spread. A bad open or close is dropped (not drawn); a bad
        low or high is pulled in to the bar's remaining open and close, or to `after` if neither is left."""
        g = s.groupby("permno", sort=False)
        next_open = g["open"].shift(-1)
        before = g["close"].shift(1).where(g.cumcount() > 0, s["open_t"].where(s["open_t"] > 0, s["open"]))
        three = np.nanmedian(np.column_stack([s["close"], s["last_mid"], next_open]), axis=1)
        after = pd.Series(np.where(next_open.notna(), three, s["last_mid"].fillna(s["close"])), index=s.index)
        low_ref, high_ref = np.fmin(before, after), np.fmax(before, after)
        dev = pd.DataFrame({
            "open": np.fmin((s["open"] - before).abs(), (s["open"] - after).abs()) / after,
            "high": ((s["high"] - high_ref) / after).clip(lower=0),
            "low": ((low_ref - s["low"]) / after).clip(lower=0),
            "close": np.fmin((s["close"] - s["last_mid"]).abs(), (s["close"] - next_open).abs()) / after,
        })
        move = (after / after.groupby(s["permno"]).shift(1) - 1).abs()
        unit = np.fmax(move.groupby(s["permno"]).transform("median"), s["spread_bp"] / 1e4)
        bad = dev.ge(V2_MIN_DEV) & dev.ge(V2_MIN_UNITS * unit, axis=0)

        o, c = s["open"].mask(bad["open"]), s["close"].mask(bad["close"])
        body_low, body_high = np.fmin(o, c).fillna(after), np.fmax(o, c).fillna(after)
        return s.assign(open=o, close=c, low=s["low"].mask(bad["low"], body_low),
                        high=s["high"].mask(bad["high"], body_high), **{f"bad_{f}": bad[f] for f in bad.columns})

    return (clean_prints,)


@app.cell
def _(PERMNO, bars, clean_prints, info, mo):
    _cleaned = (clean_prints(bars.dropna(subset=["close"]).rename_axis("bar").reset_index()
                             .assign(permno=PERMNO, open_t=info["open_t"], spread_bp=info["spread_bp"]))
                .set_index("bar").reindex(bars.index))
    clean = bars.assign(**{_f: _cleaned[_f] for _f in ["open", "high", "low", "close"]})
    _bad = _cleaned.filter(like="bad_").fillna(False).astype(bool)
    mo.vstack([mo.md(f"**Bad prints (rule V2):** {int(_bad.to_numpy().sum())} value(s) in "
                     f"{int(_bad.any(axis=1).sum())} bar(s):"),
               bars[_bad.any(axis=1)]])
    return (clean,)


@app.cell
def _(np):
    def draw(o, h, l, c, v, price_rows, volume_rows):
        """A greyscale image (uint8, 0 or 255) of bars given as arrays of open, high, low, close and volume, NaN for a
        bar without trades: 3 columns a bar; price on top, scaled to the bars' highest high and lowest low; one black
        row; volume below, scaled to the largest volume."""
        top, bottom = np.nanmax(h), np.nanmin(l)
        span = top - bottom if top > bottom else 1.0  # a flat day: every price on the top row

        def row(p):  # a price's row, 0 at the top; NaN stays NaN and draws nothing
            return np.round((top - p) / span * (price_rows - 1))

        r = np.arange(price_rows)[:, None]
        price = np.zeros((price_rows, 3 * len(o)), dtype=bool)
        price[:, 0::3] = r == row(o)
        price[:, 1::3] = (r >= row(h)) & (r <= row(l))
        price[:, 2::3] = r == row(c)

        filled = np.round(v / np.nanmax(v) * volume_rows)  # rows of volume, counted from the bottom
        volume = np.zeros((volume_rows, 3 * len(o)), dtype=bool)
        volume[:, 1::3] = np.arange(volume_rows)[:, None] >= volume_rows - filled

        gap = np.zeros((1, 3 * len(o)), dtype=bool)
        return np.vstack([price, gap, volume]).astype(np.uint8) * 255

    return (draw,)


@app.cell
def _(PRICE_ROWS, VOLUME_ROWS, clean, draw, mo, np, stored):
    def _image(b):
        return draw(*(b[c].to_numpy() for c in ["open", "high", "low", "close", "shares"]), PRICE_ROWS, VOLUME_ROWS)


    raw_image, image = _image(stored), _image(clean)  # every trade as stored; round lots cleaned by V2
    mo.md(f"**Image:** {image.shape[0]} x {image.shape[1]} pixels; {image.nbytes:,} bytes as uint8, "
          f"{np.packbits(image > 0).nbytes:,} bytes packed at 1 bit a pixel.")
    return image, raw_image


@app.cell
def _(DAY, PERMNO, image, plt, raw_image):
    ZOOM_BARS = 10
    fig, axes = plt.subplots(3, 1, figsize=(9, 13))
    for _ax, _img, _title in [(axes[0], raw_image, "every trade, as stored"),
                               (axes[1], image, "round lots, cleaned (rule V2)")]:
        _ax.imshow(_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        _ax.set_title(f"PERMNO {PERMNO} on {DAY}, session bars 09:30 to 15:50: {_title}")
        _ax.set_axis_off()
    axes[2].imshow(image[:, :3 * ZOOM_BARS], cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
    for _i in range(1, ZOOM_BARS):
        axes[2].axvline(3 * _i - 0.5, color="tab:red", linewidth=0.8)  # bar boundaries
    axes[2].set_title(f"The first {ZOOM_BARS} bars (09:30 to 11:10); each bar: open, high-low, close")
    axes[2].set_xticks([3 * _i + 1 for _i in range(ZOOM_BARS)], [f"{_i}" for _i in range(ZOOM_BARS)])
    axes[2].set_yticks([])
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Do the round-lot prices fix the images? 100 random training days

    Every panel stock of $1B or more on 100 training days drawn at random, three ways: the bars as stored (every trade), the round-lot prices (`px_`), and the round-lot prices cleaned by V2 (what is drawn).

    - **Against the official range:** each stock-day's prices are compared with CRSP's official daily high and low (`dlyhigh`, `dlylow`, days with a trade price only), which leave out odd lots. The official range runs to 16:00 and includes the auctions, so when nothing is wrong it can only be wider than the bars'; a price outside it is a trade the official tape does not count. It includes trading after the cutoff, so it serves only to check, never as an input.
    - **Squeezed image:** the share of the image's price range that lies outside the official range. At 10% or more (5 of the 51 price rows) a stray print visibly squeezes the chart; at 50% or more the day uses half the height or less. A stray print inside the day's official range is not counted, so this undercounts.
    - **Images:** as stored | drawn (round lots and V2), for one stock drawn at random from each day, for stock-days squeezed as stored, and for those still squeezed.
    """)
    return


@app.cell
def _(LARGE_CAP, TAQ_DIR, ds, mo, pd, pq):
    N_DAYS, SEED = 100, 0
    CRSP_DAILY = TAQ_DIR.parents[1] / "raw" / "crsp" / "stkdlysecuritydata_from_2013.csv"

    _days = pd.Series(pq.read_table(TAQ_DIR / "panel_train.parquet", columns=["day"]).column("day").unique()
                      .to_pandas())
    sample_days = _days.sample(N_DAYS, random_state=SEED).sort_values().tolist()
    _missing = [_d for _d in sample_days if not (TAQ_DIR / "train" / f"{_d:%Y%m%d}" / "prices.parquet").exists()]
    mo.stop(bool(_missing), mo.md(f"**Waiting for the round-lot prices:** {len(_missing)} of the {N_DAYS} sampled days "
                                  "have no `prices.parquet` yet (`scripts/taq/01_clean_TAQ_prices.py`). Run this cell "
                                  "again when it has finished."))
    sample_panel = pq.read_table(
        TAQ_DIR / "panel_train.parquet",
        columns=["day", "permno", "symbol", "cap_prev", "early_close", "open_t", "spread_bp"],
        filters=[("day", "in", [_d.to_pydatetime() for _d in sample_days])]).to_pandas()
    sample_panel = sample_panel[sample_panel["cap_prev"] >= LARGE_CAP]  # the universe of the images
    picks = sample_panel.groupby("day").sample(1, random_state=SEED)[["day", "permno"]]

    _crsp = ds.dataset(CRSP_DAILY, format="csv").to_table(
        columns=["permno", "yyyymmdd", "dlyprcflg", "dlyhigh", "dlylow"],
        filter=ds.field("yyyymmdd").isin([int(_d.strftime("%Y%m%d")) for _d in sample_days])).to_pandas()
    _crsp = _crsp[_crsp["dlyprcflg"] == "TR"]
    official = pd.DataFrame({"day": pd.to_datetime(_crsp["yyyymmdd"].astype(str)), "permno": _crsp["permno"],
                             "off_high": _crsp["dlyhigh"].astype(float), "off_low": _crsp["dlylow"].astype(float)})
    mo.md(f"**Sample:** {N_DAYS} days from {sample_days[0]:%Y-%m-%d} to {sample_days[-1]:%Y-%m-%d}, "
          f"{len(sample_panel):,} stock-days of $1B or more; official range for {len(official):,} stock-days.")
    return SEED, official, picks, sample_panel


@app.cell
def _(
    PRICES,
    TAQ_DIR,
    clean_prints,
    mo,
    official,
    pd,
    picks,
    pq,
    sample_panel,
):
    SQUEEZED = 0.10  # the share of the image's price range outside the official range that visibly squeezes it


    def outside_share(high, low, off_high, off_low):
        """The share of an image's price range, low to high, that lies outside the official range."""
        span = high - low
        out = (high - off_high).clip(lower=0) + (off_low - low).clip(lower=0)
        return (out / span).where(span > 0, 0.0)


    _stockdays, _kept = [], []
    for _day, _info in mo.status.progress_bar(list(sample_panel.groupby("day")), title="Cleaning the sampled days"):
        _folder = TAQ_DIR / "train" / f"{_day:%Y%m%d}"
        _session = [("piece", "==", "session")]
        _b = pq.read_table(_folder / "bars.parquet", columns=["permno", "bar", *PRICES, "last_mid", "shares"],
                           filters=_session).to_pandas()
        _px = pq.read_table(_folder / "prices.parquet", columns=["permno", "bar", *[f"px_{_p}" for _p in PRICES]],
                            filters=_session).to_pandas()
        _b = (_b.merge(_px, on=["permno", "bar"], how="left")
              .merge(_info[["permno", "open_t", "spread_bp"]], on="permno")  # the panel's stocks only
              .merge(official.loc[official["day"] == _day, ["permno", "off_high", "off_low"]], on="permno", how="left")
              .sort_values(["permno", "bar"], ignore_index=True))
        _b[PRICES + [f"px_{_p}" for _p in PRICES]] = _b[PRICES + [f"px_{_p}" for _p in PRICES]].astype(float)
        _b = _b.rename(columns={_p: f"raw_{_p}" for _p in PRICES}).rename(columns={f"px_{_p}": _p for _p in PRICES})
        _has = _b["close"].notna()  # bars with a round lot; V2 runs on them, in order
        _c = _b.copy()
        _cleaned = clean_prints(_b[_has].reset_index(drop=True))
        _c.loc[_has, PRICES] = _cleaned[PRICES].to_numpy()
        _bad = _cleaned[[f"bad_{_f}" for _f in PRICES]].to_numpy()
        _vals = _b.loc[_has, PRICES]
        _out = (_vals.gt(_b.loc[_has, "off_high"] + 0.01, axis=0) | _vals.lt(_b.loc[_has, "off_low"] - 0.01, axis=0)).to_numpy()
        _c["n_bad"], _c["n_bad_outside"] = 0, 0
        _c.loc[_has, "n_bad"], _c.loc[_has, "n_bad_outside"] = _bad.sum(axis=1), (_bad & _out).sum(axis=1)
        _c = _c.assign(rl_high=_b["high"], rl_low=_b["low"], no_price=~_has)
        _sd = (_c.groupby("permno")
               .agg(high=("raw_high", "max"), low=("raw_low", "min"), rl_high=("rl_high", "max"), rl_low=("rl_low", "min"),
                    clean_high=("high", "max"), clean_low=("low", "min"), off_high=("off_high", "first"),
                    off_low=("off_low", "first"), n_bad=("n_bad", "sum"), n_bad_outside=("n_bad_outside", "sum"),
                    bars=("bar", "size"), no_price=("no_price", "sum"))
               .reset_index().merge(_info[["permno", "symbol", "cap_prev", "early_close"]], on="permno")
               .assign(day=_day))
        for _name, _h, _l in [("outside", "high", "low"), ("outside_rl", "rl_high", "rl_low"),
                              ("outside_clean", "clean_high", "clean_low")]:
            _sd[_name] = outside_share(_sd[_h], _sd[_l], _sd["off_high"], _sd["off_low"])
        _stockdays.append(_sd)
        _keep = (set(_sd.loc[(_sd["n_bad"] > 0) | (_sd[["outside", "outside_clean"]].max(axis=1) >= SQUEEZED),
                                 "permno"])
                 | set(picks.loc[picks["day"] == _day, "permno"]))  # bars kept for the images
        _kept.append(_c[_c["permno"].isin(_keep)].assign(day=_day))
    stockdays = pd.concat(_stockdays, ignore_index=True)
    kept = pd.concat(_kept, ignore_index=True)
    return SQUEEZED, kept, stockdays


@app.cell
def _(SQUEEZED, mo, np, pd, stockdays):
    def _summary(d):
        three = {"as stored": "outside", "round lots": "outside_rl", "round lots + V2": "outside_clean"}
        rows = {"stock-days": len(d), "bars with trades but no round-lot price (%)": 100 * d["no_price"].sum() / d["bars"].sum()}
        for _label, _limit in [("range outside the official range at all", 0), (f"squeezed (>= {SQUEEZED:.0%} outside)", SQUEEZED),
                               ("half or less of the height (>= 50% outside)", 0.5)]:
            for _version, _col in three.items():
                rows[f"{_label}, {_version} (%)"] = 100 * ((d[_col] > 0) if _limit == 0 else (d[_col] >= _limit)).mean()
        rows["V2 on round lots: stock-days changed (%)"] = 100 * (d["n_bad"] > 0).mean()
        rows["V2 on round lots: values outside the official range (%)"] = (
            100 * d["n_bad_outside"].sum() / d["n_bad"].sum() if d["n_bad"].sum() else np.nan)
        return pd.Series(rows)


    _s = stockdays.dropna(subset=["outside"])
    check_table = _summary(_s).to_frame(">= $1B")
    mo.vstack([mo.md(f"**The check, {len(_s):,} stock-days with an official range:**"),
               check_table.round(3)])
    return


@app.cell
def _(PRICE_ROWS, SESSION_BARS, VOLUME_ROWS, draw, kept, np, plt):
    def pair(day, permno, n_bars):
        """A stock-day's image as stored (every trade) and as drawn (round lots, V2), side by side with a grey column
        between them."""
        b = kept[(kept["day"] == day) & (kept["permno"] == permno)].set_index("bar").reindex(range(n_bars))
        images = [draw(*(b[f"{prefix}{c}"].to_numpy() for c in ["open", "high", "low", "close"]),
                       b["shares"].to_numpy(dtype=float), PRICE_ROWS, VOLUME_ROWS) for prefix in ["raw_", ""]]
        return np.hstack([images[0], np.full((images[0].shape[0], 2), 128, dtype=np.uint8), images[1]])


    def gallery(rows, title, ncols=5):
        """One tile a stock-day (rows of `stockdays`): as stored | drawn. Tile title: symbol, day, size, and the share
        of the image's price range outside the official range as stored -> drawn; red when V2 changed the stock-day."""
        nrows = max(1, -(-len(rows) // ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 1.05 * nrows + 0.6), squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, r in zip(axes.flat, rows.itertuples()):
            ax.imshow(pair(r.day, r.permno, SESSION_BARS[bool(r.early_close)]), cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest")
            ax.set_title(f"{r.symbol} {r.day:%Y-%m-%d} \\${r.cap_prev / 1e6:.1f}B  "
                         f"{100 * r.outside:.0f}% -> {100 * r.outside_clean:.0f}%", fontsize=6.5,
                         color="tab:red" if r.n_bad > 0 else "black")
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        return fig

    return (gallery,)


@app.cell
def _(gallery, picks, stockdays):
    fig_random = gallery(picks.merge(stockdays, on=["day", "permno"]).sort_values("day"),
                         "One random stock a day: as stored | drawn (red: V2 changed it)")
    fig_random
    return


@app.cell
def _(SEED, SQUEEZED, gallery, mo, stockdays):
    _fixed = stockdays[stockdays["outside"] >= 0.5]
    fig_fixed = gallery(_fixed.sample(min(40, len(_fixed)), random_state=SEED).sort_values("day"),
                        f"Half the height or more outside as stored: 40 of {len(_fixed):,} at random, as stored | drawn")
    mo.vstack([mo.md(f"**Badly squeezed as stored:** {len(_fixed):,} stock-days with half or more of the "
                     f"image's price range outside the official range; {int((_fixed['outside_clean'] < SQUEEZED).sum()):,} "
                     f"of them are below {SQUEEZED:.0%} as drawn."), fig_fixed])
    return


@app.cell
def _(SEED, SQUEEZED, gallery, mo, stockdays):
    _missed = stockdays[stockdays["outside_clean"] >= SQUEEZED]
    fig_missed = gallery(_missed.sample(min(40, len(_missed)), random_state=SEED).sort_values("day"),
                         f"Still squeezed as drawn: 40 of {len(_missed):,} at random, as stored | drawn")
    mo.vstack([mo.md(f"**Still squeezed as drawn:** {len(_missed):,} stock-days with at least "
                     f"{SQUEEZED:.0%} of the image's price range outside the official range."), fig_missed])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The CNN input: four channels

    Each large-cap stock-day becomes a 4 × 64 × 114 array of 0-255 values (51 price rows, one empty row, 12 volume rows, 3 columns a bar). Price bars carry direction, volume bars carry amount.

    | Channel | Price bars | Volume bars |
    |---|---|---|
    | grey | round-lot prices (`px_`), 255; the shape is scaled to the day's highest high and lowest low | volume, every trade, 255; scaled to the day's largest bar, at least one row for a bar with trades |
    | red | retail imbalance: (buys − sells) / (buys + sells + 2) in trades, midpoint rule (below) | retail share: retail dollars / all dollars |
    | green | retail size tilt: retail imbalance in dollars − retail imbalance in trades (below): positive when the retail buys are larger orders than the sells | abnormal retail activity: log((1 + retail trades) / (1 + their mean over the same bar of the previous 20 trading days)), at least 15 days |
    | blue | exchange order flow: (buyer- − seller-initiated shares) / their sum, quote rule, exchange trades only (mostly not retail) | odd-lot share: trades under 100 shares / all off-exchange trades |

    **Standardized by bar:** within each day and bar, across every large cap, each of the six color values is winsorized at the 1st and 99th percentiles and standardized; −3 standard deviations is pixel 0, the cross-sectional average 128, +3 is 255, linear in between, and a missing value is 128. So a color says how this stock's bar differs from the other large caps' at the same moment: market-wide moves, the time of day and the drift over the years drop out, and nothing after the cutoff is used. The grey channel is a shape and is not standardized. Pixels the chart does not draw are 0 in every channel.

    **Retail imbalance as a posterior mean:** with p the probability that a retail trade in the bar is a buy, a uniform prior p ~ Beta(1, 1) and b buys and s sells give the posterior Beta(1 + b, 1 + s), whose mean of the imbalance 2p − 1 is (b − s) / (b + s + 2). A bar with few retail trades is pulled toward balanced in proportion to how few (one buy: +0.33; ten: +0.83; a hundred: +0.98), a bar without any is 0, and no value is missing. That is the raw imbalance times n / (n + 2), n the signed retail trades; the size tilt shrinks the imbalance in dollars by the same factor before taking the difference.

    In the history of abnormal retail activity, a day the stock was in the panel but did not trade in that bar counts as 0 retail trades; a day it was not in the panel does not count. A bar whose trades were all odd lots has no price pixels, so its three price values are not drawn; its volume bar still is.
    """)
    return


@app.cell
def _(PRICES, np, pd):
    HISTORY_DAYS, HISTORY_MIN = 20, 15  # the same bar on the previous 20 trading days, at least 15 of them
    Z_CLIP = 3.0  # -3 standard deviations is pixel 0, +3 is 255; the cross-sectional average is 128
    FLOW_COLUMNS = ["shares", "dollars", "mid_buy_trades", "mid_sell_trades", "mid_buy_dollars", "mid_sell_dollars",
                    "off_trades", "off_odd_trades", "exch_qr_buy_shares", "exch_qr_sell_shares"]
    PRICE_FEATURES = ["retail_imbalance", "size_tilt", "exchange_imbalance"]  # red, green, blue on price bars
    VOLUME_FEATURES = ["retail_share", "abnormal_activity", "odd_lot_share"]  # red, green, blue on volume bars


    PRIOR_TRADES = 2  # the uniform prior Beta(1, 1) on the share of retail trades that are buys: one buy, one sell


    def retail(buy_trades, sell_trades):
        """Signed retail trades and their imbalance: the posterior mean of 2p - 1, p the share of retail trades that
        are buys, under the prior Beta(1, 1): (buys - sells) / (buys + sells + 2). Never missing; 0 without trades."""
        signed = buy_trades + sell_trades
        return signed, (buy_trades - sell_trades) / (signed + PRIOR_TRADES)


    def flow(b):
        """Each bar's raw values from its bars columns (FLOW_COLUMNS): retail trades and imbalance (midpoint rule, in
        trades), the size tilt (the imbalance in dollars, shrunk by the same n / (n + 2), minus the one in trades),
        retail share of dollars, exchange order-flow imbalance (quote rule, in shares) and odd-lot share of the
        off-exchange trades."""
        trades, imbalance = retail(b["mid_buy_trades"], b["mid_sell_trades"])
        dollars = b["mid_buy_dollars"] + b["mid_sell_dollars"]
        dollar_imbalance = ((b["mid_buy_dollars"] - b["mid_sell_dollars"]) / dollars.where(dollars > 0)
                            * trades / (trades + PRIOR_TRADES)).fillna(0.0)
        exchange = b["exch_qr_buy_shares"] + b["exch_qr_sell_shares"]
        return pd.DataFrame({
            "retail_trades": trades, "retail_imbalance": imbalance, "size_tilt": dollar_imbalance - imbalance,
            "retail_share": (b["mid_buy_dollars"] + b["mid_sell_dollars"]) / b["dollars"].where(b["dollars"] > 0),
            "exchange_imbalance": (b["exch_qr_buy_shares"] - b["exch_qr_sell_shares"]) / exchange.where(exchange > 0),
            "odd_lot_share": b["off_odd_trades"] / b["off_trades"].where(b["off_trades"] > 0),
        }, index=b.index)


    def against_history(today, history):
        """Add abnormal_activity to one day's bars (permno, bar, retail_trades): log((1 + retail trades) / (1 + their
        mean over the same bar of the history)), with at least HISTORY_MIN days. history: permno, bar, retail_trades
        for the previous HISTORY_DAYS trading days, one row for every bar of every day a stock was in the panel."""
        g = history.groupby(["permno", "bar"])["retail_trades"]
        t = today.join(pd.DataFrame({"trades_mean": g.mean(), "days": g.size()}), on=["permno", "bar"])
        activity = np.log((1 + t["retail_trades"]) / (1 + t["trades_mean"]))
        return today.assign(abnormal_activity=activity.where(t["days"] >= HISTORY_MIN))


    def standardize_by_bar(df, columns):
        """Each column's pixel value (column_px, uint8), across the stocks of each bar (df holds one day): winsorized at
        the 1st and 99th percentiles, standardized, -Z_CLIP to 0, the mean to 128, +Z_CLIP to 255; missing to 128."""
        out = {}
        for c in columns:
            by_bar = df[c].groupby(df["bar"])
            w = df[c].clip(by_bar.transform(lambda x: x.quantile(0.01)), by_bar.transform(lambda x: x.quantile(0.99)))
            z = (w - w.groupby(df["bar"]).transform("mean")) / w.groupby(df["bar"]).transform("std")
            out[f"{c}_px"] = np.where(z.isna(), 128, np.round((z.clip(-Z_CLIP, Z_CLIP) + Z_CLIP) / (2 * Z_CLIP) * 255))
        return df.assign(**{k: v.astype(np.uint8) for k, v in out.items()})


    def draw_four(b, price_rows=51, volume_rows=12):
        """One stock-day's image (uint8, 4 x (price_rows + 1 + volume_rows) x 3n) from its n session bars in order (NaN
        rows for bars without trades): round-lot open, high, low, close, shares, and the six *_px pixel values."""
        o, h, l, c = (b[_k].to_numpy(dtype=float) for _k in PRICES)
        n = len(b)
        top, bottom = np.nanmax(h), np.nanmin(l)
        span = top - bottom if top > bottom else 1.0

        def row(p):  # a price's row, 0 at the top; NaN draws nothing
            return np.round((top - p) / span * (price_rows - 1))

        r = np.arange(price_rows)[:, None]
        price = np.zeros((price_rows, 3 * n), dtype=bool)
        price[:, 0::3] = r == row(o)
        price[:, 1::3] = (r >= row(h)) & (r <= row(l))
        price[:, 2::3] = r == row(c)
        v = b["shares"].to_numpy(dtype=float)
        filled = np.where(v > 0, np.maximum(1, np.round(v / np.nanmax(v) * volume_rows)), 0)
        volume = np.zeros((volume_rows, 3 * n), dtype=bool)
        volume[:, 1::3] = np.arange(volume_rows)[:, None] >= volume_rows - filled
        chart = np.vstack([price, np.zeros((1, 3 * n), dtype=bool), volume])

        def values(name):  # a bar's pixel value over its 3 columns
            return np.repeat(np.nan_to_num(b[f"{name}_px"].to_numpy(dtype=float), nan=128), 3)[None, :]

        in_price = np.arange(chart.shape[0])[:, None] < price_rows
        layers = [chart * 255] + [np.where(chart, np.where(in_price, values(_p), values(_v)), 0)
                                  for _p, _v in zip(PRICE_FEATURES, VOLUME_FEATURES)]
        return np.stack(layers).astype(np.uint8)

    return (
        FLOW_COLUMNS,
        HISTORY_DAYS,
        PRICE_FEATURES,
        VOLUME_FEATURES,
        against_history,
        draw_four,
        flow,
        retail,
        standardize_by_bar,
    )


@app.cell
def _(
    FLOW_COLUMNS,
    HISTORY_DAYS,
    LARGE_CAP,
    PRICES,
    PRICE_FEATURES,
    SEED,
    SESSION_BARS,
    TAQ_DIR,
    VOLUME_FEATURES,
    against_history,
    draw_four,
    flow,
    mo,
    np,
    pd,
    pq,
    retail,
    standardize_by_bar,
):
    N_EXAMPLE_DAYS, PER_DAY = 4, 2
    _panel = pq.read_table(TAQ_DIR / "panel_train.parquet",
                           columns=["day", "permno", "symbol", "cap_prev", "early_close"]).to_pandas()
    _days = sorted(_panel["day"].unique())
    _early = _panel.groupby("day")["early_close"].first()
    _members = _panel.groupby("day")["permno"].agg(set)
    _chosen = sorted(np.random.default_rng(SEED).choice(len(_days) - HISTORY_DAYS, N_EXAMPLE_DAYS, replace=False)
                     + HISTORY_DAYS)  # days with a full history before them
    _session = [("piece", "==", "session")]


    def _read(day, columns, table="bars"):
        return pq.read_table(TAQ_DIR / "train" / f"{pd.Timestamp(day):%Y%m%d}" / f"{table}.parquet",
                             columns=["permno", "bar", *columns], filters=_session).to_pandas()


    examples, _sections = [], []
    for _i in mo.status.progress_bar(_chosen, title="Building the example days"):
        _day = _days[_i]
        _large = _panel[(_panel["day"] == _day) & (_panel["cap_prev"] >= LARGE_CAP)]
        _stocks = set(_large["permno"])
        _b = _read(_day, FLOW_COLUMNS)
        _b = _b[_b["permno"].isin(_stocks)].reset_index(drop=True)
        _b[FLOW_COLUMNS] = _b[FLOW_COLUMNS].astype(float)
        _today = pd.concat([_b[["permno", "bar"]], flow(_b)], axis=1)

        _history = []
        for _d in _days[_i - HISTORY_DAYS:_i]:  # every bar of each day a stock was in the panel; no trades: 0
            _in = sorted(_stocks & _members[_d])
            _grid = pd.MultiIndex.from_product([_in, range(SESSION_BARS[bool(_early[_d])])],
                                               names=["permno", "bar"]).to_frame(index=False)
            _h = _grid.merge(_read(_d, ["mid_buy_trades", "mid_sell_trades"]), on=["permno", "bar"], how="left")
            _h = _h.fillna({"mid_buy_trades": 0, "mid_sell_trades": 0}).astype({"mid_buy_trades": float, "mid_sell_trades": float})
            _trades, _ = retail(_h["mid_buy_trades"], _h["mid_sell_trades"])
            _history.append(_h[["permno", "bar"]].assign(retail_trades=_trades))
        _today = standardize_by_bar(against_history(_today, pd.concat(_history, ignore_index=True)),
                                    PRICE_FEATURES + VOLUME_FEATURES)
        _sections.append(_today.assign(day=_day))

        _px = _read(_day, [f"px_{_p}" for _p in PRICES], "prices")
        _n = SESSION_BARS[bool(_early[_day])]
        for _r in _large.sample(PER_DAY, random_state=SEED).itertuples():
            _sb = (_px[_px["permno"] == _r.permno].set_index("bar").drop(columns="permno")
                   .rename(columns=lambda _c: _c.removeprefix("px_"))
                   .join(_b[_b["permno"] == _r.permno].set_index("bar")[["shares"]], how="outer")
                   .join(_today[_today["permno"] == _r.permno].set_index("bar")
                         [[f"{_c}_px" for _c in PRICE_FEATURES + VOLUME_FEATURES]], how="outer")
                   .astype(float).reindex(range(_n)))
            examples.append({"day": _day, "permno": _r.permno, "symbol": _r.symbol, "cap_prev": _r.cap_prev,
                             "image": draw_four(_sb)})
    cross_sections = pd.concat(_sections, ignore_index=True)
    mo.md(f"**Examples:** {len(examples)} stock-days on {N_EXAMPLE_DAYS} days "
          f"({', '.join(f'{pd.Timestamp(_days[_i]):%Y-%m-%d}' for _i in _chosen)}); "
          f"{len(cross_sections):,} large-cap bars standardized on those days.")
    return cross_sections, examples


@app.cell
def _(examples, np, plt):
    from matplotlib.colors import LinearSegmentedColormap

    _maps = ["gray"] + [LinearSegmentedColormap.from_list(_n, ["black", _c])
                        for _n, _c in [("red", "red"), ("green", "lime"), ("blue", "deepskyblue")]]
    _titles = ["grey: prices | volume", "red: retail imbalance | retail share",
               "green: size tilt | abnormal activity", "blue: exchange flow | odd-lot share",
               "red, green, blue together"]
    fig_four, _axes = plt.subplots(len(examples), 5, figsize=(17, 1.35 * len(examples) + 0.7), squeeze=False)
    for _i, _e in enumerate(examples):
        for _j in range(4):
            _axes[_i, _j].imshow(_e["image"][_j], cmap=_maps[_j], vmin=0, vmax=255, interpolation="nearest")
        _axes[_i, 4].imshow(np.moveaxis(_e["image"][1:], 0, -1), interpolation="nearest")
        _axes[_i, 0].set_ylabel(f"{_e['symbol']}\n{_e['day']:%Y-%m-%d}\n\\${_e['cap_prev'] / 1e6:.0f}B", fontsize=7)
        for _j, _ax in enumerate(_axes[_i]):
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _i == 0:
                _ax.set_title(_titles[_j], fontsize=8)
    fig_four.suptitle("The four channels: price bars | volume bars. Color: each bar standardized across the large caps "
                      "(128 = average, brighter = higher)", fontsize=9)
    fig_four.tight_layout()
    fig_four
    return


@app.cell
def _(PRICE_FEATURES, VOLUME_FEATURES, cross_sections, mo):
    _px = cross_sections[[f"{_c}_px" for _c in PRICE_FEATURES + VOLUME_FEATURES]].astype(float)
    _px.columns = [f"{_c} ({_w})" for _c, _w in zip(PRICE_FEATURES + VOLUME_FEATURES, ["price"] * 3 + ["volume"] * 3)]
    channel_corr = _px.corr().round(2)
    mo.vstack([mo.md(f"**How much the color values overlap:** correlations of the six pixel values across the "
                     f"{len(_px):,} large-cap bars of the example days."), channel_corr])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two bars before the session: the previous day's last ten minutes, then the night

    The session image starts at 09:30, so everything between the previous trading day's cutoff and the open is missing. Two more bars, drawn first, hold it:

    1. **The previous day's last ten minutes** (`prev_last10`: from its cutoff to its close), an ordinary ten-minute bar: its round-lot prices (`px_`, as the session bars), its volume and its colors from its `bars` row, its volume capped as below. Every image stops at the cutoff, so no other image shows these ten minutes, among the busiest of the day.
    2. **The night** (`prev_evening`, the previous day's close to midnight, and `pre_open`, midnight to 09:30 on the decision day), one bar:
        - **Its open is the previous day's last round-lot price** (the first bar's close), **its close the last trade before 09:30** (the pre-open's, else the evening's), and **its high and low are just the larger and the smaller of the two**, so it spans the overnight move and nothing else. The price-only pull has no round-lot prices outside the session and the last ten minutes, so the extremes of the evening and pre-open, often single odd lots, are left out; the close is still a trade of any size.
        - **Its colors** are the evening's and pre-open's counts summed, so pre-open retail buying shows on its own instead of inside the last ten minutes' far larger flow.

    **The session alone sets the volume scale:** both bars are scaled like the session bars but capped at the session's tallest bar. The last ten minutes are usually the busiest bar of all, and on their own scale they would flatten every session bar; capped, their colors still carry their flow.

    The auctions are not in the bars (excluded by sale condition), so neither the previous day's closing auction nor today's opening auction is in either bar. Each bar gets the six values of a session bar (`flow`), standardized across the large caps' same bar that day, and abnormal retail activity compares it with the same bar on the previous 20 trading days (at least 15; a day the stock was in the panel without trades in it counts as 0). The image becomes 4 × 64 × 120.

    The examples are the stock-days of the previous section; the table below them covers every large cap on those days.
    """)
    return


@app.cell
def _(
    FLOW_COLUMNS,
    HISTORY_DAYS,
    LARGE_CAP,
    PRICES,
    PRICE_FEATURES,
    SESSION_BARS,
    TAQ_DIR,
    VOLUME_FEATURES,
    against_history,
    cross_sections,
    draw_four,
    examples,
    flow,
    mo,
    np,
    pd,
    pq,
    retail,
    standardize_by_bar,
):
    LAST10_BAR, NIGHT_BAR = -2, -1  # their bar numbers: before the session's bar 0
    NIGHT_PIECES = ["prev_evening", "pre_open"]


    def last10_bar(b, px):
        """The previous day's last ten minutes as a bar: its round-lot prices (px: permno, px_open, px_high, px_low,
        px_close) and its counts (b: the prev_last10 rows of bars, with FLOW_COLUMNS), one row per stock that traded."""
        stocks = pd.Index(sorted(b["permno"].unique()), name="permno")
        prices = (px.set_index("permno")[[f"px_{_p}" for _p in PRICES]]
                  .rename(columns=lambda _c: _c.removeprefix("px_")).astype(float).reindex(stocks))
        return prices.join(b.set_index("permno")[FLOW_COLUMNS].astype(float)).reset_index().assign(bar=LAST10_BAR)


    def night_bar(b, start):
        """The evening and pre-open as one bar. b: their bars rows (permno, piece, close, FLOW_COLUMNS); start: each
        stock's price when the night begins (the previous day's last round-lot price). Open: that price; close: the
        last trade before 09:30 (the pre-open's, else the evening's), any size; high and low: the larger and the
        smaller of the two. Counts summed over the two pieces."""
        stocks = pd.Index(sorted(b["permno"].unique()), name="permno")
        close = {_p: b[b["piece"] == _p].set_index("permno")["close"].astype(float).reindex(stocks) for _p in NIGHT_PIECES}
        out = pd.DataFrame({"open": start.reindex(stocks), "close": close["pre_open"].fillna(close["prev_evening"])})
        out["high"], out["low"] = out[["open", "close"]].max(axis=1), out[["open", "close"]].min(axis=1)
        counts = b.groupby("permno")[FLOW_COLUMNS].sum().reindex(stocks).astype(float)
        return out[PRICES].join(counts).reset_index().assign(bar=NIGHT_BAR)


    _panel = pq.read_table(TAQ_DIR / "panel_train.parquet",
                           columns=["day", "permno", "cap_prev", "early_close"]).to_pandas()
    _days = list(pd.DatetimeIndex(sorted(_panel["day"].unique())))
    _members = _panel.groupby("day")["permno"].agg(set)
    _early = _panel.groupby("day")["early_close"].first()
    _before = ["prev_last10", *NIGHT_PIECES]


    def _read(day, columns, pieces, table="bars"):
        return pq.read_table(TAQ_DIR / "train" / f"{pd.Timestamp(day):%Y%m%d}" / f"{table}.parquet",
                             columns=["permno", "piece", "bar", *columns],
                             filters=[("piece", "in", pieces)]).to_pandas()


    _pixels = [f"{_c}_px" for _c in PRICE_FEATURES + VOLUME_FEATURES]
    overnight_examples, _stats = [], []
    for _day in mo.status.progress_bar(sorted({pd.Timestamp(_e["day"]) for _e in examples}),
                                       title="Building the two bars before the session"):
        _i = _days.index(_day)
        _stocks = set(_panel.loc[(_panel["day"] == _day) & (_panel["cap_prev"] >= LARGE_CAP), "permno"])
        _b = _read(_day, [*PRICES, *FLOW_COLUMNS], _before)
        _b = _b[_b["permno"].isin(_stocks)]
        _px = _read(_day, [f"px_{_p}" for _p in PRICES], ["prev_last10"], "prices")
        _last10 = last10_bar(_b[_b["piece"] == "prev_last10"], _px[_px["permno"].isin(_stocks)])
        _night = night_bar(_b[_b["piece"].isin(NIGHT_PIECES)], _last10.set_index("permno")["close"])
        _extra = pd.concat([_last10, _night], ignore_index=True)
        _today = pd.concat([_extra[["permno", "bar"]], flow(_extra)], axis=1)

        _history = []
        for _d in _days[_i - HISTORY_DAYS:_i]:  # each day a stock was in the panel; no trades in a bar: 0
            _in = sorted(_stocks & _members[_d])
            _h = _read(_d, ["mid_buy_trades", "mid_sell_trades"], _before)
            _h["bar"] = np.where(_h["piece"] == "prev_last10", LAST10_BAR, NIGHT_BAR)
            _h = (_h.groupby(["permno", "bar"])[["mid_buy_trades", "mid_sell_trades"]].sum()
                  .reindex(pd.MultiIndex.from_product([_in, [LAST10_BAR, NIGHT_BAR]], names=["permno", "bar"]),
                           fill_value=0).astype(float))
            _trades, _ = retail(_h["mid_buy_trades"], _h["mid_sell_trades"])
            _history.append(_h.index.to_frame(index=False).assign(retail_trades=_trades.to_numpy()))
        _today = standardize_by_bar(against_history(_today, pd.concat(_history, ignore_index=True)),
                                    PRICE_FEATURES + VOLUME_FEATURES)

        _spx = (_read(_day, [f"px_{_p}" for _p in PRICES], ["session"], "prices")
                .rename(columns=lambda _c: _c.removeprefix("px_")).astype({_p: float for _p in PRICES}))
        _sb = _read(_day, ["shares"], ["session"])
        _s = pd.DataFrame({"session_high": _spx.groupby("permno")["high"].max(),
                           "session_low": _spx.groupby("permno")["low"].min(),
                           "session_max_shares": _sb.groupby("permno")["shares"].max()}
                          ).reindex(pd.Index(sorted(_stocks), name="permno"))
        _s = (_s.join(_last10.set_index("permno")[["high", "low", "shares"]].add_prefix("last10_"))
              .join(_night.set_index("permno")[["high", "low", "shares"]].add_prefix("night_")))
        _s["evening"] = _s.index.isin(_b.loc[_b["piece"] == "prev_evening", "permno"])
        _s["pre_open"] = _s.index.isin(_b.loc[_b["piece"] == "pre_open", "permno"])
        _stats.append(_s.reset_index().assign(day=_day))

        _n = SESSION_BARS[bool(_early[_day])]
        _cs = cross_sections[pd.to_datetime(cross_sections["day"]) == _day]
        for _e in [_e for _e in examples if pd.Timestamp(_e["day"]) == _day]:
            _p = _e["permno"]
            _session = (_spx[_spx["permno"] == _p].set_index("bar")[PRICES]
                        .join(_sb[_sb["permno"] == _p].set_index("bar")[["shares"]], how="outer")
                        .join(_cs[_cs["permno"] == _p].set_index("bar")[_pixels], how="outer")
                        .astype(float).reindex(range(_n)))
            _first = (_extra[_extra["permno"] == _p].set_index("bar")[[*PRICES, "shares"]]
                      .join(_today[_today["permno"] == _p].set_index("bar")[_pixels])
                      .astype(float).reindex([LAST10_BAR, NIGHT_BAR]))
            _image = pd.concat([_first, _session])
            # the session alone sets the volume scale: the two bars before it are capped at its tallest bar
            _image.loc[[LAST10_BAR, NIGHT_BAR], "shares"] = np.fmin(_image.loc[[LAST10_BAR, NIGHT_BAR], "shares"],
                                                                    _session["shares"].max())
            overnight_examples.append({**_e, "with_overnight": draw_four(_image),
                                       "colors": _image[_pixels].to_numpy(dtype=float)})
    overnight_stats = pd.concat(_stats, ignore_index=True)
    return overnight_examples, overnight_stats


@app.cell
def _(np, overnight_examples, plt):
    _titles = ["grey, session only", "grey, with the two bars (left of the yellow line)",
               "red, green, blue, with the two bars"]
    fig_overnight, _axes = plt.subplots(len(overnight_examples), 3, figsize=(13, 1.35 * len(overnight_examples) + 0.7),
                                        squeeze=False)
    for _i, _e in enumerate(overnight_examples):
        _axes[_i, 0].imshow(_e["image"][0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        _axes[_i, 1].imshow(_e["with_overnight"][0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        _axes[_i, 2].imshow(np.moveaxis(_e["with_overnight"][1:], 0, -1), interpolation="nearest")
        _axes[_i, 0].set_ylabel(f"{_e['symbol']}\n{_e['day']:%Y-%m-%d}\n\\${_e['cap_prev'] / 1e6:.0f}B", fontsize=7)
        for _j, _ax in enumerate(_axes[_i]):
            if _j > 0:
                _ax.axvline(5.5, color="yellow", lw=0.6)
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _i == 0:
                _ax.set_title(_titles[_j], fontsize=8)
    fig_overnight.suptitle("The same stock-days with two bars first: the previous day's last ten minutes, then the "
                           "night (its evening and today's pre-open)", fontsize=9)
    fig_overnight.tight_layout()
    fig_overnight
    return


@app.cell
def _(mo, np, overnight_stats, pd):
    _s = overnight_stats
    _top = np.fmax(np.fmax(_s["session_high"], _s["last10_high"]), _s["night_high"])
    _bottom = np.fmin(np.fmin(_s["session_low"], _s["last10_low"]), _s["night_low"])
    _share = (_s["session_high"] - _s["session_low"]) / (_top - _bottom)
    _last10_beyond = (_s["last10_high"] > _s["session_high"]) | (_s["last10_low"] < _s["session_low"])
    _night_beyond = ((_s["night_high"] > np.fmax(_s["session_high"], _s["last10_high"]))
                     | (_s["night_low"] < np.fmin(_s["session_low"], _s["last10_low"])))
    overnight_summary = pd.Series({
        "large-cap stock-days": len(_s),
        "with evening trades, %": 100 * _s["evening"].mean(),
        "with pre-open trades, %": 100 * _s["pre_open"].mean(),
        "last ten minutes reach beyond the session's range, %": 100 * _last10_beyond.mean(),
        "the night's move reaches beyond both, %": 100 * _night_beyond.mean(),
        "session's share of the price rows, median, %": 100 * _share.median(),
        "session's share of the price rows, 10th percentile, %": 100 * _share.quantile(0.1),
        "session squeezed into less than half the price rows, %": 100 * (_share < 0.5).mean(),
        "last ten minutes capped (taller than the session's tallest bar), %":
            100 * (_s["last10_shares"] > _s["session_max_shares"]).mean(),
        "last ten minutes / tallest session bar, median": (_s["last10_shares"] / _s["session_max_shares"]).median(),
        "the night capped (taller than the session's tallest bar), %":
            100 * (_s["night_shares"] > _s["session_max_shares"]).mean(),
    }).round(2).to_frame("value")
    mo.vstack([mo.md("**What the two bars do to the images,** every large cap on the example days:"),
               mo.ui.table(overnight_summary.reset_index(names="measure"), selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 21 days in hourly bars

    A look at the multi-day image proposed in `explanations/CNN/cnn.md` (multi-day images), on the same example stock-days: the 21 trading days up to the decision day's cutoff, 7 bars a day, 4 × 96 × 441 (76 price rows, an empty row, 19 volume rows: over 21 days the price range is wide, and at 64 rows most hourly bars were one or two rows tall).

    - **Hourly bars:** 09:30, 10:30, 11:30, 12:30, 13:30 and 14:30, then 15:30 to the close. A past day's last bar includes its 15:50-16:00 (the next decision day's `prev_last10`); today's stops at the 15:50 cutoff. Every day takes 7 slots, so an hour sits at the same place every day; on early closes the slots after 12:50 (13:00 on past days) stay empty. Each bar is built from the ten-minute bars: the first round-lot open, the highest high, the lowest low, the last round-lot close, and the counts summed.
    - **Prices adjusted for splits** to the decision day's terms (a past price over the product of CRSP's `facprc_t` on the days after it), and scaled from the window's highest high or moving average to its lowest.
    - **The moving average line:** at each bar, the average close of the 147 hourly bars up to it (21 days, so it needs the 21 days before the window too), drawn in grey through each bar's middle column and joined column by column; it has no color.
    - **Volume:** shares, every trade, scaled to the window's tallest bar, at least one row for a bar that traded.
    - **Colors:** the six values of each hourly bar, standardized across that day's large caps in each hour; abnormal retail activity compares a bar with the same hour over the 20 trading days before its day.
    - **No night bar:** the night shows only as the jump from one day's last bar to the next day's first.

    The thin lines between days in the figure are drawn on the plot, not in the image.
    """)
    return


@app.cell
def _(
    FLOW_COLUMNS,
    HISTORY_DAYS,
    LARGE_CAP,
    PRICES,
    PRICE_FEATURES,
    SESSION_BARS,
    TAQ_DIR,
    VOLUME_FEATURES,
    against_history,
    examples,
    flow,
    mo,
    np,
    pd,
    pq,
    retail,
    standardize_by_bar,
):
    WINDOW_DAYS, DAY_SLOTS = 21, 7  # 21 days of 7 hourly bars
    MA_BARS = WINDOW_DAYS * DAY_SLOTS  # the moving average: the 147 hourly closes up to each bar
    W_PRICE_ROWS, W_VOLUME_ROWS = 76, 19  # 96 rows with the empty one, as Jiang, Kelly and Xiu's 60-day images

    _panel = pq.read_table(TAQ_DIR / "panel_train.parquet",
                           columns=["day", "permno", "cap_prev", "early_close", "facprc_t"]).to_pandas()
    _days = list(pd.DatetimeIndex(sorted(_panel["day"].unique())))
    _members = _panel.groupby("day")["permno"].agg(set)
    _early = _panel.groupby("day")["early_close"].first()
    _factor = _panel.set_index(["day", "permno"])["facprc_t"]


    def _read(day, table, columns, piece):
        return pq.read_table(TAQ_DIR / "train" / f"{pd.Timestamp(day):%Y%m%d}" / f"{table}.parquet",
                             columns=["permno", "bar", *columns], filters=[("piece", "==", piece)]).to_pandas()


    def hourly(i, last):
        """Day i's hourly bars (permno, bar = hour slot 0-6) for every stock in its files: the counts summed and the
        round-lot open, high, low and close of its ten-minute bars, plus its 15:50-16:00 from day i + 1's prev_last10
        when day i is before `last`."""
        px_cols = [f"px_{_p}" for _p in PRICES]
        rename = lambda df: df.rename(columns=lambda _c: _c.removeprefix("px_"))
        b = _read(_days[i], "bars", FLOW_COLUMNS, "session")
        px = rename(_read(_days[i], "prices", px_cols, "session"))
        rows = b.merge(px, on=["permno", "bar"], how="outer").assign(order=lambda d: d["bar"])
        last_slot = (SESSION_BARS[bool(_early[_days[i]])] - 1) // 6
        rows["bar"] = np.minimum(rows["bar"] // 6, 6)
        if i < last:  # the day's last ten minutes, from the next decision day
            b10 = _read(_days[i + 1], "bars", FLOW_COLUMNS, "prev_last10")
            px10 = rename(_read(_days[i + 1], "prices", px_cols, "prev_last10"))
            ten = b10.merge(px10, on=["permno", "bar"], how="outer").assign(bar=last_slot, order=99)
            rows = pd.concat([rows, ten], ignore_index=True)
        rows = rows.sort_values(["permno", "bar", "order"])
        rows[PRICES] = rows[PRICES].astype(float)
        g = rows.groupby(["permno", "bar"])
        out = g[FLOW_COLUMNS].sum().astype(float)
        out["open"], out["high"], out["low"], out["close"] = (g["open"].first(), g["high"].max(), g["low"].min(),
                                                              g["close"].last())
        return out.reset_index()


    def colors(i, tables):
        """The six pixel values of day i's hourly bars for its large caps (tables: hourly bars by day index, including
        the 20 days before i)."""
        stocks = set(_panel.loc[(_panel["day"] == _days[i]) & (_panel["cap_prev"] >= LARGE_CAP), "permno"])
        t = tables[i][tables[i]["permno"].isin(stocks)].reset_index(drop=True)
        today = pd.concat([t[["permno", "bar"]], flow(t)], axis=1)
        history = []
        for j in range(i - HISTORY_DAYS, i):  # each day a stock was in the panel; no trades in an hour: 0
            slots = (SESSION_BARS[bool(_early[_days[j]])] - 1) // 6 + 1
            grid = pd.MultiIndex.from_product([sorted(stocks & _members[_days[j]]), range(slots)],
                                              names=["permno", "bar"]).to_frame(index=False)
            h = grid.merge(tables[j][["permno", "bar", "mid_buy_trades", "mid_sell_trades"]], how="left").fillna(0)
            history.append(h[["permno", "bar"]].assign(
                retail_trades=retail(h["mid_buy_trades"].astype(float), h["mid_sell_trades"].astype(float))[0]))
        return standardize_by_bar(against_history(today, pd.concat(history, ignore_index=True)),
                                  PRICE_FEATURES + VOLUME_FEATURES)


    def draw_window(w):
        """One image (uint8, 4 x 96 x 3n) from a window's n hourly bars in order (open, high, low, close, shares, ma and
        the six *_px values; NaN where a bar has no trades): bars as in draw_four, the moving average as a grey line."""
        o, h, l, c, ma = (w[_k].to_numpy(dtype=float) for _k in ["open", "high", "low", "close", "ma"])
        n = len(w)
        top, bottom = np.nanmax(np.r_[h, ma]), np.nanmin(np.r_[l, ma])
        span = top - bottom if top > bottom else 1.0
        row = lambda p: np.round((top - p) / span * (W_PRICE_ROWS - 1))
        r = np.arange(W_PRICE_ROWS)[:, None]
        price = np.zeros((W_PRICE_ROWS, 3 * n), dtype=bool)
        price[:, 0::3], price[:, 2::3] = r == row(o), r == row(c)
        price[:, 1::3] = (r >= row(h)) & (r <= row(l))
        line = np.zeros_like(price)
        mid, ok = 3 * np.arange(n) + 1, ~np.isnan(ma)
        if ok.sum() > 1:
            cols = np.arange(mid[ok][0], mid[ok][-1] + 1)
            line[row(np.interp(cols, mid[ok], ma[ok])).astype(int), cols] = True
        v = w["shares"].to_numpy(dtype=float)
        filled = np.where(v > 0, np.maximum(1, np.round(v / np.nanmax(v) * W_VOLUME_ROWS)), 0)
        volume = np.zeros((W_VOLUME_ROWS, 3 * n), dtype=bool)
        volume[:, 1::3] = np.arange(W_VOLUME_ROWS)[:, None] >= W_VOLUME_ROWS - filled
        gap = np.zeros((1, 3 * n), dtype=bool)
        chart, ma_only = np.vstack([price, gap, volume]), np.vstack([line & ~price, gap, volume & False])
        values = lambda name: np.repeat(np.nan_to_num(w[f"{name}_px"].to_numpy(dtype=float), nan=128), 3)[None, :]
        in_price = np.arange(chart.shape[0])[:, None] < W_PRICE_ROWS
        layers = [(chart | ma_only) * 255] + [np.where(chart, np.where(in_price, values(_p), values(_v)), 0)
                                              for _p, _v in zip(PRICE_FEATURES, VOLUME_FEATURES)]
        return np.stack(layers).astype(np.uint8)


    _pixels = [f"{_c}_px" for _c in PRICE_FEATURES + VOLUME_FEATURES]
    window_examples = []
    for _day in mo.status.progress_bar(sorted({pd.Timestamp(_e["day"]) for _e in examples}),
                                       title="Building the 21-day windows"):
        _d = _days.index(_day)
        _first = _d - 2 * WINDOW_DAYS + 1  # the window's 21 days, and the 21 before for the moving average
        _tables = {_i: hourly(_i, _d) for _i in range(min(_first, _d - WINDOW_DAYS + 1 - HISTORY_DAYS), _d + 1)}
        _colors = {_i: colors(_i, _tables) for _i in range(_d - WINDOW_DAYS + 1, _d + 1)}
        for _e in [_e for _e in examples if pd.Timestamp(_e["day"]) == _day]:
            _p, _bars = _e["permno"], []
            for _i in range(_first, _d + 1):
                _later = [_factor.get((_days[_j], _p), 1.0) for _j in range(_i + 1, _d + 1)]
                _adj = float(np.prod(_later)) if _later else 1.0  # a past price in the decision day's terms
                _t = (_tables[_i][_tables[_i]["permno"] == _p].set_index("bar")[[*PRICES, "shares"]]
                      .reindex(range(DAY_SLOTS)))
                _t[PRICES] = _t[PRICES] / _adj
                if _i in _colors:
                    _t = _t.join(_colors[_i][_colors[_i]["permno"] == _p].set_index("bar")[_pixels])
                _bars.append(_t.assign(day=_i))
            _all = pd.concat(_bars, ignore_index=True)
            _closes = _all["close"].dropna()
            _all["ma"] = _closes.rolling(MA_BARS, min_periods=MA_BARS // 2).mean().reindex(_all.index)
            _window = _all[_all["day"] > _d - WINDOW_DAYS].reset_index(drop=True)
            window_examples.append({**_e, "window": draw_window(_window)})
    return DAY_SLOTS, WINDOW_DAYS, window_examples


@app.cell
def _(DAY_SLOTS, WINDOW_DAYS, np, plt, window_examples):
    fig_window, _axes = plt.subplots(len(window_examples), 2, figsize=(17, 1.7 * len(window_examples) + 0.8),
                                     squeeze=False)
    for _i, _e in enumerate(window_examples):
        _axes[_i, 0].imshow(_e["window"][0], cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
        _axes[_i, 1].imshow(np.moveaxis(_e["window"][1:], 0, -1), interpolation="nearest", aspect="auto")
        _axes[_i, 0].set_ylabel(f"{_e['symbol']}\n{_e['day']:%Y-%m-%d}\n\\${_e['cap_prev'] / 1e6:.0f}B", fontsize=7)
        for _ax in _axes[_i]:
            for _k in range(1, WINDOW_DAYS):
                _ax.axvline(3 * DAY_SLOTS * _k - 0.5, color="steelblue", lw=0.4, alpha=0.6)
            _ax.set_xticks([])
            _ax.set_yticks([])
    _axes[0, 0].set_title("grey: hourly prices, the 21-day moving average and volume", fontsize=8)
    _axes[0, 1].set_title("red, green, blue: the six values of each hourly bar", fontsize=8)
    fig_window.suptitle("21 trading days up to the decision day's cutoff, 7 hourly bars a day (4 x 96 x 441); "
                        "lines between days drawn on the plot only", fontsize=9)
    fig_window.tight_layout()
    fig_window
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The retail strip: three numbers for each of the 40 bars

    The short-term retail branch of the three-branch model sees no picture. Each of the one-day image's 40 bars (the previous day's last ten minutes, the night, the session's ten-minute bars) becomes a column of three numbers, read by a 1D CNN along time:

    - **retail imbalance** (midpoint rule, in trades): direction;
    - **retail share of dollars:** intensity;
    - **abnormal retail activity** (retail trades against the same bar over the previous 20 days): attention.

    Each is standardized across the large caps' same bar that day, as in the image. There is no price shape: a bar without a value (no trades, or after an early close) is 0, the large caps' mean.

    Below, each example's grey image with its strip under it, bar under bar (yellow line: the session starts). Red is above the large caps' mean, blue below, up to ±3 standard deviations.
    """)
    return


@app.cell
def _(PRICE_FEATURES, VOLUME_FEATURES, np, overnight_examples, plt):
    STRIP = {"retail_imbalance": "imbalance", "retail_share": "share", "abnormal_activity": "abnormal"}
    _cols = [(PRICE_FEATURES + VOLUME_FEATURES).index(_c) for _c in STRIP]
    _n = len(overnight_examples)
    fig_strip, _axes = plt.subplots(2 * _n, 1, figsize=(11, 1.6 * _n + 0.6), squeeze=False, layout="constrained",
                                    gridspec_kw={"height_ratios": [4, 1] * _n})
    for _i, _e in enumerate(overnight_examples):
        _ax_img, _ax_strip = _axes[2 * _i, 0], _axes[2 * _i + 1, 0]
        _strip = np.full((len(STRIP), 40), 128.0)  # 40 bars; an early close's empty bars at the mean
        _values = _e["colors"][:, _cols].T
        _strip[:, :_values.shape[1]] = np.nan_to_num(_values, nan=128)
        _ax_img.imshow(_e["with_overnight"][0], cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
        _shown = _ax_strip.imshow(_strip / 255 * 6 - 3, cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
                                  aspect="auto", extent=(-0.5, 119.5, len(STRIP) - 0.5, -0.5))
        _ax_img.set_ylabel(f"{_e['symbol']}\n{_e['day']:%Y-%m-%d}\n\\${_e['cap_prev'] / 1e6:.0f}B", fontsize=7)
        _ax_strip.set_yticks(range(len(STRIP)), list(STRIP.values()), fontsize=6)
        _ax_img.set_yticks([])
        for _ax in (_ax_img, _ax_strip):
            _ax.set_xlim(-0.5, 119.5)
            _ax.axvline(5.5, color="yellow", lw=0.6)
            _ax.set_xticks([])
    fig_strip.colorbar(_shown, ax=_axes[:, 0].tolist(), fraction=0.02, pad=0.01, label="standard deviations")
    fig_strip.suptitle("The retail strip (3 x 40) under each one-day grey image, bar under bar", fontsize=9)
    fig_strip
    return


if __name__ == "__main__":
    app.run()
