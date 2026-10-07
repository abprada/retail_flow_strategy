# TAQ One-Day Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean one day of WRDS Consolidated Trades (2019-06-04) by the rules in the cleaning spec, entirely inside the TAQ notebook, so each step and what it removes can be read in one place. The notebook ends with the head of the clean trades.

**Architecture:** One notebook, `scripts/01_explore_and_clean_TAQ.py`, with one section per spec step:
- a markdown cell saying what the step does and why;
- a code cell producing the step's table;
- a view of what the step removed.

Each step reads the previous step's table and defines a new one, so marimo's dataflow runs them in order. A running table of counts (trades, dollars, off-exchange trades) grows by one row per step. A checks cell gates the result. No module, no unit tests: correctness is shown on the real day through the counts and the checks.

**Tech Stack:** Python 3.12, pandas 3, numpy, pyarrow (parquet), marimo (edited live through `marimo._code_mode`).

**Spec:** `explanations/TAQ/cleaning_spec.md`

## Global Constraints

- No co-author or tool attribution in code, comments, docs or commit messages.
- Never commit `plan.md`, `literature/` or `data/`.
- Only fixed rules: nothing is chosen by looking at values.
- A trade feeds a decision day only if its SIP time (`time_m`) is before that day's cutoff (15:50; 12:50 on an early-close day).
- Never match securities by ticker: symbol → CUSIP (that day's master file) → PERMNO (CRSP name record valid that day, first 8 CUSIP characters).
- The notebook is open in a live marimo server (port 2720). Change it only through `marimo._code_mode` (marimo-pair skill), never by editing the file. Read a cell's current code before editing it.
- One day only: no WRDS queries beyond Task 0, and nothing written to `data/`.

## Review Focus

1. **Tick Size Pilot codes are unverified** (blank on this day). The step must stop with an error on any non-blank code outside `C`, `1`, `2`, `3`, not silently filter nothing (Task 5).
2. **Opening prints appear on several venues.** Only `O` prints on the listing exchange count; every `Q` print is excluded (Task 4). Check: AAPL's opening volume is 344,633 shares, not 689,266.
3. **Bucket edges:** a trade at exactly 09:30:00.000000 is `regular`; one at exactly 15:50:00.000000 is `last` (Task 4). The checks cell tests that no trade of decision day 2019-06-04 is at or after 15:50.
4. **Corrections must match their WCT row.** The time text can differ between tables, so match on nanoseconds after midnight; the checks cell requires every correction to match (Task 2).
5. **An empty suffix may be null:** the symbol must be `A`, not `A None` (Task 1).

## What the notebook already has

Cells (by first line): title; imports (`mo`, `pd`, `pa`, `pc`, `pq`); paths (`DATE`, `RAW_DIR`, `CRSP_NAMES_PATH`); heads of WCT and the master file; the CUSIP join (`wct_cusip`); the PERMNO join (`cusip_ambiguous`, `symbol_permno`, `wct_permno`); the checks on prices, quotes and missing values (`wct_checks`, `wct_missing`).

The new section reads the raw files itself, so it doesn't depend on the earlier joins. The earlier sections stay as exploration.

---

### Task 0: Download the corrected and cancelled trades (you run this)

- [ ] **Step 1:** `WRDS_USERNAME=<you> uv run python scripts/00_fetch_TAQ.py`, approve Duo, answer `y`. WCT and the master file are skipped because they exist.
- [ ] **Step 2:** `ls data/raw/taq/2019-06-04/ctm_corrections.parquet`. Expected: the file exists; the script prints about 1,032 rows.

---

### Task 1: Load and type the day; start the step counts

Add after the last cell, through `cm`:

- [ ] **Step 1: Markdown cell**

```python
mo.md(r"""
# Cleaning one day

The rules in `explanations/TAQ/cleaning_spec.md`, one section per step. Each step keeps a table of trades and adds a row to the counts: trades, dollars and off-exchange trades left after it.

## Load

WCT's columns are text. Here they become numbers, a symbol as the master file writes it, and `t_ns`, the SIP time in nanoseconds after midnight (`time_m` plus `time_m_nano`).
""")
```

- [ ] **Step 2: Code cell**

```python
def typed(raw):
    """Numbers, symbol and time from TAQ's text columns."""
    suffix = raw["sym_suffix"].fillna("")
    out = raw.assign(
        symbol=np.where(suffix.eq(""), raw["sym_root"], raw["sym_root"] + " " + suffix),
        t_ns=pd.to_timedelta(raw["time_m"]).to_numpy().astype("timedelta64[ns]").astype("int64")
        + pd.to_numeric(raw["time_m_nano"]).astype("int64").to_numpy(),
    )
    for col in ["price", "size", "nbb", "nbo", "tr_seqnum"]:
        if col in out:
            out[col] = pd.to_numeric(out[col].replace("", np.nan))
    return out


def count_step(step, t):
    """One row of the step counts."""
    return {"step": step, "trades": len(t), "dollars": float((t["price"] * t["size"]).sum()),
            "off_exchange_trades": int(t["ex"].eq("D").sum())}


c0_trades = typed(pd.read_parquet(RAW_DIR / "wct.parquet"))
c0_counts = [count_step("raw WCT", c0_trades)]
```

Add `import numpy as np` to the imports cell (read it first).

- [ ] **Step 3: Run and verify**

Run both cells. In the scratchpad: `print(len(c0_trades), c0_trades[["symbol", "t_ns", "price", "nbb"]].head())`.
Expected: 42,560,621 rows; `t_ns` integers; `nbb` floats with NaN where the text was empty.

- [ ] **Step 4: Commit**

```bash
git add scripts/01_explore_and_clean_TAQ.py
git commit -m "Load and type one WCT day in the TAQ notebook"
```

---

### Task 2: Remove corrected and cancelled trades

- [ ] **Step 1: Markdown cell**

```python
mo.md(r"""
## Step 1: corrected and cancelled trades

WCT has no correction flag, so the flagged trades come from the trade file (`ctm_corrections.parquet`). They are matched to WCT on symbol, `tr_seqnum` and `t_ns`, which is unique per trade in both tables. Dropped: `08` later cancelled, `10` cancel record, `07` and `11` erroneous, `12` correction record (it carries the wrong original data). Kept: `01`, the trade as corrected.
""")
```

- [ ] **Step 2: Code cell**

```python
TRADE_KEY = ["sym_root", "sym_suffix", "tr_seqnum", "t_ns"]
CORRECTIONS_TO_DROP = {"07", "08", "10", "11", "12"}

c1_corrections = typed(pd.read_parquet(RAW_DIR / "ctm_corrections.parquet"))
_drop = c1_corrections.loc[c1_corrections["tr_corr"].isin(CORRECTIONS_TO_DROP), TRADE_KEY].drop_duplicates()
_m = c0_trades.merge(_drop.assign(_correction=True), on=TRADE_KEY, how="left")
c1_matched = int(_m["_correction"].notna().sum())
c1_to_drop = len(_drop)
c1_trades = _m[_m["_correction"].isna()].drop(columns="_correction")
c1_counts = c0_counts + [count_step("without corrected and cancelled trades", c1_trades)]
```

- [ ] **Step 3: View cell**

```python
mo.vstack([
    mo.md(f"**{c1_matched:,} of {c1_to_drop:,} trades to drop found in WCT.** By correction code and venue:"),
    c1_corrections.assign(off_exchange=c1_corrections["ex"].eq("D")).groupby(["tr_corr", "off_exchange"]).size().rename("trades").reset_index(),
])
```

- [ ] **Step 4: Run and verify**

Expected: `c1_matched == c1_to_drop`, about 1,025. The table also answers whether the cancelled trades are off-exchange.

- [ ] **Step 5: Commit**

```bash
git add scripts/01_explore_and_clean_TAQ.py
git commit -m "Remove corrected and cancelled trades in the TAQ cleaning"
```

---

### Task 3: Identify the security

- [ ] **Step 1: Markdown cell**

```python
mo.md(r"""
## Step 2: PERMNO

Symbol → CUSIP from that day's master file → PERMNO from the CRSP name record valid that day, on the first 8 CUSIP characters. A CUSIP valid for two PERMNOs that day is left unmatched. Trades without a PERMNO are dropped: test symbols, ETNs, preferreds, warrants, units, structured products, ADRs that CRSP lacks.
""")
```

- [ ] **Step 2: Code cell**

```python
_names = pd.read_csv(CRSP_NAMES_PATH, dtype=str, usecols=["permno", "namedt", "nameenddt", "cusip"])
_names = _names.loc[(_names["namedt"] <= DATE) & (_names["nameenddt"] >= DATE), ["cusip", "permno"]].drop_duplicates()
_names = _names[~_names["cusip"].duplicated(keep=False)].rename(columns={"cusip": "cusip8"})
_master = pd.read_parquet(RAW_DIR / "mastm.parquet")
c2_symbols = (
    _master.assign(symbol=_master["symbol_15"], cusip8=_master["cusip"].str[:8],
                   tick_pilot_indicator=_master["tick_pilot_indicator"].fillna(""))
    .merge(_names, on="cusip8", how="left")[["symbol", "cusip", "permno", "listed_exchange", "tick_pilot_indicator", "sec_type", "sec_desc"]]
)
_t = c1_trades.merge(c2_symbols, on="symbol", how="left")
c2_dropped = _t[_t["permno"].isna()]
c2_trades = _t[_t["permno"].notna()]
c2_counts = c1_counts + [count_step("with a PERMNO", c2_trades)]
```

- [ ] **Step 3: View cell**

```python
mo.vstack([
    mo.md("**Dropped, by the master file's security type**"),
    c2_dropped.groupby(c2_dropped["sec_type"].fillna("not in master"))["symbol"].agg(symbols="nunique", trades="size").sort_values("trades", ascending=False).reset_index(),
])
```

- [ ] **Step 4: Run and verify**

Expected: about 571,500 trades dropped in 1,076 symbols, matching `explanations/TAQ/README.md` (571,978 before corrections).

- [ ] **Step 5: Commit**

```bash
git add scripts/01_explore_and_clean_TAQ.py
git commit -m "Match trades to PERMNOs in the TAQ cleaning"
```

---

### Task 4: Sale conditions, buckets and the opening auction

- [ ] **Step 1: Markdown cell**

```python
mo.md(r"""
## Step 3: sale conditions and buckets

Each trade's conditions put it in one class:
- **normal:** only blank / `@`, `I`, `F`;
- **extended:** normal codes plus `T` or `U`;
- **open:** an `O` (opening) print;
- **excluded:** anything else.

Buckets on a normal day, by `t_ns`:
- **`pre`** [04:00, 09:30), normal or extended, feeds this day;
- **`open_auction`**: `O` prints on the stock's listing exchange, feeds this day;
- **`regular`** [09:30, 15:50), normal, feeds this day;
- **`last`** [15:50, 16:00), normal, feeds the next trading day;
- **`after`** from 16:00, normal or extended, feeds the next trading day.

Early-close days (13:00 close, 12:50 cutoff) move the times; 2019-06-04 is not one. The opening auction becomes one price (its largest print) and one volume per stock; it is not counted as trades. `Q` prints (each venue's own official open) and `O` prints off the listing exchange are excluded.
""")
```

- [ ] **Step 2: Code cell**

```python
EARLY_CLOSE_DAYS = set(pd.to_datetime([
    "2018-07-03", "2018-11-23", "2018-12-24", "2019-07-03", "2019-11-29", "2019-12-24", "2020-11-27", "2020-12-24",
    "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
]))  # NYSE calendar, written from memory; checked against the data when the full sample is built
TRADE_BUCKETS = ["pre", "regular", "last", "after"]


def condition_class(code):
    codes = set(code) - {" "}
    if "O" in codes:
        return "open"
    if not codes <= set("@IFTU"):
        return "excluded"
    return "extended" if codes & set("TU") else "normal"


_dsi = pd.read_csv(CRSP_NAMES_PATH.parent / "dsi.csv", usecols=["date"], parse_dates=["date"])["date"]
NEXT_DAY = _dsi[_dsi > pd.Timestamp(DATE)].min()
_close = pd.Timedelta(hours=13 if pd.Timestamp(DATE) in EARLY_CLOSE_DAYS else 16)
SESSION = {"open": pd.Timedelta(hours=9, minutes=30), "cutoff": _close - pd.Timedelta(minutes=10), "close": _close}

_codes = c2_trades["tr_scond"].fillna("")
_unique = _codes.unique()
_cls = _codes.map(dict(zip(_unique, map(condition_class, _unique))))
_t, _n, _e = c2_trades["t_ns"], _cls.eq("normal"), _cls.eq("extended")
_bucket = np.select(
    [
        _cls.eq("open") & c2_trades["ex"].eq(c2_trades["listed_exchange"]),
        (_n | _e) & (_t < SESSION["open"].value),
        _n & (_t >= SESSION["open"].value) & (_t < SESSION["cutoff"].value),
        _n & (_t >= SESSION["cutoff"].value) & (_t < SESSION["close"].value),
        (_n | _e) & (_t >= SESSION["close"].value),
    ],
    ["open_auction", "pre", "regular", "last", "after"],
    default="",
)
c3_all = c2_trades.assign(
    condition_class=_cls.to_numpy(), bucket=_bucket,
    decision_day=np.where(np.isin(_bucket, ["last", "after"]), NEXT_DAY, pd.Timestamp(DATE)),
)
_open = c3_all[c3_all["bucket"].eq("open_auction")]
opening_auctions = pd.DataFrame({
    "open_price": _open.sort_values("size").groupby("permno").tail(1).set_index("permno")["price"],
    "open_volume": _open.groupby("permno")["size"].sum(),
}).reset_index()
c3_trades = c3_all[c3_all["bucket"].isin(TRADE_BUCKETS)]
c3_counts = c2_counts + [count_step("normal sale conditions, in a bucket", c3_trades)]
```

- [ ] **Step 3: View cell**

```python
_x = c3_all[~c3_all["bucket"].isin(TRADE_BUCKETS)]
mo.vstack([
    mo.md(f"**Opening auctions:** {len(opening_auctions):,} stocks. AAPL's volume: "
          f"{int(opening_auctions.merge(c2_symbols[c2_symbols['symbol'].eq('AAPL')], on='permno')['open_volume'].iloc[0]):,} shares"),
    mo.md("**Kept trades by bucket and decision day**"),
    c3_trades.groupby(["decision_day", "bucket"]).size().rename("trades").reset_index(),
    mo.md("**Not kept as trades, by class and sale condition** (top 20)"),
    _x.groupby(["condition_class", "bucket", "tr_scond"]).size().rename("trades").sort_values(ascending=False).head(20).reset_index(),
])
```

- [ ] **Step 4: Run and verify**

Expected:
- AAPL's opening volume is 344,633 shares;
- about 530,000 trades excluded for `4`, `B`, `W`;
- 1,046 `T` trades inside the session excluded;
- trades after 15:50 have decision day 2019-06-05.

- [ ] **Step 5: Commit**

```bash
git add scripts/01_explore_and_clean_TAQ.py
git commit -m "Classify sale conditions and assign buckets in the TAQ cleaning"
```

---

### Task 5: Trade filters, flags, checks and the head

- [ ] **Step 1: Markdown cell**

```python
mo.md(r"""
## Step 4: trade filters

Prices of at least $1 (Boehmer & Jones 2022), positive price and size, and no stock in Tick Size Pilot groups 2 or 3 (until October 2018; none on this day). An unknown pilot code stops the notebook.

## Step 5: flags

Kept on every trade, not used to drop:
- **`quote_state`:** `missing` if the bid or offer is empty, `crossed` if bid > offer, `locked` if bid = offer, else `valid`. Midpoint signing needs `valid`.
- **`outside_quote`:** a valid quote and a price more than one spread below the bid or above the offer. Left out of price-based features.
- **`large`:** over $100,000.
""")
```

- [ ] **Step 2: Code cell**

```python
_pilot = set(c3_trades["tick_pilot_indicator"]) - {"", "C", "1", "2", "3"}
if _pilot:
    raise ValueError(f"unknown tick_pilot_indicator codes: {sorted(_pilot)}")
_t = c3_trades[(c3_trades["price"] >= 1) & (c3_trades["size"] > 0)]
_counts = c3_counts + [count_step("price at least $1", _t)]
_t = _t[~_t["tick_pilot_indicator"].isin({"2", "3"})]
_counts = _counts + [count_step("outside Tick Size Pilot groups 2 and 3", _t)]

_bid, _ask, _px = _t["nbb"], _t["nbo"], _t["price"]
_state = np.select([_bid.isna() | _ask.isna(), _bid > _ask, _bid == _ask], ["missing", "crossed", "locked"], default="valid")
_spread = _ask - _bid
clean_trades = _t.assign(
    quote_state=_state,
    outside_quote=(_state == "valid") & ((_px < _bid - _spread) | (_px > _ask + _spread)),
    large=_px * _t["size"] > 100_000,
).reset_index(drop=True)
clean_counts = pd.DataFrame(_counts).assign(share_of_raw_trades=lambda d: d["trades"] / d["trades"].iloc[0])
```

- [ ] **Step 3: Checks cell**

```python
_own = clean_trades["decision_day"].eq(pd.Timestamp(DATE))
clean_checks = pd.DataFrame([
    ("every correction to drop matched one WCT trade", c1_matched == c1_to_drop),
    ("every trade has a bucket", clean_trades["bucket"].isin(TRADE_BUCKETS).all()),
    ("every trade feeds this day or the next", clean_trades["decision_day"].isin([pd.Timestamp(DATE), NEXT_DAY]).all()),
    ("this day's trades are before its cutoff", (clean_trades.loc[_own, "t_ns"] < SESSION["cutoff"].value).all()),
    ("no PERMNO under two symbols", clean_trades.groupby("permno")["symbol"].nunique().le(1).all()),
    ("trades never increase from step to step", clean_counts["trades"].is_monotonic_decreasing),
], columns=["check", "passed"])
```

- [ ] **Step 4: Results cell, ending with the head**

```python
mo.vstack([
    mo.md("## Result"),
    mo.md("**What each step leaves**"),
    mo.ui.table(clean_counts, selection=None),
    mo.md("**Checks**"),
    mo.ui.table(clean_checks, selection=None),
    mo.md("**Flags**"),
    pd.DataFrame({
        "quote_state": clean_trades["quote_state"].value_counts(),
    }).reset_index(),
    mo.md(f"outside the quote: {int(clean_trades['outside_quote'].sum()):,}; over $100k: {int(clean_trades['large'].sum()):,}"),
    mo.md(f"**Clean trades: first 20 of {len(clean_trades):,}**"),
    clean_trades.head(20),
])
```

- [ ] **Step 5: Run and verify**

Expected: every check `True`; trades never rise from step to step; the head shows the columns WCT plus `symbol`, `t_ns`, `cusip`, `permno`, `listed_exchange`, `tick_pilot_indicator`, `condition_class`, `bucket`, `decision_day`, `quote_state`, `outside_quote` and `large`.

- [ ] **Step 6: Update the spec with facts found while planning**

In `explanations/TAQ/cleaning_spec.md`:
- In Step 3, after the auctions paragraph, add: "The opening auction is the stock's `O` prints on its listing exchange (the master file's `listed_exchange`). `O` prints on other venues and every `Q` print are excluded; `Q` is each venue's own official open, often repeating the auction's size. On 2019-06-04 BRK B's NYSE opening print came at 09:32."
- In Step 4, under the extended-hours set, add: "A `T` or `U` trade inside the regular session is excluded (1,046 on 2019-06-04)."

- [ ] **Step 7: Commit**

```bash
git add scripts/01_explore_and_clean_TAQ.py explanations/TAQ/cleaning_spec.md
git commit -m "Filter, flag and check the clean WCT trades and show their head"
```
