# TAQ

NYSE's Daily TAQ (Trade and Quote) is every trade and quote printed on the US consolidated tape, with millisecond-or-better timestamps. We read it on WRDS in the schema `taqmsec`.

- **Layout:** one table per day and file type, named `<type>_YYYYMMDD`, e.g. `ctm_20190604`.
- **Download:** `scripts/00_fetch_TAQ.py` saves WCT and the master file for one day, unchanged, in `data/raw/taq/<date>/`. So far that is one full day, 2019-06-04, inside the Robintrack training period. The full sample will not be downloaded as trades; its features will be aggregated in SQL on WRDS.
- **Notebook:** `scripts/01_explore_and_clean_TAQ.py` shows the head of each raw file and gives every trade its CUSIP and PERMNO.
- **Definitions:** the column meanings and codes come from NYSE's *Daily TAQ Client Specification* (v3.3). Counts are from 2019-06-04: the full day, or, in [trades.md](trades.md) and [nbbo.md](nbbo.md), a first test minute (15:49).

## Tables for one day

| Table | Content | Downloaded | Explained in |
|---|---|---|---|
| `wct` | WRDS Consolidated Trades: every trade with the NBBO standing just before it. **The table we use**; it replaces `ctm` and `complete_nbbo` | yes | [wct.md](wct.md) |
| `ctm` | every trade on the consolidated tape | checked, then deleted | [trades.md](trades.md) |
| `complete_nbbo` | every change in the national best bid and offer (NBBO) | checked, then deleted | [nbbo.md](nbbo.md) |
| `mastm` | the master file: one row per symbol, with CUSIP, security type and listing venue | yes | [master.md](master.md) |
| `cqm` | every quote from every venue, NBBO or not; by far the largest table | no | |
| `nbbom` | NYSE's NBBO file: quotes that change the NBBO, *excluding* those that are themselves both best bid and best offer. `complete_nbbo` is WRDS's merge of `nbbom` with those quotes from `cqm` | no | |
| `luld_ctm`, `luld_cqm` | Limit Up-Limit Down price bands and trading pauses | no | |

## Size of one day (2019-06-04)

| Table | Rows | Parquet |
|---|---|---|
| `wct` | 42,560,621 | 761 MB |
| `mastm` | 8,812 | 0.35 MB |
| `ctm` (deleted) | 42,561,043 | 496 MB |
| `complete_nbbo` (deleted) | 193,018,146 | 1,446 MB |

- **Download speed:** about 1.5 to 3.3 MB/s over one WRDS connection, so one day of WCT took 20 to 45 minutes. Raw trades for the full sample would take months, which is why the features have to be computed on WRDS.
- **Quotes are the bulk:** the NBBO has 4.5 times as many rows as trades. Quotes change far more often than trades happen.

## Things that hold across files

- **Times are New York local time**, not UTC (unlike Robintrack).
- **`time_m` is when the SIP published the record.** The SIP is the securities information processor that runs the consolidated tape: CTA for NYSE- and other-listed stocks, UTP for Nasdaq-listed ones. `time_m` is when the market could first see a record, so it is the time the 15:50 cutoff applies to. `time_m_nano` adds nanoseconds (0 to 999) to `time_m`'s microseconds.
- **Symbols are split into `sym_root` and `sym_suffix`.** The master file's `symbol_15` is the two joined by a space. TAQ also splits some plain Nasdaq tickers this way: `GOOGL` is `GOOG` + `L`, `CMCSA` is `CMCS` + `A`. So a TAQ symbol is not a CRSP ticker.
- **Match to CRSP by CUSIP, never by ticker.** Each trade's symbol gets its CUSIP from that day's master file. The CUSIP, cut to 8 characters (CRSP has no check digit), matches CRSP's `cusip` on the name record valid that day (`stocknames_v2`), which gives the PERMNO. CRSP's ticker is not used: CRSP's PERMNO 92118 carries the ticker `CIG`, but its CUSIP and price ($4.68 that day) are those of TAQ's `CIG C`, the other Cemig ADR, not of `CIG` ($3.84).

## Matching on 2019-06-04

- **CUSIP:** 99.97% of trades. The 12,055 without one are all SIP test symbols (22 symbols).
- **PERMNO:** 41,988,643 trades (98.66%), 99.20% of dollar volume. 7,496 of 8,572 traded symbols match.
- **No ambiguity:** no CUSIP matches two PERMNOs on the day, and no PERMNO is reached by two symbols.
- **The 15:50 return universe that day:** all 3,832 PERMNOs match.

**The 1,076 traded symbols without a PERMNO**, by the master file's security type:

| `sec_type` | Symbols | Trades | Why |
|---|---|---|---|
| ETF | 30 | 241,388 | exchange-traded notes listed as ETFs (TVIX, VXX), not in CRSP |
| H (ADR) | 22 | 141,980 | ADRs CRSP does not have under any CUSIP or ticker (ITUB, BBD, GOL, PBR A) |
| ETN | 123 | 64,277 | exchange-traded notes |
| B | 476 | 57,255 | preferred stock |
| G | 191 | 24,916 | structured products |
| U | 54 | 23,884 | unknown type (e.g. UGAZ, an ETN) |
| T, A, blank, not in the master file | 23 | 12,077 | test symbols (`ZVZZ T`, `NTEST`, `ATEST`, `ZWZZ T`) |
| C, R, D, I | 157 | 6,201 | warrants, units, rights, preferred ADRs |
