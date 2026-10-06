# TAQ

NYSE's Daily TAQ (Trade and Quote) is every trade and quote printed on the US consolidated tape, with millisecond-or-better timestamps. We read it on WRDS in the schema `taqmsec`.

- **Layout:** one table per day and file type, named `<type>_YYYYMMDD`, e.g. `ctm_20190604`.
- **Download:** `scripts/00_fetch_TAQ.py` saves files unchanged in `data/raw/taq/<date>/`. So far that is one test minute: 15:49:00 to 15:49:59 New York time on 2019-06-04.
- **First look:** `scripts/01_explore_and_clean_TAQ.py` shows the head of each raw file.
- **Definitions:** the column meanings and codes below come from NYSE's *Daily TAQ Client Specification* (v3.3). The counts come from the test minute.

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

## Size of one minute (15:49, 2019-06-04)

| | Rows | Size |
|---|---|---|
| Trades (`ctm`) | 242,992 | 26 MB |
| NBBO changes (`complete_nbbo`) | 616,999 | 40 MB |
| Master (`mastm`, whole day) | 8,812 | 1.2 MB |

## Things that hold across files

- **Times are New York local time**, not UTC (unlike Robintrack).
- **`time_m` is when the SIP published the record.** The SIP is the securities information processor that runs the consolidated tape: CTA for NYSE- and other-listed stocks, UTP for Nasdaq-listed ones. `time_m` is when the market could first see a record, so it is the time the 15:50 cutoff applies to. `time_m_nano` adds nanoseconds (0 to 999) to `time_m`'s microseconds.
- **Symbols are split into `sym_root` and `sym_suffix`.** The master file's `symbol_15` is the two joined by a space. TAQ also splits some plain Nasdaq tickers this way: `GOOGL` is `GOOG` + `L`, `CMCSA` is `CMCS` + `A`. So a TAQ symbol is not a CRSP ticker.
- **Match to CRSP by CUSIP, never by ticker.** The master file's 9-digit CUSIP, cut to 8 digits, matches CRSP's `cusip` on the name record valid that day. On 2019-06-04:
  - all 3,830 PERMNOs in the 15:50 universe matched, one to one;
  - 99.1% of trades and 99.5% of dollar volume matched a PERMNO.
