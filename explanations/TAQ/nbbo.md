# NBBO (`taqmsec.complete_nbbo_YYYYMMDD`)

**Replaced by WCT.** We no longer download `complete_nbbo`: [WRDS Consolidated Trades](wct.md) attaches the NBBO standing before each trade, checked against this table. This file stays as the reference for the quote codes.

The national best bid and offer (NBBO): the highest bid and the lowest offer across all exchanges for a stock. One row each time it changes.

The table is WRDS's merge of two NYSE files, following Appendix I of the TAQ spec:

- every record of NYSE's NBBO file (`nbbom`);
- every quote in the quote file (`cqm`) that is itself both the best bid and the best offer. Those are left out of `nbbom` by design.

The latest row at or before time *s* is the NBBO standing at *s*.

The test minute (15:49, 2019-06-04) has 616,999 rows, 2.5 times as many as trades.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `date` | date | trading day |
| `time_m`, `time_m_nano` | time, smallint | when the SIP published the change, New York time; nanoseconds added to `time_m` |
| `sym_root`, `sym_suffix` | text | symbol, as in the trade file |
| `qu_cond` | text | condition of the quote that caused the change (below) |
| `natbbo_ind` | text | how this row carries the NBBO (below) |
| `qu_source` | text | which SIP: `C` CTA (Tapes A, B), `N` UTP (Tape C) |
| `nbbo_qu_cond` | text | the NBBO's own quote condition; blank on every row of the test minute |
| `best_bid`, `best_ask` | numeric | national best bid and offer after the change, in dollars |
| `best_bidsizeshares`, `best_asksizeshares` | integer | shares at the best bid and offer, *at the venue setting it* (not summed across venues tied at that price) |

The table has no venue column, so it doesn't say which exchange sets the NBBO. To get that, or the total size at the best price across venues, you need `cqm`.

## `natbbo_ind`: where the row comes from

| Code | SIP | Meaning | Rows |
|---|---|---|---|
| `G` | CTA | the quote itself is both the best bid and the best offer (from `cqm`) | 174,725 |
| `U` | CTA | the quote created a new NBBO, carried as a long appendage (from `nbbom`) | 285,168 |
| `4` | UTP | the quote itself is the new NBBO (from `cqm`) | 70,640 |
| `2` | UTP | new NBBO in a short-form appendage (from `nbbom`) | 85,189 |
| `3` | UTP | new NBBO in a long-form appendage (from `nbbom`) | 1,277 |

Either way, `best_bid` and `best_ask` are the NBBO after the change; the code only says how NYSE recorded it.

The spec also lists codes NYSE reserved for future use, but only these five appear.

## `qu_cond`

Every row in the test minute is `R`: a regular, two-sided, firm quote, eligible for the NBBO. Other codes mark quotes that are not:

- **Both SIPs:** opening (`O`), closing (`C`), non-firm (`N`), closed market maker (`L`), intraday auction (`4`).
- **CTA, slow quotes:** `A`, `B`, `E`, `F`, `H`, `U`, `W`.
- **UTP:** fast trading (`F`), order imbalance (`I`), order influx (`X`), one-sided (`Y`), no open or no resume (`Z`).

Check `qu_cond` before trusting a spread at the open or the close.

## Quality in the test minute

- No zero bids or asks.
- 163 rows with one side missing (84 without a bid, 151 without an offer): no one was quoting that side.
- 22 crossed rows (bid above ask).
- Median quoted spread 3.4 bp.
- Sizes are in shares, almost always multiples of 100 (round lots).
