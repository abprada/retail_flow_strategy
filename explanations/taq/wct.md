# WRDS Consolidated Trades (`taqmsec.wct_YYYYMMDD`)

A table WRDS builds from Daily TAQ: every trade on the consolidated tape, each with the national best bid and offer (NBBO) standing just before it. It replaces the trade file (`ctm`, see [trades.md](trades.md)) and the NBBO file (`complete_nbbo`, see [nbbo.md](nbbo.md)). Matching trades to quotes ourselves would mean an as-of join of tens of millions of trades against hundreds of millions of quotes every day. WCT has that join done already.

Barber et al. (2024) use WCT for the same reason. They note that since 2017 WRDS computes the NBBO in the nanosecond before each trade.

On 2019-06-04 it has 42,560,621 rows (761 MB as parquet).

## Columns

| Column | Meaning |
|---|---|
| `date` | trading day |
| `time_m`, `time_m_nano` | when the SIP published the trade, New York time; nanoseconds added to `time_m`. As in `ctm` |
| `sym_root`, `sym_suffix` | symbol, as in `ctm` |
| `type` | `T` on every row |
| `ex` | reporting venue; `D` is off-exchange, via a FINRA TRF |
| `tr_scond` | sale conditions, as in `ctm` |
| `size`, `price` | shares and price, as in `ctm` |
| `tr_seqnum` | the SIP's message sequence number |
| `qtime`, `qtime_nano` | time of the prevailing NBBO: that quote's SIP time plus 1 nanosecond |
| `nbb`, `nbo` | national best bid and offer at the trade, in dollars |
| `nbbqty`, `nboqty` | shares at the best bid and offer, at the venue setting it |

**Not in WCT:** the correction flag `tr_corr`, the TRF code `tr_rf`, the trade-through flag `tte_ind`, the SIP `tr_source`, and any CUSIP. The master file (`mastm`) is still needed to match symbols to PERMNOs.

## Check against `ctm` and `complete_nbbo` (2019-06-04)

Run once on the full day, before both files were deleted.

- **Same trades.**
  - Matched on symbol, sequence number and time, every WCT row is in `ctm`, with identical price, size, venue and sale conditions on all 42.56M rows.
  - `ctm` has 422 more. All of them carry the extended-hours code `T` (04:00 to 17:00), and several are test symbols. None matter for a 9:30 to 15:50 window.
- **Correct NBBO, with no lookahead.**
  - On 300 random symbols (9.0M trades), `nbb` and `nbo` equal the last `complete_nbbo` row strictly before the trade 99.89% of the time. On 10 liquid symbols (1.1M trades) it's 99.96%.
  - Matched on `qtime`, it's 99.9997%.
  - `qtime` is before the trade time on 99.998% of rows and never after.
- **Quality, 9:30 to 16:00.**
  - 0.02% of trades have no bid or offer;
  - 0.06% have a crossed quote (bid above offer);
  - none has a zero bid or offer.

## Known limitation: cancelled trades stay in

Without `tr_corr`, cancelled and corrected trades can't be removed. On 2019-06-04 WCT keeps:
- the 509 trades later cancelled;
- the 509 cancel records that repeat them;
- 7 corrected trades and their 7 correction records.

That's 1,032 rows of 42.56M (0.002%). Removing them would take a second scan of `ctm` every day, so they are left in.
