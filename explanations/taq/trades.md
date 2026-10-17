# Trades (`taqmsec.ctm_YYYYMMDD`)

**Replaced by WCT.** We no longer download `ctm`: [WRDS Consolidated Trades](wct.md) holds the same trades with the same columns for venue, sale conditions, price and size, so this file stays as the reference for those codes.

Every trade reported to the consolidated tape by every venue: the exchanges, and FINRA's trade reporting facilities (TRFs) for off-exchange trades. One row per trade report, sorted by symbol, time and sequence number.

The test minute (15:49, 2019-06-04) has 242,992 trades in 4,871 symbols.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `date` | date | trading day |
| `time_m` | time | when the SIP published the trade, New York time, to the microsecond. The time to use for the 15:50 cutoff |
| `time_m_nano` | smallint | nanoseconds added to `time_m` (0 to 999) |
| `ex` | text | the venue that reported the trade (codes below) |
| `sym_root`, `sym_suffix` | text | symbol root and suffix, e.g. `BRK` + `B`. The suffix is blank for most stocks |
| `tr_scond` | text | up to four sale-condition codes (below) |
| `size` | integer | shares traded |
| `price` | numeric | price per share in dollars, as traded (not split-adjusted). Can have sub-penny digits |
| `tr_stop_ind` | text | stopped-stock flag, CTA only: `N` not stopped, `Y` stopped. Blank on UTP trades |
| `tr_corr` | text | correction status (below) |
| `tr_seqnum` | bigint | the SIP's message sequence number. Unique only within a SIP, not across the file |
| `tr_id` | text | trade identifier, unique per venue, symbol and session. Links a trade to its later correction or cancellation |
| `tr_source` | text | which SIP published it: `C` CTA (NYSE- and other-listed, Tapes A and B), `N` UTP (Nasdaq-listed, Tape C) |
| `tr_rf` | text | for off-exchange trades, which TRF it was reported to (below). Blank for exchange trades |
| `part_time`, `part_time_nano` | time, smallint | the venue's own timestamp, when it reported the trade to the SIP |
| `trf_time`, `trf_time_nano` | time, smallint | the TRF's timestamp. Filled only for off-exchange trades (`ex = D`) |
| `tte_ind` | text | `1` if the trade is exempt from the Reg NMS trade-through rule (Rule 611), else `0` |

## Timestamps

There are three clocks:

- the venue's (`part_time`);
- the TRF's, for off-exchange trades (`trf_time`);
- the SIP's publication time (`time_m`).

In the test minute, `time_m` trails `part_time` by a median of 0.47 ms and a 99th percentile of 192 ms.

Late reports, flagged with sale conditions `L` or `Z`, carry an execution time long before `time_m`. A cutoff on `time_m` treats them as known only once they were published, which is what point in time requires.

## Venues (`ex`)

| Code | Venue | Trades in the test minute |
|---|---|---|
| D | FINRA: off-exchange trades (wholesalers, dark pools), via a TRF | 50,348 (21%; 36% of shares) |
| T | Nasdaq (in Tape A and B stocks) | 31,255 |
| Q | Nasdaq (in Tape C stocks) | 21,648 |
| Y | Cboe BYX | 20,686 |
| P | NYSE Arca | 20,231 |
| N | NYSE | 20,133 |
| Z | Cboe BZX | 18,742 |
| K | Cboe EDGX | 16,369 |
| J | Cboe EDGA | 15,715 |
| V | IEX | 10,986 |
| B | Nasdaq BX | 8,582 |
| C | NYSE National | 4,865 |
| A | NYSE American | 1,692 |
| X | Nasdaq PSX | 1,680 |
| M | NYSE Chicago | 60 |

Codes that appear only in later years: `H` MIAX Pearl, `L` LTSE, `U` MEMX.

## Off-exchange trades: `tr_rf`

| Code | TRF | Trades |
|---|---|---|
| T | FINRA/Nasdaq TRF Carteret, CTA stocks | 31,294 |
| Q | FINRA/Nasdaq TRF Carteret, UTP stocks | 14,227 |
| N | FINRA/NYSE TRF | 4,700 |
| B | FINRA/Nasdaq TRF Chicago | 127 |
| blank | reported directly by an exchange | 192,644 |

Every `ex = D` trade has a TRF code, and no exchange trade has one.

Retail orders routed to wholesalers (Citadel, Virtu and others, which is where Robinhood sends its orders) print here. They are often at a sub-penny price, because the wholesaler gives the customer a fraction of a cent of price improvement. Boehmer, Jones, Zhang and Zhang use that to identify retail trades:

- the trade is off-exchange (`ex = D`);
- the fraction of a cent in its price is between 0.6 and 1: a retail buy;
- between 0 and 0.4: a retail sell;
- near 0.5 (a midpoint execution): excluded.

In the test minute that flags 3,678 buys and 3,597 sells.

## Sale conditions (`tr_scond`)

Up to four one-character codes, positionally padded with spaces, e.g. `@F I` is regular + intermarket sweep + odd lot. The two SIPs use different code lists. The ones that matter here:

| Code | CTA (Tapes A, B) | UTP (Tape C) |
|---|---|---|
| blank | regular trade | |
| `@` | | regular trade |
| `F` | intermarket sweep order | intermarket sweep |
| `I` | odd lot (under 100 shares) | odd lot |
| `4` | derivatively priced | derivatively priced |
| `B` | average-price trade | bunched trade |
| `W` | | average-price trade |
| `C` | cash trade (same-day clearing) | cash sale |
| `R` | seller | seller |
| `L` | sold last (late report) | sold last |
| `Z` | sold out of sequence | sold out of sequence |
| `T` | extended-hours trade | Form T (extended hours) |
| `U` | extended hours, out of sequence | extended trading hours |
| `O`, `Q` | opening trade; official open | opening prints; official open |
| `M`, `6` | official close; closing trade | official close; closing prints |
| `5` | re-opening trade | re-opening prints |
| `7` | qualified contingent trade | exempt qualified contingent trade |
| `V` | contingent trade | stock-option trade |
| `X` | cross trade | cross trade |
| `N` | next-day clearing | next day |
| `H` | price-variation trade | price-variation trade |
| `P` | prior reference price | prior reference price |

**Most common in the test minute:**

| `tr_scond` | Trades |
|---|---|
| blank | 71,102 |
| `I` | 44,990 |
| `@` | 28,923 |
| `F` | 28,799 |
| `F I` | 21,589 |
| `@  I` | 21,235 |
| `@F` | 13,111 |
| `@F I` | 11,770 |
| `4 B` | 582 |
| `@4 W` | 322 |

- **Odd lots:** 41% of trades.
- **Intermarket sweeps (`F`):** always come with `tte_ind = 1`, because they are exempt from the trade-through rule.

Prints with `4`, `B`, `W`, `C`, `N`, `R`, `T`, `U`, `L` or `Z` are not normal-way trades at the current price. A last-trade price or VWAP should leave them out. Volume measures may keep them.

## Corrections (`tr_corr`)

| Code | Meaning |
|---|---|
| 00 | regular trade, never corrected or cancelled |
| 01 | original trade, later corrected (carries the corrected data) |
| 07 | original trade, later marked erroneous |
| 08 | original trade, later cancelled |
| 10 | cancel record (follows an `08`) |
| 11 | error record (follows a `07`) |
| 12 | correction record (follows an `01`; carries the original, incorrect data) |

In the test minute every trade is `00`. Over the full day: 509 trades later cancelled (`08`), 509 cancel records (`10`), 7 corrected trades (`01`) and 7 correction records (`12`), out of 42,561,043. WCT has no `tr_corr`, so these stay in it (see [wct.md](wct.md)).

The flag is written at the end of the day, so a trade cancelled at 15:55 already shows `08` at 15:49. Keeping only `00` therefore uses a little information from after the trade. At these frequencies that is negligible, but it is the one place the file is not point in time.
