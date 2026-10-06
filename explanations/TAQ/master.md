# Master file (`taqmsec.mastm_YYYYMMDD`)

One row per symbol on the consolidated tape that day: what the security is, its CUSIP, where it is listed and which venues traded it.

It is the bridge from TAQ symbols to CRSP PERMNOs. Because it is published per day, its CUSIPs are as of that day, not a later corrected history.

On 2019-06-04 it has 8,812 symbols.

## Columns

| Column | Meaning |
|---|---|
| `date` | trading day |
| `symbol_15` | symbol in NYSE format: `sym_root` and `sym_suffix` of the trade and quote files, joined by a space (`BRK B`, `GOOG L`, `BFS PRC`). Joins this file to the others |
| `sec_desc` | security name |
| `cusip` | 9-digit CUSIP. The first 8 match CRSP's `cusip`. Blank for 59 symbols, all of them test symbols |
| `sec_type` | security type (below) |
| `cta_symbol` | the symbol as the SIP publishes it (`BRK.B`, `GOOGL`, `BFSpC`). Closer to CRSP's ticker than `symbol_15`, but still not a key |
| `old_symbol` | the previous symbol, if it changed that day. Only one on 2019-06-04: `SRL`, formerly `MFCB` |
| `test_symbol_flag` | `Y` for SIP test symbols, which are not real securities (59, e.g. `ZVZZT`) |
| `listed_exchange` | primary listing venue, same codes as the trade file's `ex`: `Q` Nasdaq 3,428, `N` NYSE 3,139, `P` NYSE Arca 1,598, `Z` Cboe BZX 328, `A` NYSE American 315, `V` IEX 4 |
| `tape` | `A` NYSE-listed, `B` other exchange-listed (Arca, American, Cboe, IEX), `C` Nasdaq-listed. Decides which SIP (CTA for A and B, UTP for C) and therefore which sale-condition codes apply |
| `uot` | unit of trade. The spec defines it as the minimum trade increment in shares, but the values are `1` (5,358) and `4` (3,454), which doesn't fit that reading. Not used |
| `round_lot` | shares in a round lot: 100 for all but 12 symbols (10 for high-priced stocks such as NVR and Markel and some preferreds; 1 for BRK A and Seaboard) |
| `nyse_industry_code` | NYSE's industry code; NYSE-group listings only, blank otherwise |
| `shares_outstanding` | shares outstanding that day, in millions (Apple 4,601.08) |
| `halt_delay_reason` | reason for a trading halt or delay; blank unless halted (one `P`, news pending, that day) |
| `specialist_clearing_agent`, `specialist_clearing_num`, `specialist_post_num`, `specialist_panel` | the NYSE or NYSE American designated market maker (DMM) firm and its floor post; blank for other listings |
| `tron_*` | "traded on": `1` if the symbol traded on that venue that day, e.g. `tron_finra`, `tron_nyse`, `tron_iex`. Reflects the whole day, so it is not point in time at 15:50 |
| `tick_pilot_indicator` | group in the SEC tick-size pilot (2016 to 2018): `C` control, `1`, `2`, `3` test groups. Blank on 2019-06-04, after the pilot ended |
| `effective_date` | date the symbol's current master record took effect (often years earlier) |

## Security types (`sec_type`)

| Code | Meaning | Symbols on 2019-06-04 |
|---|---|---|
| A | common stock | 5,033 |
| ETF | exchange-traded fund (all exchange-traded products listed on Nasdaq or BATS) | 2,069 |
| B | preferred stock | 494 |
| H | ADR, common | 394 |
| G | structured product | 196 |
| C | warrant | 171 |
| ETN | exchange-traded note | 125 |
| R | unit | 91 |
| ETV | not in the spec; likely exchange-traded vehicles | 77 |
| U | unknown | 67 |
| P | fund | 44 |
| D | rights | 22 |
| T | not in the spec; test symbols (12 of the 59 flagged by `test_symbol_flag`) | 12 |
| I | ADR, preferred | 4 |
| M | not in the spec; New York registry shares of Dutch N.V.s (Aegon, ArcelorMittal, Philips, STMicro, Unilever) | 5 |
| blank | | 8 |

The spec also lists `E` corporate bond, `J` ADR warrant, `K` ADR right, `Q` basket and `S` liquidating trust; none appear that day.

The universe is chosen from CRSP's classification, not this one. `sec_type` is only a cross-check.

## Point in time

Most fields describe the security as of that morning:

- `shares_outstanding` is that day's count, which makes it a candidate replacement for CRSP's share count. The return panel's market cap uses CRSP's, which can be dated before it was public.
- `tron_*` and `halt_delay_reason` reflect the whole day, so they are known only after the close.
