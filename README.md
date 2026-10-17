# daily-stock-returns

Predicting next-day **cross-sectional** stock returns (relative performance within a universe
of liquid equities, not market direction) from a single public data source, examined in depth.

## Layout

```
scripts/
  data_acquisition/  # raw data, downloaded unchanged: TAQ (wrds_cloud/), CRSP, Robintrack
  taq/               # clean and explore (01), the model panel (02), checks/
  robintrack/        # clean and explore (01), holder changes and returns (02)
  models/            # tuning (03), monthly walk-forward (04), backtest (05)
src/dsr/             # reusable code
tests/
notebooks/           # exploration
reports/             # write-up and figures
data/                # local only (gitignored)
```

## Setup

```bash
uv sync
uv run pytest
```
