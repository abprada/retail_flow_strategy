# daily-stock-returns

Predicting next-day **cross-sectional** stock returns (relative performance within a universe
of liquid equities, not market direction) from a single public data source, examined in depth.

## Layout

```
src/dsr/
  data.py        # ingestion + point-in-time alignment
  universe.py    # liquid universe construction
  features.py    # raw fields -> one number per stock per day
  evaluation.py  # IC, quantile spreads, turnover, decay
notebooks/       # exploration
reports/         # write-up and figures
data/            # local only (gitignored)
tests/
```

## Setup

```bash
uv sync
uv run pytest
```
