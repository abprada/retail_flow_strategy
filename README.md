# daily-stock-returns

Predicting next-day **cross-sectional** US stock returns: how liquid stocks do relative to each other, not where the market goes.

**Data:**
- TAQ trades and quotes are examined in depth, including retail flow signed by the midpoint rule and by BJZZ.
- CRSP gives the returns and the universe.
- Robintrack holder counts are used for a side study and as a check on the retail signing.

Every input is known before the 15:50 ET cutoff. Research only; nothing trades live.

## Where it stands (2026-10-25)

**The final model is LightGBM.** It uses 17 features and is refit yearly.

| | IC (t) | Decile long-short, bp a day (t) | Sharpe |
|---|---|---|---|
| Validation, 2019-2023 (settings chosen here) | 0.0128 (4.60) | 9.2 (3.22) | 1.41 |

- The long-short holds the top tenth long and the bottom tenth short, equal-weighted, $1 a side, before costs.
- The test years, 2024-2025, are not read. A LightGBM run on them, made early (2026-10-24) so the test would not hold up the schedule, is sealed unread ([explanations/models/test.md](explanations/models/test.md)).
- CNNs on chart images and a two-branch model (a chart CNN with a tabular MLP) were tried. None beat LightGBM on 2019-2023, so none was adopted.
- The CNN family's pick is the two-branch model (a chart CNN and an MLP trained together, λ = 0.1): IC 0.0124 and Sharpe 1.60 on 2019-2023, against LightGBM's 0.0128 and 1.41.

**Where to read more:**
- [explanations/summary.md](explanations/summary.md): every model and result.
- [explanations/README.md](explanations/README.md): one file per topic.

## Layout

```
scripts/
  data_acquisition/  # raw data, downloaded unchanged: TAQ (the pull runs on WRDS Cloud, wrds_cloud/), CRSP, Robintrack
  taq/               # clean and explore (01), the model panel and the CNNs' image data (02), checks/
  robintrack/        # clean and explore (01), holder changes and returns (02), the retail-signing rules against it (03)
  models/            # sweeps (03), walk-forward (04), backtest (05), CNNs (06-08), two-branch model (10),
                     # stress test (11)
src/dsr/             # reusable code
tests/               # its tests
notebooks/CNN/       # inside the CNNs: the images, kernels and saliency
explanations/        # how everything works and every result, one topic per file
reports/             # the odd-lot bad prints, and figures
output/              # figures for presentations
data/                # local only (gitignored)
```

The rules every input follows are in [AGENTS.md](AGENTS.md):
- no lookahead past the 15:50 cutoff;
- no survivorship bias;
- settings chosen on the training years only.

## Setup

```bash
uv sync
uv run pytest
```

The scripts are marimo notebooks.
- **To open one:** `uv run marimo edit scripts/models/05_backtest.py`.
- **To run a long job from the terminal:** each script's "Running it" cell lists its commands. For example, LightGBM's yearly walk-forward is `uv run python scripts/models/04_walkforward.py run --models lgbm --every year`.
