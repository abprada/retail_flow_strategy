# daily-stock-returns

Predicting next-day **cross-sectional** US stock returns: how liquid stocks do relative to each other, not where the market goes.

**Data:**
- TAQ trades and quotes are examined in depth, including retail flow signed by the midpoint rule and by BJZZ.
- CRSP gives the returns and the universe.
- Robintrack holder counts are used for a side study and as a check on the retail signing.

Every input is known before the 15:50 ET cutoff. Research only; nothing trades live.

## Where it stands (2026-10-26)

**The final model is the two-branch model:** a CNN on the day's grey price chart and an MLP on 17 tabular features, trained together and refit yearly. It was chosen over LightGBM, the benchmark, for its Sharpe ratio on 2019-2023.

| | IC (t) | Decile long-short, bp a day (t) | Sharpe (t) |
|---|---|---|---|
| Validation, 2019-2023 (settings chosen here): two-branch | 0.0124 (5.34) | 8.7 (3.67) | 1.60 (3.72) |
| Validation: LightGBM | 0.0128 (4.60) | 9.2 (3.22) | 1.41 (3.28) |
| **Test, 2024-2025 (run once): two-branch** | 0.0137 (2.94) | 2.7 (0.54) | 0.41 (0.54) |
| Test: LightGBM | 0.0135 (2.95) | 2.5 (0.53) | 0.37 (0.54) |

- The long-short holds the top tenth long and the bottom tenth short, equal-weighted, $1 a side, before costs, on stocks of $1B or more.
- Every t is Newey-West (5 lags); the Sharpe ratio's by the delta method (`dsr.backtest.sharpe_t`).
- **In the test the two models cannot be told apart:** the Sharpe difference is +0.04 (95% interval −0.65 to +0.71). The ranking held up (IC t about 2.9), but both long-shorts earned about a third of what they did in validation, and neither Sharpe can be told from zero ([explanations/models/test.md](explanations/models/test.md)).
- A LightGBM run made on the test years early (2026-10-24), so the test would not hold up the schedule, stays sealed unread.

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
                     # stress test (11), the test (12)
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
