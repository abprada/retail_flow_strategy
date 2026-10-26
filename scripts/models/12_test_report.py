"""The test (explanations/models/test.md): the final model, the two-branch model after step 3 at λ = 0.1, against the
benchmark, LightGBM refit yearly, on 2024-2025.

    uv run python scripts/models/12_test_report.py                 # the test years, once both runs are written
    uv run python scripts/models/12_test_report.py --validation    # the same tables on 2019-2023, as a check

Every measure is on the >= $1B stocks, as in validation: rank IC and R² (t-statistics as `dsr.evaluation`), and the
daily decile long-short (long the top decile, short the bottom one, equal-weighted, $1 a side, before costs; summed,
not compounded): mean bp a day and its Newey-West t, Sharpe ratio and its t (`dsr.backtest.sharpe_t`: delta method,
Newey-West), annual volatility, turnover and the largest drawdown (`dsr.backtest.stress`), overall and by year. The model against the benchmark, paired by day: the
differences in IC and in spread (Newey-West t), in Sharpe ratio (95% block bootstrap, 21-day blocks) and the
long-shorts' correlation. Writes output/test_report.csv, output/test_by_year.csv and output/test_paired.csv (with
--validation: only prints).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dsr.backtest import DAYS_A_YEAR, books, daily, positions, stress
from dsr.evaluation import daily_decile_spread, daily_rank_ic, ic_summary, r2_summary

ROOT = Path(__file__).resolve().parents[2]
WALKFORWARD = ROOT / "data" / "walkforward"
OUT = ROOT / "output"
MODELS = {"CNN + MLP": "two_branch_day_cnn_lam0.1/comb_s3", "LightGBM": "lgbm_yearly"}  # the model, the benchmark
BUCKETS = 10  # deciles
BLOCK, REPS = 21, 10_000  # the bootstrap's block length (trading days) and draws


def measures(p: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """One model's >= $1B predictions: its measures, its measures by year, and its daily long-short and IC."""
    ic_days = daily_rank_ic(p["day"], p["pred"], p["ret_next"])
    spread = daily_decile_spread(p["day"], p["pred"], p["ret_next"])
    held, average = positions(p["day"], p["permno"], p["pred"], p["ret_next"], BUCKETS)
    ls, turnover = books(daily(held, average))["long-short"]
    ic, r2, s = ic_summary(ic_days), r2_summary(p["day"], p["pred"], p["target_demeaned"]), stress(ls, average)
    row = {"days": int(ls.notna().sum()), "ic": ic["ic_mean"], "ic_t": ic["ic_t"], "r2_pct": r2["r2_pct"],
           "r2_t": r2["r2_t"], "spread_bp": 1e4 * ic_summary(spread)["ic_mean"], "spread_t": ic_summary(spread)["ic_t"],
           **{k: s[k] for k in ("mean_bp", "t", "annual_vol_pct", "sharpe", "sharpe_t", "max_drawdown_pct")},
           "turnover": float(turnover.mean())}
    ls = ls.dropna()
    by_year = pd.DataFrame({"ic": ic_days.groupby(ic_days.index.year).mean(),
                            "mean_bp": 1e4 * ls.groupby(ls.index.year).mean(),
                            "sharpe": ls.groupby(ls.index.year).apply(lambda r: np.sqrt(DAYS_A_YEAR) * r.mean() / r.std())})
    return row, by_year.rename_axis("year"), ls, ic_days


def sharpe_difference(a: pd.Series, b: pd.Series, seed: int = 0) -> tuple[float, float, float]:
    """The annualized Sharpe ratio of a minus b's on their common days, and its 95% interval from a block bootstrap
    (blocks of BLOCK days drawn with replacement, both series resampled together)."""
    d = pd.concat([a, b], axis=1, join="inner").to_numpy()
    sharpe = lambda x: np.sqrt(DAYS_A_YEAR) * x.mean(0) / x.std(0, ddof=1)
    n, rng = len(d), np.random.default_rng(seed)
    starts = rng.integers(0, n - BLOCK + 1, (REPS, int(np.ceil(n / BLOCK))))
    draws = np.empty(REPS)
    for i in range(REPS):
        s = sharpe(d[(starts[i][:, None] + np.arange(BLOCK)).ravel()[:n]])
        draws[i] = s[0] - s[1]
    s = sharpe(d)
    return float(s[0] - s[1]), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def report(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, years, ls, ic = {}, {}, {}, {}
    for label, folder in MODELS.items():
        p = pd.read_parquet(root / folder / "predictions.parquet",
                            columns=["day", "permno", "pred", "target_demeaned", "ret_next", "large_cap"])
        rows[label], years[label], ls[label], ic[label] = measures(p[p["large_cap"]])
    (a, b), paired = MODELS, {}
    di = (ic[a] - ic[b]).dropna()
    dl = pd.concat([ls[a], ls[b]], axis=1, join="inner")
    diff, low, high = sharpe_difference(ls[a], ls[b])
    paired[f"{a} − {b}"] = {"ic": di.mean(), "ic_t": ic_summary(di)["ic_t"],
                            "spread_bp": 1e4 * (dl.iloc[:, 0] - dl.iloc[:, 1]).mean(),
                            "spread_t": ic_summary(dl.iloc[:, 0] - dl.iloc[:, 1])["ic_t"],
                            "sharpe": diff, "sharpe_low": low, "sharpe_high": high,
                            "correlation": float(dl.corr().iloc[0, 1])}
    return (pd.DataFrame(rows).T, pd.concat(years, names=["model"]), pd.DataFrame(paired).T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="12_test_report.py")
    parser.add_argument("--validation", action="store_true", help="the same tables on 2019-2023, printed only")
    args = parser.parse_args()
    table, by_year, paired = report(WALKFORWARD if args.validation else WALKFORWARD / "test")
    if not args.validation:
        OUT.mkdir(exist_ok=True)
        table.to_csv(OUT / "test_report.csv")
        by_year.to_csv(OUT / "test_by_year.csv")
        paired.to_csv(OUT / "test_paired.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).to_string(), by_year.round(4).to_string(), paired.round(4).to_string(), sep="\n\n")
