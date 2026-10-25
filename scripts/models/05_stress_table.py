"""Risk and return table of a set of models' long-short, the legs and the signal, from the backtest code.

    uv run python scripts/models/05_stress_table.py [tabular|classes]

Each model's walk-forward predictions on the >= $1B stocks (2019-2023), long the top decile and short the bottom one,
equal-weighted, $1 a side, before costs, returns summed (`dsr.backtest`, as 05_backtest.py). `tabular`: linear, LightGBM
refit monthly and MLP; `classes`: one model per class, LightGBM refit yearly (the final model), the MLP, linear and the
two-branch CNN + MLP (step 3 at λ = 0.1, the CNN family's pick). Writes output/stress_<set>.csv (one column a model),
output/stress_<set>_by_year.csv (the long-short's Sharpe a year) and output/stress_<set>_slide.csv (the rows in SLIDE).

The legs are the long book's return and the short book's (minus the short stocks' return); "vs average" takes each
against the day's equal-weighted >= $1B average, which is what each leg adds to the long-short. Breakeven cost: the
one-way cost per dollar traded at which the long-short's mean return is used up (a day trades 4 x the mean turnover per $1
a side: each leg buys and sells its replaced share).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dsr.backtest import books, daily, positions, stress
from dsr.evaluation import daily_rank_ic, ic_summary, r2_summary

ROOT = Path(__file__).resolve().parents[2]
WALKFORWARD = ROOT / "data" / "walkforward"
OUT = ROOT / "output"
SETS = {"tabular": {"Linear": "linear", "LightGBM": "lgbm", "MLP": "mlp"},
        "classes": {"LightGBM": "lgbm_yearly", "MLP": "mlp", "Linear": "linear",
                    "CNN + MLP": "two_branch_day_cnn_lam0.1/comb_s3"}}
SLIDE = {"Max DD (%)": "Max drawdown (%)", "Max DD length (days)": "Max drawdown length (days)",  # the slide's rows
         "Avg. DD length (days)": "Avg drawdown length (days)", "Long Sharpe": "Long Sharpe",
         "Short Sharpe": "Short Sharpe", "Long return (% a year)": "Long return (% a year)",
         "Short return (% a year)": "Short return (% a year)", "Sortino": "Sortino",
         "Hit rate (share of winning days)": "Hit rate (share of winning days)", "Skew": "Skew",
         "Excess kurtosis": "Excess kurtosis", "Average win of winners (bp)": "Average win (bp)",
         "Average loss of losers (bp)": "Average loss (bp)", "Breakeven one-way cost (bp)": "Breakeven one-way cost (bp)"}
BUCKETS, DAYS = 10, 252


def column(folder: str) -> tuple[dict, pd.Series]:
    p = pd.read_parquet(WALKFORWARD / folder / "predictions.parquet",
                        columns=["day", "permno", "pred", "target_demeaned", "ret_next", "large_cap"])
    p = p[p["large_cap"]]
    held, average = positions(p["day"], p["permno"], p["pred"], p["ret_next"], BUCKETS)
    b = books(daily(held, average))
    ic = ic_summary(daily_rank_ic(p["day"], p["pred"], p["ret_next"]))
    ls, turnover = b["long-short"]
    s = stress(ls, average)
    ls = ls.dropna()
    month = ls.groupby(ls.index.to_period("M")).sum()
    long_s, short_s = stress(b["long leg"][0], average), stress(b["short leg"][0], average)
    long_x, short_x = stress(b["long leg − average"][0], average), stress(b["average − short leg"][0], average)
    out = {
        "R² (%)": r2_summary(p["day"], p["pred"], p["target_demeaned"])["r2_pct"],
        "Rank IC": ic["ic_mean"], "Rank IC t": ic["ic_t"],
        "Sharpe": s["sharpe"], "Sharpe t": s["sharpe_t"], "Sortino": s["sortino"],
        "Annual return (%)": s["annual_mean_pct"], "Annual vol (%)": s["annual_vol_pct"],
        "Turnover (share of book a day)": float(turnover.mean()),
        "Breakeven one-way cost (bp)": 1e4 * float(ls.mean()) / (4 * float(turnover.mean())),
        "Max drawdown (%)": s["max_drawdown_pct"],
        "Calmar (return / max drawdown)": s["annual_mean_pct"] / s["max_drawdown_pct"],
        "Drawdowns (count)": s["drawdowns"],
        "Avg drawdown length (days)": s["drawdown_days_mean"], "Max drawdown length (days)": s["drawdown_days_max"],
        "Long Sharpe": long_s["sharpe"], "Short Sharpe": short_s["sharpe"],
        "Long return (% a year)": long_s["annual_mean_pct"], "Short return (% a year)": short_s["annual_mean_pct"],
        "Long vs average (% a year)": long_x["annual_mean_pct"], "Short vs average (% a year)": short_x["annual_mean_pct"],
        "Hit rate (share of winning days)": s["win_days"],
        "Skew": s["skew"], "Excess kurtosis": s["excess_kurtosis"],
        "Average win (bp)": s["average_win_bp"], "Average loss (bp)": s["average_loss_bp"],
        "Win / loss ratio": s["average_win_bp"] / -s["average_loss_bp"],
        "Worst day (bp)": s["worst_bp"], "Best day (bp)": s["best_bp"], "1st percentile (bp)": s["p1_bp"],
        "Worst month (%)": 100 * float(month.min()), "Positive months (share)": float((month > 0).mean()),
        "Sharpe without best 1% of days": s["sharpe_without_best_days"],
        "Correlation with universe": s["corr_universe"],
    }
    return out, ls.groupby(ls.index.year).apply(lambda r: float(np.sqrt(DAYS) * r.mean() / r.std()))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tabular"
    columns, years = {}, {}
    for label, folder in SETS[which].items():
        columns[label], years[label] = column(folder)
    table, by_year = pd.DataFrame(columns), pd.DataFrame(years)
    slide = table.loc[list(SLIDE.values())].set_axis(list(SLIDE), axis=0)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / f"stress_{which}.csv")
    by_year.to_csv(OUT / f"stress_{which}_by_year.csv")
    slide.to_csv(OUT / f"stress_{which}_slide.csv")
    print(table.round(3).to_string())
    print("\nSharpe by year\n" + by_year.round(2).to_string())
