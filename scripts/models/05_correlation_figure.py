"""Correlation of the models' daily long-short returns, as a lower-triangle heat map.

    uv run python scripts/models/05_correlation_figure.py [tabular]

With no argument, the five models; `tabular`: LightGBM, MLP and Linear only (output/correlation_long_short_tabular.*).

Each model's daily long-short is as in `05_backtest.py` (its walk-forward predictions on the >= $1B stocks, long the top
decile and short the bottom one, equal-weighted, $1 a side, before costs). The correlation is over the days all the
models have a return. Writes output/correlation_long_short.{png,csv}.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from dsr.backtest import daily, positions

ROOT = Path(__file__).resolve().parents[2]
WALKFORWARD = ROOT / "data" / "walkforward"
OUT = ROOT / "output"
MODELS = {"LightGBM": "lgbm_yearly", "MLP": "mlp", "Linear": "linear", "CNN": "cnn_one_day_grey",
          "CNN + MLP": "two_branch_day_cnn_lam0.1/comb_s3"}
TABULAR = ("LightGBM", "MLP", "Linear")
BUCKETS = 10  # deciles
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#81c5ea", "#019df1", "#002438"])
NAVY, GAP = "#002438", 0.04


def long_short(models: dict[str, str]) -> pd.DataFrame:
    found = {}
    for label, folder in models.items():
        p = pd.read_parquet(WALKFORWARD / folder / "predictions.parquet", columns=["day", "permno", "pred", "ret_next", "large_cap"])
        p = p[p["large_cap"]]
        held, average = positions(p["day"], p["permno"], p["pred"], p["ret_next"], BUCKETS)
        found[label] = daily(held, average)["long_short"]
    return pd.DataFrame(found).dropna()


def draw(corr: pd.DataFrame) -> plt.Figure:
    names, n = list(corr.index), len(corr) - 1  # a staircase: the first model has no row, the last no column
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16})
    fig, ax = plt.subplots(figsize=(max(1.6 * n + 2.4, 7.2), 1.6 * n + 1.6), facecolor="white")
    for i in range(1, n + 1):
        for j in range(i):  # the cells below the diagonal only
            v = corr.iloc[i, j]
            x, y = j, n - i
            ax.add_patch(Rectangle((x + GAP, y + GAP), 1 - 2 * GAP, 1 - 2 * GAP, color=BLUES(0.15 + 0.85 * v), lw=0))
            ax.text(x + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=19, fontweight="bold",
                    color="white" if v > 0.6 else NAVY)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(n)], names[:-1])
    ax.set_yticks([n - i + 0.5 for i in range(1, n + 1)], names[1:])
    ax.tick_params(length=0, labelsize=15, colors=NAVY)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_aspect("equal")
    fig.suptitle("Correlation of daily long-short returns", fontsize=19, color=NAVY, x=0.02, ha="left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    tabular = sys.argv[1:] == ["tabular"]
    models = {k: v for k, v in MODELS.items() if k in TABULAR} if tabular else MODELS
    suffix = "_tabular" if tabular else ""
    ls = long_short(models)
    print(len(ls), "days,", ls.index.min().date(), "to", ls.index.max().date())
    corr = ls.corr()
    print(corr.round(2))
    OUT.mkdir(exist_ok=True)
    corr.to_csv(OUT / f"correlation_long_short{suffix}.csv")
    draw(corr).savefig(OUT / f"correlation_long_short{suffix}.png", dpi=200, facecolor="white")
