"""Rolling one-year Sharpe ratio and rank IC of a set of models' decile long-short, from the backtest code.

    uv run python scripts/models/05_rolling_sharpe.py [tabular|classes]

The long-short of `05_backtest.py` (top decile minus bottom, >= $1B stocks, $1 a side, before costs, 2019-2023); the
Sharpe at each day is that of the previous 252 trading days (annualized, no risk-free rate), so the first value is one
year in. The IC at each day is the mean daily rank IC over the same 252 days. `tabular`: linear, LightGBM refit monthly
and MLP; `classes`: LightGBM refit yearly, the MLP, linear and the two-branch CNN + MLP (step 3 at λ = 0.1), drawn
one point a week (each week's last trading day, still over the year before it). Writes
output/rolling_sharpe.{png,csv} and output/rolling_ic.{png,csv}, with the set's name after them for `classes`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsr.backtest import daily, positions
from dsr.evaluation import daily_rank_ic

ROOT = Path(__file__).resolve().parents[2]
WALKFORWARD = ROOT / "data" / "walkforward"
OUT = ROOT / "output"
SETS = {"tabular": {"Linear": "linear", "LightGBM": "lgbm", "MLP": "mlp"},
        "classes": {"LightGBM": "lgbm_yearly", "MLP": "mlp", "Linear": "linear",
                    "CNN + MLP": "two_branch_day_cnn_lam0.1/comb_s3"}}
COLORS = {"tabular": {"Linear": "#81c5ea", "LightGBM": "#019df1", "MLP": "#002438"},
          "classes": {"Linear": "#81c5ea", "LightGBM": "#019df1", "MLP": "#2c5a74",  # the muted blue darkened: as
                      "CNN + MLP": "#002438"}}  # #467c9a it was too close to LightGBM's
WINDOW, BUCKETS = 252, 10
STEP = {"classes": "W"}  # the sets drawn one point a period (pandas period, "W" a week): the year to its last day
LABEL_GAP = 0.06  # share of the y range: names of lines ending together are never closer than this


def long_short(folder: str) -> tuple[pd.Series, pd.Series]:
    """The model's daily decile long-short and its daily rank IC."""
    p = pd.read_parquet(WALKFORWARD / folder / "predictions.parquet", columns=["day", "permno", "pred", "ret_next", "large_cap"])
    p = p[p["large_cap"]]
    held, average = positions(p["day"], p["permno"], p["pred"], p["ret_next"], BUCKETS)
    return daily(held, average)["long_short"].dropna(), daily_rank_ic(p["day"], p["pred"], p["ret_next"])


def rolling_sharpe(r: pd.Series) -> pd.Series:
    roll = r.rolling(WINDOW)
    return np.sqrt(252) * roll.mean() / roll.std()


def draw(rolling: pd.DataFrame, colors: dict[str, str], title: str):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="white")
    for name in rolling:
        ax.plot(rolling.index, rolling[name], color=colors[name], lw=2.6 if name == "CNN + MLP" else 2)
    ax.axhline(0, color="#d1d5db", lw=1)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(length=0, colors="#6b7280")
    last, ys = rolling.index[-1], {}
    gap = LABEL_GAP * float(rolling.max().max() - rolling.min().min())
    for name, y in rolling.iloc[-1].sort_values().items():
        ys[name] = max(y, max(ys.values(), default=-1e9) + gap)
    for name, y in ys.items():
        ax.text(last + pd.Timedelta(days=15), y, name, color=colors[name], va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(rolling.index[0], last + pd.Timedelta(days=170))
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color="#002438", pad=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tabular"
    suffix = "" if which == "tabular" else f"_{which}"
    series = {label: long_short(folder) for label, folder in SETS[which].items()}
    rolling = {"sharpe": pd.DataFrame({k: rolling_sharpe(ls) for k, (ls, _) in series.items()}).dropna(),
               "ic": pd.DataFrame({k: ic.rolling(WINDOW).mean() for k, (_, ic) in series.items()}).dropna()}
    if which in STEP:
        rolling = {kind: frame.groupby(frame.index.to_period(STEP[which])).tail(1) for kind, frame in rolling.items()}
    titles = {"sharpe": "Rolling one-year Sharpe", "ic": "Rolling one-year rank IC"}
    OUT.mkdir(exist_ok=True)
    for kind, frame in rolling.items():
        frame.to_csv(OUT / f"rolling_{kind}{suffix}.csv")
        draw(frame, COLORS[which], titles[kind]).savefig(OUT / f"rolling_{kind}{suffix}.png", dpi=200, facecolor="white")
        print(titles[kind] + "\n" + frame.resample("QE").last().round(3).to_string())
