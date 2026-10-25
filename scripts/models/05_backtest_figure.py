"""Cumulative long-short return of a set of models against the >= $1B universe held long, and their summary table, from
the backtest code.

    uv run python scripts/models/05_backtest_figure.py [tabular|cnn|cnn_mlp]

Same calculation as `05_backtest.py`: each model's walk-forward predictions on the >= $1B stocks, long the top decile and
short the bottom one, equal-weighted, $1 a side, before costs; returns are summed, not compounded. The universe is every
>= $1B stock held long, equal-weighted. `tabular`: linear, LightGBM and MLP (2019-2023); `cnn`: the one-day chart CNN with
four channels and with the grey channel alone; `cnn_mlp`: the MLP and the two-branch model (a CNN and an MLP
trained together, step 3 at λ = 0.1), the CNN family's pick. Writes output/cumulative_returns_<set>.{png,csv} and
output/summary_<set>.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsr.backtest import daily, positions, universe
from dsr.evaluation import daily_rank_ic, ic_summary, r2_zero

ROOT = Path(__file__).resolve().parents[2]
WALKFORWARD = ROOT / "data" / "walkforward"
PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
OUT = ROOT / "output"
SETS = {
    "tabular": {"Linear": "linear", "LightGBM": "lgbm", "MLP": "mlp"},
    "cnn": {"4 channels": "cnn_one_day", "Grey only": "cnn_one_day_grey"},
    "cnn_mlp": {"MLP": "mlp", "CNN + MLP": "two_branch_day_cnn_lam0.1/comb_s3"},
}
COLORS = {"Linear": "#81c5ea", "LightGBM": "#019df1", "MLP": "#002438", "4 channels": "#019df1",
          "Grey only": "#002438", "Universe (EW)": "#6b7280"}
SET_COLORS = {"cnn_mlp": {"MLP": "#019df1", "CNN + MLP": "#002438"}}  # over COLORS
BUCKETS = 10  # deciles
LABEL_GAP = 5.5  # percentage points: names of lines ending on the same day are never closer than this


def backtest(models: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each model's summed daily long-short return (%) and the universe's, by day; and a table of R², rank IC, Sharpe and turnover."""
    cum, rows = {}, []
    for label, folder in models.items():
        p = pd.read_parquet(WALKFORWARD / folder / "predictions.parquet",
                            columns=["day", "permno", "pred", "ret_next", "target_demeaned", "large_cap"])
        p = p[p["large_cap"]]
        held, average = positions(p["day"], p["permno"], p["pred"], p["ret_next"], BUCKETS)
        days = daily(held, average)
        ls = days["long_short"].dropna()
        cum[label] = 100 * days["long_short"].fillna(0).cumsum()
        ic = ic_summary(daily_rank_ic(p["day"], p["pred"], p["ret_next"]))
        rows.append({"model": label, "days": len(ls), "r2_pct": 100 * r2_zero(p["pred"], p["target_demeaned"]),
                     "rank_ic": ic["ic_mean"], "rank_ic_t": ic["ic_t"],
                     "sharpe": float(np.sqrt(252) * ls.mean() / ls.std()),
                     "turnover": float(((days["turnover_long"] + days["turnover_short"]) / 2).mean())})
    first = min(c.index.min() for c in cum.values())
    last = max(c.index.max() for c in cum.values())
    u = pd.read_parquet(PANEL, columns=["day", "ret_next", "cap_prev", "large_cap"])
    u = u[u["large_cap"] & u["day"].between(first, last)]
    cum["Universe (EW)"] = 100 * universe(u["day"], u["ret_next"], u["cap_prev"])["equal"].cumsum()
    return pd.DataFrame(cum), pd.DataFrame(rows).set_index("model")


def label_positions(ends: pd.Series) -> dict[str, float]:
    """Where each name goes: at its line's end, moved only as far as needed to keep LABEL_GAP between names."""
    out, last = {}, float("-inf")
    for name, y in ends.sort_values().items():
        last = max(y, last + LABEL_GAP)
        out[name] = last
    return out


def draw(cum: pd.DataFrame, colors: dict[str, str] = COLORS):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="white")
    for name in cum:
        ax.plot(cum.index, cum[name], color=colors[name], lw=2, ls=(0, (4, 3)) if name == "Universe (EW)" else "-")
    ends = pd.DataFrame({"day": {n: cum[n].last_valid_index() for n in cum}, "y": {n: cum[n].dropna().iloc[-1] for n in cum}})
    for day, group in ends.groupby("day"):  # lines that end together share one set of non-overlapping names
        for name, y in label_positions(group["y"]).items():
            ax.text(day + pd.Timedelta(days=15), y, name, color=colors[name], va="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="#d1d5db", lw=1)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(length=0, colors="#6b7280")
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=230))
    ax.set_title("Cumulative return (%)", loc="left", fontsize=15, fontweight="bold", color="#002438", pad=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tabular"
    cum, table = backtest(SETS[which])
    OUT.mkdir(exist_ok=True)
    name = "models" if which == "tabular" else which
    cum.to_csv(OUT / f"cumulative_returns_{name}.csv")
    table.to_csv(OUT / f"summary_{which}.csv")
    draw(cum, {**COLORS, **SET_COLORS.get(which, {})}).savefig(OUT / f"cumulative_returns_{name}.png", dpi=200, facecolor="white")
    print(table.round(4).to_string())
