"""Cumulative long-short return of a set of models against the >= $1B universe held long, and their summary table, from
the backtest code.

    uv run python scripts/models/05_backtest_figure.py [tabular|cnn|cnn_mlp|test|final]

Same calculation as `05_backtest.py`: each model's walk-forward predictions on the >= $1B stocks, long the top decile and
short the bottom one, equal-weighted, $1 a side, before costs; returns are summed, not compounded. The universe is every
>= $1B stock held long, equal-weighted. `tabular`: linear, LightGBM and MLP (2019-2023); `cnn`: the one-day chart CNN with
four channels and with the grey channel alone; `cnn_mlp`: the MLP and the two-branch model (a CNN and an MLP
trained together, step 3 at λ = 0.1), the CNN family's pick; `test`: that model, the final one, against LightGBM refit
yearly on the test years, 2024-2025 (explanations/models/test.md); `final`: the final model alone over all its
walk-forward years, validation (2019-2023) then test (2024-2025), with a dashed line where the test starts. There are
no predictions before 2019: 2014-2018 is only ever training data. Writes output/cumulative_returns_<set>.{png,csv} and
output/summary_<set>.csv.
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dsr.backtest import daily, positions, universe
from dsr.evaluation import daily_rank_ic, ic_summary, r2_zero

ROOT = Path(__file__).resolve().parents[2]
WALKFORWARD = ROOT / "data" / "walkforward"
PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
TEST_PANEL = ROOT / "data" / "processed" / "model_panel_test.parquet"  # the universe of the test set
OUT = ROOT / "output"
SETS = {
    "tabular": {"Linear": "linear", "LightGBM": "lgbm", "MLP": "mlp"},
    "cnn": {"4 channels": "cnn_one_day", "Grey only": "cnn_one_day_grey"},
    "cnn_mlp": {"MLP": "mlp", "CNN + MLP": "two_branch_day_cnn_lam0.1/comb_s3"},
    "test": {"LightGBM": "test/lgbm_yearly", "CNN + MLP": "test/two_branch_day_cnn_lam0.1/comb_s3"},
    "final": {"CNN + MLP": ("two_branch_day_cnn_lam0.1/comb_s3", "test/two_branch_day_cnn_lam0.1/comb_s3")},
}  # a tuple of folders: one model's predictions, joined in time
PANELS = {"test": [TEST_PANEL], "final": [PANEL, TEST_PANEL]}  # the universe's days, PANEL otherwise
SPLITS = {"final": pd.Timestamp("2024-01-01")}  # the dashed line: where the test years start
COLORS = {"Linear": "#81c5ea", "LightGBM": "#019df1", "MLP": "#002438", "4 channels": "#019df1",
          "Grey only": "#002438", "Universe (EW)": "#6b7280"}
SET_COLORS = {"cnn_mlp": {"MLP": "#019df1", "CNN + MLP": "#002438"},
              "test": {"LightGBM": "#019df1", "CNN + MLP": "#002438"}, "final": {"CNN + MLP": "#019df1"}}  # over COLORS
BUCKETS = 10  # deciles
LABEL_GAP = 5.5  # percentage points: names of lines ending on the same day are never closer than this
LABEL_ROOM, LABEL_OFFSET = 0.126, 0.008  # the room right of the lines for their names and the names' gap, as fractions of
                                          # the days shown (230 and 15 days on 2019-2023)


def backtest(models: dict[str, str | tuple[str, ...]], panels: list[Path] = [PANEL]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each model's summed daily long-short return (%) and the universe's, by day; and a table of R², rank IC, Sharpe and turnover."""
    cum, rows = {}, []
    for label, folders in models.items():
        p = pd.concat([pd.read_parquet(WALKFORWARD / folder / "predictions.parquet",
                                       columns=["day", "permno", "pred", "ret_next", "target_demeaned", "large_cap"])
                       for folder in ([folders] if isinstance(folders, str) else folders)], ignore_index=True)
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
    u = pd.concat([pd.read_parquet(panel, columns=["day", "ret_next", "cap_prev", "large_cap"]) for panel in panels],
                  ignore_index=True)
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


def draw(cum: pd.DataFrame, colors: dict[str, str] = COLORS, split: pd.Timestamp | None = None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="white")
    span = cum.index[-1] - cum.index[0]
    for name in cum:
        ax.plot(cum.index, cum[name], color=colors[name], lw=2, ls=(0, (4, 3)) if name == "Universe (EW)" else "-")
    ends = pd.DataFrame({"day": {n: cum[n].last_valid_index() for n in cum}, "y": {n: cum[n].dropna().iloc[-1] for n in cum}})
    for day, group in ends.groupby("day"):  # lines that end together share one set of non-overlapping names
        for name, y in label_positions(group["y"]).items():
            ax.text(day + LABEL_OFFSET * span, y, name, color=colors[name], va="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="#d1d5db", lw=1)
    if split is not None:  # validation on its left, test on its right
        ax.axvline(split, color="#6b7280", lw=1, ls=(0, (4, 3)))
        for text, side in (("Validation", "right"), ("Test", "left")):
            ax.text(split + (1 if side == "left" else -1) * LABEL_OFFSET * span, 0.97, text, ha=side, va="top",
                    transform=ax.get_xaxis_transform(), color="#6b7280", fontsize=11)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(length=0, colors="#6b7280")
    ax.set_xlim(cum.index[0], cum.index[-1] + LABEL_ROOM * span)
    ax.set_xticks([t for t in ax.get_xticks() if t <= mdates.date2num(cum.index[-1])])  # no dates past the data
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if span > pd.Timedelta(days=3 * 365) else "%Y-%m"))
    ax.set_title("Cumulative return (%)", loc="left", fontsize=15, fontweight="bold", color="#002438", pad=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "tabular"
    cum, table = backtest(SETS[which], PANELS.get(which, [PANEL]))
    OUT.mkdir(exist_ok=True)
    name = "models" if which == "tabular" else which
    cum.to_csv(OUT / f"cumulative_returns_{name}.csv")
    table.to_csv(OUT / f"summary_{which}.csv")
    draw(cum, {**COLORS, **SET_COLORS.get(which, {})}, SPLITS.get(which)).savefig(OUT / f"cumulative_returns_{name}.png",
                                                                                  dpi=200, facecolor="white")
    print(table.round(4).to_string())
