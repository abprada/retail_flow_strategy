"""Does each model's long-short return load on the Fama-French factors, and is there alpha left over?

    uv run python scripts/models/05_factor_alpha.py

Each model's daily long-short (the top decile minus the bottom one of its walk-forward predictions on the >= $1B stocks,
equal-weighted, $1 a side, before costs; the same series as `05_backtest.py`) is regressed on the market, size, value,
profitability, investment (CMA) and momentum factors. The long-short is self-financing, so its return is not reduced by
the risk-free rate. A day's long-short is earned from its close to the next trading day's close, so it is paired with the
factors dated the next trading day. Newey-West standard errors, 5 lags, as for the other t-statistics.

Needs data/raw/factors/factors_daily_from_2013.csv (00_data_factors.py). Writes output/factor_alpha.csv: per model, the
annualized alpha (% a year: the daily intercept times 252), its t-statistic, each loading and its t-statistic, and R².
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from dsr.backtest import daily, positions

ROOT = Path(__file__).resolve().parents[2]
WALKFORWARD = ROOT / "data" / "walkforward"
FACTORS = ROOT / "data" / "raw" / "factors" / "factors_daily_from_2013.csv"
OUT = ROOT / "output"
MODELS = {"Linear": "linear", "LightGBM": "lgbm", "MLP": "mlp"}
FACTOR_NAMES = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
BUCKETS = 10  # deciles
NW_LAGS = 5


def long_short(folder: str) -> pd.Series:
    """The model's daily long-short return (decimal), indexed by the day the position is taken."""
    p = pd.read_parquet(WALKFORWARD / folder / "predictions.parquet", columns=["day", "permno", "pred", "ret_next", "large_cap"])
    p = p[p["large_cap"]]
    held, average = positions(p["day"], p["permno"], p["pred"], p["ret_next"], BUCKETS)
    return daily(held, average)["long_short"].dropna()


def next_day_factors() -> pd.DataFrame:
    """The factors, indexed by the day before they are earned: row t holds the next trading day's."""
    f = pd.read_csv(FACTORS, parse_dates=["date"]).set_index("date").sort_index()
    out = f[FACTOR_NAMES].iloc[1:].copy()  # each row moves to the previous trading day's label
    out.index = pd.Index(f.index[:-1], name="day")
    return out


def regress(y: pd.Series, x: pd.DataFrame) -> dict:
    both = pd.concat([y.rename("y"), x], axis=1, join="inner").dropna()
    fit = sm.OLS(both["y"], sm.add_constant(both[FACTOR_NAMES])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    row = {"days": len(both), "alpha_pct_year": 100 * 252 * fit.params["const"], "alpha_t": fit.tvalues["const"]}
    for name in FACTOR_NAMES:
        row[f"{name}_beta"], row[f"{name}_t"] = fit.params[name], fit.tvalues[name]
    row["r2"] = fit.rsquared
    row["sharpe_raw"] = float(np.sqrt(252) * both["y"].mean() / both["y"].std())
    return row


if __name__ == "__main__":
    x = next_day_factors()
    table = pd.DataFrame({label: regress(long_short(folder), x) for label, folder in MODELS.items()}).T
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "factor_alpha.csv")
    print(table.round(3).T.to_string())
