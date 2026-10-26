"""The test's fallback (explanations/models/test.md), fixed before any test result: if the 2025 refit cannot finish in
the time available, 2025 is scored by the 2024 fold's final model, the two-branch model after step 3 at λ = 0.1
(trained through 2023), and the report says so.

    uv run python scripts/models/12_test_fallback.py            # 2025 from the 2024 fold's step 3 weights
    uv run python scripts/models/12_test_fallback.py --check    # 2023 from the validation run's 2023 weights, on the
                                                                # CPU: must reproduce the predictions that run wrote
                                                                # (a sample of 20,000 stock-days)

The stock-days are the walk-forward's (`load_day` in 10_two_branch_walkforward.py): the large caps with a chart and a
panel row. The predictions are scaled back with the weights' own fold's target scale (the sd of the target over its
training years' large caps), as that fold's predictions are. The fallback writes them as fold 2025 of comb_s3 (its
`fold` column says "2024 model"), collects comb_s3 again so the report reads them with 2024's, and leaves
comb_s3/fallback.json as the record. It prints no measure of them.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dsr.cnn import Images
from dsr.two_branch import COMBINED, WIDTH, TwoBranchNet
from dsr.two_branch_train import TwoBranchData, collect, fold_indices, outputs, scored_frame, write_fold
from dsr.walkforward import feature_columns, folds

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
DAY_BARS, DAY_BARS_TEST = PROCESSED / "cnn_bars_train.npz", PROCESSED / "cnn_bars_test.npz"
PANEL, TEST_PANEL = PROCESSED / "model_panel_train.parquet", PROCESSED / "model_panel_test.parquet"
VALIDATION = ROOT / "data" / "walkforward" / "two_branch_day_cnn_lam0.1"  # --check
TEST = ROOT / "data" / "walkforward" / "test" / "two_branch_day_cnn_lam0.1"
TARGET, HOLD_OUT_MONTHS = "target_demeaned", 12
FIRST, TEST_FIRST = "2019-01-01", "2024-01-01"  # the validation's first fold, the test's
HIDDEN = COMBINED["16-8"]  # the combined MLP of step 2's setting, lr0.0003_h16-8
CHECK_SAMPLE = 20_000  # --check: the 2023 stock-days scored again (the whole year takes long on the CPU)


def stock_days(test: bool) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """As load_day: the large-cap stock-days with drawing data and a panel row (`test`: the test years' after the
    training ones), sorted by day and PERMNO, with their features and day_pos; the feature columns; and the drawing
    data's arrays, which keys["pos"] indexes."""
    files = [DAY_BARS, DAY_BARS_TEST] if test else [DAY_BARS]
    parts, arrays = [], {}
    for f in files:
        with np.load(f) as z:
            parts.append(pd.DataFrame({"day": pd.to_datetime(z["day"]), "permno": z["permno"]}))
            for k in ("rows", "volume", "colors"):
                arrays.setdefault(k, []).append(z[k])
    keys = pd.concat(parts, ignore_index=True)
    arrays = {k: np.concatenate(a) for k, a in arrays.items()}
    panel = pd.concat([pd.read_parquet(p) for p in ([PANEL, TEST_PANEL] if test else [PANEL])], ignore_index=True)
    features = feature_columns(panel.columns)
    panel["day"] = pd.to_datetime(panel["day"])
    keys = keys.reset_index(names="pos").merge(panel, on=["day", "permno"], how="inner", validate="one_to_one")
    keys = keys[keys["large_cap"]].sort_values(["day", "permno"]).reset_index(drop=True)
    keys["day_pos"] = np.searchsorted(np.unique(panel["day"]), keys["day"])
    return keys, features, arrays


def score(weights: Path, keys: pd.DataFrame, features, arrays: dict, scale_fold, rows, label: str,
          device: str) -> pd.DataFrame:
    """The step 3 model in `weights` on the stock-days `rows` (positions in keys), scaled back with the target scale of
    `scale_fold`, the fold it was trained in."""
    scale = fold_indices(keys, scale_fold, TARGET, HOLD_OUT_MONTHS)["scale"]
    k = keys.iloc[rows].reset_index(drop=True)
    pos = k["pos"].to_numpy()
    data = TwoBranchData(Images(*(arrays[a][pos] for a in ("rows", "volume", "colors")), device),
                         k[features].to_numpy(np.float32), device)
    net = TwoBranchNet(WIDTH, HIDDEN, chart="day_cnn")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    pred = outputs(net.to(device), data, np.arange(len(k)))["comb"]
    return scored_frame(k, np.arange(len(k)), pred, scale, label, TARGET)


def check(sample: int = CHECK_SAMPLE) -> None:
    """Fold 2023 of the validation run, `sample` of its stock-days scored again on the CPU from its step 3 weights:
    the same predictions, and the same stock-days in the year."""
    keys, features, arrays = stock_days(test=False)
    fold = {f.name: f for f in folds(np.unique(keys["day"]), first=FIRST, every="year")}["2023"]
    year = np.flatnonzero(keys["day"].isin(fold.predict_days).to_numpy())
    written = pd.read_parquet(VALIDATION / "comb_s3" / "folds" / "2023.parquet")
    rows = np.sort(np.random.default_rng(0).choice(year, sample, replace=False))
    again = score(VALIDATION / "checkpoints" / "2023" / "stage3.pt", keys, features, arrays, fold, rows, "2023", "cpu")
    both = written.merge(again, on=["day", "permno"], suffixes=("", "_again"), validate="one_to_one")
    gap = (both["pred"] - both["pred_again"]).abs().max() / written["pred"].std()
    rank = both["pred"].corr(both["pred_again"], method="spearman")
    print(f"stock-days in 2023: written {len(written):,}, here {len(year):,}; scored again {len(again):,}, "
          f"matched {len(both):,}")
    print(f"largest difference {gap:.2e} of the predictions' sd, rank correlation {rank:.8f}")
    print("REPRODUCED" if len(written) == len(year) and len(both) == sample and gap < 1e-3 and rank > 0.9999
          else "NOT REPRODUCED")


def fallback() -> None:
    """2025 from the 2024 fold's step 3 weights, written as comb_s3's fold 2025."""
    if (TEST / "comb_s3" / "folds" / "2025.parquet").exists():
        raise SystemExit("comb_s3 already has 2025: the refit finished, the fallback is not used")
    keys, features, arrays = stock_days(test=True)
    found = {f.name: f for f in folds(np.unique(keys["day"]), first=TEST_FIRST, every="year")}
    rows = np.flatnonzero(keys["day"].isin(found["2025"].predict_days).to_numpy())
    frame = score(TEST / "checkpoints" / "2024" / "stage3.pt", keys, features, arrays, found["2024"], rows,
                  "2024 model", "mps")
    write_fold(TEST / "comb_s3", "2025", frame)
    collect(TEST / "comb_s3", TARGET)
    (TEST / "comb_s3" / "fallback.json").write_text(json.dumps(
        {"2025": "scored by the 2024 fold's step 3 model (trained through 2023): the refit did not finish in time",
         "rows": len(frame), "written": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"fallback: 2025 scored by the 2024 model, {len(frame):,} stock-days, written to {TEST / 'comb_s3'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="12_test_fallback.py")
    parser.add_argument("--check", action="store_true", help="reproduce validation fold 2023 on the CPU")
    args = parser.parse_args()
    check() if args.check else fallback()
