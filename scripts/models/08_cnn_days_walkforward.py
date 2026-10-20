import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Walk-forward of the CNN on 20-day daily images

    Jiang, Kelly and Xiu's 20-day image (64 × 60): 20 trading days of daily bars up to the decision day's cutoff (each past day to its close, the decision day to 15:50), a 20-day moving average, volume, and the six retail colors per day, drawn on the GPU from `data/processed/cnn_days_train.npz` (`scripts/taq/02_cnn_days.py`). Refit every year over 2019-2023 and scored like the tabular models and the other CNNs (`06_cnn_walkforward.py`, the one-day images; `07_cnn_hours_walkforward.py`, the 21-day hourly images).

    - **Rows:** every large-cap stock-day (`cap_prev` at least $1B) with a model-panel row; it trains on all of them with a target and predicts all of them.
    - **As Jiang, Kelly and Xiu:** early stopping on the hold-out's squared error (the training window's last 12 months), patience 2 epochs, at most 10. One seed, as the other CNN runs (about 6,400 images a second: 6 to 11 minutes an epoch from the 2019 fold to the 2023 one, about 4 hours in all). The network of `explanations/CNN/cnn.md` (32, 64 and 128 filters; their 20-day model has 64, 128 and 256), AdamW with learning rate 1e-4 and weight decay 0.05, batches of 1,024.

    **The test years (2024-2025) are not read.**
    """)
    return


@app.cell
def _():
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch

    import dsr.models  # noqa: F401  lightgbm before torch: torch first crashes lightgbm on macOS (two OpenMP runtimes)
    from dsr.cnn import ChartCNN, DayWindows, fit, predict
    from dsr.evaluation import daily_rank_ic, measures, summarize
    from dsr.walkforward import fold_rows, folds

    return (
        ChartCNN,
        DayWindows,
        Path,
        argparse,
        daily_rank_ic,
        fit,
        fold_rows,
        folds,
        json,
        measures,
        mo,
        np,
        pd,
        predict,
        summarize,
        sys,
        time,
        torch,
    )


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    DAYS = ROOT / "data" / "processed" / "cnn_days_train.npz"
    PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
    OUT = ROOT / "data" / "walkforward" / "cnn_days"
    TARGET = "target_demeaned"
    FIRST, EVERY, HOLD_OUT_MONTHS = "2019-01-01", "year", 12
    TRAIN_EVERY = 1  # every day
    SEEDS = 1  # as the other CNN runs, so they compare like for like; seeds once it shows something
    PARAMS = {"filters": [32, 64, 128], "dropout": 0.5, "batch_size": 1024, "max_epochs": 10, "patience": 2,
              "learning_rate": 1e-4, "weight_decay": 0.05, "norm_sample": 20_000}
    DEVICE = "mps"
    return DEVICE, EVERY, FIRST, HOLD_OUT_MONTHS, DAYS, OUT, PANEL, PARAMS, SEEDS, TARGET, TRAIN_EVERY


@app.cell
def _(
    ChartCNN,
    DEVICE,
    EVERY,
    FIRST,
    HOLD_OUT_MONTHS,
    DAYS,
    DayWindows,
    PANEL,
    PARAMS,
    TARGET,
    TRAIN_EVERY,
    daily_rank_ic,
    fit,
    fold_rows,
    folds,
    json,
    measures,
    np,
    pd,
    predict,
    summarize,
    time,
    torch,
):
    def load():
        """The large-cap stock-days with a model-panel row (their keys, target and return), and every stock-day's
        hourly data on the GPU."""
        with np.load(DAYS) as z:
            stored = {k: z[k] for k in ("day", "permno", "prices", "shares", "colors", "days", "early")}
        permnos = np.unique(stored["permno"])
        stock_of = pd.Series(np.arange(len(permnos)), index=permnos)
        row_of = np.full((len(permnos), len(stored["days"])), -1, np.int32)
        row_of[stock_of[stored["permno"]].to_numpy(), stored["day"]] = np.arange(len(stored["permno"]), dtype=np.int32)
        day_of = pd.Series(np.arange(len(stored["days"])), index=pd.to_datetime(stored["days"]))
        panel = pd.read_parquet(PANEL, columns=["day", "permno", "large_cap", "ret_next", TARGET])
        panel["day"] = pd.to_datetime(panel["day"])
        keys = panel[panel["large_cap"] & panel["day"].isin(day_of.index) & panel["permno"].isin(permnos)]
        keys = keys.sort_values(["day", "permno"]).reset_index(drop=True)
        keys["day_pos"], keys["stock"] = day_of[keys["day"]].to_numpy(), stock_of[keys["permno"]].to_numpy()
        images = DayWindows(stored["prices"], stored["shares"], stored["colors"], row_of, stored["early"],
                             keys["stock"].to_numpy(), keys["day_pos"].to_numpy(), DEVICE)
        return keys, images


    def speed(batches=40):
        """Images a second through one training step (draw, forward, backward)."""
        keys, images = load()
        y = keys[TARGET].to_numpy(np.float32)
        train = np.flatnonzero(~np.isnan(y))
        net = ChartCNN(tuple(PARAMS["filters"]), PARAMS["dropout"], height=images.height, width=images.width).to(DEVICE)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        yt = torch.as_tensor(np.nan_to_num(y), device=DEVICE)
        b = PARAMS["batch_size"]
        idx = torch.as_tensor(np.random.default_rng(0).choice(train, b * batches), device=DEVICE)
        for warm in (True, False):
            torch.mps.synchronize()
            start = time.perf_counter()
            for i in range(3 if warm else batches):
                x = images.draw(idx[i * b:(i + 1) * b])
                opt.zero_grad()
                torch.nn.functional.mse_loss(net(x), yt[idx[i * b:(i + 1) * b]]).backward()
                opt.step()
            torch.mps.synchronize()
        rate = batches * b / (time.perf_counter() - start)
        sampled = int(((keys["day_pos"] + keys["permno"]) % TRAIN_EVERY == 0).sum())
        return {"images_a_second": round(rate), "stock_days": len(keys), "training_sample_all_days": sampled,
                "minutes_an_epoch_on_the_sample_of_all_days": round(sampled / rate / 60, 1)}


    def walk_forward(out, n_folds=None, seeds=1):
        """Fit every yearly fold, stopping on the hold-out's squared error, average the seeds' predictions of its year,
        score it, and write the predictions, the folds' and epochs' measures (after every fold) and the summary to
        `out`."""
        keys, images = load()
        day, y = keys["day"].to_numpy(), keys[TARGET].to_numpy(np.float32)
        ret = keys["ret_next"].to_numpy()
        sampled = ((keys["day_pos"] + keys["permno"]) % TRAIN_EVERY == 0).to_numpy()
        out.mkdir(parents=True, exist_ok=True)
        (out / "settings.json").write_text(json.dumps({"params": PARAMS, "seeds": seeds, "every": EVERY,
                                                       "hold_out_months": HOLD_OUT_MONTHS, "target": TARGET,
                                                       "train_every": TRAIN_EVERY, "stop_on": "holdout squared error"},
                                                      indent=2))
        predictions, per_fold, epochs = [], [], []
        for fold in folds(np.unique(day), first=FIRST, every=EVERY)[:n_folds]:
            start = time.perf_counter()
            train, hold, rows = fold_rows(day, y, np.ones(len(y), bool), fold, HOLD_OUT_MONTHS, holds_out=True)
            train, hold, rows = np.flatnonzero(train & sampled), np.flatnonzero(hold & sampled), np.flatnonzero(rows)
            hold_ic = lambda pred: float(daily_rank_ic(day[hold], pred, ret[hold]).mean())
            preds, chosen = [], []
            for seed in range(seeds):
                def log(e, seed=seed, t0=time.perf_counter()):
                    epochs.append({"fold": fold.name, "seed": seed, "epoch": e["epoch"], "train_loss": e["train_loss"],
                                   "holdout_loss": e["holdout_loss"], "holdout_ic": hold_ic(e["holdout_pred"]),
                                   "minutes": (time.perf_counter() - t0) / 60})
                    print(f"cnn-days {fold.name} seed {seed} epoch {e['epoch']}: train {e['train_loss']:.4f}, hold-out "
                          f"{e['holdout_loss']:.4f}, hold-out IC {epochs[-1]['holdout_ic']:.4f}, "
                          f"{epochs[-1]['minutes']:.0f} min", flush=True)

                net, scale, info = fit(images, y, train, hold, PARAMS, seed=seed, on_epoch=log)
                preds.append(predict(net, images, rows) * scale)
                chosen.append(info["epochs"])
            scored = pd.DataFrame({"day": day[rows], "permno": keys["permno"].to_numpy()[rows], "fold": fold.name,
                                   "pred": np.mean(preds, axis=0), TARGET: y[rows], "ret_next": ret[rows],
                                   "large_cap": True})
            predictions.append(scored)
            m = measures(scored, TARGET)[0]
            per_fold.append({"fold": fold.name, "train_rows": len(train), "hold_rows": len(hold),
                             "epochs": float(np.mean(chosen)), "minutes": (time.perf_counter() - start) / 60,
                             **{f"oos_1b_{k}": m[k] for k in ("ic_mean", "r2_pct", "mse", "decile_spread_bp")}})
            print(f"cnn-days {fold.name}: {per_fold[-1]['minutes']:.0f} min, epoch {per_fold[-1]['epochs']:.0f} kept; "
                  f">= $1B: IC {m['ic_mean']:.4f}, R² {m['r2_pct']:.4f}%, spread {m['decile_spread_bp']:.1f} bp",
                  flush=True)
            pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet")
            pd.DataFrame(per_fold).to_parquet(out / "folds.parquet")
            pd.DataFrame(epochs).to_parquet(out / "epochs.parquet")
        summary = {"oos_1b": summarize(pd.concat(predictions, ignore_index=True), TARGET)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    return speed, walk_forward


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it (from the terminal)

    ```
    uv run python scripts/models/08_cnn_days_walkforward.py speed                               # images a second
    uv run python scripts/models/08_cnn_days_walkforward.py run                                 # every yearly fold
    uv run python scripts/models/08_cnn_days_walkforward.py run --folds 1 --out /tmp/cnn_days   # the first fold
    ```

    Opened in marimo, this cell only shows the commands; the cell below reads the results.
    """)
    return


@app.cell
def _(OUT, Path, SEEDS, argparse, mo, speed, sys, walk_forward):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="08_cnn_days_walkforward.py")
        sub = parser.add_subparsers(dest="command", required=True)
        sub.add_parser("speed", help="measure images a second through a training step")
        r = sub.add_parser("run", help="the yearly walk-forward")
        r.add_argument("--folds", type=int, help="only the first N folds")
        r.add_argument("--seeds", type=int, default=SEEDS)
        r.add_argument("--out", type=Path, default=OUT)
        args = parser.parse_args(argv)
        print(speed() if args.command == "speed" else walk_forward(args.out, args.folds, args.seeds), flush=True)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


@app.cell
def _(OUT, json, mo, pd):
    if (OUT / "folds.parquet").exists():
        _summary = json.loads((OUT / "summary.json").read_text())["oos_1b"] if (OUT / "summary.json").exists() else None
        _keys = ["ic_mean", "ic_t", "decile_spread_bp", "decile_spread_t", "r2_pct", "long_leg_bp", "short_leg_bp"]
        _out = mo.vstack([
            mo.md("**Out of sample, ≥ $1B, 2019-2023:**") if _summary else mo.md("**Still running:** the folds so far."),
            pd.Series({_k: _summary[_k] for _k in _keys}).round(4).to_frame("value") if _summary else mo.md(""),
            mo.md("**By fold:**"), pd.read_parquet(OUT / "folds.parquet").round(4),
            mo.md("**By epoch:**"), pd.read_parquet(OUT / "epochs.parquet").round(4),
        ])
    else:
        _out = mo.md("No results yet: run `uv run python scripts/models/08_cnn_days_walkforward.py run`.")
    _out
    return


if __name__ == "__main__":
    app.run()
