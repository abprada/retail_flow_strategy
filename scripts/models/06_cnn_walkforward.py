import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Walk-forward of the CNN on chart images

    The main CNN of `explanations/CNN/` (one network on the four channels; `architecture.md`, `training_plan.md`), with its default settings and nothing tuned, refit every year over 2019-2023 and scored like the tabular models (`scripts/models/04_walkforward.py`).

    - **Rows:** every large-cap stock-day (`cap_prev` at least $1B) with drawing data (`scripts/taq/02_cnn_bars.py`) and a row in the model panel; it trains on those with a target, `target_demeaned`.
    - **Folds:** yearly, an expanding window from 2014 (`dsr.walkforward`): train on every earlier day, predict the year. The early-stopping hold-out is the training window's last 12 months, as for the tabular models.
    - **Each fit** (`dsr.cnn.fit`): squared error on the target scaled by its training standard deviation, AdamW, batches of 1,024, stopped after 2 epochs without improving the hold-out's squared error (at most 10), keeping the best epoch. The images are drawn on the GPU from the drawing data, batch by batch.
    - **An initial run:** one seed. The plan's final run averages 3 to 5.

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

    from dsr.cnn import ChartCNN, Images, fit, predict
    from dsr.evaluation import daily_rank_ic, measures, summarize
    from dsr.walkforward import fold_rows, folds

    return (
        ChartCNN,
        Images,
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
    BARS = ROOT / "data" / "processed" / "cnn_bars_train.npz"
    PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
    OUT = ROOT / "data" / "walkforward" / "cnn"
    TARGET = "target_demeaned"
    FIRST, EVERY, HOLD_OUT_MONTHS = "2019-01-01", "year", 12
    SEEDS = 1  # an initial run; the plan's final run averages 3 to 5
    PARAMS = {  # training_plan.md: fixed, with the defaults for the two settings it would tune
        "filters": [32, 64, 128], "dropout": 0.5, "batch_size": 1024, "max_epochs": 10, "patience": 2,
        "learning_rate": 1e-4, "weight_decay": 0.05, "norm_sample": 20_000,
    }
    DEVICE = "mps"
    return BARS, DEVICE, EVERY, FIRST, HOLD_OUT_MONTHS, OUT, PANEL, PARAMS, SEEDS, TARGET


@app.cell
def _(
    BARS,
    ChartCNN,
    DEVICE,
    EVERY,
    FIRST,
    HOLD_OUT_MONTHS,
    Images,
    PANEL,
    PARAMS,
    TARGET,
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
        """The stock-days with drawing data and a model-panel row: their keys, target and return, and their drawing
        data on the GPU."""
        with np.load(BARS) as z:
            keys = pd.DataFrame({"day": pd.to_datetime(z["day"]), "permno": z["permno"]})
            arrays = {k: z[k] for k in ("rows", "volume", "colors")}
        panel = pd.read_parquet(PANEL, columns=["day", "permno", "large_cap", "ret_next", TARGET])
        panel["day"] = pd.to_datetime(panel["day"])
        keys = keys.reset_index(names="pos").merge(panel, on=["day", "permno"], how="inner", validate="one_to_one")
        keys = keys[keys["large_cap"]].reset_index(drop=True)
        images = Images(*(arrays[k][keys["pos"].to_numpy()] for k in ("rows", "volume", "colors")), DEVICE)
        return keys, images


    def speed(batches=100):
        """Images a second through one training step (draw, forward, backward), and the epoch time that implies."""
        keys, images = load()
        y = keys[TARGET].to_numpy(np.float32)
        train = np.flatnonzero(~np.isnan(y))
        net = ChartCNN(tuple(PARAMS["filters"]), PARAMS["dropout"]).to(images.device)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        yt = torch.as_tensor(np.nan_to_num(y), device=images.device)
        idx = torch.as_tensor(np.random.default_rng(0).choice(train, PARAMS["batch_size"] * batches), device=images.device)
        for warm in (True, False):
            torch.mps.synchronize()
            start = time.perf_counter()
            for i in range(5 if warm else batches):
                b = idx[i * PARAMS["batch_size"]:(i + 1) * PARAMS["batch_size"]]
                opt.zero_grad()
                torch.nn.functional.mse_loss(net(images.draw(b)), yt[b]).backward()
                opt.step()
            torch.mps.synchronize()
        rate = batches * PARAMS["batch_size"] / (time.perf_counter() - start)
        return {"images_a_second": round(rate), "stock_days": len(keys),
                "minutes_an_epoch_on_all_training_days": round(len(train) / rate / 60, 1)}


    def walk_forward(out, n_folds=None, seeds=1):
        """Fit every yearly fold, predict its year, score it, and write the predictions, the folds' and epochs'
        measures (after every fold) and the summary to `out`."""
        keys, images = load()
        day, y = keys["day"].to_numpy(), keys[TARGET].to_numpy(np.float32)
        out.mkdir(parents=True, exist_ok=True)
        (out / "settings.json").write_text(json.dumps({"params": PARAMS, "seeds": seeds, "every": EVERY,
                                                       "hold_out_months": HOLD_OUT_MONTHS, "target": TARGET}, indent=2))
        predictions, per_fold, epochs = [], [], []
        for fold in folds(np.unique(day), first=FIRST, every=EVERY)[:n_folds]:
            start = time.perf_counter()
            train, hold, rows = fold_rows(day, y, np.ones(len(y), bool), fold, HOLD_OUT_MONTHS, holds_out=True)
            train, hold, rows = np.flatnonzero(train), np.flatnonzero(hold), np.flatnonzero(rows)
            preds, chosen = [], []
            for seed in range(seeds):
                def log(e, seed=seed, t0=time.perf_counter()):
                    ic = daily_rank_ic(day[hold], e["holdout_pred"], keys["ret_next"].to_numpy()[hold]).mean()
                    epochs.append({"fold": fold.name, "seed": seed, "epoch": e["epoch"], "train_loss": e["train_loss"],
                                   "holdout_loss": e["holdout_loss"], "holdout_ic": float(ic),
                                   "minutes": (time.perf_counter() - t0) / 60})
                    print(f"cnn {fold.name} seed {seed} epoch {e['epoch']}: train {e['train_loss']:.4f}, hold-out "
                          f"{e['holdout_loss']:.4f}, hold-out IC {ic:.4f}, {epochs[-1]['minutes']:.0f} min", flush=True)

                net, scale, info = fit(images, y, train, hold, PARAMS, seed=seed, on_epoch=log)
                preds.append(predict(net, images, rows) * scale)
                chosen.append(info["epochs"])
            scored = pd.DataFrame({"day": day[rows], "permno": keys["permno"].to_numpy()[rows], "fold": fold.name,
                                   "pred": np.mean(preds, axis=0), TARGET: y[rows],
                                   "ret_next": keys["ret_next"].to_numpy()[rows], "large_cap": True})
            predictions.append(scored)
            m = measures(scored, TARGET)[0]
            per_fold.append({"fold": fold.name, "train_rows": len(train), "hold_rows": len(hold),
                             "epochs": float(np.mean(chosen)), "minutes": (time.perf_counter() - start) / 60,
                             **{f"oos_1b_{k}": m[k] for k in ("ic_mean", "r2_pct", "mse", "decile_spread_bp")}})
            print(f"cnn {fold.name}: {per_fold[-1]['minutes']:.0f} min; >= $1B: IC {m['ic_mean']:.4f}, "
                  f"R² {m['r2_pct']:.4f}%, spread {m['decile_spread_bp']:.1f} bp", flush=True)
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
    uv run python scripts/models/06_cnn_walkforward.py speed                         # images a second, epoch time
    uv run python scripts/models/06_cnn_walkforward.py run                           # every yearly fold, one seed
    uv run python scripts/models/06_cnn_walkforward.py run --folds 1 --out /tmp/cnn  # the first fold only
    ```

    Opened in marimo, this cell only shows the commands; the cells below read the results.
    """)
    return


@app.cell
def _(OUT, Path, SEEDS, argparse, mo, speed, sys, walk_forward):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="06_cnn_walkforward.py")
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
        _out = mo.md("No results yet: run `uv run python scripts/models/06_cnn_walkforward.py run`.")
    _out
    return


if __name__ == "__main__":
    app.run()
