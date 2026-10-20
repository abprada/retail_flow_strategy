import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Walk-forward of the two-branch model

    The model of `explanations/CNN/fusion.md`: the 21-day hourly chart (grey) through a densely linked CNN and the 1D block, the 17 tabular features through the tuned MLP's layer, and a combined MLP on the two 16-number summaries (`dsr.two_branch`, trained by `dsr.two_branch_train`). Yearly folds over 2019-2023, run in the spec's order:

    1. `delta`: step 1b (the tabular branch alone) in every fold with δ = 0.8, 1, 1.5 and 2 and squared error; the δ with the highest `oos_1b/ic_mean` is kept (squared error is never chosen);
    2. `stage12`: fold by fold, step 1a (the CNN alone: every stock of $100M or more, one day in nine) and step 2 (the combined MLP on the frozen branches) with its 6 settings;
    3. `choose`: the step 2 setting with the highest `oos_1b/ic_mean`;
    4. `stage3`: fold by fold, the fine-tuning, once (`--lam 0 --out .../two_branch_lambda0` for the control).

    Each scored output gets its own folder under `data/walkforward/two_branch/` (`predictions.parquet` and `summary.json`, as the other walk-forwards), the balance diagnostics go to `balance.parquet` and each fold's weights to `checkpoints/`. A fold already written is skipped, so a stopped command resumes. **The test years (2024-2025) are not read.**
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

    import dsr.models  # noqa: F401  lightgbm before torch: torch first crashes lightgbm on macOS (two OpenMP runtimes)
    import torch

    from dsr.cnn import HourWindows, xavier
    from dsr.evaluation import daily_rank_ic
    from dsr.two_branch import COMBINED, CombinedMLP, TwoBranchNet, loss, prepare, step3_loss, summary_drift, weight_share
    from dsr.two_branch_train import (Balance, TwoBranchData, collect, finetune, fold_done, fold_indices, folds_done,
                                      ignores_chart, learns_nothing, outputs, pushes_chart_out, same_folds,
                                      scored_frame, tabular_predictions, train_combined, train_price, train_tabular,
                                      write_fold)
    from dsr.walkforward import feature_columns, folds

    return (
        Balance, COMBINED, CombinedMLP, HourWindows, Path, TwoBranchData, TwoBranchNet, argparse, collect,
        daily_rank_ic, feature_columns, finetune, fold_done, fold_indices, folds, folds_done, ignores_chart, json,
        learns_nothing, loss, mo, np, outputs, pd, prepare, pushes_chart_out, same_folds, scored_frame, step3_loss,
        summary_drift, sys, tabular_predictions, time, torch, train_combined, train_price, train_tabular,
        weight_share, write_fold, xavier,
    )


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    HOURS = ROOT / "data" / "processed" / "cnn_hours_train.npz"
    PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
    OUT = ROOT / "data" / "walkforward" / "two_branch"
    TARGET = "target_demeaned"
    FIRST, EVERY, HOLD_OUT_MONTHS = "2019-01-01", "year", 12
    DEVICE = "mps"
    DELTAS = {"0.8": 0.8, "1": 1.0, "1.5": 1.5, "2": 2.0, "mse": None}  # step 1b's grid; squared error is reported, never chosen
    PRICE = {"learning_rate": 1e-4, "weight_decay": 0.05, "batch_size": 512, "max_epochs": 5, "patience": 2}
    TABULAR = {"learning_rate": 2.9e-4, "weight_decay": 0.035, "batch_size": 4096, "epochs": 12}
    STEP2 = {f"lr{lr:g}_h{h}": {"learning_rate": lr, "hidden": h} for lr in (1e-4, 3e-4, 1e-3) for h in ("16", "16-8")}
    COMBINED_FIT = {"weight_decay": 0.035, "batch_size": 4096, "max_epochs": 40, "patience": 3}
    FINETUNE = {"epochs": 2, "batch_size": 512,
                "weight_decay": {"one_d": 0.05, "price_head": 0.05, "tab_body": 0.035, "tab_head": 0.035,
                                 "combined": 0.035}}
    QUICK = {"price_epochs": 1, "tabular_epochs": 1, "combined_epochs": 2, "finetune_epochs": 1}  # --quick
    CORR_SAMPLE = 20_000  # the fixed hold-out sample for the correlations of the model with each head
    return (COMBINED_FIT, CORR_SAMPLE, DELTAS, DEVICE, EVERY, FINETUNE, FIRST, HOLD_OUT_MONTHS, HOURS, OUT, PANEL,
            PRICE, QUICK, STEP2, TABULAR, TARGET)


@app.cell
def _(DEVICE, HOURS, HourWindows, PANEL, TARGET, TwoBranchData, feature_columns, np, pd):
    def load(stocks=None):
        """Every panel stock-day with hourly data (all of them, or `stocks` PERMNOs spread over them, as a quick check): their
        keys, target and return, and the data on the GPU (the charts drawn from the hourly data, the 17 features)."""
        with np.load(HOURS) as z:
            stored = {k: z[k] for k in ("day", "permno", "prices", "shares", "colors", "days", "early")}
        permnos = np.unique(stored["permno"])
        if stocks:  # spread over the PERMNOs, so the sample has large caps in every year
            permnos = permnos[::max(1, len(permnos) // stocks)][:stocks]
            keep = np.isin(stored["permno"], permnos)
            stored = {k: (v if k in ("days", "early") else v[keep]) for k, v in stored.items()}
        stock_of = pd.Series(np.arange(len(permnos)), index=permnos)
        row_of = np.full((len(permnos), len(stored["days"])), -1, np.int32)
        row_of[stock_of[stored["permno"]].to_numpy(), stored["day"]] = np.arange(len(stored["permno"]), dtype=np.int32)
        day_of = pd.Series(np.arange(len(stored["days"])), index=pd.to_datetime(stored["days"]))
        panel = pd.read_parquet(PANEL)
        features = feature_columns(panel.columns)
        panel["day"] = pd.to_datetime(panel["day"])
        keys = panel[panel["day"].isin(day_of.index) & panel["permno"].isin(permnos)]
        keys = keys.sort_values(["day", "permno"]).reset_index(drop=True)
        keys["day_pos"], stock = day_of[keys["day"]].to_numpy(), stock_of[keys["permno"]].to_numpy()
        windows = HourWindows(stored["prices"], stored["shares"], stored["colors"], row_of, stored["early"], stock,
                              keys["day_pos"].to_numpy(), DEVICE)
        data = TwoBranchData(windows, keys[features].to_numpy(np.float32), DEVICE)
        return keys[["day", "permno", "day_pos", "large_cap", "ret_next", TARGET]], data

    return (load,)


@app.cell
def _(
    COMBINED, COMBINED_FIT, CORR_SAMPLE, CombinedMLP, DELTAS, DEVICE, EVERY, FINETUNE, FIRST, HOLD_OUT_MONTHS, PRICE,
    QUICK, STEP2, TABULAR, TARGET, Balance, TwoBranchNet, collect, daily_rank_ic, finetune, fold_done, fold_indices,
    folds, folds_done, ignores_chart, json, learns_nothing, load, loss, np, outputs, pd, prepare, pushes_chart_out,
    same_folds, scored_frame, step3_loss, summary_drift, tabular_predictions, time, torch, train_combined, train_price,
    train_tabular, weight_share, write_fold, xavier,
):
    def spearman(a, b):
        return float(pd.Series(a.cpu().numpy()).corr(pd.Series(b.cpu().numpy()), method="spearman"))


    def read_choices(out):
        path = out / "choices.json"
        return json.loads(path.read_text()) if path.exists() else {}


    def write_choices(out, **found):
        out.mkdir(parents=True, exist_ok=True)
        (out / "choices.json").write_text(json.dumps({**read_choices(out), **found}, indent=2))


    def fold_list(keys, n_folds):
        return folds(np.unique(keys["day"].to_numpy()), first=FIRST, every=EVERY)[:n_folds]


    def scaled_target(keys, ix, device):
        y = keys[TARGET].to_numpy(np.float32) / ix["scale"]
        return torch.as_tensor(np.nan_to_num(y), device=device)


    def run_delta(out, n_folds=None, stocks=None, quick=False):
        """Step 1b in every fold for every δ (and squared error), then the δ with the highest oos_1b/ic_mean."""
        keys, data = load(stocks)
        balance = Balance(out / "balance.parquet")
        p = {**TABULAR, **({"epochs": QUICK["tabular_epochs"]} if quick else {})}
        for fold in fold_list(keys, n_folds):
            ix = fold_indices(keys, fold, TARGET, HOLD_OUT_MONTHS)
            y = scaled_target(keys, ix, data.device)
            for name, delta in DELTAS.items():
                folder = out / f"tab_s1_delta_{name}"
                if fold_done(folder, fold.name):
                    continue
                start = time.perf_counter()
                torch.manual_seed(0)
                net = TwoBranchNet().to(DEVICE)
                train_tabular(net, data.features, y, ix["train"], p, delta,
                              on_epoch=lambda e: balance.add({"fold": fold.name, "step": "1b", "delta": name, **e}))
                (out / "checkpoints" / fold.name).mkdir(parents=True, exist_ok=True)
                torch.save(net.tabular.state_dict(), out / "checkpoints" / fold.name / f"tab_delta_{name}.pt")
                pred = tabular_predictions(net, data.features, ix["test"])
                write_fold(folder, fold.name, scored_frame(keys, ix["test"], pred, ix["scale"], fold.name, TARGET))
                print(f"two-branch {fold.name} step 1b delta {name}: {time.perf_counter() - start:.0f} s", flush=True)
        same_folds([out / f"tab_s1_delta_{name}" for name in DELTAS])
        table = {name: collect(out / f"tab_s1_delta_{name}", TARGET)["ic_mean"] for name in DELTAS}
        chosen = max((n for n, d in DELTAS.items() if d is not None), key=table.get)
        write_choices(out, delta=DELTAS[chosen], delta_name=chosen, delta_ic=table)
        return {"oos_1b/ic_mean by delta": table, "chosen": chosen}


    def run_stage12(out, n_folds=None, stocks=None, channels=4, quick=False, max_epochs=None):
        """Fold by fold: step 1a, then step 2 with every setting not yet written; prints warning signs 1 and 2 on the
        first fold. A fold's trained price branch (price.pt) and its summaries (summaries.pt) are saved and reused, so
        rerunning step 2 (e.g. with a higher epoch cap, max_epochs) never retrains or reruns the CNN."""
        choices = read_choices(out)
        delta, delta_name = choices["delta"], choices["delta_name"]
        keys, data = load(stocks)
        day, ret = keys["day"].to_numpy(), keys["ret_next"].to_numpy()
        balance = Balance(out / "balance.parquet")
        price_p = {**PRICE, **({"max_epochs": QUICK["price_epochs"]} if quick else {})}
        comb_p = {**COMBINED_FIT, **({"max_epochs": QUICK["combined_epochs"]} if quick else {}),
                  **({"max_epochs": max_epochs} if max_epochs else {})}
        for i, fold in enumerate(fold_list(keys, n_folds)):
            if folds_done([out / f"comb_s2_{s}" for s in STEP2], fold.name):
                continue
            start = time.perf_counter()
            ix = fold_indices(keys, fold, TARGET, HOLD_OUT_MONTHS)
            y = scaled_target(keys, ix, data.device)
            ckpt = out / "checkpoints" / fold.name
            torch.manual_seed(0)
            net = TwoBranchNet(channels).to(DEVICE)
            net.tabular.load_state_dict(torch.load(ckpt / f"tab_delta_{delta_name}.pt"))
            if (ckpt / "price.pt").exists():  # step 1a done earlier: reuse it, never retrain the CNN
                saved = torch.load(ckpt / "price.pt")
                net.price.load_state_dict(saved["state"])
                info = {"epochs": saved["epochs"], "score": saved["score"]}
            else:
                hold_ic = lambda pred: float(daily_rank_ic(day[ix["hold_ic"]], pred, ret[ix["hold_ic"]]).mean())
                info = train_price(net, data, y, ix["train_price"], ix["hold_ic"], hold_ic, price_p, delta,
                                   on_epoch=lambda e: balance.add({"fold": fold.name, "step": "1a", **e}))
                torch.save({"state": net.price.state_dict(), **info}, ckpt / "price.pt")
                if i == 0 and learns_nothing(info["score"]):
                    print("two-branch WARNING SIGN 1: the chart learns nothing alone (best hold-out IC "
                          f"{info['score']:.4f}); stop and look", flush=True)
            if (ckpt / "summaries.pt").exists():  # the frozen branches' outputs, computed once per fold
                saved = torch.load(ckpt / "summaries.pt")
                hold_out, test_out = ({k: v.to(data.device) for k, v in saved[part].items()} for part in ("hold", "test"))
            else:
                hold_out, test_out = outputs(net, data, ix["hold"]), outputs(net, data, ix["test"])
                torch.save({part: {k: v.cpu() for k, v in found.items()}
                            for part, found in (("hold", hold_out), ("test", test_out))}, ckpt / "summaries.pt")
            for name, head in (("price_s1", "price"), ("tab_s1", "tab")):
                if not fold_done(out / name, fold.name):
                    write_fold(out / name, fold.name,
                               scored_frame(keys, ix["test"], test_out[head], ix["scale"], fold.name, TARGET))
            at = lambda rows: torch.as_tensor(np.searchsorted(ix["hold"], rows), device=data.device)
            fit_at, stop_at = at(ix["hold_fit"]), at(ix["hold_stop"])
            sample = torch.as_tensor(np.sort(np.random.default_rng(0).choice(
                len(ix["hold"]), min(CORR_SAMPLE, len(ix["hold"])), replace=False)), device=data.device)
            y_hold = y[torch.as_tensor(ix["hold"], device=data.device)]
            fit = (hold_out["s_p"][fit_at], hold_out["s_t"][fit_at], y_hold[fit_at])
            stop = (hold_out["s_p"][stop_at], hold_out["s_t"][stop_at], y_hold[stop_at])
            for name, setting in STEP2.items():
                if fold_done(out / f"comb_s2_{name}", fold.name):
                    continue
                torch.manual_seed(0)
                net.combined = CombinedMLP(COMBINED[setting["hidden"]]).to(DEVICE)
                xavier(net.combined)
                info2 = train_combined(net, fit, stop, {**comb_p, "learning_rate": setting["learning_rate"]}, delta,
                                       on_epoch=lambda e: balance.add({"fold": fold.name, "step": "2",
                                                                       "setting": name, **e}))
                torch.save({"state": net.combined.state_dict(), **info2, "max_epochs": comb_p["max_epochs"]},
                           ckpt / f"comb_{name}.pt")
                net.combined.eval()
                with torch.no_grad():
                    comb_test = net.combined(test_out["s_p"], test_out["s_t"])
                    comb_sample = net.combined(hold_out["s_p"][sample], hold_out["s_t"][sample])
                end = {"corr_price": spearman(comb_sample, hold_out["price"][sample]),
                       "corr_tab": spearman(comb_sample, hold_out["tab"][sample]), **weight_share(net.combined)}
                balance.add({"fold": fold.name, "step": "2 end", "setting": name, **info2, **end})
                if i == 0 and ignores_chart(end["corr_price"], end["weight_share_price"]):
                    print(f"two-branch WARNING SIGN 2 ({name}): the combination ignores the chart; stop and look",
                          flush=True)
                write_fold(out / f"comb_s2_{name}", fold.name,
                           scored_frame(keys, ix["test"], comb_test, ix["scale"], fold.name, TARGET))
            print(f"two-branch {fold.name} steps 1a and 2: {(time.perf_counter() - start) / 60:.0f} min, "
                  f"price epoch {info['epochs']} (hold-out IC {info['score']:.4f})", flush=True)
        return {name: collect(out / name, TARGET)["ic_mean"]
                for name in ["price_s1", "tab_s1", *(f"comb_s2_{s}" for s in STEP2)]}


    def run_choose(out):
        """The step 2 setting with the highest oos_1b/ic_mean; notes a setting whose best epoch reached the cap."""
        same_folds([out / f"comb_s2_{name}" for name in STEP2])
        table = {name: collect(out / f"comb_s2_{name}", TARGET)["ic_mean"] for name in STEP2}
        chosen = max(table, key=table.get)
        saved = {f.name: torch.load(f / f"comb_{chosen}.pt") for f in sorted((out / "checkpoints").iterdir())
                 if (f / f"comb_{chosen}.pt").exists()}
        capped = [fold for fold, s in saved.items() if s["epochs"] >= s["max_epochs"]]
        write_choices(out, step2=chosen, step2_ic=table)
        spread = max(table.values()) - min(table.values())
        return {"oos_1b/ic_mean by setting": table, "chosen": chosen, "spread": spread,
                "spread under 0.003 (noise)": spread < 0.003,
                "folds whose best epoch hit the cap (delete comb_s2_*/folds, rerun stage12 --max-epochs)": capped}


    def run_stage3(out, source=None, n_folds=None, stocks=None, channels=4, lam=0.5, quick=False):
        """Fold by fold, step 3 from the checkpoints in `source` (out by default; the λ = 0 control writes elsewhere);
        prints warning sign 3 on the first fold."""
        source = source or out
        choices = read_choices(source)
        delta, setting = choices["delta"], choices["step2"]
        keys, data = load(stocks)
        balance = Balance(out / "balance.parquet")
        p = {**FINETUNE, **({"epochs": QUICK["finetune_epochs"]} if quick else {}),
             "learning_rate": {"one_d": PRICE["learning_rate"] / 10, "price_head": PRICE["learning_rate"] / 10,
                               "tab_body": TABULAR["learning_rate"] / 10, "tab_head": TABULAR["learning_rate"] / 10,
                               "combined": STEP2[setting]["learning_rate"] / 10}}
        for i, fold in enumerate(fold_list(keys, n_folds)):
            if folds_done([out / "comb_s3", out / "price_s3", out / "tab_s3"], fold.name):
                continue
            start = time.perf_counter()
            ix = fold_indices(keys, fold, TARGET, HOLD_OUT_MONTHS)
            y = scaled_target(keys, ix, data.device)
            ckpt = source / "checkpoints" / fold.name
            net = TwoBranchNet(channels, COMBINED[STEP2[setting]["hidden"]]).to(DEVICE)
            net.price.load_state_dict(torch.load(ckpt / "price.pt")["state"])
            net.tabular.load_state_dict(torch.load(ckpt / f"tab_delta_{choices['delta_name']}.pt"))
            net.combined.load_state_dict(torch.load(ckpt / f"comb_{setting}.pt")["state"])
            share_2 = weight_share(net.combined)["weight_share_price"]
            bn = net.combined.layers[0]
            ref_mean, ref_sd = bn.running_mean.clone(), torch.sqrt(bn.running_var + bn.eps)
            sample = np.sort(np.random.default_rng(0).choice(ix["hold"], min(CORR_SAMPLE, len(ix["hold"])),
                                                             replace=False))
            warned = []

            def on_log(e):
                balance.add({"fold": fold.name, "step": "3", "lam": lam, **e})
                if i == 0 and not warned and pushes_chart_out(e["weight_share_price"], share_2):
                    warned.append(True)
                    print("two-branch WARNING SIGN 3: fine-tuning halves the chart's weight share "
                          f"({share_2:.3f} -> {e['weight_share_price']:.3f}); stop and look", flush=True)

            def on_epoch(e):
                s = outputs(net, data, sample)
                balance.add({"fold": fold.name, "step": "3 epoch end", "lam": lam, **e,
                             "corr_price": spearman(s["comb"], s["price"]), "corr_tab": spearman(s["comb"], s["tab"]),
                             **summary_drift(s["s_p"], s["s_t"], ref_mean, ref_sd), **weight_share(net.combined)})

            finetune(net, data, y, ix["hold"], p, delta, lam, on_log=on_log, on_epoch=on_epoch)
            (out / "checkpoints" / fold.name).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), out / "checkpoints" / fold.name / "stage3.pt")
            test_out = outputs(net, data, ix["test"])
            for name, head in (("price_s3", "price"), ("tab_s3", "tab"), ("comb_s3", "comb")):  # comb_s3 last
                write_fold(out / name, fold.name,
                           scored_frame(keys, ix["test"], test_out[head], ix["scale"], fold.name, TARGET))
            print(f"two-branch {fold.name} step 3: {(time.perf_counter() - start) / 60:.0f} min", flush=True)
        return {name: collect(out / name, TARGET)["ic_mean"] for name in ("comb_s3", "price_s3", "tab_s3")}


    def run_speed(stocks=None, channels=4, n_batches=20):
        """Images a second through a step 1a training batch and a step 3 one, and the hours that implies."""
        keys, data = load(stocks)
        net = TwoBranchNet(channels).to(DEVICE)
        y = torch.zeros(len(keys), device=data.device)
        idx = torch.as_tensor(np.random.default_rng(0).choice(len(keys), 512 * (n_batches + 3)), device=data.device)
        rates = {}
        for step in ("1a", "3"):
            opt = torch.optim.AdamW([w for ws in prepare(net, step).values() for w in ws], lr=1e-5)
            for warm in (True, False):
                torch.mps.synchronize()
                start = time.perf_counter()
                for i in range(3 if warm else n_batches):
                    b = idx[i * 512:(i + 1) * 512]
                    img, tab = data.draw(b)
                    batch_loss = (loss(net.price(img)[1], y[b], 1.0) if step == "1a"
                                  else step3_loss(net(img, tab), y[b], 1.0))
                    opt.zero_grad()
                    batch_loss.backward()
                    opt.step()
                torch.mps.synchronize()
            rates[step] = n_batches * 512 / (time.perf_counter() - start)
        sizes = [fold_indices(keys, f, TARGET, HOLD_OUT_MONTHS) for f in fold_list(keys, None)]
        epochs_1a = sum(len(ix["train_price"]) for ix in sizes) / rates["1a"] / 3600
        hours_3 = sum(len(ix["hold"]) for ix in sizes) * FINETUNE["epochs"] / rates["3"] / 3600
        return {"images_a_second_step_1a": round(rates["1a"]), "images_a_second_step_3": round(rates["3"]),
                "hours_for_one_epoch_of_step_1a_over_5_folds": round(epochs_1a, 1),
                "hours_for_step_3_over_5_folds": round(hours_3, 1)}

    return run_choose, run_delta, run_speed, run_stage12, run_stage3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it (from the terminal)

    ```
    uv run python scripts/models/10_two_branch_walkforward.py speed                         # images a second, hours
    uv run python scripts/models/10_two_branch_walkforward.py delta                         # step 1b, chooses δ
    uv run python scripts/models/10_two_branch_walkforward.py stage12                       # steps 1a and 2
    uv run python scripts/models/10_two_branch_walkforward.py choose                        # step 2's setting
    uv run python scripts/models/10_two_branch_walkforward.py stage3                        # step 3
    uv run python scripts/models/10_two_branch_walkforward.py stage3 --lam 0 \
        --source data/walkforward/two_branch --out data/walkforward/two_branch_lambda0      # the control
    ```

    A quick check end to end: `--stocks 300 --folds 1 --quick --out /tmp/two_branch_check` on every command. The backup (`explanations/CNN/fusion.md`, backup): `--channels 8 --out data/walkforward/two_branch_c8` on every command.
    """)
    return


@app.cell
def _(OUT, Path, argparse, mo, run_choose, run_delta, run_speed, run_stage12, run_stage3, sys):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="10_two_branch_walkforward.py")
        sub = parser.add_subparsers(dest="command", required=True)
        commands = {name: sub.add_parser(name) for name in ("speed", "delta", "stage12", "choose", "stage3")}
        for name, c in commands.items():
            c.add_argument("--out", type=Path, default=OUT)
            if name != "choose":
                c.add_argument("--stocks", type=int, help="only N PERMNOs, spread over them, as a quick check")
            if name in ("delta", "stage12", "stage3"):
                c.add_argument("--folds", type=int, help="only the first N folds")
                c.add_argument("--quick", action="store_true", help="one or two epochs a step, as a quick check")
            if name in ("speed", "stage12", "stage3"):
                c.add_argument("--channels", type=int, default=4, help="the 1D block's channels (8: the backup)")
        commands["stage12"].add_argument("--max-epochs", type=int,
                                         help="step 2's epoch cap (default 40); the saved price branches are reused")
        commands["stage3"].add_argument("--lam", type=float, default=0.5, help="λ; 0 is the control")
        commands["stage3"].add_argument("--source", type=Path, help="where the checkpoints are (default: --out)")
        args = parser.parse_args(argv)
        if args.command == "speed":
            result = run_speed(args.stocks, args.channels)
        elif args.command == "delta":
            result = run_delta(args.out, args.folds, args.stocks, args.quick)
        elif args.command == "stage12":
            result = run_stage12(args.out, args.folds, args.stocks, args.channels, args.quick, args.max_epochs)
        elif args.command == "choose":
            result = run_choose(args.out)
        else:
            result = run_stage3(args.out, args.source, args.folds, args.stocks, args.channels, args.lam, args.quick)
        print(result, flush=True)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


if __name__ == "__main__":
    app.run()
