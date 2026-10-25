import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Walk-forward of the two-branch model

    The model of `explanations/CNN/fusion.md`: the one-day grey chart through the one-day CNN as it ran alone (`--chart day_cnn`; `hours` and `day` are the earlier designs, the 21-day chart and the one-day chart with the 1D block), the 17 tabular features through the tuned MLP's layer, and a combined MLP on the CNN's prediction and the tabular summary, each normalized as a branch (`dsr.two_branch`, trained by `dsr.two_branch_train`). Every loss is squared error. Yearly folds over 2019-2023, each with the year before its test year held out for stopping only, run in the spec's order:

    1. `stage12`: fold by fold, step 1b (the tabular branch alone), step 1a (the CNN alone, on the large caps) and step 2 (the combined MLP on the frozen branches, fit on the training years' large caps, stopped on the hold-out year) with its 6 settings;
    2. `choose`: the step 2 setting with the highest `oos_1b/ic_mean`;
    3. `stage3`: fold by fold, block 3, the readout, the tabular layer and the combined MLP trained together on the training years' large caps, stopped on the hold-out year, once (`--lam 0 --out .../two_branch_day_cnn_lambda0` for the control); alongside, success rule 1's comparison, the tabular branch through the same step alone.

    Each scored output gets its own folder under `data/walkforward/two_branch_day_cnn/` (`predictions.parquet` and `summary.json`, as the other walk-forwards), the balance diagnostics go to `balance.parquet` and each fold's weights to `checkpoints/`. A fold already written is skipped, so a stopped command resumes. **The test years (2024-2025) are not read.**

    4. `joint` (a last check, 2026-10-25), on the training years' large caps as step 3. First a two-layer tabular MLP (17 -> 16 -> 8 -> 1) alone on its own loss (`tab_alone`, also the control), stopped on the hold-out year; then, with no step 2, the fine-tuning: the CNN from its step 1a weights, the MLP's 17 -> 16 layer frozen, its 16 -> 8 layer and a new head on the CNN's prediction and the MLP's 8 numbers (9 -> 8 -> 1) trained together. The CNN learns from λ L_price + (1 - λ) L_comb, the MLP's 16 -> 8 and the head from L_comb alone (`dsr.two_branch.joint_loss`); λ 0.25, 0.5 and 0.75; stopped on the hold-out year's large caps. Output in `data/walkforward/two_branch_joint/`.
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

    from dsr.cnn import GreyImages, HourWindows, Images, xavier
    from dsr.cnn import fit as fit_chart_cnn
    from dsr.evaluation import daily_rank_ic
    from dsr.two_branch import (COMBINED, JOINT_COMBINED, JOINT_TABULAR, WIDTH, CombinedMLP, TwoBranchNet,
                                copy_chart_cnn, loss, norm_reference, prepare, step3_loss, summary_drift, weight_share)
    from dsr.two_branch_train import (Balance, JointData, TwoBranchData, collect, finetune, fold_done, fold_indices,
                                      folds_done,
                                      ignores_chart, learns_nothing, min_weight_share, outputs, pushes_chart_out,
                                      same_folds,
                                      scored_frame, tabular_predictions, train_combined, train_joint, train_price,
                                      train_tabular, write_fold)
    from dsr.walkforward import crossfit_folds, feature_columns, folds

    return (
        Balance, COMBINED, CombinedMLP, GreyImages, HourWindows, Images, JOINT_COMBINED, JOINT_TABULAR, JointData, Path,
        TwoBranchData, TwoBranchNet, WIDTH,
        argparse, collect, copy_chart_cnn, crossfit_folds, daily_rank_ic, fit_chart_cnn, feature_columns, finetune, fold_done,
        fold_indices, folds, folds_done, ignores_chart, json, min_weight_share,
        learns_nothing, loss, mo, norm_reference, np, outputs, pd, prepare, pushes_chart_out, same_folds, scored_frame,
        step3_loss, summary_drift, sys, tabular_predictions, time, torch, train_combined, train_joint, train_price,
        train_tabular, weight_share, write_fold, xavier,
    )


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    HOURS = ROOT / "data" / "processed" / "cnn_hours_train.npz"
    DAY_BARS = ROOT / "data" / "processed" / "cnn_bars_train.npz"  # the one-day charts' drawing data (large caps)
    PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
    OUT = ROOT / "data" / "walkforward" / "two_branch"
    OUT_DAY = ROOT / "data" / "walkforward" / "two_branch_day"  # --chart day
    OUT_DAY_CNN = ROOT / "data" / "walkforward" / "two_branch_day_cnn"  # --chart day_cnn
    OUT_JOINT = ROOT / "data" / "walkforward" / "two_branch_joint"  # joint
    CHART_CNN = {"filters": [32, 64, 128], "dropout": 0.5, "batch_size": 1024, "max_epochs": 10, "patience": 2,
                 "learning_rate": 1e-4, "weight_decay": 0.05, "norm_sample": 20_000}  # --chart day_cnn's step 1a: 06's
    TARGET = "target_demeaned"
    FIRST, EVERY, HOLD_OUT_MONTHS = "2019-01-01", "year", 12
    CROSSFIT_YEARS = (2014, 2015, 2016, 2017, 2018)  # --crossfit: each year predicted from the other four
    EMBARGO_DAYS = 20  # --crossfit: trading days after the year not trained on (the features look back 20 days)
    DEVICE = "mps"
    PRICE = {"learning_rate": 1e-4, "weight_decay": 0.05, "batch_size": 512, "max_epochs": 5, "patience": 2}
    TABULAR = {"learning_rate": 2.9e-4, "weight_decay": 0.035, "batch_size": 4096, "epochs": 12}
    STEP2 = {f"lr{lr:g}_h{h}": {"learning_rate": lr, "hidden": h} for lr in (1e-4, 3e-4, 1e-3) for h in ("16", "16-8")}
    COMBINED_FIT = {"weight_decay": 0.035, "batch_size": 4096, "max_epochs": 40, "patience": 5}
    FINETUNE = {"max_epochs": 10, "patience": 2, "batch_size": 512,  # stopped on the hold-out year's combined loss
                "weight_decay": {"one_d": 0.05, "price_head": 0.05, "tab_body": 0.035, "tab_head": 0.035,
                                 "combined": 0.035}}
    JOINT = {"max_epochs": 10, "patience": 2, "batch_size": 512,  # step 3's
             "learning_rate": {"one_d": PRICE["learning_rate"] / 10, "price_head": PRICE["learning_rate"] / 10,
                               "tab_deep": TABULAR["learning_rate"] / 10,  # the trained parts as in step 3
                               "combined": 3e-4},  # the new head at full rate (step 2's)
             "weight_decay": {"one_d": 0.05, "price_head": 0.05, "tab_deep": 0.035, "combined": 0.035}}
    JOINT_LAMS = (0.25, 0.5, 0.75)
    QUICK = {"price_epochs": 1, "tabular_epochs": 1, "combined_epochs": 2, "finetune_epochs": 1}  # --quick
    CORR_SAMPLE = 20_000  # the fixed hold-out sample for the correlations of the model with each head
    return (CHART_CNN, COMBINED_FIT, CORR_SAMPLE, CROSSFIT_YEARS, DAY_BARS, DEVICE, EMBARGO_DAYS, EVERY, FINETUNE,
            FIRST, HOLD_OUT_MONTHS, HOURS,
            JOINT, JOINT_LAMS, OUT, OUT_DAY, OUT_DAY_CNN, OUT_JOINT, PANEL, PRICE, QUICK, STEP2, TABULAR, TARGET)


@app.cell
def _(DAY_BARS, DEVICE, HOURS, HourWindows, Images, JointData, PANEL, TARGET, TwoBranchData, feature_columns, np,
      pd):
    def load(stocks=None, chart="hours"):
        """The stock-days and their data on the GPU for the chart: "hours" (load_hours) or "day" (load_day)."""
        return load_hours(stocks) if chart == "hours" else load_day(stocks)


    def load_day(stocks=None):
        """Every large-cap stock-day with one-day drawing data and a model-panel row (all of them, or `stocks` PERMNOs
        spread over them, as a quick check): their keys, target and return, and the data on the GPU (the one-day charts
        drawn from the drawing data, the 17 features)."""
        with np.load(DAY_BARS) as z:
            keys = pd.DataFrame({"day": pd.to_datetime(z["day"]), "permno": z["permno"]})
            arrays = {k: z[k] for k in ("rows", "volume", "colors")}
        if stocks:  # spread over the PERMNOs
            permnos = np.unique(keys["permno"])
            keys = keys[keys["permno"].isin(permnos[::max(1, len(permnos) // stocks)][:stocks])]
        panel = pd.read_parquet(PANEL)
        features = feature_columns(panel.columns)
        panel["day"] = pd.to_datetime(panel["day"])
        keys = keys.reset_index(names="pos").merge(panel, on=["day", "permno"], how="inner", validate="one_to_one")
        keys = keys[keys["large_cap"]].sort_values(["day", "permno"]).reset_index(drop=True)
        keys["day_pos"] = np.searchsorted(np.unique(panel["day"]), keys["day"])
        images = Images(*(arrays[k][keys["pos"].to_numpy()] for k in ("rows", "volume", "colors")), DEVICE)
        data = TwoBranchData(images, keys[features].to_numpy(np.float32), DEVICE)
        return keys[["day", "permno", "day_pos", "large_cap", "ret_next", TARGET]], data


    def load_joint(stocks=None):
        """Every model-panel stock-day (the ≥ $100M universe; all of them, or `stocks` PERMNOs spread over them, as a
        quick check) and the one-day charts of its large caps: their keys, target and return, and the data on the GPU
        (every stock-day's features, the large caps' charts). A large cap without a chart is left out, as load_day
        leaves it out, so the large caps are the stock-days with a chart."""
        panel = pd.read_parquet(PANEL)
        features = feature_columns(panel.columns)
        panel["day"] = pd.to_datetime(panel["day"])
        if stocks:  # spread over the PERMNOs
            permnos = np.unique(panel["permno"])
            panel = panel[panel["permno"].isin(permnos[::max(1, len(permnos) // stocks)][:stocks])]
        with np.load(DAY_BARS) as z:
            charts = pd.DataFrame({"day": pd.to_datetime(z["day"]), "permno": z["permno"]}).reset_index(names="pos")
            arrays = {k: z[k] for k in ("rows", "volume", "colors")}
        keys = panel.merge(charts, on=["day", "permno"], how="left", validate="one_to_one")
        missing = keys["large_cap"] & keys["pos"].isna()
        print(f"two-branch joint: {int(missing.sum()):,} large-cap stock-days without a chart left out", flush=True)
        keys = keys[~missing].sort_values(["day", "permno"]).reset_index(drop=True)
        keys["day_pos"] = np.searchsorted(np.unique(keys["day"]), keys["day"])
        large = keys["large_cap"].to_numpy(bool)
        chart_row = np.where(large, np.cumsum(large) - 1, -1)
        pos = keys.loc[large, "pos"].to_numpy(np.int64)
        images = Images(*(arrays[k][pos] for k in ("rows", "volume", "colors")), DEVICE)
        data = JointData(images, chart_row, keys[features].to_numpy(np.float32), DEVICE)
        return keys[["day", "permno", "day_pos", "large_cap", "ret_next", TARGET]], data


    def load_hours(stocks=None):
        """Every panel stock-day with hourly data (all of them, or `stocks` PERMNOs spread over them, as a quick check):
        their keys, target and return, and the data on the GPU (the charts drawn from the hourly data, the 17
        features)."""
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

    return load, load_joint


@app.cell
def _(
    CHART_CNN, COMBINED, COMBINED_FIT, CORR_SAMPLE, CROSSFIT_YEARS, CombinedMLP, DEVICE, EMBARGO_DAYS, EVERY,
    FINETUNE, FIRST, GreyImages, crossfit_folds,
    HOLD_OUT_MONTHS, JOINT, JOINT_COMBINED, JOINT_LAMS, JOINT_TABULAR, OUT_DAY_CNN, PRICE, QUICK, copy_chart_cnn,
    fit_chart_cnn,
    STEP2, TABULAR, TARGET, WIDTH, Balance, TwoBranchNet, collect, daily_rank_ic, finetune, fold_done, fold_indices,
    folds, folds_done, ignores_chart, json, learns_nothing, load, load_joint, loss, min_weight_share, norm_reference,
    np, outputs, pd, prepare, pushes_chart_out, same_folds, scored_frame, step3_loss, summary_drift,
    tabular_predictions, time, torch,
    train_combined, train_joint, train_price, train_tabular, weight_share, write_fold, xavier,
):
    def spearman(a, b):
        return float(pd.Series(a.cpu().numpy()).corr(pd.Series(b.cpu().numpy()), method="spearman"))


    def read_choices(out):
        path = out / "choices.json"
        return json.loads(path.read_text()) if path.exists() else {}


    def write_choices(out, **found):
        out.mkdir(parents=True, exist_ok=True)
        (out / "choices.json").write_text(json.dumps({**read_choices(out), **found}, indent=2))


    def fold_list(keys, n_folds, crossfit=False):
        """The yearly walk-forward folds from FIRST, or with `crossfit` each of CROSSFIT_YEARS predicted from the other
        four (the out-of-sample predictions of the years before the first fold)."""
        days = np.unique(keys["day"].to_numpy())
        found = (crossfit_folds(days, CROSSFIT_YEARS, EMBARGO_DAYS) if crossfit
                 else folds(days, first=FIRST, every=EVERY))
        return found[:n_folds]


    def scaled_target(keys, ix, device):
        y = keys[TARGET].to_numpy(np.float32) / ix["scale"]
        return torch.as_tensor(np.nan_to_num(y), device=device)


    def run_stage12(out, n_folds=None, stocks=None, width=WIDTH, quick=False, max_epochs=None, chart="hours",
                    crossfit=False, step2=None):
        """Fold by fold: step 1b, step 1a, then step 2 with every setting not yet written; prints warning signs 1 and 2
        on the first fold. A fold's trained branches (tab.pt, price.pt) and the frozen branches' outputs (summaries.pt)
        are saved and reused, so rerunning step 2 (e.g. with a higher epoch cap, max_epochs) never retrains them.
        `crossfit`: the cross-fitted folds (fold_list). `step2`: only that step 2 setting, written to choices.json as
        the one step 3 uses (the cross-fit reuses the validation run's choice instead of choosing again)."""
        settings = {step2: STEP2[step2]} if step2 else STEP2
        if step2:
            write_choices(out, step2=step2)
        keys, data = load(stocks, chart)
        day, ret = keys["day"].to_numpy(), keys["ret_next"].to_numpy()
        balance = Balance(out / "balance.parquet")
        price_p = {**PRICE, **({"max_epochs": QUICK["price_epochs"]} if quick else {})}
        tab_p = {**TABULAR, **({"epochs": QUICK["tabular_epochs"]} if quick else {})}
        comb_p = {**COMBINED_FIT, **({"max_epochs": QUICK["combined_epochs"]} if quick else {}),
                  **({"max_epochs": max_epochs} if max_epochs else {})}
        for i, fold in enumerate(fold_list(keys, n_folds, crossfit)):
            if folds_done([out / f"comb_s2_{s}" for s in settings], fold.name):
                continue
            start = time.perf_counter()
            ix = fold_indices(keys, fold, TARGET, HOLD_OUT_MONTHS)
            y = scaled_target(keys, ix, data.device)
            ckpt = out / "checkpoints" / fold.name
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(0)
            net = TwoBranchNet(width, chart=chart).to(DEVICE)
            if (ckpt / "tab.pt").exists():  # step 1b done earlier: reuse it
                net.tabular.load_state_dict(torch.load(ckpt / "tab.pt"))
            else:
                train_tabular(net, data.features, y, ix["train"], tab_p,
                              on_epoch=lambda e: balance.add({"fold": fold.name, "step": "1b", **e}))
                torch.save(net.tabular.state_dict(), ckpt / "tab.pt")
            if (ckpt / "price.pt").exists():  # step 1a done earlier: reuse it, never retrain the CNN
                saved = torch.load(ckpt / "price.pt")
                net.price.load_state_dict(saved["state"])
                info = {"epochs": saved["epochs"], "score": saved["score"]}
            elif chart == "day_cnn":  # the one-day CNN as it ran alone (script 06): dsr.cnn.fit, stopped on the loss
                w = data.windows
                hold_ic = lambda pred: float(daily_rank_ic(day[ix["hold"]], pred, ret[ix["hold"]]).mean())
                cnn, _, chosen = fit_chart_cnn(
                    GreyImages(w.rows, w.volume, w.colors, DEVICE), keys[TARGET].to_numpy(np.float32),
                    ix["train_price"], ix["hold"], {**CHART_CNN, **({"max_epochs": 1} if quick else {})}, seed=0,
                    on_epoch=lambda e: balance.add({"fold": fold.name, "step": "1a", "epoch": e["epoch"],
                                                    "train_loss": e["train_loss"], "holdout_loss": e["holdout_loss"],
                                                    "holdout_score": hold_ic(e["holdout_pred"])}))
                copy_chart_cnn(net.price, cnn)
                kept = outputs(net, data, ix["hold"])["price"].cpu().numpy()
                info = {"epochs": chosen["epochs"], "score": hold_ic(kept)}  # the kept epoch's hold-out IC
                torch.save({"state": net.price.state_dict(), **info}, ckpt / "price.pt")
            else:
                hold_ic = lambda pred: float(daily_rank_ic(day[ix["hold_ic"]], pred, ret[ix["hold_ic"]]).mean())
                info = train_price(net, data, y, ix["train_price"], ix["hold_ic"], hold_ic, price_p,
                                   on_epoch=lambda e: balance.add({"fold": fold.name, "step": "1a", **e}))
                torch.save({"state": net.price.state_dict(), **info}, ckpt / "price.pt")
            if i == 0 and learns_nothing(info["score"]):
                print("two-branch WARNING SIGN 1: the chart learns nothing alone (best hold-out IC "
                      f"{info['score']:.4f}); stop and look", flush=True)
            if (ckpt / "summaries.pt").exists():  # the frozen branches' outputs, computed once per fold
                saved = torch.load(ckpt / "summaries.pt")
                train_out, hold_out, test_out = ({k: v.to(data.device) for k, v in saved[part].items()}
                                                 for part in ("train", "hold", "test"))
            else:  # the training years' large caps (step 2 fits on their summaries), the hold-out year, the test year
                train_out = {k: v for k, v in outputs(net, data, ix["train_comb"]).items() if k != "comb"}
                hold_out, test_out = outputs(net, data, ix["hold"]), outputs(net, data, ix["test"])
                torch.save({part: {k: v.cpu() for k, v in found.items()}
                            for part, found in (("train", train_out), ("hold", hold_out), ("test", test_out))},
                           ckpt / "summaries.pt")
            # each head's IC where step 2 fits (in sample for the branches) and where it stops
            ic = lambda rows, pred: float(daily_rank_ic(day[rows], pred.cpu().numpy(), ret[rows]).mean())
            balance.add({"fold": fold.name, "step": "summaries",
                         **{f"ic_train_{h}": ic(ix["train_comb"], train_out[h]) for h in ("price", "tab")},
                         **{f"ic_hold_{h}": ic(ix["hold"], hold_out[h]) for h in ("price", "tab")}})
            for name, head in (("price_s1", "price"), ("tab_s1", "tab")):
                if not fold_done(out / name, fold.name):
                    write_fold(out / name, fold.name,
                               scored_frame(keys, ix["test"], test_out[head], ix["scale"], fold.name, TARGET))
            sample = torch.as_tensor(np.sort(np.random.default_rng(0).choice(
                len(ix["hold"]), min(CORR_SAMPLE, len(ix["hold"])), replace=False)), device=data.device)
            at = lambda rows: y[torch.as_tensor(rows, device=data.device)]
            fit = (train_out["s_p"], train_out["s_t"], at(ix["train_comb"]))
            stop = (hold_out["s_p"], hold_out["s_t"], at(ix["hold"]))
            for name, setting in settings.items():
                if fold_done(out / f"comb_s2_{name}", fold.name):
                    continue
                torch.manual_seed(0)
                net.combined = CombinedMLP(COMBINED[setting["hidden"]], train_out["s_p"].shape[1]).to(DEVICE)
                xavier(net.combined)
                info2 = train_combined(net, fit, stop, {**comb_p, "learning_rate": setting["learning_rate"]},
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
                if i == 0 and ignores_chart(end["corr_price"], end["weight_share_price"],
                                            min_weight_share(train_out["s_p"].shape[1])):
                    print(f"two-branch WARNING SIGN 2 ({name}): the combination ignores the chart; stop and look",
                          flush=True)
                write_fold(out / f"comb_s2_{name}", fold.name,
                           scored_frame(keys, ix["test"], comb_test, ix["scale"], fold.name, TARGET))
            print(f"two-branch {fold.name} steps 1a and 2: {(time.perf_counter() - start) / 60:.0f} min, "
                  f"price epoch {info['epochs']} (hold-out IC {info['score']:.4f})", flush=True)
        return {name: collect(out / name, TARGET)["ic_mean"]
                for name in ["price_s1", "tab_s1", *(f"comb_s2_{s}" for s in settings)]}


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


    def run_stage3(out, source=None, n_folds=None, stocks=None, width=WIDTH, lam=0.5, quick=False, chart="hours",
                   crossfit=False):
        """Fold by fold, step 3 from the checkpoints in `source` (out by default; the λ = 0 control writes elsewhere);
        prints warning sign 3 on the first fold. Also the tabular branch fine-tuned alone on the hold-out year with its
        own loss and step 3's settings (tab_ref_s3): success rule 1's comparison, since step 3 trains the tabular branch
        through the combined loss only."""
        source = source or out
        setting = read_choices(source)["step2"]
        keys, data = load(stocks, chart)
        balance = Balance(out / "balance.parquet")
        p = {**FINETUNE, **({"max_epochs": QUICK["finetune_epochs"]} if quick else {}),
             "learning_rate": {"one_d": PRICE["learning_rate"] / 10, "price_head": PRICE["learning_rate"] / 10,
                               "tab_body": TABULAR["learning_rate"] / 10, "tab_head": TABULAR["learning_rate"] / 10,
                               "combined": STEP2[setting]["learning_rate"] / 10}}
        for i, fold in enumerate(fold_list(keys, n_folds, crossfit)):
            if folds_done([out / "comb_s3", out / "price_s3", out / "tab_s3", out / "tab_ref_s3"], fold.name):
                continue
            start = time.perf_counter()
            ix = fold_indices(keys, fold, TARGET, HOLD_OUT_MONTHS)
            y = scaled_target(keys, ix, data.device)
            ckpt = source / "checkpoints" / fold.name
            torch.manual_seed(0)
            ref = TwoBranchNet(width, chart=chart).to(DEVICE)  # only its tabular branch is used
            ref.tabular.load_state_dict(torch.load(ckpt / "tab.pt"))
            train_tabular(ref, data.features, y, ix["train_comb"],
                          {"learning_rate": TABULAR["learning_rate"] / 10, "weight_decay": TABULAR["weight_decay"],
                           "batch_size": FINETUNE["batch_size"], "max_epochs": p["max_epochs"],
                           "patience": FINETUNE["patience"]}, stop=ix["hold"])
            write_fold(out / "tab_ref_s3", fold.name, scored_frame(keys, ix["test"], tabular_predictions(
                ref, data.features, ix["test"]), ix["scale"], fold.name, TARGET))
            net = TwoBranchNet(width, COMBINED[STEP2[setting]["hidden"]], chart=chart).to(DEVICE)
            net.price.load_state_dict(torch.load(ckpt / "price.pt")["state"])
            net.tabular.load_state_dict(torch.load(ckpt / "tab.pt"))
            net.combined.load_state_dict(torch.load(ckpt / f"comb_{setting}.pt")["state"])
            share_2 = weight_share(net.combined)["weight_share_price"]
            ref_mean, ref_scale = norm_reference(net.combined)
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
                             **summary_drift(s["s_p"], s["s_t"], ref_mean, ref_scale), **weight_share(net.combined)})

            info3 = finetune(net, data, y, ix["train_comb"], p, lam, on_log=on_log, on_epoch=on_epoch,
                             stop=ix["hold"])
            balance.add({"fold": fold.name, "step": "3 end", "lam": lam, **info3})
            (out / "checkpoints" / fold.name).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), out / "checkpoints" / fold.name / "stage3.pt")
            test_out = outputs(net, data, ix["test"])
            for name, head in (("price_s3", "price"), ("tab_s3", "tab"), ("comb_s3", "comb")):  # comb_s3 last
                write_fold(out / name, fold.name,
                           scored_frame(keys, ix["test"], test_out[head], ix["scale"], fold.name, TARGET))
            print(f"two-branch {fold.name} step 3: {(time.perf_counter() - start) / 60:.0f} min", flush=True)
        return {name: collect(out / name, TARGET)["ic_mean"]
                for name in ("comb_s3", "price_s3", "tab_s3", "tab_ref_s3")}


    def run_joint(out, lams=JOINT_LAMS, n_folds=None, stocks=None, quick=False):
        """Fold by fold, the joint run: first the two-layer tabular MLP alone on its own loss (tab_alone, also the
        control; its weights saved), then each λ: the CNN from its step 1a weights (OUT_DAY_CNN's checkpoints) and
        that MLP, its first layer frozen, with a new head, trained together (train_joint) on the training years' large
        caps, stopped on the hold-out year's, into out/lam<λ>/ (comb, price). Every λ starts from the same weights and
        the same shuffling. A fold already written is skipped."""
        keys, data = load_joint(stocks)
        day, ret = keys["day"].to_numpy(), keys["ret_next"].to_numpy()
        balance = Balance(out / "balance.parquet")
        p = {**JOINT, **({"max_epochs": QUICK["finetune_epochs"]} if quick else {})}
        tab_p = {"learning_rate": TABULAR["learning_rate"], "weight_decay": TABULAR["weight_decay"],
                 "batch_size": p["batch_size"], "max_epochs": p["max_epochs"], "patience": p["patience"]}

        def new_net():
            torch.manual_seed(0)
            return TwoBranchNet(hidden=JOINT_COMBINED, chart="day_cnn", tab_hidden=JOINT_TABULAR).to(DEVICE)

        for fold in fold_list(keys, n_folds):
            ix = fold_indices(keys, fold, TARGET, HOLD_OUT_MONTHS)
            y = scaled_target(keys, ix, data.device)
            hold_ic = lambda pred: float(daily_rank_ic(day[ix["hold"]], pred, ret[ix["hold"]]).mean())
            if not fold_done(out / "tab_alone", fold.name):
                start, net = time.perf_counter(), new_net()
                info = train_tabular(net, data.features, y, ix["train_comb"], tab_p, stop=ix["hold"],
                                     on_epoch=lambda e: balance.add({"fold": fold.name, "step": "tab alone", **e}))
                balance.add({"fold": fold.name, "step": "tab alone end", **info})
                (out / "tab_alone" / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(net.tabular.state_dict(), out / "tab_alone" / "checkpoints" / f"{fold.name}.pt")
                write_fold(out / "tab_alone", fold.name, scored_frame(
                    keys, ix["test"], tabular_predictions(net, data.features, ix["test"]), ix["scale"], fold.name,
                    TARGET))
                print(f"two-branch joint {fold.name} tab alone: {(time.perf_counter() - start) / 60:.0f} min",
                      flush=True)
            for lam in lams:
                folder = out / f"lam{lam:g}"
                if folds_done([folder / h for h in ("comb", "price")], fold.name):
                    continue
                start, net = time.perf_counter(), new_net()
                net.price.load_state_dict(torch.load(OUT_DAY_CNN / "checkpoints" / fold.name / "price.pt")["state"])
                net.tabular.load_state_dict(torch.load(out / "tab_alone" / "checkpoints" / f"{fold.name}.pt"))
                info = train_joint(net, data, y, ix["train_comb"], p, lam, stop=ix["hold"], score=hold_ic,
                                   on_epoch=lambda e: balance.add({"fold": fold.name, "step": "joint", "lam": lam, **e,
                                                                   **weight_share(net.combined)}))
                balance.add({"fold": fold.name, "step": "joint end", "lam": lam, **info})
                (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), folder / "checkpoints" / f"{fold.name}.pt")
                test_out = outputs(net, data, ix["test"])
                for head in ("price", "comb"):  # comb last
                    write_fold(folder / head, fold.name,
                               scored_frame(keys, ix["test"], test_out[head], ix["scale"], fold.name, TARGET))
                print(f"two-branch joint {fold.name} λ {lam:g}: {(time.perf_counter() - start) / 60:.0f} min",
                      flush=True)
        names = ["tab_alone", *(f"lam{lam:g}/{head}" for lam in lams for head in ("comb", "price"))]
        return {name: collect(out / name, TARGET)["ic_mean"] for name in names}


    def run_speed(stocks=None, width=WIDTH, n_batches=20, chart="hours"):
        """Images a second through a step 1a training batch, a step 3 one and prediction (step 2's summaries), and the
        hours that implies: an epoch of step 1a at one day in 9, 5, 3 or 1, the summaries and step 3, over the folds."""
        keys, data = load(stocks, chart)
        net = TwoBranchNet(width, chart=chart).to(DEVICE)
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
                    batch_loss = loss(net.price(img)[1], y[b]) if step == "1a" else step3_loss(net(img, tab), y[b])
                    opt.zero_grad()
                    batch_loss.backward()
                    opt.step()
                torch.mps.synchronize()
            rates[step] = n_batches * 512 / (time.perf_counter() - start)
        rows = idx.cpu().numpy()
        outputs(net, data, rows[:1024])  # warm up
        torch.mps.synchronize()
        start = time.perf_counter()
        outputs(net, data, rows[:512 * n_batches])
        torch.mps.synchronize()
        rates["predict"] = n_batches * 512 / (time.perf_counter() - start)
        sizes = [fold_indices(keys, f, TARGET, HOLD_OUT_MONTHS) for f in fold_list(keys, None)]
        train = sum(len(ix["train"]) for ix in sizes)
        summaries = sum(len(ix["train_comb"]) + len(ix["hold"]) + len(ix["test"]) for ix in sizes)
        hours_3 = sum(len(ix["train_comb"]) for ix in sizes) / rates["3"] / 3600  # one epoch
        return {"images_a_second": {k: round(v) for k, v in rates.items()},
                "hours_for_one_epoch_of_step_1a_over_5_folds": {
                    f"1 day in {k}": round(train / k / rates["1a"] / 3600, 1) for k in (9, 5, 3, 1)},
                "hours_for_step_2_summaries_over_5_folds": round(summaries / rates["predict"] / 3600, 1),
                "hours_for_one_epoch_of_step_3_over_5_folds": round(hours_3, 1)}

    return run_choose, run_joint, run_speed, run_stage12, run_stage3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running it (from the terminal)

    ```
    uv run python scripts/models/10_two_branch_walkforward.py speed                         # images a second, hours
    uv run python scripts/models/10_two_branch_walkforward.py stage12                       # steps 1b, 1a and 2
    uv run python scripts/models/10_two_branch_walkforward.py choose                        # step 2's setting
    uv run python scripts/models/10_two_branch_walkforward.py stage3                        # step 3
    uv run python scripts/models/10_two_branch_walkforward.py stage3 --lam 0 \
        --source data/walkforward/two_branch --out data/walkforward/two_branch_lambda0      # the control
    uv run python scripts/models/10_two_branch_walkforward.py joint                         # the joint run
    uv run python scripts/models/10_two_branch_walkforward.py stage12 --chart day_cnn --crossfit --step2 lr0.0003_h16-8 \
        --out data/walkforward/crossfit/two_branch_day_cnn                                  # 2014-2018, cross-fitted
    uv run python scripts/models/10_two_branch_walkforward.py stage3 --chart day_cnn --crossfit --lam 0.1 \
        --source data/walkforward/crossfit/two_branch_day_cnn --out data/walkforward/crossfit/two_branch_day_cnn_lam0.1
    ```

    A quick check end to end: `--stocks 300 --folds 1 --quick --out /tmp/two_branch_check` on every command (`joint`: `--lams 0.5`, one λ, works too). The backup (`explanations/CNN/fusion.md`, backup): `--width 32 --out data/walkforward/two_branch_w32` on every command. The one-day chart instead of the 21-day one: `--chart day` on every command (output in `data/walkforward/two_branch_day`).
    """)
    return


@app.cell
def _(JOINT_LAMS, OUT, OUT_DAY, OUT_DAY_CNN, OUT_JOINT, Path, STEP2, WIDTH, argparse, mo, run_choose, run_joint,
      run_speed, run_stage12, run_stage3, sys):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="10_two_branch_walkforward.py")
        sub = parser.add_subparsers(dest="command", required=True)
        commands = {name: sub.add_parser(name) for name in ("speed", "stage12", "choose", "stage3")}
        for name, c in commands.items():
            c.add_argument("--out", type=Path, help="where the outputs go (default: two_branch, _day or _day_cnn)")
            c.add_argument("--chart", choices=["hours", "day", "day_cnn"], default="hours",
                           help="the 21-day hourly chart (default), the one-day chart, or the one-day CNN itself")
            if name != "choose":
                c.add_argument("--stocks", type=int, help="only N PERMNOs, spread over them, as a quick check")
            if name in ("stage12", "stage3"):
                c.add_argument("--folds", type=int, help="only the first N folds")
                c.add_argument("--crossfit", action="store_true",
                               help="2014-2018, each year predicted from the other four (give --out)")
                c.add_argument("--quick", action="store_true", help="one or two epochs a step, as a quick check")
            if name in ("speed", "stage12", "stage3"):
                c.add_argument("--width", type=int, default=WIDTH, help="the 1D block's width (32: the backup)")
        commands["stage12"].add_argument("--max-epochs", type=int,
                                         help="step 2's epoch cap (default 40); the saved price branches are reused")
        commands["stage12"].add_argument("--step2", choices=list(STEP2),
                                         help="only this step 2 setting, recorded as the one step 3 uses")
        commands["stage3"].add_argument("--lam", type=float, default=0.5, help="λ; 0 is the control")
        commands["stage3"].add_argument("--source", type=Path, help="where the checkpoints are (default: --out)")
        j = sub.add_parser("joint")
        j.add_argument("--out", type=Path, default=OUT_JOINT, help="where the outputs go (default: two_branch_joint)")
        j.add_argument("--lams", type=float, nargs="+", default=list(JOINT_LAMS),
                       help="the λs (default 0.25 0.5 0.75)")
        j.add_argument("--stocks", type=int, help="only N PERMNOs, spread over them, as a quick check")
        j.add_argument("--folds", type=int, help="only the first N folds")
        j.add_argument("--quick", action="store_true", help="one epoch, as a quick check")
        args = parser.parse_args(argv)
        if args.command == "joint":
            print(run_joint(args.out, tuple(args.lams), args.folds, args.stocks, args.quick), flush=True)
            return
        args.out = args.out or {"hours": OUT, "day": OUT_DAY, "day_cnn": OUT_DAY_CNN}[args.chart]
        if args.command == "speed":
            result = run_speed(args.stocks, args.width, chart=args.chart)
        elif args.command == "stage12":
            result = run_stage12(args.out, args.folds, args.stocks, args.width, args.quick, args.max_epochs, args.chart,
                                 args.crossfit, args.step2)
        elif args.command == "choose":
            result = run_choose(args.out)
        else:
            result = run_stage3(args.out, args.source, args.folds, args.stocks, args.width, args.lam, args.quick,
                                args.chart, args.crossfit)
        print(result, flush=True)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


if __name__ == "__main__":
    app.run()
