import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training: elastic net, LightGBM and MLPs, tuned with W&B sweeps

    Trains every model family on the model panel (`data/processed/model_panel_train.parquet`, from `scripts/02_features_taq_trees_linear.py`) with the same walk-forward, features and target, and tunes each with a W&B sweep. The plan is in `explanations/models/training.md`.

    **The walk-forward:** quarterly refits over 2019-2023 (20 folds). Each fold trains on every day from 2014 to just before its quarter, on all rows with a target; LightGBM and the MLP hold out the window's last 12 months to stop early. Predictions are scored on the stocks with previous cap of $1B or more only, the trading universe (`large_cap`; v4 was scored on the liquid stocks instead, those with a 20-day median quoted spread of at most 20 bp).

    **The objective** each sweep minimizes is the out-of-sample MSE against the target on the ≥ $1B stocks, `oos_1b/mse`: the loss every model trains on and stops early on, now also the one its settings are chosen on. IC, R² and the deciles are logged alongside it, not optimized.

    **The test years (2024-2025) are not read.**
    """)
    return


@app.cell
def _():
    import argparse
    import functools
    import itertools
    import sys
    import time

    import marimo as mo
    import numpy as np
    import pandas as pd
    import wandb

    from dsr.evaluation import daily_decile_spread, daily_ic_and_deciles, daily_rank_ic, decile_means, ic_summary, mse, r2_zero
    from dsr.models import fit_predict
    from dsr.walkforward import feature_columns, fold_rows, folds

    return (
        argparse,
        daily_decile_spread,
        daily_ic_and_deciles,
        daily_rank_ic,
        decile_means,
        feature_columns,
        fit_predict,
        fold_rows,
        folds,
        functools,
        ic_summary,
        itertools,
        mo,
        mse,
        np,
        pd,
        r2_zero,
        sys,
        time,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Settings and search spaces

    Every run uses the baseline target `target_demeaned` and the midpoint imbalances (`bjzz` swaps in the BJZZ benchmark).

    **v5 goes back to the ≥ $1B universe** (user decision: the standard one), still training on all rows with a target, and its search spaces reach further toward regularization than v4's: v4's chosen runs sat at or near the regularized end of every range.

    - **Elastic net**, a fixed grid of 60 runs: the penalty α at 0 and 10⁻⁵ to 30 (v4's best Ridge was at its largest α, 1; v3's, on the ≥ $1B stocks, at 0.1), times the L1 share at 0 (Ridge), 0.1, 0.5 and 1 (LASSO); α = 0 is OLS. Features and target are scaled to unit variance on the training rows, so the penalty treats every feature alike, the 0/1 flags included. At the largest α the L1 runs keep no feature: harmless, and seconds each.
    - **LightGBM** (30 runs) and **the MLP** (40 runs, on the M5's GPU) use Bayesian search; LightGBM stops after 50 trees and the MLP after 3 epochs without improving the hold-out. Both depend on a random seed (row and feature sampling; starting weights, batch order and dropout): a sweep run uses seed 0 alone, the comparison averages seeds 0-4.
        - **LightGBM** (v4 in brackets): learning rate 0.003-0.05 (0.01-0.2), 3-31 leaves (7-255: every v4 run with 20 or more scored at or below zero), L2 penalty 0.01-1,000 (0.01-100), feature fraction 0.3-1 (0.5-1), up to 5,000 trees (2,000), so that the slow learning rates stop early instead of at the cap. Minimum rows per leaf (100-20,000) and bagging fraction (0.3-1) as in v4.
        - **The MLP:** hidden layers halving layer by layer from the first one's width (256, 128, 64, ...), with 1-6 layers, a first width of 8-256 (16-512), ReLU, LeakyReLU or GELU, dropout 0.1-0.7 (0-0.5), learning rate 3·10⁻⁵ to 3·10⁻³ (10⁻⁴ to 10⁻²), weight decay 10⁻⁵ to 10⁻¹ (10⁻⁶ to 10⁻²), batch as in v4, up to 40 epochs (20).
    """)
    return


@app.cell
def _(mo):
    PANEL = mo.notebook_dir().parent / "data" / "processed" / "model_panel_train.parquet"
    PROJECT = "daily-stock-returns-v5"  # a fresh W&B project per version: v1 and v2 are in daily-stock-returns, v3 in -v3
    FIRST = "2019-01-01"
    HOLD_OUT_MONTHS = 12
    OBJECTIVE, GOAL = "oos_1b/mse", "minimize"  # out of sample, on the >= $1B stocks that will be traded
    VERSION = "v5"  # the sweeps' name suffix and tag: v5 is back on the >= $1B stocks (v4 was scored on the liquid ones)

    COMMON = {"target": {"value": "target_demeaned"}, "bjzz": {"value": False},
              "train_rows": {"value": "all"}}  # every row with a target; "universe" would train on the >= $1B stocks only
    SWEEPS = {
        "linear": {  # an elastic net (dsr.models): alpha 0 is OLS, l1_ratio 0 Ridge, 1 LASSO
            "alpha": {"values": [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]},
            "l1_ratio": {"values": [0.0, 0.1, 0.5, 1.0]},
        },
        "lgbm": {
            "learning_rate": {"distribution": "log_uniform_values", "min": 0.003, "max": 0.05},
            "num_leaves": {"distribution": "q_log_uniform_values", "min": 3, "max": 31, "q": 1},
            "min_data_in_leaf": {"distribution": "q_log_uniform_values", "min": 100, "max": 20000, "q": 1},
            "feature_fraction": {"min": 0.3, "max": 1.0},
            "bagging_fraction": {"min": 0.3, "max": 1.0},
            "lambda_l2": {"distribution": "log_uniform_values", "min": 0.01, "max": 1000.0},
            "linear_tree": {"value": False},  # constant leaves only (explanations/models/training.md)
            "max_trees": {"value": 5000},  # learning rates down to 0.003 need more than v4's 2,000
            "seeds": {"value": 1},
        },
        "mlp": {  # width is the first hidden layer's, halved layer by layer
            "layers": {"values": [1, 2, 3, 4, 5, 6]},
            "width": {"values": [8, 16, 32, 64, 128, 256]},
            "shape": {"value": "decreasing"},
            "activation": {"values": ["relu", "leaky_relu", "gelu"]},
            "dropout": {"min": 0.1, "max": 0.7},
            "learning_rate": {"distribution": "log_uniform_values", "min": 3e-5, "max": 3e-3},
            "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
            "batch_size": {"values": [4096, 16384, 65536]},
            "max_epochs": {"value": 40},  # learning rates down to 3e-5 need more than v4's 20
            "seeds": {"value": 1},
        },
    }
    METHODS = {"linear": "grid", "lgbm": "bayes", "mlp": "bayes"}  # linear: every grid value once
    return (
        COMMON,
        FIRST,
        GOAL,
        HOLD_OUT_MONTHS,
        METHODS,
        OBJECTIVE,
        PANEL,
        PROJECT,
        SWEEPS,
        VERSION,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. One run: the walk-forward

    `walk_forward` fits and predicts every fold. Every fold trains on all rows with a target (all stocks above $100M; `train_rows` "all") or only on those in the trading universe on their own day ("universe"), the early-stopping hold-out included, and is scored **out of sample** on the quarter it predicts, on **all stocks** and on the **≥ $1B stocks** that will be traded, and **in sample** on the rows it was fit on (the ≥ $1B ones only, when it trained on those only). LightGBM's and the MLP's early-stopping hold-out is neither. W&B groups what is logged into sections:

    - **`fold/`**, as each fold finishes: IC, R², MSE and decile spread in each of the four groups (`fold/oos_1b_mse`, `fold/is_all_ic_mean`, ...), and what the fit chose (trees, epochs, nonzero coefficients).
    - **`oos_1b/`** and **`oos_all/`**, over all 1,257 predicted days:
        - `mse` and `r2_pct`, R² against the target in percent (against a prediction of 0; on the same rows R² = 1 − MSE ÷ the mean squared target, so the two rank runs identically). **`oos_1b/mse` is the objective.**
        - `decile_spread` (also `decile_spread_bp`), its Newey-West t-statistic `decile_spread_t`, the share of days it is positive, and `spread_<year>_bp`;
        - `long_leg_bp` and `short_leg_bp`, what the top and bottom decile each add against the average decile (the short leg's is positive when the bottom decile underperforms), and every decile's `decile_<d>_vs_avg_bp`;
        - `ic_mean`, its Newey-West t-statistic `ic_t`, `ic_positive` and `ic_<year>`.
    - **`is_1b/`** and **`is_all/`** (only when it trained on all rows): the same IC, R², MSE and decile measures in sample, each averaged over the 20 folds. In-sample and out-of-sample MSE cover different periods with different volatility; R² is the one to compare.
    - **`charts/`**: the four groups side by side, IC by fold, R² by fold, and the decile profile (each decile's `ret_next` minus the average decile, decile 1 holding the lowest predictions); and `charts/deciles_oos_1b`, the out-of-sample decile profile of the ≥ $1B stocks on its own, with its raw values in a table.
    - **`epoch/`** (the MLP), live as it trains: each epoch's training loss (the average of its batch losses) and hold-out loss, on the target scaled to unit variance, with the fold, seed and epoch number. Every epoch of every fold and seed is one point along `epoch/step`, so each fold shows as one stretch of the line; early stopping keeps the epoch with the lowest hold-out loss.

    It returns the out-of-sample predictions (all stocks, with `large_cap`) and the in-sample summaries, for the comparison.
    """)
    return


@app.cell
def _(
    FIRST,
    HOLD_OUT_MONTHS,
    PANEL,
    daily_decile_spread,
    daily_ic_and_deciles,
    decile_means,
    feature_columns,
    fit_predict,
    fold_rows,
    folds,
    functools,
    ic_summary,
    itertools,
    mse,
    np,
    pd,
    r2_zero,
    time,
    wandb,
):
    @functools.cache
    def load_panel():
        """The training panel, read once per process (a sweep agent runs many runs in one process)."""
        return pd.read_parquet(PANEL)


    @functools.cache
    def arrays(bjzz, target):
        """The panel as compact arrays, made once per process for each feature set and target: the features as one
        float32 matrix, so each fold slices rows of it instead of copying the whole table."""
        df = load_panel()
        features = feature_columns(df.columns, bjzz=bjzz)
        return {"features": features, "x": df[features].to_numpy("float32"), "y": df[target].to_numpy("float32"),
                "day": df["day"].to_numpy(), "large": df["large_cap"].to_numpy(bool),
                "ret": df["ret_next"].to_numpy()}


    def measures(scored, target):
        """IC, R², MSE and the decile profile of one set of scored predictions (day, ret_next, target, pred); also
        returns the daily ICs."""
        ic_days, deciles = daily_ic_and_deciles(scored["day"], scored["pred"], scored["ret_next"])
        vs_average = 1e4 * (deciles - deciles.mean())  # bp per day, so the market's drift drops out
        return {
            "ic_mean": float(ic_days.mean()),
            "r2_pct": 100 * r2_zero(scored["pred"], scored[target]),  # in percent: raw values are ~0.0001
            "mse": mse(scored["pred"], scored[target]),
            "decile_spread_bp": 1e4 * float(deciles.loc[10] - deciles.loc[1]),
            "long_leg_bp": float(vs_average.loc[10]),
            "short_leg_bp": -float(vs_average.loc[1]),
            **{f"decile_{d}_vs_avg_bp": float(v) for d, v in vs_average.items()},
        }, ic_days


    def summarize(scored, target):
        """The out-of-sample summary over every fold's predictions: the measures, plus the Newey-West t-statistic and
        share of positive days of the daily IC and of the daily decile long-short, and both by year."""
        summary, ic_days = measures(scored, target)
        spread_days = daily_decile_spread(scored["day"], scored["pred"], scored["ret_next"])
        ic, spread = ic_summary(ic_days), ic_summary(spread_days)
        summary.update({"ic_t": ic["ic_t"], "ic_positive": ic["ic_positive"], "days": ic["days"],
                        "decile_spread": spread["ic_mean"],  # the objective, in return units
                        "decile_spread_t": spread["ic_t"], "decile_spread_positive": spread["ic_positive"]})
        for year, value in ic_days.groupby(ic_days.index.year).mean().items():
            summary[f"ic_{year}"] = float(value)
        for year, value in spread_days.groupby(spread_days.index.year).mean().items():
            summary[f"spread_{year}_bp"] = 1e4 * float(value)
        return summary


    GROUPS = {"is_all": "in sample, all", "is_1b": "in sample, >= $1B", "oos_all": "out of sample, all",
              "oos_1b": "out of sample, >= $1B"}


    def log_charts(per_fold, overall, scored_1b, groups):
        """The groups side by side: IC and R² by fold, and the decile profile; and the out-of-sample decile profile of
        the >= $1B stocks on its own, the one to be traded."""
        by_fold = lambda k: [[f[g][k] for f in per_fold] for g in groups]
        oos_1b = decile_means(scored_1b["day"], scored_1b["pred"], scored_1b["ret_next"])  # raw too, for the table
        fold_axis = f"fold (0 = {per_fold[0]['name']})"
        deciles = list(range(1, 11))
        wandb.log({
            "charts/ic_by_fold": wandb.plot.line_series(
                xs=list(range(len(per_fold))), ys=by_fold("ic_mean"), keys=[GROUPS[g] for g in groups],
                title="Mean daily rank IC by fold", xname=fold_axis),
            "charts/r2_by_fold": wandb.plot.line_series(
                xs=list(range(len(per_fold))), ys=by_fold("r2_pct"), keys=[GROUPS[g] for g in groups],
                title="R² against the target by fold, %", xname=fold_axis),
            # a line from decile 1 (lowest predictions) to 10 (highest), so the order can't be read upside down
            "charts/deciles": wandb.plot.line_series(
                xs=deciles, ys=[[overall[g][f"decile_{d}_vs_avg_bp"] for d in deciles] for g in groups],
                keys=[GROUPS[g] for g in groups], title="ret_next by prediction decile minus the average decile, bp per day",
                xname="decile"),
            "charts/deciles_oos_1b": wandb.plot.line(
                wandb.Table(data=[[int(d), float(r) * 1e4, float(r - oos_1b.mean()) * 1e4] for d, r in oos_1b.items()],
                            columns=["decile", "ret_next_bp", "vs_average_bp"]),
                "decile", "vs_average_bp",
                title="Out of sample, >= $1B: ret_next by prediction decile minus the average decile (bp per day)"),
        })


    def walk_forward(kind, cfg, every="quarter"):
        """Fit and predict every fold on the rows `cfg["train_rows"]` says (all rows with a target, or the >= $1B ones;
        all when it isn't set), score it out of sample on all stocks and on the >= $1B stocks and in sample on the
        rows it was fit on, log it as it finishes, then the summaries and charts. Returns the out-of-sample scored
        predictions (all stocks, with large_cap) and the in-sample summaries."""
        target, train_rows = cfg["target"], cfg.get("train_rows", "all")
        a = arrays(cfg["bjzz"], target)
        x, y, day, large = a["x"], a["y"], a["day"], a["large"]
        # in sample on all stocks only when it trained on all of them: otherwise the fit rows are the >= $1B ones
        groups = [g for g in GROUPS if not (g == "is_all" and train_rows == "universe")]
        if wandb.run is not None:
            wandb.config.update({"features": a["features"]}, allow_val_change=True)
            wandb.define_metric("fold/number")
            wandb.define_metric("fold/*", step_metric="fold/number")
            wandb.define_metric("epoch/step")
            wandb.define_metric("epoch/*", step_metric="epoch/step")
        steps = itertools.count()  # one point per MLP epoch, across folds and seeds

        def log_epoch(i):
            """Logs fold i's MLP epochs to W&B as they finish."""
            return lambda e: wandb.log({"epoch/step": next(steps), "epoch/fold": i, **{f"epoch/{k}": v for k, v in e.items()}})

        frame = lambda r, p: pd.DataFrame({"day": day[r], "ret_next": a["ret"][r], target: y[r], "pred": p,
                                           "large_cap": large[r]})
        out, per_fold = [], []
        for i, fold in enumerate(folds(np.unique(day), first=FIRST, every=every)):
            start = time.perf_counter()
            fit, hold, rows = fold_rows(day, y, large, fold, HOLD_OUT_MONTHS, holds_out=kind != "linear",
                                        train_rows=train_rows)
            # one call predicts both the quarter (out of sample) and the rows the model was fit on (in sample)
            pred, info = fit_predict(kind, cfg, x[fit], y[fit], x[hold], y[hold], np.concatenate([x[rows], x[fit]]),
                                     on_epoch=log_epoch(i) if wandb.run is not None else None)  # the MLP's epochs, live
            n = int(rows.sum())
            scored = {"oos": frame(rows, pred[:n]), "is": frame(fit, pred[n:])}
            rows_of = lambda g: scored[g.split("_")[0]].pipe(lambda df: df[df["large_cap"]] if g.endswith("_1b") else df)
            m = {g: measures(rows_of(g), target)[0] for g in groups}  # "oos_1b": the quarter's >= $1B stocks, ...
            per_fold.append({"name": fold.name, **m})
            if wandb.run is not None:
                wandb.log({"fold/number": i, "fold/name": fold.name, "fold/seconds": time.perf_counter() - start,
                           **{f"fold/{g}_{k}": m[g][k] for g in groups for k in ("ic_mean", "r2_pct", "mse", "decile_spread_bp")},
                           **{f"fold/{k}": v for k, v in info.items()}})
            out.append(scored["oos"])
        scored = pd.concat(out, ignore_index=True)
        overall = {"oos_all": summarize(scored, target), "oos_1b": summarize(scored[scored["large_cap"]], target),
                   # in sample: each measure averaged over the folds
                   **{g: pd.DataFrame([f[g] for f in per_fold]).mean().to_dict() for g in groups if g.startswith("is_")}}
        if wandb.run is not None:
            wandb.summary.update({f"{g}/{k}": v for g, values in overall.items() for k, v in values.items()})
            log_charts(per_fold, overall, scored[scored["large_cap"]], groups)
        return scored, {g: overall[g] for g in groups if g.startswith("is_")}

    return summarize, walk_forward


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Running it (from the terminal)

    Sweeps run for hours, so they are launched from the terminal, where this notebook runs as a script; opened in marimo, this cell only shows the commands.

    ```
    uv run python scripts/03_train_wandb_linear_trees_nn.py sweep --model lgbm --count 30
    uv run python scripts/03_train_wandb_linear_trees_nn.py sweep --model lgbm --count 10 --sweep-id <id>   # add runs
    uv run python scripts/03_train_wandb_linear_trees_nn.py compare --sweeps <linear id> <lgbm id> <mlp id>
    uv run python scripts/03_train_wandb_linear_trees_nn.py check --project daily-stock-returns-v4 --runs fearless-sweep-27
    ```

    `sweep` creates a W&B sweep for one model family (or joins one), named `<model>-quarterly-v5`, and runs its agent here. `compare` takes each sweep's best run by the metric that sweep optimized (the lowest `oos_1b/mse` for the v5 sweeps), reruns it (LightGBM and the MLP averaging seeds 0-4 instead of the sweep's seed 0), and logs one comparison run, on the ≥ $1B stocks: every model's out-of-sample measures and in-sample IC, R², MSE and spread side by side, and paired tests between every two models, on their daily IC, their daily long-short return and their daily MSE (the mean daily difference and its Newey-West t-statistic). `check` reruns runs of an earlier version's sweeps by name, each twice with everything else as it was: trained on all rows, and on the trading universe only. It logs both, and the same paired tests between them on the universe: what training on the universe alone changes at fixed settings. (It was run on 2026-10-14 for v4's chosen runs, when the universe was the liquid stocks; explanations/models/training.md.) Run without a command, the script only runs its cells.
    """)
    return


@app.cell
def _(
    COMMON,
    GOAL,
    METHODS,
    OBJECTIVE,
    PROJECT,
    SWEEPS,
    VERSION,
    argparse,
    daily_decile_spread,
    daily_rank_ic,
    ic_summary,
    mo,
    pd,
    summarize,
    sys,
    walk_forward,
    wandb,
):
    def paired(a, b, target):
        """Paired tests between two sets of predictions of the same rows (day, ret_next, target, pred): the mean daily
        difference, a minus b, of the rank IC, the decile long-short and the MSE, with Newey-West t-statistics."""
        ic, spread = (ic_summary(f(a["day"], a["pred"], a["ret_next"]) - f(b["day"], b["pred"], b["ret_next"]))
                      for f in (daily_rank_ic, daily_decile_spread))
        daily_mse = lambda s: (s["pred"] - s[target]).pow(2).groupby(s["day"]).mean()
        error = ic_summary(daily_mse(a) - daily_mse(b))
        return {"ic_difference": ic["ic_mean"], "ic_difference_t": ic["ic_t"],
                "spread_difference_bp": 1e4 * spread["ic_mean"], "spread_difference_t": spread["ic_t"],
                "mse_difference": error["ic_mean"], "mse_difference_t": error["ic_t"]}


    def compare(sweep_ids):
        """Rerun each sweep's best run and log every model side by side with paired tests between them."""
        api = wandb.Api()
        scored, in_sample = {}, {}
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{api.default_entity}/{PROJECT}/{sweep_id}")
            kind = sweep.name.split("-")[0]
            metric, goal = sweep.config["metric"]["name"], sweep.config["metric"]["goal"]  # what the sweep optimized
            runs = [r for r in sweep.runs if r.state == "finished" and metric in r.summary]
            best = (min if goal == "minimize" else max)(runs, key=lambda r: r.summary[metric])
            cfg = {k: v for k, v in best.config.items() if k != "features"}
            if kind in ("lgbm", "mlp"):  # a fit depends on its seed: average seeds 0-4
                cfg["seeds"] = 5
            print(f"{kind}: best of {len(runs)} runs is {best.name} ({metric} {best.summary[metric]:.6f}); rerunning",
                  flush=True)
            with wandb.init(project=PROJECT, job_type="final", tags=[kind, "final"], config=cfg,
                            name=f"{kind}-final-{best.id}"):
                s, in_sample[kind] = walk_forward(kind, cfg)
                scored[kind] = s[s["large_cap"]]  # the models are compared on the >= $1B stocks
        target = COMMON["target"]["value"]
        table = pd.DataFrame({name: {**summarize(s, target),
                                     **{f"is_{k}": in_sample[name]["is_1b"][k] for k in ("ic_mean", "r2_pct", "mse", "decile_spread_bp")}}
                              for name, s in scored.items()}).T
        names = list(scored)
        pairs = pd.DataFrame([{"model": a, "minus": b, **paired(scored[a], scored[b], target)}
                              for i, a in enumerate(names) for b in names[i + 1:]])
        with wandb.init(project=PROJECT, job_type="comparison", name="comparison", config={"sweeps": sweep_ids}):
            wandb.log({"models": wandb.Table(dataframe=table.reset_index(names="model")),
                       "paired_tests": wandb.Table(dataframe=pairs)})
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(table.round(5).T.to_string())
            print(pairs.round(5).to_string(index=False))


    def check(project, run_names):
        """Rerun runs of an earlier version's sweeps (by name) with their settings and seeds, once trained on all rows
        and once on the trading universe only, and test the difference on the universe."""
        api = wandb.Api()
        target = COMMON["target"]["value"]
        keys = ("mse", "r2_pct", "ic_mean", "ic_t", "decile_spread_bp", "decile_spread_t")
        table = []
        for name in run_names:
            run = api.runs(f"{api.default_entity}/{project}", filters={"display_name": name})[0]
            kind = run.sweep.name.split("-")[0]
            scored = {}
            for train_rows in ("all", "universe"):
                cfg = {**{k: v for k, v in run.config.items() if k != "features"}, "train_rows": train_rows}
                print(f"{kind}: {name} from {project}, trained on {train_rows} rows", flush=True)
                with wandb.init(project=PROJECT, job_type="check", tags=[kind, "check", train_rows],
                                config={**cfg, "from_run": f"{project}/{name}"}, name=f"{kind}-{train_rows}-{name}"):
                    s, _ = walk_forward(kind, cfg)
                scored[train_rows] = s[s["large_cap"]]
            summary = {tr: summarize(scored[tr], target) for tr in scored}
            table.append({"model": kind, "run": name, **{f"{k}_{tr}": summary[tr][k] for tr in scored for k in keys},
                          **paired(scored["universe"], scored["all"], target)})  # universe minus all
        table = pd.DataFrame(table)
        with wandb.init(project=PROJECT, job_type="check", name=f"check-{'-'.join(run_names)}",
                        config={"project": project, "runs": run_names}):
            wandb.log({"check": wandb.Table(dataframe=table)})
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(table.round(6).T.to_string())


    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="03_train_wandb_linear_trees_nn.py")
        sub = parser.add_subparsers(dest="command", required=True)
        s = sub.add_parser("sweep", help="create a W&B sweep for one model family and run its agent here")
        s.add_argument("--model", choices=SWEEPS, required=True)
        s.add_argument("--count", type=int, default=30, help="runs for this agent")
        s.add_argument("--sweep-id", help="join an existing sweep instead of creating one")
        c = sub.add_parser("compare", help="rerun each sweep's best run and compare them, with paired tests")
        c.add_argument("--sweeps", nargs="+", required=True, help="sweep ids (their names say the model family)")
        k = sub.add_parser("check", help="rerun earlier sweep runs trained on all rows and on the universe only")
        k.add_argument("--project", required=True, help="the W&B project of the earlier sweeps")
        k.add_argument("--runs", nargs="+", required=True, help="run names, e.g. fearless-sweep-27")
        args = parser.parse_args(argv)

        if args.command == "sweep":
            config = {"method": METHODS[args.model], "metric": {"name": OBJECTIVE, "goal": GOAL},
                      "parameters": {**COMMON, **SWEEPS[args.model]}, "name": f"{args.model}-quarterly-{VERSION}"}
            sweep_id = args.sweep_id or wandb.sweep(config, project=PROJECT)

            def one_run():
                with wandb.init(job_type="sweep", tags=[args.model, VERSION]) as run:
                    walk_forward(args.model, dict(run.config))

            wandb.agent(sweep_id, function=one_run, count=args.count, project=PROJECT)
        elif args.command == "compare":
            compare(args.sweeps)
        else:
            check(args.project, args.runs)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Results so far (from W&B)

    Every sweep of this version (`-v5`), with its best run by the objective (the lowest out-of-sample MSE on the ≥ $1B stocks); measures on the ≥ $1B stocks. Rerun this cell to refresh while a sweep is going; the full comparison is the run named `comparison`.
    """)
    return


@app.cell
def _(GOAL, OBJECTIVE, PROJECT, VERSION, mo, pd, wandb):
    _api = wandb.Api()
    _rows = []
    try:
        _sweeps = list(_api.project(PROJECT).sweeps())
    except ValueError:  # the project is created by its first sweep
        _sweeps = []
    for _sweep in _sweeps:
        if not _sweep.name.endswith(f"-{VERSION}"):
            continue
        _runs = [_r for _r in _sweep.runs if _r.state == "finished" and OBJECTIVE in _r.summary]
        if not _runs:
            continue
        _best = (min if GOAL == "minimize" else max)(_runs, key=lambda _r: _r.summary[OBJECTIVE])
        _rows.append({
            "sweep": _sweep.name, "id": _sweep.id, "finished runs": len(_runs), "best run": _best.name,
            **{_k: _best.summary.get(f"oos_1b/{_k}") for _k in ["mse", "r2_pct", "ic_mean", "ic_t", "decile_spread_bp",
                                                                 "decile_spread_t", "long_leg_bp", "short_leg_bp"]},
            **{f"is_{_k}": _best.summary.get(f"is_1b/{_k}") for _k in ["r2_pct", "ic_mean", "decile_spread_bp"]},
            "settings": {_k: _v for _k, _v in _best.config.items() if _k not in ("target", "bjzz", "features")},
        })
    sweep_results = pd.DataFrame(_rows)
    sweep_results if len(sweep_results) else mo.md(f"No finished runs in the {VERSION} sweeps yet.")
    return


if __name__ == "__main__":
    app.run()
