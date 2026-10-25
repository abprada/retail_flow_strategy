import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Walk-forward with the tuned settings: monthly refits, coefficients, SHAP and SAGE

    Reruns each model family's chosen run from the v5 W&B sweeps (`scripts/models/03_train_wandb_linear_trees_nn.py`) over 2019-2023, refitting every month instead of every quarter, since nothing is tuned here, and records what each model learned at every refit. The plan is in `explanations/models/models.md`.

    - **The elastic net:** its coefficients and their t-statistics.
    - **LightGBM and the MLP:** SHAP values (what each feature adds to a prediction) and SAGE values (what each feature adds to the out-of-sample fit).

    **The test years (2024-2025) are not read.** The settings were chosen on these same years, so the performance below is not out of sample for the settings.
    """)
    return


@app.cell
def _():
    import argparse
    import functools
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import shap
    import wandb

    from dsr.evaluation import ic_summary, measures, summarize
    from dsr.explain import linear_inference, sage_values, shap_values
    from dsr.models import fit
    from dsr.walkforward import crossfit_folds, feature_columns, fold_rows, folds

    return (
        Path,
        alt,
        argparse,
        crossfit_folds,
        feature_columns,
        fit,
        fold_rows,
        folds,
        functools,
        ic_summary,
        json,
        linear_inference,
        measures,
        mo,
        np,
        pd,
        plt,
        sage_values,
        shap,
        shap_values,
        subprocess,
        summarize,
        sys,
        time,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Settings
    """)
    return


@app.cell
def _(mo):
    PANEL = mo.notebook_dir().parents[1] / "data" / "processed" / "model_panel_train.parquet"
    TEST_PANEL = mo.notebook_dir().parents[1] / "data" / "processed" / "model_panel_test.parquet"  # read only by --test
    OUT = mo.notebook_dir().parents[1] / "data" / "walkforward"  # one folder per model
    PROJECT = "daily-stock-returns-v5"
    SWEEP_SUFFIX = "-quarterly-v5"  # the sweeps whose chosen runs are rerun: linear-quarterly-v5, ...
    MODELS = ("linear", "mlp", "lgbm")
    # the MLP's and LightGBM's: their sweep's highest out-of-sample rank IC on the >= $1B stocks (oos_1b/ic_mean);
    # the elastic net's: the region choice by MSE, which backtested better than its highest rank IC (section 2)
    RUNS = {"linear": "driven-sweep-41", "mlp": "major-sweep-29", "lgbm": "winter-sweep-17"}
    FIRST = "2019-01-01"
    TEST_FIRST = "2024-01-01"  # --test: the test years, 2024-2025, run once with the validation run's settings
    CROSSFIT_YEARS = (2014, 2015, 2016, 2017, 2018)  # --crossfit: each year predicted from the other four
    EMBARGO_DAYS = 20  # --crossfit: trading days after the year not trained on (the features look back 20 days)
    HOLD_OUT_MONTHS = 12  # LightGBM's and the MLP's early-stopping hold-out, as in the sweeps
    SEEDS = 5  # LightGBM and the MLP average seeds 0-4, as in 03's comparison
    SHAP_ROWS = 5_000  # >= $1B rows whose SHAP values are kept each month, drawn at random from the month's
    SAGE_PERMUTATIONS = 1_000  # SAGE: rows drawn at random from each day's >= $1B rows, each with its own feature order
    BACKGROUND_ROWS = 32  # >= $1B training rows whose values stand in for a feature left out (SHAP and SAGE)
    return (
        BACKGROUND_ROWS,
        CROSSFIT_YEARS,
        EMBARGO_DAYS,
        FIRST,
        HOLD_OUT_MONTHS,
        MODELS,
        OUT,
        PANEL,
        PROJECT,
        RUNS,
        SAGE_PERMUTATIONS,
        SEEDS,
        SHAP_ROWS,
        SWEEP_SUFFIX,
        TEST_FIRST,
        TEST_PANEL,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Each sweep's chosen run

    The run named in `RUNS` from each `<model>-quarterly-v5` sweep, shown with its rank by the metric the sweep optimized (the lowest out-of-sample MSE on the ≥ $1B stocks) and the sweep's best run by that metric.

    - **The MLP's and LightGBM's chosen run is the sweep's highest out-of-sample rank IC on the ≥ $1B stocks** (`oos_1b/ic_mean`; user decision, 2026-10-17). Until then each sweep's chosen run sat in the middle of its best region by MSE (`explanations/models/models.md`); those results are in `data/walkforward/previous_choice/`.
    - **The elastic net keeps its region choice,** Ridge with α = 0.3 (user decision, 2026-10-17): its highest rank IC, α = 30 at the top of the grid, had the higher IC in the walk-forward (0.0150 against 0.0131) but the weaker decile long-short (Sharpe 0.96 against 1.10). Its results are in `data/walkforward/rank_ic_choice/linear/`.
    - **LightGBM and the MLP average 5 seeds** instead of the sweep run's one.
    - **The settings a walk-forward used are written with its results.**
    """)
    return


@app.cell
def _(MODELS, PROJECT, RUNS, SWEEP_SUFFIX, mo, pd, wandb):
    def sweep_run(kind):
        """The model's chosen run (`RUNS`) in its sweep, with its settings, its value of the sweep's metric, its rank
        among the finished runs, and the sweep's best run; None before the sweep has it. If several sweeps have the
        name, the one with the most finished runs."""
        api = wandb.Api()
        try:
            sweeps = [s for s in api.project(PROJECT).sweeps() if s.name == f"{kind}{SWEEP_SUFFIX}"]
        except ValueError:  # the project is created by its first sweep
            return None
        found = []
        for sweep in sweeps:
            metric, goal = sweep.config["metric"]["name"], sweep.config["metric"]["goal"]
            runs = sorted([r for r in sweep.runs if r.state == "finished" and metric in r.summary],
                          key=lambda r: r.summary[metric], reverse=goal == "maximize")
            rank = next((i for i, r in enumerate(runs) if r.name == RUNS[kind]), None)
            if rank is not None:
                run = runs[rank]
                found.append({"sweep": sweep.id, "run": run.name, "run_id": run.id, "metric": metric,
                              "value": run.summary[metric], "rank": rank + 1, "finished_runs": len(runs),
                              "best_run": runs[0].name, "best_value": runs[0].summary[metric],
                              "settings": {k: v for k, v in run.config.items() if k != "features"}})
        return max(found, key=lambda b: b["finished_runs"]) if found else None


    _rows = [{"model": _k, **_b} for _k in MODELS if (_b := sweep_run(_k)) is not None]
    pd.DataFrame(_rows) if _rows else mo.md("No chosen sweep runs yet.")
    return (sweep_run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One model's walk-forward

    `walk_forward` refits the model at the start of every month of 2019-2023 (60 folds) on every earlier day, on the rows its sweep run trained on (`train_rows`: all rows with a target, for every v5 run); LightGBM and the MLP hold out the window's last 12 months to stop early, as in the sweeps. It predicts the month and scores it on all stocks and on the ≥ $1B stocks (the trading universe, `large_cap`) with 03's measures. Then, at each refit:

    - **The elastic net:** each coefficient in bp of next-day demeaned return per unit of the feature (one cross-sectional standard deviation for the standardized features; on against off for the 0/1 flags), its standard error clustered by day (stocks' errors on the same day are correlated), and its t-statistic, on the training window the refit used (`dsr.explain.linear_inference`).
    - **LightGBM and the MLP**, on their 5-seed average, with a feature left out taking the values of 32 ≥ $1B training rows in turn:
        - **SHAP** (shap's `PermutationExplainer`) for 5,000 of the month's ≥ $1B rows, drawn at random: what each feature adds to each row's prediction. It only calls the model, so it works for both models and for LightGBM's linear leaves, which shap's `TreeExplainer` reads wrongly.
        - **SAGE** (`sage-importance`) for every day of the month, from 1,000 of the day's ≥ $1B rows drawn at random: what each feature adds to the fall in squared error against the target. One estimate per day, so its uncertainty is measured across days, as the IC's is.

    It writes to `data/walkforward/<model>/`:

    - `settings.json`: the sweep run and settings used;
    - `predictions.parquet` (out of sample, all stocks), `folds.parquet` (each month's measures), `summary.json`;
    - the elastic net: `betas.parquet`;
    - LightGBM and the MLP: `shap.parquet` (each month's mean |SHAP| and SHAP slope per feature), `shap_rows.parquet` (every explained row's SHAP values) and `sage_days.parquet` (each day's SAGE values).
    """)
    return


@app.cell
def _(
    BACKGROUND_ROWS,
    CROSSFIT_YEARS,
    EMBARGO_DAYS,
    FIRST,
    HOLD_OUT_MONTHS,
    PANEL,
    SAGE_PERMUTATIONS,
    SEEDS,
    SHAP_ROWS,
    TEST_FIRST,
    TEST_PANEL,
    crossfit_folds,
    feature_columns,
    fit,
    fold_rows,
    folds,
    functools,
    json,
    linear_inference,
    measures,
    np,
    pd,
    sage_values,
    shap_values,
    summarize,
    time,
):
    @functools.cache
    def arrays(bjzz, target, test=False):
        """The training panel as compact arrays, made once per process: the features as one float32 matrix. With
        `test`, the test panel follows it, for the test years' walk-forward."""
        df = pd.read_parquet(PANEL)
        if test:
            df = pd.concat([df, pd.read_parquet(TEST_PANEL)], ignore_index=True)
        features = feature_columns(df.columns, bjzz=bjzz)
        return {"features": features, "x": df[features].to_numpy("float32"), "y": df[target].to_numpy("float32"),
                "day": df["day"].to_numpy(), "permno": df["permno"].to_numpy(),
                "large": df["large_cap"].to_numpy(bool), "ret": df["ret_next"].to_numpy()}


    def explain(model, x, explained, background, seed):
        """SHAP for the explained rows, in bp: every row's values, and each feature's mean |SHAP| and SHAP slope (the
        SHAP values' regression on the feature, comparable to a linear coefficient)."""
        shap = 1e4 * shap_values(model.predict, x[explained], background, seed=seed)
        xc = x[explained] - x[explained].mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):  # a flag never on in the rows has no slope
            slope = (shap * xc).sum(axis=0) / (xc ** 2).sum(axis=0)
        return shap, np.abs(shap).mean(axis=0), slope


    def walk_forward(kind, chosen, out, months=None, every="month", refit=False, test=False, crossfit=False):
        """Refit every month (or every year, `every`) with the chosen run's settings, predict and score the period,
        explain the refit, and write it all to `out`. `months` keeps only the first periods, as a quick check. With
        `refit` (LightGBM), each fit is redone on the training rows and the hold-out with the early-stopped tree count.
        With `test`, the periods are the test years', on the training and test panels. With `crossfit`, each of
        CROSSFIT_YEARS is predicted from the other four (crossfit_folds), without the SHAP and SAGE explanations: the
        out-of-sample predictions of the years before the first fold."""
        cfg = {**chosen["settings"], **({"seeds": SEEDS} if kind != "linear" else {}),
               **({"refit_with_holdout": True} if refit else {})}
        target = cfg["target"]
        a = arrays(cfg["bjzz"], target, test)
        x, y, day, large, features = a["x"], a["y"], a["day"], a["large"], a["features"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "settings.json").write_text(json.dumps(
            {**chosen, "settings": cfg, "shap_rows": SHAP_ROWS, "sage_permutations": SAGE_PERMUTATIONS,
             "background_rows": BACKGROUND_ROWS}, indent=2))
        predictions, per_fold, betas, shap_folds, shap_rows, sage_days = [], [], [], [], [], []
        periods = (crossfit_folds(np.unique(day), CROSSFIT_YEARS, EMBARGO_DAYS) if crossfit
                   else folds(np.unique(day), first=TEST_FIRST if test else FIRST, every=every))
        for i, fold in enumerate(periods[:months]):
            start = time.perf_counter()
            fit_rows, hold, rows = fold_rows(day, y, large, fold, HOLD_OUT_MONTHS, holds_out=kind != "linear",
                                             train_rows=cfg.get("train_rows", "all"))
            model = fit(kind, cfg, x[fit_rows], y[fit_rows], x[hold], y[hold])
            scored = pd.DataFrame({"day": day[rows], "permno": a["permno"][rows], "fold": fold.name,
                                   "pred": model.predict(x[rows]), target: y[rows], "ret_next": a["ret"][rows],
                                   "large_cap": large[rows]})
            predictions.append(scored)
            m = {u: measures(df, target)[0] for u, df in (("all", scored), ("1b", scored[scored["large_cap"]]))}
            per_fold.append({"fold": fold.name, **model.info,
                             **{f"oos_{u}_{k}": m[u][k] for u in m for k in ("ic_mean", "r2_pct", "mse", "decile_spread_bp")}})
            if kind == "linear":
                beta, se = linear_inference(x[fit_rows], y[fit_rows], day[fit_rows], **model.linear)
                betas.append(pd.DataFrame({"fold": fold.name, "feature": features, "beta_bp": 1e4 * beta,
                                           "se_bp": 1e4 * se, "t": beta / se}))
            elif not crossfit:
                rng = np.random.default_rng(i)
                pool = np.flatnonzero(rows & large & ~np.isnan(y))  # the month's >= $1B rows with a target
                background = x[rng.choice(np.flatnonzero(fit_rows & large), BACKGROUND_ROWS, replace=False)]
                explained = np.sort(rng.choice(pool, min(SHAP_ROWS, len(pool)), replace=False))
                shap, mean_abs, slope = explain(model, x, explained, background, seed=i)
                shap_folds.append(pd.DataFrame({"fold": fold.name, "feature": features, "mean_abs_bp": mean_abs,
                                                "slope_bp": slope}))
                shap_rows.append(pd.DataFrame(shap.astype("float32"), columns=features)
                                 .assign(day=day[explained], permno=a["permno"][explained]))
                for k, d in enumerate(np.unique(day[pool])):  # SAGE day by day
                    on_day = pool[day[pool] == d]
                    values, _ = sage_values(model.predict, x[on_day], y[on_day], background, SAGE_PERMUTATIONS,
                                            seed=1000 * i + k)
                    sage_days.append({"day": d, **dict(zip(features, values)),
                                      "y2": float(np.mean(y[on_day].astype(float) ** 2)), "rows": len(on_day)})
            print(f"{kind} {fold.name}: {time.perf_counter() - start:.0f}s; >= $1B: IC {m['1b']['ic_mean']:.4f}, "
                  f"R² {m['1b']['r2_pct']:.4f}%, spread {m['1b']['decile_spread_bp']:.1f} bp", flush=True)
        scored = pd.concat(predictions, ignore_index=True)
        summary = {"oos_all": summarize(scored, target), "oos_1b": summarize(scored[scored["large_cap"]], target)}
        scored.to_parquet(out / "predictions.parquet")
        pd.DataFrame(per_fold).to_parquet(out / "folds.parquet")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        if betas:
            pd.concat(betas, ignore_index=True).to_parquet(out / "betas.parquet")
        if shap_folds:
            pd.concat(shap_folds, ignore_index=True).to_parquet(out / "shap.parquet")
            pd.concat(shap_rows, ignore_index=True).to_parquet(out / "shap_rows.parquet")
            pd.DataFrame(sage_days).to_parquet(out / "sage_days.parquet")
        return summary

    return (walk_forward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Running it (from the terminal)

    ```
    uv run python scripts/models/04_walkforward.py run                                  # every model: linear, mlp, lgbm, one process each
    uv run python scripts/models/04_walkforward.py run --models lgbm
    uv run python scripts/models/04_walkforward.py run --months 2 --out /tmp/check      # a quick check
    uv run python scripts/models/04_walkforward.py run --models lgbm --every year --test  # the test years, once
    uv run python scripts/models/04_walkforward.py run --models lgbm --every year --crossfit  # 2014-2018, cross-fitted
    uv run python scripts/models/04_walkforward.py run --models linear --crossfit
    ```

    Opened in marimo, this cell only shows the commands.
    """)
    return


@app.cell
def _(MODELS, OUT, Path, RUNS, argparse, json, mo, subprocess, sweep_run, sys, walk_forward):
    def run_from_terminal(argv):
        parser = argparse.ArgumentParser(prog="04_walkforward.py")
        sub = parser.add_subparsers(dest="command", required=True)
        r = sub.add_parser("run", help="rerun each model's chosen sweep run with monthly refits and explain it")
        r.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
        r.add_argument("--months", type=int, help="only the first N months, as a quick check")
        r.add_argument("--out", type=Path, default=OUT, help="where to write the results, one folder per model")
        r.add_argument("--every", choices=["month", "year"], default="month",
                       help="refit every month (default) or every year, as the CNNs; yearly results go to <model>_yearly")
        r.add_argument("--refit", action="store_true",
                       help="LightGBM only: fit again on the training rows and the hold-out with the early-stopped tree "
                            "count; results go to <model>[_yearly]_refit")
        r.add_argument("--test", action="store_true",
                       help="the test years (2024-2025), once, with the settings the validation run wrote to "
                            "<out>/<model>[_yearly][_refit]/settings.json; results go to <out>/test/")
        r.add_argument("--crossfit", action="store_true",
                       help="2014-2018, each year predicted from the other four, with the settings the validation run "
                            "wrote to <out>/<model>[_yearly][_refit]/settings.json; results go to <out>/crossfit/")
        args = parser.parse_args(argv)
        if args.refit and args.models != ["lgbm"]:
            raise SystemExit("--refit is for LightGBM only: run it with --models lgbm")
        if len(args.models) > 1:
            # one process per model: PyTorch on the CPU after LightGBM in one process deadlocks or crashes on macOS (two
            # OpenMP runtimes), and each process frees its memory when its model is done
            for kind in args.models:
                subprocess.run([sys.executable, sys.argv[0], "run", "--models", kind, "--out", str(args.out),
                                "--every", args.every, *(["--refit"] if args.refit else []),
                                *(["--test"] if args.test else []), *(["--crossfit"] if args.crossfit else []),
                                *(["--months", str(args.months)] if args.months else [])],
                               check=True)
            return
        kind = args.models[0]
        name = kind + ("_yearly" if args.every == "year" else "") + ("_refit" if args.refit else "")
        out = args.out / "test" / name if args.test else args.out / "crossfit" / name if args.crossfit else args.out / name
        if args.test and (out / "summary.json").exists():
            raise SystemExit(f"{out} already holds the test's results: the test runs once")
        # the test and the cross-fit reuse the validation run's settings as written, so nothing is looked up again
        frozen = args.test or args.crossfit
        chosen = json.loads((args.out / name / "settings.json").read_text()) if frozen else sweep_run(kind)
        if chosen is None:
            raise SystemExit(f"{kind}: its sweep has no finished run named {RUNS[kind]}")
        print(f"{kind}: {chosen['run']} ({chosen['metric']} {chosen['value']:.6g}, rank {chosen['rank']} of "
              f"{chosen['finished_runs']}; best {chosen['best_run']} {chosen['best_value']:.6g})", flush=True)
        walk_forward(kind, chosen, out, months=args.months, every=args.every, refit=args.refit, test=args.test,
                     crossfit=args.crossfit)


    if mo.app_meta().mode == "script" and len(sys.argv) > 1:  # run as a script with a command
        run_from_terminal(sys.argv[1:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Results

    Out of sample over 2019-2023, from each model's saved walk-forward.
    """)
    return


@app.cell
def _(MODELS, OUT, json, pd):
    def _read(kind, name):
        path = OUT / kind / name
        if not path.exists():
            return None
        return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_parquet(path)


    results = {_k: {_n: _read(_k, _n) for _n in ("settings.json", "summary.json", "betas.parquet", "shap.parquet",
                                                 "sage_days.parquet")} for _k in MODELS}
    done = [_k for _k in MODELS if results[_k]["summary.json"] is not None]
    return done, results


@app.cell
def _(done, mo, pd, results):
    _keys = ["ic_mean", "ic_t", "r2_pct", "mse", "decile_spread_bp", "decile_spread_t", "long_leg_bp", "short_leg_bp",
             *[f"spread_{_y}_bp" for _y in range(2019, 2024)]]
    _table = pd.DataFrame({f"{_k}, {_g}": {"run": results[_k]["settings.json"]["run"],
                                           **{_m: results[_k]["summary.json"][_g].get(_m) for _m in _keys}}
                           for _k in done for _g in ("oos_1b", "oos_all")})
    _table if done else mo.md("No results yet: run `uv run python scripts/models/04_walkforward.py run`.")
    return


@app.cell
def _(alt, done, pd, results):
    _deciles = pd.DataFrame([{"model": _k, "decile": _d, "vs_average_bp": results[_k]["summary.json"]["oos_1b"][f"decile_{_d}_vs_avg_bp"]}
                             for _k in done for _d in range(1, 11)])
    (alt.Chart(_deciles).mark_line(point=True).encode(
        x=alt.X("decile:O", title="prediction decile (1 = lowest)"),
        y=alt.Y("vs_average_bp:Q", title="ret_next minus the average decile, bp per day"), color="model:N")
     .properties(title="Out of sample, >= $1B: next-day return by prediction decile", width=500, height=250)
     if done else None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The elastic net's coefficients at each monthly refit

    In bp of next-day demeaned return per unit of the feature, with a 95% band (± 1.96 day-clustered standard errors); then the t-statistics, with lines at ± 1.96. Each refit's window holds every earlier day, so consecutive refits share all but one month of data and the lines are smooth by construction.
    """)
    return


@app.cell
def _(alt, mo, pd, results):
    _b = results["linear"]["betas.parquet"]
    if _b is None:
        _out = mo.md("No elastic-net results yet.")
    else:
        _b = _b.assign(month=pd.PeriodIndex(_b["fold"], freq="M").to_timestamp(),
                       lo=_b["beta_bp"] - 1.96 * _b["se_bp"], hi=_b["beta_bp"] + 1.96 * _b["se_bp"])
        _x = alt.X("month:T", title=None)
        _facet = alt.Facet("feature:N", sort=list(_b["feature"].unique()), title=None)
        _betas = alt.layer(
            alt.Chart().mark_area(opacity=0.25).encode(x=_x, y=alt.Y("lo:Q", title="bp per unit"), y2="hi:Q"),
            alt.Chart().mark_line().encode(x=_x, y="beta_bp:Q"),
            alt.Chart().mark_rule(color="gray").encode(y=alt.datum(0)),
            data=_b).properties(width=170, height=100).facet(facet=_facet, columns=4).resolve_scale(y="independent")
        _t = alt.layer(
            alt.Chart().mark_line().encode(x=_x, y=alt.Y("t:Q", title="t-statistic")),
            alt.Chart().mark_rule(color="gray", strokeDash=[4, 4]).encode(y=alt.datum(1.96)),
            alt.Chart().mark_rule(color="gray", strokeDash=[4, 4]).encode(y=alt.datum(-1.96)),
            data=_b).properties(width=170, height=100).facet(facet=_facet, columns=4).resolve_scale(y="independent")
        _out = mo.vstack([_betas, _t])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SHAP at each monthly refit

    On each month's explained ≥ $1B rows: first the mean |SHAP|, how much the feature moves predictions, in bp; then the SHAP slope, the SHAP values' regression on the feature, in bp per unit of the feature like the elastic net's coefficients (shown with them, black).
    """)
    return


@app.cell
def _(alt, mo, pd, results):
    _s = [results[_k]["shap.parquet"].assign(model=_k) for _k in ("lgbm", "mlp") if results[_k]["shap.parquet"] is not None]
    if not _s:
        _out = mo.md("No LightGBM or MLP results yet.")
    else:
        _s = pd.concat(_s)
        _b = results["linear"]["betas.parquet"]
        _slopes = pd.concat([_s[["fold", "feature", "model", "slope_bp"]],
                             _b.rename(columns={"beta_bp": "slope_bp"}).assign(model="linear (coefficient)")[
                                 ["fold", "feature", "model", "slope_bp"]] if _b is not None else None])
        _month = lambda d: d.assign(month=pd.PeriodIndex(d["fold"], freq="M").to_timestamp())
        _facet = alt.Facet("feature:N", sort=list(_s["feature"].unique()), title=None)
        _color = alt.Color("model:N", scale=alt.Scale(domain=["lgbm", "mlp", "linear (coefficient)"],
                                                      range=["#1f77b4", "#ff7f0e", "black"]))
        _abs = (alt.Chart(_month(_s)).mark_line()
                .encode(x=alt.X("month:T", title=None), y=alt.Y("mean_abs_bp:Q", title="mean |SHAP|, bp"), color=_color)
                .properties(width=170, height=100).facet(facet=_facet, columns=4).resolve_scale(y="independent"))
        _slope = (alt.Chart(_month(_slopes)).mark_line()
                  .encode(x=alt.X("month:T", title=None), y=alt.Y("slope_bp:Q", title="bp per unit"), color=_color)
                  .properties(width=170, height=100).facet(facet=_facet, columns=4).resolve_scale(y="independent"))
        _out = mo.vstack([_abs, _slope])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SHAP beeswarm

    Every explained row of every month, 5,000 ≥ $1B rows a month (the same rows for both models). Each dot is one row and one feature: its SHAP value in bp, colored by the feature's value, which is taken from the panel. Each row's values come from the refit that predicted its month. Features are ordered by mean |SHAP|. The two models have different scales.
    """)
    return


@app.cell
def _(OUT, PANEL, mo, pd, plt, shap):
    _script = mo.app_meta().mode == "script"  # run from the terminal: skip, it reads the panel and draws for a while
    _rows = {_k: pd.read_parquet(OUT / _k / "shap_rows.parquet") for _k in ("lgbm", "mlp")
             if not _script and (OUT / _k / "shap_rows.parquet").exists()}
    if _script:
        _out = None
    elif not _rows:
        _out = mo.md("No LightGBM or MLP results yet.")
    else:
        _features = [_c for _c in next(iter(_rows.values())).columns if _c not in ("day", "permno")]
        _keys = pd.concat([_r[["day", "permno"]] for _r in _rows.values()]).drop_duplicates()
        _values = pd.read_parquet(PANEL, columns=["day", "permno", *_features]).merge(_keys, on=["day", "permno"])
        _fig, _axes = plt.subplots(1, len(_rows), figsize=(7 * len(_rows), 7), squeeze=False)
        for _ax, (_k, _r) in zip(_axes[0], _rows.items()):
            _x = _r[["day", "permno"]].merge(_values, on=["day", "permno"], how="left")
            shap.plots.beeswarm(shap.Explanation(values=_r[_features].to_numpy(), data=_x[_features].to_numpy(),
                                                 feature_names=_features),
                                max_display=len(_features), ax=_ax, show=False, plot_size=None, s=4)
            _ax.set_title(_k)
            _ax.set_xlabel("SHAP value, bp of next-day demeaned return")
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SAGE: what each feature adds to the out-of-sample fit

    Each feature's share of the fall in squared error from the background's average prediction to the model's, on the ≥ $1B stocks, as a percentage of the mean squared target (the units of R²: the features add up to the model's total). The average of the daily SAGE values, with its Newey-West t-statistic over days, by year and over 2019-2023. The target's noise sets how precise these are: like the R² itself, a year's values for single features are mostly noise.
    """)
    return


@app.cell
def _(alt, ic_summary, mo, pd, results):
    def sage_summary(daily):
        """Each feature's SAGE value and the model's total by year and over 2019-2023, in % of the mean squared
        target, with Newey-West t-statistics over days."""
        daily = daily.set_index("day")
        features = [c for c in daily.columns if c not in ("y2", "rows")]
        daily = daily.assign(total=daily[features].sum(axis=1))
        periods = {str(y): daily.index.year == y for y in sorted(set(daily.index.year))} | {"2019-2023": slice(None)}
        rows = []
        for period, keep in periods.items():
            d = daily.loc[keep]
            for f in [*features, "total"]:
                s = ic_summary(d[f])
                rows.append({"period": period, "feature": f, "sage_pct": 100 * s["ic_mean"] / d["y2"].mean(),
                             "t": s["ic_t"]})
        return pd.DataFrame(rows)


    _s = {_k: sage_summary(results[_k]["sage_days.parquet"]) for _k in ("lgbm", "mlp")
          if results[_k]["sage_days.parquet"] is not None}
    if not _s:
        _out = mo.md("No LightGBM or MLP results yet.")
    else:
        _all = pd.concat([_v.assign(model=_k) for _k, _v in _s.items()])
        _table = _all.pivot_table(index="feature", columns=["model", "period"], values=["sage_pct", "t"], sort=False)
        _overall = _all[_all["period"] == "2019-2023"].assign(se=lambda d: d["sage_pct"] / d["t"])
        _overall = _overall.assign(lo=_overall["sage_pct"] - 1.96 * _overall["se"], hi=_overall["sage_pct"] + 1.96 * _overall["se"])
        _y = alt.Y("feature:N", sort=list(_overall["feature"].unique()), title=None)
        _chart = alt.layer(
            alt.Chart().mark_bar().encode(y=_y, x=alt.X("sage_pct:Q", title="SAGE, % of the mean squared target")),
            alt.Chart().mark_errorbar().encode(y=_y, x="lo:Q", x2="hi:Q"),
            data=_overall).properties(width=300, height=320).facet(column=alt.Column("model:N", title=None))
        _out = mo.vstack([_chart, _table.round(4)])
    _out
    return


if __name__ == "__main__":
    app.run()
