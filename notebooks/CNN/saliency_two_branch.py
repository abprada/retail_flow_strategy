import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saliency of the two-branch model: the chart alone, combined, fine-tuned with and without the chart's own loss

    **The networks** (`explanations/CNN/fusion.md`), all from fold 2019 (trained on 2014-2017, stopped on 2018), shown on charts of 2019, the year they predict:

    1. **The CNN alone** (step 1a, `price.pt`): the one-day grey CNN; its prediction ŷ_price.
    2. **CNN + MLP, before fine-tuning** (step 2: the CNN and the MLP frozen, the combined head `comb_lr0.0003_h16-8.pt` on top): the combined prediction ŷ. The chart reaches ŷ only through ŷ_price, one number, so this map is the CNN's map times the head's slope on ŷ_price, one number per chart.
    3. **CNN + MLP fine-tuned on the combined loss alone** (step 3 with λ = 0, `two_branch_day_cnn_lambda0`): ŷ.
    4. **CNN + MLP fine-tuned on the combined loss and the CNN's own loss** (step 3 with λ = 0.5, the run): ŷ.

    Step 3 trains block 3, the readout, the MLP's layer and the combined head; blocks 1-2 stay as step 1a left them. Section 4 looks inside the MLP heads: how much of the prediction comes from each branch, how the combined head responds to the chart, which features matter, and what the tabular layer's 16 units read.

    **Saliency** is the gradient of a prediction with respect to each pixel of the chart (0-255): how much the prediction would move if that pixel got a little more ink. **The image** (64 × 120): rows 0-50 price, row 51 empty, rows 52-63 volume; 40 bars of 3 columns: the previous day's last ten minutes, the night, then the decision day's ten-minute bars from 09:30 to 15:50. In the plots, a yellow line marks the open and grey lines the hours.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import torch  # no LightGBM here: after it, CPU work in torch crashes on macOS (two OpenMP runtimes)

    from dsr.cnn import FIRST_SESSION_SLOT, PRICE_ROWS, ChartCNN, GreyImages
    from dsr.evaluation import daily_rank_ic
    from dsr.two_branch import COMBINED, DAY_DROPOUT, DAY_FILTERS, TwoBranchNet, loss
    from dsr.walkforward import feature_columns

    return (COMBINED, ChartCNN, DAY_DROPOUT, DAY_FILTERS, FIRST_SESSION_SLOT, GreyImages, PRICE_ROWS, TwoBranchNet,
            daily_rank_ic, feature_columns, loss, mo, np, pd, plt, torch)


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    DAY_BARS = ROOT / "data" / "processed" / "cnn_bars_train.npz"
    PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
    RUN = ROOT / "data" / "walkforward" / "two_branch_day_cnn" / "checkpoints" / "2019"
    CONTROL = ROOT / "data" / "walkforward" / "two_branch_day_cnn_lambda0" / "checkpoints" / "2019"
    SETTING = "lr0.0003_h16-8"  # step 2's chosen setting
    DEVICE = "mps"
    STOCKS = 1500  # PERMNOs spread over all of them
    SAMPLE = 2000  # random 2019 stock-days for the averages
    BATCH = 256
    TRAIN_BATCHES = 40  # training-year batches of 512 for the gradient flow
    return BATCH, CONTROL, DAY_BARS, DEVICE, PANEL, RUN, SAMPLE, SETTING, STOCKS, TRAIN_BATCHES


@app.cell
def _(DAY_BARS, DEVICE, GreyImages, PANEL, STOCKS, feature_columns, np, pd, torch):
    def _load():
        """The large-cap stock-days of STOCKS PERMNOs with one-day drawing data and a panel row: keys, the scaled
        target, the features and the grey charts on the GPU."""
        with np.load(DAY_BARS) as z:
            keys = pd.DataFrame({"day": pd.to_datetime(z["day"]), "permno": z["permno"]})
            arrays = {k: z[k] for k in ("rows", "volume", "colors")}
        permnos = np.unique(keys["permno"])
        keys = keys[keys["permno"].isin(permnos[::max(1, len(permnos) // STOCKS)][:STOCKS])]
        panel = pd.read_parquet(PANEL)
        features = feature_columns(panel.columns)
        panel["day"] = pd.to_datetime(panel["day"])
        large = panel[panel["large_cap"].astype(bool)]
        train_years = large["day"].dt.year.between(2014, 2017) & large["target_demeaned"].notna()
        scale = float(large.loc[train_years, "target_demeaned"].std())  # the fold's target scale (its training years)
        keys = keys.reset_index(names="pos").merge(panel, on=["day", "permno"], how="inner", validate="one_to_one")
        keys = keys[keys["large_cap"].astype(bool)].sort_values(["day", "permno"]).reset_index(drop=True)
        charts = GreyImages(*(arrays[k][keys["pos"].to_numpy()] for k in ("rows", "volume", "colors")), DEVICE)
        feats = torch.as_tensor(keys[features].to_numpy(np.float32), device=DEVICE)
        y = torch.as_tensor(np.nan_to_num(keys["target_demeaned"].to_numpy(np.float32) / scale), device=DEVICE)
        return keys[["day", "permno", "ret_next", "target_demeaned"]], charts, feats, y, features


    keys, charts, feats, y, FEATURES = _load()
    return FEATURES, charts, feats, keys, y


@app.cell
def _(COMBINED, CONTROL, DEVICE, RUN, SETTING, TwoBranchNet, torch):
    def _net(states):
        net = TwoBranchNet(chart="day_cnn", hidden=COMBINED["16-8"])
        for part, state in states.items():
            (net if part == "all" else getattr(net, part)).load_state_dict(state)
        net = net.to(DEVICE).eval()  # eval: BatchNorm's stored statistics, no dropout
        for w in net.parameters():
            w.requires_grad_(False)
        return net


    _load = lambda path: torch.load(path, map_location="cpu")
    _before = {"price": _load(RUN / "price.pt")["state"], "tabular": _load(RUN / "tab.pt"),
               "combined": _load(RUN / f"comb_{SETTING}.pt")["state"]}
    nets = {"CNN alone": (_net(_before), "price"), "CNN + MLP, before fine-tuning": (_net(_before), "comb")}
    if (CONTROL / "stage3.pt").exists():
        nets["fine-tuned, combined loss only (λ = 0)"] = (_net({"all": _load(CONTROL / "stage3.pt")}), "comb")
    nets["fine-tuned, combined + CNN loss (λ = 0.5)"] = (_net({"all": _load(RUN / "stage3.pt")}), "comb")
    return (nets,)


@app.cell(hide_code=True)
def _(mo, nets):
    _missing = ("" if len(nets) == 4
                else " **The λ = 0 network is not there yet** (`stage3 --lam 0`); its column is left out.")
    mo.md(f"Networks loaded: {', '.join(nets)}.{_missing}")
    return


@app.cell
def _(BATCH, FIRST_SESSION_SLOT, PRICE_ROWS, charts, feats, np, torch):
    def saliency(net, output, idx):
        """The gradient of the network's `output` ("price" or "comb") with respect to every pixel of the charts idx
        (B x 64 x 120) and to the 17 features (B x 17). Positive: more ink on that pixel raises the prediction."""
        grads, fgrads = [], []
        for i in range(0, len(idx), BATCH):
            b = torch.as_tensor(np.asarray(idx[i:i + BATCH]), device=charts.device)
            img = charts.draw(b).clone().requires_grad_(True)
            tab = feats[b].clone().requires_grad_(True)
            net(img, tab)[output].sum().backward()
            grads.append(img.grad[:, 0].cpu().numpy())
            fgrads.append(tab.grad.cpu().numpy() if tab.grad is not None else np.zeros(tuple(tab.shape)))  # CNN alone
        return np.concatenate(grads), np.concatenate(fgrads)


    @torch.no_grad()
    def predict(net, output, idx):
        found = []
        for i in range(0, len(idx), BATCH):
            b = torch.as_tensor(np.asarray(idx[i:i + BATCH]), device=charts.device)
            found.append(net(charts.draw(b), feats[b])[output].cpu().numpy())
        return np.concatenate(found)


    OPEN = 3 * FIRST_SESSION_SLOT  # the first session column


    def mark(ax, scale=1, rows=True):
        """A yellow line at the open, grey lines at the hours (6 ten-minute bars), and one under the price rows."""
        ax.axvline(OPEN / scale - 0.5, color="yellow", lw=0.8)
        for h in range(1, 7):
            ax.axvline((OPEN + 18 * h) / scale - 0.5, color="grey", lw=0.3)
        if rows:
            ax.axhline(PRICE_ROWS + 0.5, color="grey", lw=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    return OPEN, mark, predict, saliency


@app.cell
def _(keys, nets, np, predict):
    test = np.flatnonzero(keys["day"].dt.year.to_numpy() == 2019)
    scored = keys.iloc[test].assign(
        **{name: predict(net, output, test) for name, (net, output) in nets.items()})
    return scored, test


@app.cell(hide_code=True)
def _(daily_rank_ic, mo, nets, pd, scored):
    _rows = [{"network": n, "output": o, "2019 rank IC (these stocks)": round(float(daily_rank_ic(
        scored["day"].to_numpy(), scored[n].to_numpy(), scored["ret_next"].to_numpy()).mean()), 4),
              "prediction sd": round(float(scored[n].std()), 4)} for n, (_, o) in nets.items()]
    mo.vstack([mo.md(f"**{len(scored):,} stock-days of 2019** ({scored['permno'].nunique():,} stocks). Each network's "
                     "prediction on them (in the target's units over its training sd):"),
               mo.ui.table(pd.DataFrame(_rows), selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One chart, four networks

    For the chart picked below: the chart, then each network's gradient (red: more ink raises the prediction; blue: lowers it). **Each map has its own color scale,** so the shapes compare; the gradient's total size, how strongly the prediction depends on the chart at all, is printed above each map and compared in section 2.

    Pick a chart: the five highest and five lowest predictions of the fine-tuned model (λ = 0.5), and five at random.
    """)
    return


@app.cell
def _(mo, nets, np, scored):
    _model = list(nets)[-1]
    _pct = scored.groupby("day")[_model].rank(pct=True)
    _order = _pct.sort_values().index
    _picks = ([("highest", i) for i in _order[::-1][:5]] + [("lowest", i) for i in _order[:5]]
              + [("random", i) for i in np.random.default_rng(0).choice(scored.index, 5, replace=False)])
    _label = lambda kind, i: f"{kind}: {scored.at[i, 'day']:%Y-%m-%d}, PERMNO {scored.at[i, 'permno']}"
    example = mo.ui.dropdown(options={_label(k, i): int(i) for k, i in _picks}, value=_label(*_picks[0]),
                             label="chart")
    example
    return (example,)


@app.cell
def _(charts, example, keys, mark, nets, np, plt, saliency, scored, torch):
    _i = example.value  # a position in keys
    with torch.no_grad():
        _img = charts.draw(torch.as_tensor([_i], device=charts.device))[0, 0].cpu().numpy()
    _fig, _axes = plt.subplots(len(nets) + 1, 1, figsize=(10, 2.3 * (len(nets) + 1)), layout="constrained")
    _axes[0].imshow(_img, cmap="gray", vmin=0, vmax=255, aspect="auto", interpolation="nearest")
    _axes[0].set_title(f"{keys.at[_i, 'day']:%Y-%m-%d}, PERMNO {keys.at[_i, 'permno']}: next-day return "
                       f"{keys.at[_i, 'ret_next'] * 100:.2f}%", fontsize=9)
    mark(_axes[0])
    for _ax, (_name, (_net, _out)) in zip(_axes[1:], nets.items()):
        _g = saliency(_net, _out, np.array([_i]))[0][0]
        _lim = np.quantile(np.abs(_g), 0.999) or 1
        _shown = _ax.imshow(_g, cmap="RdBu_r", vmin=-_lim, vmax=_lim, aspect="auto", interpolation="nearest")
        _ax.set_title(f"{_name}: prediction {scored.at[_i, _name]:.3f}, total |gradient| {np.abs(_g).sum():.4f}",
                      fontsize=9)
        _fig.colorbar(_shown, ax=_ax, fraction=0.02, pad=0.01)
        mark(_ax)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. How much of the prediction reaches the chart, and where

    Over random 2019 charts, for each network:

    - **The gradient's total size per chart** (the sum of its absolute values over the pixels): how strongly the prediction moves with the chart. Next to it, the same for the 17 features, times each feature's sd (the features are standardized, so about 1): how strongly it moves with them. Their ratio is the chart's share of the prediction's sensitivity.
    - **Where:** the absolute gradient, each chart's divided by its total so every chart counts the same, averaged: a heat map, and its sum by bar and by row.
    """)
    return


@app.cell
def _(SAMPLE, nets, np, saliency, test):
    sample = np.sort(np.random.default_rng(1).choice(test, min(SAMPLE, len(test)), replace=False))
    maps = {}
    for _name, (_net, _out) in nets.items():
        _g, _f = saliency(_net, _out, sample)
        _total = np.abs(_g).sum(axis=(1, 2))
        maps[_name] = {"where": (np.abs(_g) / _total[:, None, None]).mean(0), "chart": _total,
                       "features": np.abs(_f).sum(1)}
    return maps, sample


@app.cell
def _(maps, mo, np, pd):
    _rows = [{"network": n, "chart: total |gradient| per chart (median)": float(np.median(m["chart"])),
              "features: total |gradient| per chart (median)": float(np.median(m["features"])),
              "chart / features": float(np.median(m["chart"]) / np.median(m["features"]))
              if np.median(m["features"]) > 0 else float("nan")} for n, m in maps.items()]
    mo.ui.table(pd.DataFrame(_rows).round(4), selection=None)
    return


@app.cell
def _(PRICE_ROWS, maps, mark, np, plt):
    _fig, _axes = plt.subplots(len(maps), 2, figsize=(12, 2.4 * len(maps)), layout="constrained",
                               gridspec_kw={"width_ratios": [5, 1]}, squeeze=False)
    _vmax = max(m["where"].max() for m in maps.values())
    for (_ax, _side), (_name, _m) in zip(_axes, maps.items()):
        _shown = _ax.imshow(_m["where"], cmap="magma", vmin=0, vmax=_vmax, aspect="auto", interpolation="nearest")
        _ax.set_title(f"{_name}: mean absolute saliency (each chart's sums to 1)", fontsize=9)
        mark(_ax)
        _rows = _m["where"].sum(1)
        _side.barh(np.arange(len(_rows)), _rows, height=1, color="0.3")
        _side.invert_yaxis()
        _side.axhline(PRICE_ROWS + 0.5, color="grey", lw=0.5)
        _side.set_yticks([])
        _side.set_title(f"price {_rows[:PRICE_ROWS].sum():.0%}, volume {_rows[PRICE_ROWS + 1:].sum():.0%}", fontsize=8)
    _fig.colorbar(_shown, ax=_axes[:, 0].tolist(), fraction=0.02, pad=0.01)
    _fig
    return


@app.cell
def _(OPEN, maps, np, plt):
    _fig, _ax = plt.subplots(figsize=(12, 3), layout="constrained")
    for _name, _m in maps.items():
        _ax.plot(np.arange(40), _m["where"].sum(0).reshape(40, 3).sum(1), marker=".", label=_name)
    _ax.axvline(OPEN / 3 - 0.5, color="goldenrod", lw=0.8)
    _ax.set_xlabel("bar (0: the previous day's last ten minutes, 1: the night, 2-39: 09:30 to 15:50)", fontsize=8)
    _ax.set_ylabel("share of the saliency", fontsize=8)
    _ax.legend(fontsize=7)
    _ax.set_title("where in the day each network looks", fontsize=9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How much gradient each loss sends into the CNN

    Step 3 trains the CNN's block 3 and readout. With λ = 0 they learn only from the combined loss, whose gradient reaches the CNN through the combined head's slope on ŷ_price; with λ = 0.5 the CNN's own loss adds its gradient directly. Over training-year batches (2014-2017, the rows step 3 trains on), the norm of each loss's gradient at the CNN's trainable weights, at the start of fine-tuning (the step 2 network) and at the end of each fine-tuning. **The share of the CNN's own loss** is what the extra loss adds to the CNN's learning signal.
    """)
    return


@app.cell
def _(TRAIN_BATCHES, charts, feats, keys, loss, nets, np, pd, torch, y):
    _train = np.flatnonzero(keys["day"].dt.year.between(2014, 2017).to_numpy() & keys["target_demeaned"].notna())
    _batches = np.random.default_rng(2).choice(_train, (TRAIN_BATCHES, 512), replace=False)


    def _flow(net):
        params = [w for part in (net.price.one_d, net.price.head) for w in part.parameters()]
        for w in params:
            w.requires_grad_(True)
        norms = {"combined loss": 0.0, "CNN's own loss": 0.0}
        for b in _batches:
            b = torch.as_tensor(b, device=charts.device)
            out = net(charts.draw(b), feats[b])
            for name, head in (("combined loss", "comb"), ("CNN's own loss", "price")):
                g = torch.autograd.grad(loss(out[head], y[b]), params, retain_graph=True)
                norms[name] += float(torch.sqrt(sum((x ** 2).sum() for x in g))) / len(_batches)
        for w in params:
            w.requires_grad_(False)
        return norms


    _rows = []
    for _name, (_net, _out) in nets.items():
        if _out != "comb":
            continue
        _n = _flow(_net)
        _rows.append({"network": _name, **{f"|gradient| at block 3 + readout, {k}": v for k, v in _n.items()},
                      "share of the CNN's own loss": _n["CNN's own loss"] / sum(_n.values())})
    gradient_flow = pd.DataFrame(_rows)
    gradient_flow.round(5)
    return (gradient_flow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inside the MLP heads

    The combined head reads two things: ŷ_price (the CNN's prediction, one number) and s_t (the MLP layer's 16 numbers). On the same 2,000 charts of 2019, for each network that has a combined head, and for **the MLP alone** (step 1b's tabular head, ŷ_tab) as the reference:

    - **How much of ŷ comes from each branch:** ŷ again with ŷ_price replaced by its average (the chart taken out), and with s_t replaced by its average (the features taken out). The change's sd over ŷ's sd is the branch's share of the prediction's variation; the rank correlation of ŷ with the version without the chart says how much of the ranking survives without it.
    - **How the combined head responds to the chart:** ŷ averaged over the charts with ŷ_price set to each of its percentiles, the features as they are: the slope is how much the head trusts the chart.
    - **Which features matter:** each feature set to 0, the day's average (the features are standardized within each day), one at a time: the mean absolute change in ŷ over ŷ's sd. And each feature's average gradient (∂ŷ per sd of the feature): its direction.
    - **The tabular layer's 16 units:** what each reads from the 17 features (step 1b's weights, and after fine-tuning with λ = 0.5), the MLP's own head's weight on each, and how strongly the fine-tuned combined head moves with each (its gradient times the unit's sd).
    """)
    return


@app.cell
def _(BATCH, charts, feats, nets, np, sample, torch):
    @torch.no_grad()
    def heads(net, idx):
        """The network's two summaries and its predictions for the stock-days idx."""
        found = []
        for i in range(0, len(idx), BATCH):
            b = torch.as_tensor(np.asarray(idx[i:i + BATCH]), device=charts.device)
            found.append(net(charts.draw(b), feats[b]))
        return {k: torch.cat([f[k] for f in found]) for k in found[0]}


    combined_nets = {n: net for n, (net, out) in nets.items() if out == "comb"}
    inside_heads = {n: heads(net, sample) for n, net in combined_nets.items()}
    sample_feats = feats[torch.as_tensor(sample, device=charts.device)]
    return combined_nets, inside_heads, sample_feats


@app.cell
def _(combined_nets, inside_heads, mo, pd, torch):
    def _rank_corr(a, b):
        return float(pd.Series(a.cpu().numpy()).rank().corr(pd.Series(b.cpu().numpy()).rank()))


    _rows = []
    with torch.no_grad():
        for _name, _net in combined_nets.items():
            _o = inside_heads[_name]
            _comb = _o["comb"]
            _no_chart = _net.combined(_o["s_p"].mean(0, keepdim=True).expand_as(_o["s_p"]), _o["s_t"])
            _no_tab = _net.combined(_o["s_p"], _o["s_t"].mean(0, keepdim=True).expand_as(_o["s_t"]))
            _rows.append({"network": _name,
                          "chart's share of ŷ's variation": float((_comb - _no_chart).std() / _comb.std()),
                          "features' share of ŷ's variation": float((_comb - _no_tab).std() / _comb.std()),
                          "rank correlation of ŷ with ŷ without the chart": _rank_corr(_comb, _no_chart),
                          "rank correlation of ŷ with ŷ_price": _rank_corr(_comb, _o["price"]),
                          "rank correlation of ŷ with ŷ_tab": _rank_corr(_comb, _o["tab"])})
    mo.ui.table(pd.DataFrame(_rows).round(3), selection=None)
    return


@app.cell
def _(combined_nets, inside_heads, np, plt, torch):
    _fig, _ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    _q = np.linspace(0.01, 0.99, 25)
    with torch.no_grad():
        for _name, _net in combined_nets.items():
            _o = inside_heads[_name]
            _p = _o["s_p"][:, 0]
            _levels = torch.quantile(_p, torch.as_tensor(_q, device=_p.device, dtype=_p.dtype))
            _curve = torch.stack([_net.combined(torch.full_like(_o["s_p"], float(v)), _o["s_t"]).mean()
                                  for v in _levels])
            _x = ((_levels - _p.mean()) / _p.std()).cpu().numpy()
            _ax.plot(_x, ((_curve - _o["comb"].mean()) / _o["comb"].std()).cpu().numpy(), marker=".", label=_name)
    _ax.axhline(0, color="grey", lw=0.5)
    _ax.set_xlabel("ŷ_price, in its sds from its mean (its 1st to 99th percentile)", fontsize=8)
    _ax.set_ylabel("average ŷ, in its sds from its mean", fontsize=8)
    _ax.set_title("how the combined head responds to the chart (features as they are)", fontsize=9)
    _ax.legend(fontsize=7)
    _fig
    return


@app.cell
def _(FEATURES, combined_nets, inside_heads, nets, np, pd, plt, sample_feats, torch):
    def _attribution(predict):
        """Each feature set to 0 (the day's average), one at a time: the mean absolute change in the prediction over
        its sd; and the average gradient of the prediction with respect to the feature."""
        x = sample_feats.clone().requires_grad_(True)
        base = predict(x)
        grad = torch.autograd.grad(base.sum(), x)[0].mean(0)
        base = base.detach()
        with torch.no_grad():
            removed = []
            for j in range(x.shape[1]):
                x0 = sample_feats.clone()
                x0[:, j] = 0
                removed.append(float((predict(x0) - base).abs().mean() / base.std()))
        return np.array(removed), grad.cpu().numpy()


    _mlp = list(nets.values())[1][0]  # the before-fine-tuning network holds step 1b's tabular branch
    _found = {"MLP alone (ŷ_tab)": _attribution(lambda x: _mlp.tabular(x)[1])}
    for _name, _net in combined_nets.items():
        _s_p = inside_heads[_name]["s_p"]
        _found[_name] = _attribution(lambda x, n=_net, s=_s_p: n.combined(s, n.tabular(x)[0]))
    feature_effects = pd.DataFrame({n: r for n, (r, _) in _found.items()}, index=FEATURES)
    feature_direction = pd.DataFrame({n: g for n, (_, g) in _found.items()}, index=FEATURES)
    _order = feature_effects.iloc[:, 0].sort_values().index
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained", sharey=True)
    feature_effects.loc[_order].plot.barh(ax=_a1, width=0.8, legend=False)
    _a1.set_title("taking the feature out: mean |change in the prediction| / its sd", fontsize=9)
    (feature_direction.loc[_order] / feature_direction.abs().max()).plot.barh(ax=_a2, width=0.8)
    _a2.axvline(0, color="grey", lw=0.5)
    _a2.set_title("direction: average gradient (each network scaled to its largest)", fontsize=9)
    _a2.legend(fontsize=7, loc="lower right")
    for _ax in (_a1, _a2):
        _ax.tick_params(labelsize=7)
    _fig
    return feature_direction, feature_effects


@app.cell
def _(FEATURES, combined_nets, inside_heads, nets, np, plt, torch):
    _before = list(nets.values())[1][0]
    _after = list(combined_nets.values())[-1]
    _w = {"step 1b": _before.tabular.body[0].weight.detach().cpu().numpy(),
          "after fine-tuning (λ = 0.5)": _after.tabular.body[0].weight.detach().cpu().numpy()}
    _head = _before.tabular.head[1].weight.detach().cpu().numpy()[0]
    _o = inside_heads[list(combined_nets)[-1]]
    _s_t = _o["s_t"].clone().requires_grad_(True)
    with torch.enable_grad():
        _g = torch.autograd.grad(_after.combined(_o["s_p"], _s_t).sum(), _s_t)[0]
    _use = (_g.abs().mean(0) * _o["s_t"].std(0)).cpu().numpy()  # how strongly ŷ moves with each unit
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 5), layout="constrained", gridspec_kw={"width_ratios": [5, 5, 1, 1]})
    _lim = max(np.abs(w).max() for w in _w.values())
    for _ax, (_title, _m) in zip(_axes[:2], _w.items()):
        _shown = _ax.imshow(_m, cmap="RdBu_r", vmin=-_lim, vmax=_lim, aspect="auto")
        _ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=6)
        _ax.set_yticks(range(_m.shape[0]), [f"unit {_k}" for _k in range(_m.shape[0])], fontsize=6)
        _ax.set_title(f"the tabular layer's weights, {_title}", fontsize=8)
    _fig.colorbar(_shown, ax=_axes[:2].tolist(), fraction=0.02, pad=0.01)
    _axes[2].barh(np.arange(len(_head)), _head, color=np.where(_head > 0, "firebrick", "steelblue"))
    _axes[2].set_title("MLP head's\nweight", fontsize=8)
    _axes[3].barh(np.arange(len(_use)), _use, color="0.3")
    _axes[3].set_title("combined head\nmoves with it", fontsize=8)
    for _ax in _axes[2:]:
        _ax.invert_yaxis()
        _ax.set_yticks([])
        _ax.tick_params(labelsize=6)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The CNN's last block and readout, before and after fine-tuning

    Blocks 1-2 are the same in every network (step 3 leaves them as step 1a trained them), and the CNN alone and the network before fine-tuning share one CNN, so there are three CNNs: step 1a's, the one fine-tuned on the combined loss alone (λ = 0) and the one fine-tuned with the CNN's own loss too (λ = 0.5). Their block 3 gives 128 maps of 3 rows (the top, middle and bottom of the image: the day's high, its low, and the volume) × 120 columns; the readout weighs each of those 46,080 values and sums them into ŷ_price.

    - **The readout's evidence:** each block 3 value times its readout weight, summed over the 128 maps: a 3 × 120 map that adds up (with the bias) to ŷ_price. Red pushes the prediction up, blue down.
    - **Block 3's maps:** the six that contribute most to this chart's prediction (by the absolute sum of weight × value), each with its own color scale.
    - **Over the 2,000 charts:** the mean absolute evidence by place, and by bar.
    - **How much fine-tuning rewired the last block:** block 3's and the readout's weights against step 1a's (the change's size over the weights'), and how closely each of block 3's 128 maps still moves with step 1a's over 512 charts (the correlation of their values, averaged over the maps).
    """)
    return


@app.cell
def _(BATCH, charts, nets, np, torch):
    cnns = {"step 1a (the CNN alone; also before fine-tuning)": nets["CNN alone"][0].price}
    for _name, (_net, _out) in list(nets.items())[2:]:
        cnns[_name.replace("fine-tuned, ", "fine-tuned: ")] = _net.price


    @torch.no_grad()
    def last_block(price, idx):
        """Block 3's output for the charts idx (B x 128 x 3 x 120) and the readout's weights as 128 x 3 x 120."""
        b = torch.as_tensor(np.asarray(idx), device=charts.device)
        a = price.one_d(price.blocks((charts.draw(b) - price.mean) / price.sd))
        return a, price.head[2].weight.view(a.shape[1:])


    def evidence_stats(price, idx):
        """Over the charts idx: the mean absolute evidence (weight x value summed over the maps) by place, 3 x 120."""
        total = 0
        for i in range(0, len(idx), BATCH):
            a, w = last_block(price, idx[i:i + BATCH])
            total = total + (a * w).sum(1).abs().sum(0)
        return (total / len(idx)).cpu().numpy()

    return cnns, evidence_stats, last_block


@app.cell
def _(cnns, example, last_block, mark, np, plt):
    _n_maps = 6
    _fig, _axes = plt.subplots(len(cnns) * (_n_maps + 1), 1, figsize=(10, 0.55 * len(cnns) * (_n_maps + 1) + 1),
                               layout="constrained")
    _k = 0
    for _name, _price in cnns.items():
        _a, _w = last_block(_price, [example.value])
        _a, _w = _a[0].cpu().numpy(), _w.cpu().numpy()
        _ev = (_a * _w).sum(0)
        _lim = np.abs(_ev).max() or 1
        _axes[_k].imshow(_ev, cmap="RdBu_r", vmin=-_lim, vmax=_lim, aspect="auto", interpolation="nearest")
        _axes[_k].set_title(f"{_name}: the readout's evidence (sum {_ev.sum():+.3f})", fontsize=8)
        mark(_axes[_k], rows=False)
        _top = np.abs((_a * _w).sum((1, 2))).argsort()[::-1][:_n_maps]
        for _j, _c in enumerate(_top, start=1):
            _axes[_k + _j].imshow(_a[_c], cmap="magma", aspect="auto", interpolation="nearest")
            _axes[_k + _j].set_ylabel(f"map {_c}", fontsize=6, rotation=0, labelpad=14)
            mark(_axes[_k + _j], rows=False)
        _k += _n_maps + 1
    _fig.suptitle("the chart picked in section 1: block 3's evidence, then its six most important maps (3 rows x 120)",
                  fontsize=9)
    _fig
    return


@app.cell
def _(OPEN, cnns, evidence_stats, mark, np, plt, sample):
    _ev = {n: evidence_stats(p, sample) for n, p in cnns.items()}
    _fig, _axes = plt.subplots(len(_ev) + 1, 1, figsize=(10, 1.3 * len(_ev) + 3), layout="constrained",
                               gridspec_kw={"height_ratios": [1] * len(_ev) + [2.5]})
    _vmax = max(e.max() for e in _ev.values())
    for _ax, (_name, _e) in zip(_axes, _ev.items()):
        _ax.imshow(_e, cmap="magma", vmin=0, vmax=_vmax, aspect="auto", interpolation="nearest")
        _ax.set_title(f"{_name}: mean absolute evidence by place (rows: high, low, volume)", fontsize=8)
        mark(_ax, rows=False)
    for _name, _e in _ev.items():
        _by_bar = _e.sum(0).reshape(40, 3).sum(1)
        _axes[-1].plot(np.arange(40), _by_bar / _by_bar.sum(), marker=".", label=_name)
    _axes[-1].axvline(OPEN / 3 - 0.5, color="goldenrod", lw=0.8)
    _axes[-1].set_xlabel("bar (0: the previous day's last ten minutes, 1: the night, 2-39: 09:30 to 15:50)", fontsize=8)
    _axes[-1].set_ylabel("share of the evidence", fontsize=8)
    _axes[-1].legend(fontsize=7)
    _fig
    return


@app.cell
def _(BATCH, cnns, last_block, mo, np, pd, sample, torch):
    _names = list(cnns)
    _base = cnns[_names[0]]
    _sub = sample[:512]
    _rows = []


    def _change(m0, m1):
        """The size of the weights' change from m0 to m1 over the size of m0's weights."""
        diff = sum(((a - b) ** 2).sum() for a, b in zip(m0.parameters(), m1.parameters()))
        return float(torch.sqrt(diff / sum((a ** 2).sum() for a in m0.parameters())))


    for _name in _names[1:]:
        _price = cnns[_name]
        _sums = None
        for _i in range(0, len(_sub), BATCH):
            _a0, _ = last_block(_base, _sub[_i:_i + BATCH])
            _a1, _ = last_block(_price, _sub[_i:_i + BATCH])
            _a0, _a1 = (t.transpose(0, 1).flatten(1) for t in (_a0, _a1))  # 128 maps x values
            _part = torch.stack([_a0.sum(1), _a1.sum(1), (_a0 ** 2).sum(1), (_a1 ** 2).sum(1), (_a0 * _a1).sum(1),
                                 torch.full_like(_a0[:, 0], _a0.shape[1])])
            _sums = _part if _sums is None else _sums + _part
        _s0, _s1, _s00, _s11, _s01, _n = (v / _sums[5] for v in _sums)  # means over the values
        _corr = (_s01 - _s0 * _s1) / torch.sqrt((_s00 - _s0 ** 2) * (_s11 - _s1 ** 2))
        _rows.append({"network": _name, "block 3 weights: change / size": _change(_base.one_d, _price.one_d),
                      "readout weights: change / size": _change(_base.head, _price.head),
                      "block 3 maps: mean correlation with step 1a's": float(torch.nanmean(_corr)),
                      "block 3 maps: lowest correlation": float(_corr[torch.isfinite(_corr)].min())})
    mo.ui.table(pd.DataFrame(_rows).round(3), selection=None) if _rows else mo.md("Only step 1a's CNN is loaded.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The kernels

    Each block's convolution has one 5 × 3 kernel (5 rows × 3 columns, one bar's width) for each filter and input channel: block 1 has 32 filters on the grey chart (32 kernels; their rows are 2 pixels apart, so a kernel spans 9 rows, and it moves 3 rows at a time), block 2 has 64 on block 1's 32 maps (2,048 kernels), and block 3 has 128 on block 2's 64 maps (8,192). Blocks 1-2 are step 1a's in every network; block 3 and the readout are also fine-tuned in step 3.

    - **The start:** `dsr.cnn.fit` seeds torch with 0 before it builds the network, so building it the same way here gives the random (Xavier) weights each kernel started from.
    - **The colors:** red is a positive weight (ink, or activity in the input map, at that place raises the filter's output), blue a negative one. BatchNorm after each convolution rescales each filter's output, so a filter's overall size doesn't matter, only its pattern. Each filter is drawn on its own scale (its start's largest absolute weight), the same for its start, its trained kernels and their change, so the change shows in proportion.
    - **The same in all 5 rows:** the part of a kernel that is a vertical stripe responds to which of a bar's 3 columns (the open tick, the high-low line, the close tick) has ink, whatever the price level. Random kernels have 3/15 = 0.20 of their sum of squares there.
    """)
    return


@app.cell
def _(ChartCNN, DAY_DROPOUT, DAY_FILTERS, cnns, np, torch):
    def weights(price):
        """A CNN's three convolutions' weights (filters x inputs x 5 x 3) and its readout's (1 x 46,080), as arrays."""
        convs = [m.weight for m in price.modules() if isinstance(m, torch.nn.Conv2d)]
        return [w.detach().cpu().numpy() for w in (*convs, price.head[-1].weight)]


    def mosaic(w, scale):
        """Kernels w (filters x inputs x 5 x 3) as one image: a row of tiles per filter, a column per input channel, each
        divided by its filter's scale, with blank lines between the tiles."""
        o, i = w.shape[:2]
        tiles = np.full((o, i, 6, 4), np.nan)
        tiles[:, :, :5, :3] = w / scale[:, None, None, None]
        return tiles.transpose(0, 2, 1, 3).reshape(o * 6, i * 4)[:-1, :-1]


    torch.manual_seed(0)  # as dsr.cnn.fit before it builds the network
    start = weights(ChartCNN(DAY_FILTERS, DAY_DROPOUT, channels=1))
    trained = {name: weights(price) for name, price in cnns.items()}
    kernel_scale = [np.abs(w).reshape(len(w), -1).max(1) for w in start[:3]]  # each filter's start's largest weight
    return kernel_scale, mosaic, start, trained


@app.cell
def _(mo, np, pd, start, trained):
    def _rel(a, b):
        """The size of a - b over the size of b."""
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))


    def _stripes(d):
        """The share of kernels' sum of squares that is the same in all 5 rows."""
        return float(5 * (d.mean(2) ** 2).sum() / (d ** 2).sum())


    _names = list(trained)
    _first = trained[_names[0]]
    _rows = []
    for _name in _names:
        for _k, _layer in enumerate(["block 1", "block 2", "block 3", "readout"]):
            if _name != _names[0] and _k < 2:
                continue  # blocks 1-2 are step 1a's in every network
            _w, _s = trained[_name][_k], start[_k]
            _later = _name != _names[0]
            _rows.append({
                "network": _name, "layer": _layer, "weights": _w.size,
                "change from the start / start's size": _rel(_w, _s),
                "correlation with the start": float(np.corrcoef(_w.ravel(), _s.ravel())[0, 1]),
                "change from the start: same in all 5 rows": _stripes(_w - _s) if _k < 3 else None,
                "change since step 1a / step 1a's size": _rel(_w, _first[_k]) if _later else None,
                "change since step 1a: same in all 5 rows": _stripes(_w - _first[_k]) if _later and _k < 3 else None})
    mo.ui.table(pd.DataFrame(_rows).round(3), selection=None)
    return


@app.cell
def _(kernel_scale, mosaic, np, plt, start, trained):
    _s = start[0][:, 0] / kernel_scale[0][:, None, None]
    _t = next(iter(trained.values()))[0][:, 0] / kernel_scale[0][:, None, None]
    _fig, _ax = plt.subplots(figsize=(12, 2.6), layout="constrained")
    _ax.imshow(mosaic(np.stack([_s, _t, _t - _s]), np.ones(3)), cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest")
    _ax.set_yticks([2, 8, 14], ["start", "trained", "change"], fontsize=8)
    _ax.set_xticks(1 + 4 * np.arange(32), range(32), fontsize=6)
    _ax.set_xlabel("block 1's filters (the same in every network)", fontsize=8)
    _ax.set_title("block 1: the 32 kernels on the chart (5 rows, 2 pixels apart, × 3 columns)", fontsize=9)
    _fig
    return


@app.cell
def _(kernel_scale, mosaic, plt, start, trained):
    _t = next(iter(trained.values()))
    _panels = [(f"block {_k + 1}: {_what}", _w, _k) for _k in (1, 2)
               for _what, _w in (("trained", _t[_k]), ("change from the start", _t[_k] - start[_k]))]
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 10), layout="constrained")
    for _ax, (_title, _w, _k) in zip(_axes, _panels):
        _ax.imshow(mosaic(_w, kernel_scale[_k]), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        _ax.set_title(_title, fontsize=8)
        _ax.set_xlabel(f"input channel (0-{_w.shape[1] - 1})", fontsize=7)
        _ax.set_ylabel(f"filter (0-{_w.shape[0] - 1})", fontsize=7)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.suptitle("blocks 2 and 3 of step 1a, every kernel (zoom in below)", fontsize=9)
    _fig
    return


@app.cell
def _(mo):
    zoom_block = mo.ui.dropdown(["block 2", "block 3"], value="block 3", label="block")
    zoom_filter = mo.ui.slider(0, 112, step=16, value=0, label="filters from")
    zoom_input = mo.ui.slider(0, 48, step=16, value=0, label="input channels from")
    mo.hstack([zoom_block, zoom_filter, zoom_input], justify="start")
    return zoom_block, zoom_filter, zoom_input


@app.cell
def _(kernel_scale, mosaic, np, plt, start, trained, zoom_block, zoom_filter, zoom_input):
    _k = int(zoom_block.value[-1]) - 1
    _names = list(trained)
    _first = trained[_names[0]][_k]
    _f = min(zoom_filter.value, _first.shape[0] - 16)
    _i = min(zoom_input.value, _first.shape[1] - 16)
    _window = (slice(_f, _f + 16), slice(_i, _i + 16))
    _panels = [("start", start[_k]), ("trained (step 1a)", _first), ("change from the start", _first - start[_k])]
    if _k == 2:
        _panels += [(f"{_n}:\nchange since step 1a", trained[_n][_k] - _first) for _n in _names[1:]]
    _fig, _axes = plt.subplots(1, len(_panels), figsize=(3 * len(_panels), 5), layout="constrained")
    for _ax, (_title, _w) in zip(np.atleast_1d(_axes), _panels):
        _ax.imshow(mosaic(_w[_window], kernel_scale[_k][_window[0]]), cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
        _ax.set_title(_title, fontsize=8)
        _ax.set_xticks(1 + 4 * np.arange(16), range(_i, _i + 16), fontsize=5)
        _ax.set_yticks(2 + 6 * np.arange(16), range(_f, _f + 16), fontsize=5)
        _ax.set_xlabel("input channel", fontsize=7)
    np.atleast_1d(_axes)[0].set_ylabel("filter", fontsize=7)
    _fig.suptitle(f"{zoom_block.value}: filters {_f}-{_f + 15} × input channels {_i}-{_i + 15}", fontsize=9)
    _fig
    return


if __name__ == "__main__":
    app.run()
