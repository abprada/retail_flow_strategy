import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inside the chart CNN: saliency and feature maps

    **The network:** the one-day chart CNN of the final two-branch model (`dsr.two_branch.PriceBranch`, chart `day_cnn`, `explanations/CNN/fusion.md`) as step 1a left it in fold 2019 (`data/walkforward/two_branch_day_cnn/checkpoints/2019/price.pt`): trained on 2014-2017's large caps, every day, and kept at its best epoch by the 2018 large caps' rank IC (both shown below). It is the CNN alone, before the tabular branch and the fine-tuning; `saliency_two_branch.py` compares it with the fused model.

    **The image** (grey channel, 64 × 120 pixels, 0-255): 40 bars of 3 columns each (open tick, high-low bar, close tick), oldest on the left:
    - bar 0: the previous day's last 10 minutes (columns 0-2);
    - bar 1: the night, the previous evening and the pre-open (columns 3-5);
    - bars 2-39: the decision day's 10-minute bars from 09:30 to 15:50 (columns 6-119).

    Rows 0-50 hold price (the day's highest high at the top, its lowest low at the bottom), row 51 is empty, rows 52-63 hold volume.

    **Stock-days:** 2019's large caps (`cap_prev` ≥ $1B) of a sample of stocks spread over all PERMNOs. 2019 is the year this network predicts: it was never trained or stopped on it.

    In every plot, darker grey lines mark the night, thin grey lines the hours of the session, and the grey line across the price rows' end.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    from dsr.cnn import FIRST_SESSION_SLOT, HEIGHT, N_SLOTS, PRICE_ROWS, WIDTH, GreyImages
    from dsr.two_branch import COMBINED, TwoBranchNet
    from dsr.walkforward import feature_columns

    return (
        COMBINED,
        FIRST_SESSION_SLOT,
        GreyImages,
        HEIGHT,
        N_SLOTS,
        PRICE_ROWS,
        TwoBranchNet,
        WIDTH,
        feature_columns,
        mo,
        np,
        pd,
        plt,
        torch,
    )


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    BARS = ROOT / "data" / "processed" / "cnn_bars_train.npz"
    PANEL = ROOT / "data" / "processed" / "model_panel_train.parquet"
    CHECKPOINT = ROOT / "data" / "walkforward" / "two_branch_day_cnn" / "checkpoints" / "2019" / "price.pt"
    DEVICE = "mps"
    YEAR = 2019  # the stock-days shown: the year the network predicts
    STOCKS = 1500  # PERMNOs spread over all of them
    SAMPLE = 2000  # random stock-days for the averages
    OCCLUDE_SAMPLE = 500  # random stock-days for the occlusion test
    BATCH = 256
    RETURNS = ["ret_open_1450", "ret_prev", "ret_2_5", "ret_6_20", "vol_20"]  # model features to compare with
    return (
        BARS,
        BATCH,
        CHECKPOINT,
        DEVICE,
        OCCLUDE_SAMPLE,
        PANEL,
        RETURNS,
        SAMPLE,
        STOCKS,
        YEAR,
    )


@app.cell
def _(BARS, DEVICE, GreyImages, PANEL, RETURNS, STOCKS, YEAR, feature_columns, np, pd):
    def _load():
        """YEAR's large-cap stock-days of STOCKS PERMNOs (keys, target, return) and their one-day charts on the GPU."""
        with np.load(BARS) as z:
            stored = pd.DataFrame({"day": pd.to_datetime(z["day"]), "permno": z["permno"]})
            arrays = {k: z[k] for k in ("rows", "volume", "colors")}
        permnos = np.unique(stored["permno"])
        stored = stored[stored["permno"].isin(permnos[::max(1, len(permnos) // STOCKS)][:STOCKS])
                        & (stored["day"].dt.year == YEAR)]
        panel = pd.read_parquet(PANEL, columns=["day", "permno", "large_cap", "ret_next", "target_demeaned", *RETURNS])
        panel["day"] = pd.to_datetime(panel["day"])
        keys = stored.reset_index(names="pos").merge(panel, on=["day", "permno"], how="inner", validate="one_to_one")
        keys = keys[keys["large_cap"].astype(bool)].sort_values(["day", "permno"]).reset_index(drop=True)
        charts = GreyImages(*(arrays[k][keys["pos"].to_numpy()] for k in ("rows", "volume", "colors")), DEVICE)
        return keys, charts


    keys, charts = _load()
    return charts, keys


@app.cell
def _(CHECKPOINT, COMBINED, DEVICE, TwoBranchNet, torch):
    _ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = TwoBranchNet(chart="day_cnn", hidden=COMBINED["16-8"]).price
    net.load_state_dict(_ckpt["state"])
    trained = {"epochs": _ckpt["epochs"], "holdout_ic": _ckpt["score"]}
    torch.manual_seed(0)  # as the run built its network before step 1a: the weights training started from
    untrained = TwoBranchNet(chart="day_cnn", hidden=COMBINED["16-8"]).price
    untrained.mean.copy_(net.mean)  # the same pixel standardization
    untrained.sd.copy_(net.sd)
    for _model in (net, untrained):
        _model.to(DEVICE).eval()  # eval: BatchNorm uses stored statistics, so each chart is processed alone
        for _w in _model.parameters():
            _w.requires_grad_(False)
    return net, trained, untrained


@app.cell
def _(BATCH, charts, net, np, torch):
    def draw(idx):
        """The grey charts of the stock-days idx (positions in keys): B x 1 x 64 x 120, pixels 0-255."""
        return charts.draw(torch.as_tensor(np.asarray(idx), device=charts.device))


    @torch.no_grad()
    def predict(idx, model=net):
        """The CNN's predictions (in the target's units over its training sd)."""
        return np.concatenate([model(draw(idx[i:i + BATCH]))[1].cpu().numpy() for i in range(0, len(idx), BATCH)])


    def saliency(idx):
        """The gradient of the prediction with respect to every pixel of the charts idx: B x 64 x 120. Positive: more
        ink on that pixel raises the prediction."""
        found = []
        for i in range(0, len(idx), BATCH):
            img = draw(idx[i:i + BATCH]).clone().requires_grad_(True)
            net(img)[1].sum().backward()
            found.append(img.grad[:, 0].cpu().numpy())
        return np.concatenate(found)

    return draw, predict, saliency


@app.cell
def _(BATCH, draw, keys, np, torch, untrained):
    # the untrained network's BatchNorm statistics, measured on these charts (it has none of its own: in training,
    # BatchNorm used each batch's)
    for _m in untrained.modules():
        if isinstance(_m, torch.nn.BatchNorm2d):
            _m.reset_running_stats()
            _m.momentum = None  # a plain average over the batches
    untrained.train()
    _idx = np.sort(np.random.default_rng(9).choice(len(keys), 4096, replace=False))
    with torch.no_grad():
        for _i in range(0, len(_idx), BATCH):
            untrained(draw(_idx[_i:_i + BATCH]))
    untrained.eval()
    calibrated = True
    return


@app.cell
def _(RETURNS, keys, np, pd, predict, untrained):
    _all = np.arange(len(keys))
    scored = keys.assign(pred=predict(_all), untrained=predict(_all, untrained))
    _by_day = scored.groupby("day")
    scored["pct"] = _by_day["pred"].rank(pct=True)  # the prediction's percentile within its day
    scored["pct_untrained"] = _by_day["untrained"].rank(pct=True)


    def _daily_corr(a, b):
        """The mean over days of the rank correlation of columns a and b within the day."""
        if a == b:
            return pd.Series(1.0)
        ranks = scored[[a, b]].groupby(scored["day"]).rank()
        return ranks.groupby(scored["day"]).corr().xs(a, level=1)[b]


    daily_ic = _daily_corr("pred", "target_demeaned")
    correlations = pd.DataFrame({
        "with": ["next-day return (target_demeaned)", "the network before training", *RETURNS],
        "trained": [_daily_corr("pred", c).mean() for c in ("target_demeaned", "untrained", *RETURNS)],
        "before training": [_daily_corr("untrained", c).mean() for c in ("target_demeaned", "untrained", *RETURNS)],
    }).round(3)
    return correlations, daily_ic, scored


@app.cell(hide_code=True)
def _(daily_ic, mo, np, scored, trained):
    mo.md(f"""
    **{len(scored):,} stock-days** ({scored['permno'].nunique():,} stocks, about {scored.groupby('day').size().mean():.0f} a day). The network: best epoch {trained['epochs']}, 2018 hold-out rank IC {trained['holdout_ic']:.4f}. On these stock-days its mean daily rank IC is **{daily_ic.mean():.4f}** (t {daily_ic.mean() / daily_ic.std() * np.sqrt(len(daily_ic)):.2f}), and its predictions have sd {scored['pred'].std():.3f} (the target's sd is 1 in these units).
    """)
    return


@app.cell
def _(FIRST_SESSION_SLOT, N_SLOTS, PRICE_ROWS, np):
    HOUR_BARS = 6  # session bars (10 minutes) per hour group of the plots
    _groups = [range(s, min(s + HOUR_BARS, N_SLOTS - FIRST_SESSION_SLOT)) for s in
               range(0, N_SLOTS - FIRST_SESSION_SLOT, HOUR_BARS)]
    _clock = lambda b: f"{9 + (30 + 10 * b) // 60:02d}:{(30 + 10 * b) % 60:02d}"
    SEGMENTS = ["prev last 10 min", "night", *[f"{_clock(g[0])}" for g in _groups]]
    SEGMENT_COLS = ([range(0, 3), range(3, 6)]
                    + [range(3 * (FIRST_SESSION_SLOT + g[0]), 3 * (FIRST_SESSION_SLOT + g[-1] + 1)) for g in _groups])
    NIGHT_COL = 3  # the night's first column
    HOUR_COLS = [c.start for c in SEGMENT_COLS[3:]]  # the session's hour groups after the first


    def mark(ax, scale=1, rows=True):
        """Lines between the session's hours (thin grey), around the night (darker grey) on an image's columns divided by
        `scale`, and under the price rows."""
        for c in HOUR_COLS:
            ax.axvline(c / scale - 0.5, color="grey", lw=0.3)
        for c in (NIGHT_COL, NIGHT_COL + 3):
            ax.axvline(c / scale - 0.5, color="0.5", lw=0.6)
        if rows:
            ax.axhline(PRICE_ROWS, color="grey", lw=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    return NIGHT_COL, SEGMENTS, SEGMENT_COLS, mark


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The average chart of the highest and lowest predictions

    **What it shows:** the mean of the charts in the top tenth of each day's predictions, the mean of those in the bottom tenth, and the difference (top minus bottom). Light pixels have ink more often.

    **What to look at:** where the two differ. Red in the difference means the top tenth has ink there more often: for example, prices at the bottom of the day's range at the close would mean the network ranks stocks that fell during the day high (a reversal bet); volume rows would mean it reacts to trading activity.

    **The control:** the same picture for the network as it was before training: its starting weights, rebuilt from the run's seed, with BatchNorm statistics measured on these charts. Any network ranks charts by some shape, so the difference is never blank; what training added is the change from the control to the trained picture. The table below gives the mean daily rank correlation of each network's predictions with the next-day return, with each other and with the return features of the tabular models.
    """)
    return


@app.cell
def _(SAMPLE, draw, mark, np, plt, scored, torch):
    def _mean_image(mask, seed):
        idx = np.flatnonzero(mask)
        idx = np.sort(np.random.default_rng(seed).choice(idx, min(SAMPLE, len(idx)), replace=False))
        total = 0
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                total = total + draw(idx[i:i + 256])[:, 0].sum(0)
        return (total / len(idx)).cpu().numpy()


    _top, _bottom = _mean_image(scored["pct"] > 0.9, 0), _mean_image(scored["pct"] <= 0.1, 1)
    _diffs = {"trained network: top minus bottom (red: ink more often in the top tenth)": _top - _bottom,
              "before training (the control): top minus bottom":
                  _mean_image(scored["pct_untrained"] > 0.9, 2) - _mean_image(scored["pct_untrained"] <= 0.1, 3)}
    _fig, _axes = plt.subplots(4, 1, figsize=(12, 9), layout="constrained")
    for _ax, _img, _title in zip(_axes[:2], (_top, _bottom), ("top tenth of each day's predictions",
                                                             "bottom tenth")):
        _ax.imshow(_img, cmap="gray", vmin=0, vmax=255, aspect="auto", interpolation="nearest")
        _ax.set_title(f"trained network, mean chart, {_title}", fontsize=9)
        mark(_ax)
    _lim = max(np.abs(_d).max() for _d in _diffs.values())
    for _ax, (_title, _d) in zip(_axes[2:], _diffs.items()):
        _shown = _ax.imshow(_d, cmap="RdBu_r", vmin=-_lim, vmax=_lim, aspect="auto", interpolation="nearest")
        _ax.set_title(_title, fontsize=9)
        mark(_ax)
        _fig.colorbar(_shown, ax=_ax, fraction=0.02, pad=0.01, label="pixel value difference")
    _fig
    return


@app.cell
def _(correlations, mo):
    mo.ui.table(correlations, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Saliency of single charts

    **Saliency** is the gradient of the prediction with respect to each pixel: how much the prediction would move if that pixel got a little more ink. Three panels per chart:
    - **the chart;**
    - **the gradient**, on every pixel, inked or not (red: more ink there raises the prediction; blue: lowers it);
    - **gradient × pixel**, on the inked pixels only: what the ink that is there adds to the prediction, to first order.

    A gradient is local: it says how the prediction responds to small changes around this chart, not why the network rates it as it does. Section 3's occlusion test checks it with whole hours removed.

    Pick a chart: the five highest and five lowest predictions of the year (by percentile within the day, then the raw prediction), and five at random.
    """)
    return


@app.cell
def _(mo, np, scored):
    _order = scored.sort_values(["pct", "pred"])
    _rng = np.random.default_rng(0)
    _picks = ([("highest", i) for i in _order.index[::-1][:5]] + [("lowest", i) for i in _order.index[:5]]
              + [("random", i) for i in _rng.choice(scored.index, 5, replace=False)])
    _label = lambda kind, i: (f"{kind}: {scored.at[i, 'day']:%Y-%m-%d}, PERMNO {scored.at[i, 'permno']}, "
                              f"percentile {scored.at[i, 'pct']:.3f}")
    example = mo.ui.dropdown(options={_label(k, i): int(i) for k, i in _picks}, value=_label(*_picks[0]),
                             label="chart")
    example
    return (example,)


@app.cell
def _(draw, example, mark, np, plt, saliency, scored, torch):
    _i = example.value
    with torch.no_grad():
        _img = draw([_i])[0, 0].cpu().numpy()
    _g = saliency(np.array([_i]))[0]
    _gx = _g * _img / 255
    _fig, _axes = plt.subplots(3, 1, figsize=(12, 7), layout="constrained")
    _axes[0].imshow(_img, cmap="gray", vmin=0, vmax=255, aspect="auto", interpolation="nearest")
    _axes[0].set_title(f"{scored.at[_i, 'day']:%Y-%m-%d}, PERMNO {scored.at[_i, 'permno']}: prediction "
                       f"{scored.at[_i, 'pred']:.3f} (percentile {scored.at[_i, 'pct']:.3f} of its day), next-day "
                       f"return {scored.at[_i, 'ret_next'] * 100:.2f}%", fontsize=9)
    for _ax, _v, _title in ((_axes[1], _g, "gradient (red: more ink raises the prediction)"),
                            (_axes[2], _gx, "gradient x pixel (inked pixels only)")):
        _lim = np.quantile(np.abs(_v), 0.999) or 1
        _shown = _ax.imshow(_v, cmap="RdBu_r", vmin=-_lim, vmax=_lim, aspect="auto", interpolation="nearest")
        _ax.set_title(_title, fontsize=9)
        _fig.colorbar(_shown, ax=_ax, fraction=0.02, pad=0.01)
    for _ax in _axes:
        mark(_ax)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Where it looks, on average

    **Saliency averaged over random charts:** each chart's absolute gradient divided by its total, so every chart counts the same, then averaged. Below the heat map, the same summed by bar (time) and by row (price against volume), and each segment's share.

    **The signed average** keeps the gradient's sign: red where more ink raises the prediction on most charts. Where signs differ from chart to chart it averages out.

    **Occlusion:** each segment (the previous last 10 minutes, the night, each hour of the session) blanked out in turn (its columns set to black, as a period without trades), and the mean absolute change in the prediction, relative to the predictions' spread across stocks. Unlike the gradient, this measures what removing a whole period does.

    **What to look at:** whether the network uses the whole day or mostly the last hours, and price or volume.

    **Every pool is a max,** so the gradient flows through the pixel that set each maximum; rows the pool did not choose get none.
    """)
    return


@app.cell
def _(SAMPLE, np, saliency, scored):
    sample = np.sort(np.random.default_rng(2).choice(len(scored), min(SAMPLE, len(scored)), replace=False))
    _g = saliency(sample)
    _norm = np.abs(_g).sum(axis=(1, 2), keepdims=True)
    mean_abs = (np.abs(_g) / _norm).mean(0)
    mean_signed = (_g / _norm).mean(0)
    return mean_abs, mean_signed, sample


@app.cell
def _(
    N_SLOTS,
    PRICE_ROWS,
    SEGMENTS,
    SEGMENT_COLS,
    mark,
    mean_abs,
    mean_signed,
    np,
    plt,
):
    _fig = plt.figure(figsize=(12, 9), layout="constrained")
    _grid = _fig.add_gridspec(4, 2, width_ratios=[5, 1], height_ratios=[3, 3, 1.6, 1.6])
    _ax = _fig.add_subplot(_grid[0, 0])
    _shown = _ax.imshow(mean_abs, cmap="magma", aspect="auto", interpolation="nearest")
    _ax.set_title("mean absolute saliency (each chart's sums to 1)", fontsize=9)
    mark(_ax)
    _fig.colorbar(_shown, ax=_ax, fraction=0.02, pad=0.01)
    _ax = _fig.add_subplot(_grid[0, 1])
    _rows = mean_abs.sum(1)
    _ax.barh(np.arange(len(_rows)), _rows, height=1, color="0.3")
    _ax.invert_yaxis()
    _ax.axhline(PRICE_ROWS, color="grey", lw=0.5)
    _ax.set_title(f"by row: price {_rows[:PRICE_ROWS].sum():.0%}, volume {_rows[PRICE_ROWS + 1:].sum():.0%}",
                  fontsize=8)
    _ax.set_yticks([])
    _ax = _fig.add_subplot(_grid[1, 0])
    _lim = np.abs(mean_signed).max()
    _shown = _ax.imshow(mean_signed, cmap="RdBu_r", vmin=-_lim, vmax=_lim, aspect="auto", interpolation="nearest")
    _ax.set_title("mean signed saliency (red: more ink raises the prediction on most charts)", fontsize=9)
    mark(_ax)
    _fig.colorbar(_shown, ax=_ax, fraction=0.02, pad=0.01)
    _ax = _fig.add_subplot(_grid[2, :])
    _by_bar = mean_abs.sum(0).reshape(N_SLOTS, 3).sum(1)
    _ax.bar(np.arange(N_SLOTS), _by_bar, width=1, color="0.3")
    _ax.set_xlim(-0.5, N_SLOTS - 0.5)
    mark(_ax, scale=3, rows=False)
    _ax.set_title("by bar (the 3 columns of each bar summed)", fontsize=9)
    _ax = _fig.add_subplot(_grid[3, :])
    _share = [mean_abs[:, list(c)].sum() for c in SEGMENT_COLS]
    _ax.bar(SEGMENTS, _share, color=["0.3"] * len(SEGMENTS))
    _ax.tick_params(axis="x", labelsize=7, rotation=60)
    _ax.set_title("share by segment (the session in hours from the label's time; the night 3 columns)", fontsize=9)
    _fig
    return


@app.cell
def _(
    BATCH,
    OCCLUDE_SAMPLE,
    SEGMENTS,
    SEGMENT_COLS,
    draw,
    net,
    np,
    pd,
    plt,
    sample,
    torch,
):
    @torch.no_grad()
    def _occlusion(idx):
        found = {s: [] for s in ["none", *SEGMENTS]}
        for i in range(0, len(idx), BATCH):
            img = draw(idx[i:i + BATCH])
            found["none"].append(net(img)[1])
            for s, cols in zip(SEGMENTS, SEGMENT_COLS):
                blanked = img.clone()
                blanked[..., cols.start:cols.stop] = 0
                found[s].append(net(blanked)[1])
        return {s: torch.cat(v).cpu().numpy() for s, v in found.items()}


    _p = _occlusion(sample[:OCCLUDE_SAMPLE])
    occlusion = pd.DataFrame({"segment": SEGMENTS,
                              "mean_abs_change": [np.abs(_p[s] - _p["none"]).mean() / _p["none"].std()
                                                  for s in SEGMENTS],
                              "mean_change": [(_p[s] - _p["none"]).mean() / _p["none"].std() for s in SEGMENTS]})
    _fig, _ax = plt.subplots(figsize=(12, 2.6), layout="constrained")
    _ax.bar(occlusion["segment"], occlusion["mean_abs_change"],
            color=["0.3"] * len(SEGMENTS))
    _ax.tick_params(axis="x", labelsize=7, rotation=60)
    _ax.set_ylabel("|change| / sd of predictions", fontsize=8)
    _ax.set_title("occlusion: each segment blanked out in turn", fontsize=9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Feature maps: the chart chosen in section 2, layer by layer

    **Block 1's 32 filters**, each a 5 × 3 grid of weights (5 rows, 3 columns): the first convolution slides it down the chart 3 rows at a time, with a gap of one row between its taps (dilation 2), over the chart standardized with the training pixels' mean and sd. Red weights respond to ink, blue to its absence. A filter with a red middle column and blue sides, say, responds to the high-low bar of a bar.

    **The three blocks' outputs** (after their convolution, BatchNorm, LeakyReLU and 2 × 1 max-pool): 32 maps of 13 rows × 120 columns in block 1, 64 maps of 6 rows in block 2, 128 maps of 3 rows in block 3. Each map is one filter's response at every place on the chart; the 12 that vary most on this chart are shown per block, each with its own color scale (bright: strong response). Time stays at full resolution (120 columns): only the rows shrink, and each filter spans 3 columns, one bar.

    **The readout:** the prediction is one weighted sum of block 3's 128 × 3 × 120 = 46,080 numbers (dropout is off at prediction) plus a bias. Its weight times the activation, summed over the 128 maps, is shown as a 3 × 120 picture and by bar; summed over everything with the bias it is the prediction.
    """)
    return


@app.cell
def _(draw, example, net, torch):
    @torch.no_grad()
    def inside(i):
        """The chart i's activations at every stage of the CNN, on the CPU."""
        img = draw([i])
        x, outs = (img - net.mean) / net.sd, []
        for k, layer in enumerate(net.blocks):
            x = layer(x)
            if k % 4 == 3:
                outs.append(x[0])  # block 1, block 2: after the pool
        for layer in net.one_d:
            x = layer(x)
        outs.append(x[0])  # block 3
        weight = net.head[2].weight.view(*x.shape[1:])  # 128 x 3 x 120
        readout = (weight * x[0]).sum(0)  # the 3 x 120 weighted activations, over the 128 maps
        pred = readout.sum() + net.head[2].bias[0]
        assert torch.allclose(pred, net(img)[1][0], atol=1e-4), "the steps don't add up to the CNN's prediction"
        return {"image": img[0, 0], "blocks": outs, "readout": readout, "pred": pred}


    layers = {k: (v.cpu().numpy() if torch.is_tensor(v) else [b.cpu().numpy() for b in v])
              for k, v in inside(example.value).items()}
    return (layers,)


@app.cell
def _(net, np, plt):
    _w = net.blocks[0].weight.detach().cpu().numpy()[:, 0]  # filters x 5 x 3
    _fig, _axes = plt.subplots(4, len(_w) // 4, figsize=(12, 3.4), layout="constrained")
    for _k, _ax in enumerate(_axes.flat):
        _lim = np.abs(_w[_k]).max()
        _ax.imshow(_w[_k], cmap="RdBu_r", vmin=-_lim, vmax=_lim, interpolation="nearest")
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_title(str(_k), fontsize=7)
    _fig.suptitle(f"block 1's {len(_w)} filters (5 rows x 3 columns; red: responds to ink)", fontsize=9)
    _fig
    return


@app.cell
def _(layers, mark, plt):
    def _maps(acts, title, n=12):
        """The chart, then the n maps of one block that vary most on it."""
        order = acts.reshape(len(acts), -1).std(1).argsort()[::-1][:n]
        fig, axes = plt.subplots(n + 1, 1, figsize=(12, 0.55 * (n + 1) + 2.4), layout="constrained",
                                 gridspec_kw={"height_ratios": [3] + [1] * n})
        axes[0].imshow(layers["image"], cmap="gray", vmin=0, vmax=255, aspect="auto", interpolation="nearest")
        mark(axes[0])
        for ax, k in zip(axes[1:], order):
            ax.imshow(acts[k], cmap="magma", aspect="auto", interpolation="nearest")
            mark(ax, rows=False)
            ax.set_ylabel(str(k), fontsize=7, rotation=0, labelpad=8)
        fig.suptitle(title, fontsize=9)
        return fig


    _names = ("block 1 (32 maps of 13 rows)", "block 2 (64 maps of 6 rows)", "block 3 (128 maps of 3 rows)")
    block_figures = [_maps(_a, f"{_n}: the 12 maps that vary most on this chart")
                     for _a, _n in zip(layers["blocks"], _names)]
    return (block_figures,)


@app.cell
def _(block_figures, mo):
    mo.vstack(block_figures)
    return


@app.cell
def _(layers, mark, net, np, plt):
    _r = layers["readout"]
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 3.6), layout="constrained", gridspec_kw={"height_ratios": [1, 1]})
    _lim = np.abs(_r).max()
    _shown = _axes[0].imshow(_r, cmap="RdBu_r", vmin=-_lim, vmax=_lim, aspect="auto", interpolation="nearest")
    mark(_axes[0], rows=False)
    _axes[0].set_title("the readout: weight x activation summed over block 3's 128 maps, by row (3) and column (120)",
                       fontsize=9)
    _fig.colorbar(_shown, ax=_axes[0], fraction=0.02, pad=0.01)
    _by_col = _r.sum(0)
    _axes[1].bar(np.arange(len(_by_col)), _by_col, width=1, color=np.where(_by_col > 0, "firebrick", "steelblue"))
    _axes[1].set_xlim(-0.5, len(_by_col) - 0.5)
    mark(_axes[1], rows=False)
    _axes[1].set_title(f"by column (sum {_by_col.sum():.3f} + bias {net.head[2].bias.item():.3f} = "
                       f"{float(layers['pred']):.3f}, the prediction)", fontsize=9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What the readout can read, over all charts

    **Where it can look:** the readout has one weight for each of block 3's 128 maps at each of 3 rows and 120 columns. Its absolute size summed over the maps, by column and row, shows which part of the chart the readout *can* use. The weights alone ignore how large the activations are, so read it as where the layer can look; section 3 shows where the prediction responds.

    **How far training moved the weights:** each layer's weights against those training started from (rebuilt from the run's seed): the size of the change relative to the starting weights' size.
    """)
    return


@app.cell
def _(mark, net, np, plt):
    _w = np.abs(net.head[2].weight.detach().cpu().numpy()).reshape(128, 3, 120)
    _fig, _axes = plt.subplots(2, 1, figsize=(12, 3.4), layout="constrained")
    _shown = _axes[0].imshow(_w.sum(0), cmap="magma", aspect="auto", interpolation="nearest")
    mark(_axes[0], rows=False)
    _axes[0].set_title("the readout's absolute weights over block 3's maps, by row (3) and column (120)", fontsize=9)
    _fig.colorbar(_shown, ax=_axes[0], fraction=0.02, pad=0.01)
    _by_col = _w.sum((0, 1))
    _axes[1].bar(np.arange(len(_by_col)), _by_col, width=1, color="0.3")
    _axes[1].set_xlim(-0.5, len(_by_col) - 0.5)
    mark(_axes[1], rows=False)
    _axes[1].set_title("by column", fontsize=9)
    _fig
    return


@app.cell
def _(mo, net, pd, untrained):
    def _layers(model):
        found = [("block 1 convolution", model.blocks[0].weight), ("block 2 convolution", model.blocks[4].weight),
                 ("block 3 convolution", model.one_d[0].weight), ("readout", model.head[2].weight)]
        return [(n, w.detach().cpu()) for n, w in found]


    _t = pd.DataFrame([{"layer": _n, "weights": _w.numel(), "mean |w| now": _w.abs().mean().item(),
                        "mean |w| at the start": _w0.abs().mean().item(),
                        "change / start size": ((_w - _w0).norm() / _w0.norm()).item()}
                       for (_n, _w), (_, _w0) in zip(_layers(net), _layers(untrained))])
    mo.ui.table(_t.round(4), selection=None)
    return


if __name__ == "__main__":
    app.run()
