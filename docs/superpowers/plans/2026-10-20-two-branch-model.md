# Two-Branch Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test and run the two-branch model (the 21-day hourly chart through a densely linked CNN and a 1D block, the 17 tabular features through the tuned MLP's layer, a combined MLP on the two summaries) in its three training steps over the 2019-2023 yearly walk-forward, and backtest it against the fixed benchmarks.

**Architecture:** Two new library modules: `src/dsr/two_branch.py` holds the network, the loss, which parts each step trains (`prepare`) and the balance diagnostics; `src/dsr/two_branch_train.py` holds the data on the GPU, each fold's rows, the four step trainers and the output writers. A marimo script, `scripts/models/10_two_branch_walkforward.py`, runs the spec's order as commands (`speed`, `delta`, `stage12`, `choose`, `stage3`), writing one folder per scored output under `data/walkforward/two_branch/` that `scripts/models/05_backtest.py` reads. The first fused model's code is removed.

**Tech Stack:** Python 3 with `uv`, PyTorch on the Mac's GPU (`mps`), pandas, NumPy, marimo, pytest.

**Spec:** `explanations/CNN/fusion.md` (approved 2026-10-20). Read it before starting; every number below comes from it.

## Global Constraints

- Run everything through `uv run` (`uv run pytest`, `uv run python scripts/...`).
- In every test file and script that imports torch, import `dsr.models` (LightGBM) **before** torch: torch first crashes LightGBM on macOS (two OpenMP runtimes).
- Never run a convolution on the CPU in a process that has loaded LightGBM: tests use `DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"`, as `tests/test_cnn.py` does.
- Weight counts, exactly: **194,483** (4 channels in the 1D block, combined MLP 32→16→1); **194,611** with the combined MLP 32→16→8→1; **233,099** with 8 channels in the 1D block. The blocks alone: **154,920**; the 1D block: **38,632**; step 3 trains **39,563**.
- Filters 32 / 64 / 84; first convolution 5×3, stride 3×1, padding 2×1, no dilation; time dilation 1, 3, 9 in blocks 1, 2, 3; pixels ÷ 255; LeakyReLU (slope 0.01) everywhere; dropout 0.54 in the tabular head and the combined MLP only; Xavier-uniform weights, zero biases.
- Every loss is a Huber loss with the δ chosen in step 1b (`torch.nn.functional.huber_loss`); squared error only as step 1b's unselectable reference.
- The target is `target_demeaned` divided by the sd of the fold's large-cap training rows.
- Yearly folds from 2019 (`dsr.walkforward.folds(..., first="2019-01-01", every="year")`), 12-month hold-out (`fold_rows(..., holds_out=True)`). The test years 2024-2025 are never read: `data/processed/cnn_hours_train.npz` holds 2014-2023 only.
- Do not commit unless the user asks; no `Co-Authored-By` trailers in commit messages.
- `data/` is local only and never committed.
- Match the code around it: short docstrings that say what a thing is and why, few comments, names spelled out.

## Review Focus

1. **A training batch of one row in a BatchNorm1d step** (the last batch of an epoch, steps 2 and 3): it must be skipped, not crash ("Expected more than 1 value per channel"). Pinned in Task 3 (`test_train_combined_stops_on_the_stop_rows_and_skips_a_last_batch_of_one`).
2. **Rows with a missing target** (the training window's last days, delistings): never fitted or stopped on, but test-year rows are predicted whatever their target. Pinned in Task 4 (`test_fold_indices_split_the_fold_as_the_spec_says`).
3. **A chart that is mostly empty** (a stock new to the panel, early closes): the network's outputs must stay finite. Pinned in Task 1 (`test_an_empty_chart_gives_finite_outputs`).
4. **A command stopped and started again:** finished folds are skipped and a fold written twice appears once in the collected predictions. Pinned in Task 4 (`test_a_fold_written_twice_is_collected_once`).
5. **Predicting with BatchNorm in training mode:** a stock-day's prediction would depend on the rest of its batch. Pinned in Task 1 (`test_a_prediction_does_not_depend_on_the_rest_of_its_batch`) and Task 3 (`outputs` puts the net in eval mode; `prepare` puts the trained parts back).

---

### Task 1: The network

**Files:**
- Create: `src/dsr/two_branch.py`
- Test: `tests/test_two_branch.py`

**Interfaces:**
- Consumes: `dsr.cnn.W_WIDTH` (444), `dsr.cnn.xavier(net)`.
- Produces: constants `FILTERS`, `TIME_DILATION`, `ROWS`, `BARS`, `SUMMARY`, `DROPOUT`, `N_FEATURES`, `COMBINED` (`{"16": (16,), "16-8": (16, 8)}`); `pool_rows(x, rows) -> Tensor`; classes `DenseBlocks()`, `OneDBlock(c_in, channels=4)`, `PriceBranch(channels=4)` (`forward(img) -> (s_p, price)`), `TabularBranch(n_features=17)` (`forward(x) -> (s_t, tab)`), `CombinedMLP(hidden=(16,))` (`forward(s_p, s_t) -> comb`; `layers[0]` its BatchNorm, `layers[1]` its first linear layer), `TwoBranchNet(channels=4, hidden=(16,), n_features=17)` (`forward(img, tab) -> {"s_p", "s_t", "price", "tab", "comb"}`; attributes `price`, `tabular`, `combined`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_two_branch.py`:

```python
import numpy as np
import pytest

import dsr.models  # noqa: F401  lightgbm before torch: torch first crashes lightgbm on macOS (two OpenMP runtimes)
import torch  # noqa: E402

from dsr.cnn import W_HEIGHT, W_WIDTH  # noqa: E402
from dsr.two_branch import COMBINED, DenseBlocks, TwoBranchNet  # noqa: E402

# the GPU when there is one: CPU convolutions after LightGBM crash on macOS (two OpenMP runtimes)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def _charts(n, seed):
    """n random charts (n x 1 x 96 x 444, pixels 0 or 255), made with NumPy: elementwise torch work on the CPU starts
    OpenMP threads, which crash after LightGBM on macOS."""
    rng = np.random.default_rng(seed)
    return torch.as_tensor((rng.random((n, 1, W_HEIGHT, W_WIDTH)) > 0.97).astype(np.float32) * 255, device=DEVICE)


def _normal(*shape, seed):
    return torch.as_tensor(np.random.default_rng(seed).normal(size=shape).astype(np.float32), device=DEVICE)


def _weights(module):
    return sum(w.numel() for w in module.parameters())


def test_the_network_has_the_spec_s_weights():
    net = TwoBranchNet()
    assert _weights(net) == 194_483
    assert _weights(net.price.blocks) == 154_920 and _weights(net.price.one_d) == 38_632
    assert _weights(net.tabular) == 305 and _weights(net.combined) == 609
    assert _weights(TwoBranchNet(hidden=COMBINED["16-8"])) == 194_611
    assert _weights(TwoBranchNet(channels=8)) == 233_099


def test_the_network_gives_two_summaries_and_three_predictions():
    net = TwoBranchNet().to(DEVICE).eval()
    out = net(torch.zeros(2, 1, W_HEIGHT, W_WIDTH, device=DEVICE), torch.zeros(2, 17, device=DEVICE))
    assert out["s_p"].shape == (2, 16) and out["s_t"].shape == (2, 16)
    assert out["price"].shape == out["tab"].shape == out["comb"].shape == (2,)


def test_the_dense_blocks_put_the_image_and_the_three_blocks_side_by_side():
    blocks = DenseBlocks().to(DEVICE).eval()
    assert blocks(torch.zeros(2, 1, W_HEIGHT, W_WIDTH, device=DEVICE)).shape == (2, 181, 4, W_WIDTH)
    assert [b[0].in_channels for b in blocks.blocks] == [1, 33, 97]
    assert [b[0].dilation for b in blocks.blocks] == [(1, 1), (1, 3), (1, 9)]


def test_one_block_3_value_reads_27_columns_without_gaps():
    blocks = DenseBlocks().to(DEVICE).eval()
    with torch.no_grad():
        for conv in (b[0] for b in blocks.blocks):
            conv.weight.fill_(1.0)
            conv.bias.zero_()
    x = torch.zeros(1, 1, W_HEIGHT, W_WIDTH, device=DEVICE, requires_grad=True)
    blocks(x)[0, -84:, :, 222].sum().backward()  # the last 84 channels are block 3
    read = (x.grad[0, 0].abs().sum(0) > 0).nonzero().flatten().cpu().numpy()
    assert read.min() == 209 and read.max() == 235 and len(read) == 27


def test_an_empty_chart_gives_finite_outputs():
    net = TwoBranchNet().to(DEVICE).eval()
    with torch.no_grad():
        out = net(torch.zeros(3, 1, W_HEIGHT, W_WIDTH, device=DEVICE), torch.zeros(3, 17, device=DEVICE))
    assert all(torch.isfinite(v).all() for v in out.values())


def test_a_prediction_does_not_depend_on_the_rest_of_its_batch():
    img, tab = _charts(3, seed=0), _normal(3, 17, seed=0)
    net = TwoBranchNet().to(DEVICE).eval()
    with torch.no_grad():
        a = net(img[[0, 1]], tab[[0, 1]])["comb"][0]
        b = net(img[[0, 2]], tab[[0, 2]])["comb"][0]
    assert a.item() == pytest.approx(b.item(), abs=1e-6)
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_two_branch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dsr.two_branch'`.

- [ ] **Step 3: Write the network**

Create `src/dsr/two_branch.py`:

```python
"""The two-branch model (explanations/CNN/fusion.md): the 21-day hourly chart through a densely linked CNN and the 1D
block, the 17 tabular features through the tuned MLP's layer, each ending in a 16-number summary and its own head, and
a combined MLP on the two summaries; the loss, which parts each training step trains, and the balance diagnostics."""

import torch
from torch import nn
from torch.nn import functional as F

from dsr.cnn import W_WIDTH, xavier

FILTERS = (32, 64, 84)  # the three blocks' filters
TIME_DILATION = (1, 3, 9)  # their time dilations: a block 3 value sees 27 columns (9 bars), with no gaps
ROWS = 4  # the image rows left after the blocks: 96 -> 32 (stride 3) -> 16 -> 8 -> 4 (the pools)
BARS = W_WIDTH // 3  # 148: each bar is 3 columns
SUMMARY = 16  # each branch's summary
DROPOUT = 0.54  # the tabular head's and the combined MLP's
N_FEATURES = 17
COMBINED = {"16": (16,), "16-8": (16, 8)}  # the combined MLP's hidden layers, the two step 2 tries


def pool_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """x (B x C x H x W) max-pooled over its rows down to `rows` (H a multiple of rows), the columns untouched."""
    return F.max_pool2d(x, (x.shape[2] // rows, 1))


class DenseBlocks(nn.Module):
    """Three blocks of a 5x3 convolution, BatchNorm, LeakyReLU and a 2x1 max-pool (the rows only), densely linked: each
    block reads the image and every earlier block, max-pooled to its height. The first block moves 3 rows at a time;
    `dilation` spaces each block's 3 time taps. Returns the image and the three blocks, each max-pooled to ROWS rows,
    side by side: B x (1 + sum(filters)) x ROWS x W."""

    def __init__(self, filters=FILTERS, dilation=TIME_DILATION):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i, (c, d) in enumerate(zip(filters, dilation)):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(1 + sum(filters[:i]), c, (5, 3), stride=(3, 1) if i == 0 else 1, padding=(2, d),
                          dilation=(1, d)),
                nn.BatchNorm2d(c), nn.LeakyReLU(), nn.MaxPool2d((2, 1))))
        self.channels = 1 + sum(filters)

    def forward(self, x):
        outs = [x]
        for block in self.blocks:
            rows = outs[-1].shape[2]
            outs.append(block(torch.cat([pool_rows(o, rows) for o in outs], 1)))
        return torch.cat([pool_rows(o, ROWS) for o in outs], 1)


class OneDBlock(nn.Module):
    """The blocks' output to the 16-number summary: a max-pool over each bar's 3 columns, a 1x1 convolution to
    `channels`, LeakyReLU, flatten, and a linear layer with LeakyReLU (explanations/CNN/fusion.md, the 1D block)."""

    def __init__(self, c_in: int, channels: int = 4):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d((1, 3)), nn.Conv2d(c_in, channels, 1), nn.LeakyReLU(), nn.Flatten(),
                                    nn.Linear(channels * ROWS * BARS, SUMMARY), nn.LeakyReLU())

    def forward(self, x):
        return self.layers(x)


class PriceBranch(nn.Module):
    """The chart (B x 1 x 96 x 444, pixels 0-255) divided by 255, the dense blocks and the 1D block: the summary s_p,
    and the price head's prediction from it."""

    def __init__(self, channels: int = 4):
        super().__init__()
        self.blocks = DenseBlocks()
        self.one_d = OneDBlock(self.blocks.channels, channels)
        self.head = nn.Linear(SUMMARY, 1)

    def forward(self, img):
        s = self.one_d(self.blocks(img / 255))
        return s, self.head(s).squeeze(1)


class TabularBranch(nn.Module):
    """The tuned MLP's layer (17 -> 16, LeakyReLU) giving the summary s_t, taken before the head's dropout, and the
    tabular head's prediction."""

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_features, SUMMARY), nn.LeakyReLU())
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(SUMMARY, 1))

    def forward(self, x):
        s = self.body(x)
        return s, self.head(s).squeeze(1)


class CombinedMLP(nn.Module):
    """The two summaries side by side, a BatchNorm, the hidden layers (each linear, LeakyReLU and dropout) and one
    output: the model's prediction."""

    def __init__(self, hidden=(16,)):
        super().__init__()
        layers, c = [nn.BatchNorm1d(2 * SUMMARY)], 2 * SUMMARY
        for h in hidden:
            layers += [nn.Linear(c, h), nn.LeakyReLU(), nn.Dropout(DROPOUT)]
            c = h
        self.layers = nn.Sequential(*layers, nn.Linear(c, 1))

    def forward(self, s_p, s_t):
        return self.layers(torch.cat([s_p, s_t], 1)).squeeze(1)


class TwoBranchNet(nn.Module):
    """The price branch, the tabular branch and the combined MLP. forward gives the summaries s_p and s_t and the three
    predictions: price, tab and comb (the model's)."""

    def __init__(self, channels: int = 4, hidden=(16,), n_features: int = N_FEATURES):
        super().__init__()
        self.price, self.tabular, self.combined = PriceBranch(channels), TabularBranch(n_features), CombinedMLP(hidden)
        xavier(self)

    def forward(self, img, tab):
        s_p, price = self.price(img)
        s_t, tab_pred = self.tabular(tab)
        return {"s_p": s_p, "s_t": s_t, "price": price, "tab": tab_pred, "comb": self.combined(s_p, s_t)}
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_two_branch.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: every test passes (125 before this task, 131 after).

---

### Task 2: The loss, which parts each step trains, and the balance diagnostics

**Files:**
- Modify: `src/dsr/two_branch.py` (append)
- Test: `tests/test_two_branch.py` (append)

**Interfaces:**
- Consumes: Task 1's `TwoBranchNet`, `SUMMARY`.
- Produces: `TRAINED` (step → part names); `parts(net) -> dict[str, nn.Module]` with keys `blocks`, `one_d`, `price_head`, `tab_body`, `tab_head`, `combined`; `prepare(net, step: str) -> dict[str, list[Parameter]]` for steps `"1a"`, `"1b"`, `"2"`, `"3"`; `loss(pred, y, delta) -> Tensor` (squared error when `delta is None`); `step3_loss(out, y, delta, lam=0.5) -> Tensor`; `grad_share(out, y, delta) -> dict` with `grad_price`, `grad_tab`, `grad_share_price`; `weight_share(combined) -> dict` with `weight_price`, `weight_tab`, `weight_share_price`; `summary_drift(s_p, s_t, ref_mean, ref_sd) -> dict` with `drift_shift_price`, `drift_shift_tab`, `drift_ratio_price`, `drift_ratio_tab`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_two_branch.py`:

```python
from dsr.two_branch import (TRAINED, grad_share, loss, parts, prepare, step3_loss, summary_drift,  # noqa: E402
                            weight_share)


def test_the_loss_is_huber_with_delta_and_squared_error_without():
    pred, y = torch.tensor([0.5, 3.0]), torch.zeros(2)
    assert loss(pred, y, 1.0).item() == pytest.approx((0.5 * 0.5 ** 2 + (3.0 - 0.5)) / 2)  # squared, then linear
    assert loss(pred, y, None).item() == pytest.approx((0.5 ** 2 + 3.0 ** 2) / 2)


def test_step_3_s_loss_weighs_all_three_alike_at_one_half_and_is_the_combined_loss_at_zero():
    y = torch.zeros(3)
    out = {"price": torch.ones(3), "tab": 2 * torch.ones(3), "comb": 3 * torch.ones(3)}
    full = loss(out["comb"], y, 1.0) + loss(out["price"], y, 1.0) + loss(out["tab"], y, 1.0)
    assert step3_loss(out, y, 1.0, 0.5).item() == pytest.approx(full.item())
    assert step3_loss(out, y, 1.0, 0.0).item() == pytest.approx(loss(out["comb"], y, 1.0).item())


def test_prepare_trains_only_the_step_s_parts_and_freezes_the_rest_in_eval_mode():
    net = TwoBranchNet()
    for step, trained in TRAINED.items():
        prepare(net, step)
        for name, module in parts(net).items():
            assert module.training == (name in trained), (step, name)
            assert all(w.requires_grad == (name in trained) for w in module.parameters()), (step, name)
    assert TRAINED["3"] == ("one_d", "price_head", "tab_body", "tab_head", "combined")


def test_step_3_leaves_the_blocks_and_their_batchnorm_alone_and_updates_the_combined_batchnorm():
    img, tab, y = _charts(8, seed=1), _normal(8, 17, seed=1), _normal(8, seed=2)
    net = TwoBranchNet().to(DEVICE)
    blocks_before = {k: v.clone() for k, v in net.price.blocks.state_dict().items()}
    one_d_before = net.price.one_d.layers[4].weight.detach().clone()
    bn_mean_before = net.combined.layers[0].running_mean.clone()
    groups = prepare(net, "3")
    opt = torch.optim.AdamW([w for ws in groups.values() for w in ws], lr=1e-2)
    step3_loss(net(img, tab), y, 1.0).backward()
    opt.step()
    assert all(torch.equal(blocks_before[k], v) for k, v in net.price.blocks.state_dict().items())
    assert not torch.equal(one_d_before, net.price.one_d.layers[4].weight)
    assert not torch.equal(bn_mean_before, net.combined.layers[0].running_mean)


def test_the_gradient_share_uses_the_combined_loss_only():
    img, tab, y = _charts(8, seed=3), _normal(8, 17, seed=3), _normal(8, seed=4)
    net = TwoBranchNet().to(DEVICE)
    prepare(net, "3")
    net.eval()  # no dropout, so both passes see the same network
    first = grad_share(net(img, tab), y, 1.0)
    with torch.no_grad():
        net.price.head.weight.mul_(100.0)  # the price head changes L_price, not L_comb
    second = grad_share(net(img, tab), y, 1.0)
    assert first["grad_share_price"] == pytest.approx(second["grad_share_price"])
    assert 0 < first["grad_share_price"] < 1


def test_the_weight_share_scales_each_input_by_its_batchnorm():
    net = TwoBranchNet()
    bn, lin = net.combined.layers[0], net.combined.layers[1]
    with torch.no_grad():
        lin.weight.fill_(1.0)
        bn.weight.fill_(1.0)
        bn.weight[:16] = 2.0  # the price inputs count twice as much
        bn.running_var.fill_(1.0 - bn.eps)
    assert weight_share(net.combined)["weight_share_price"] == pytest.approx(2 / 3)


def test_the_summary_drift_is_the_shift_in_step_2_sds_and_the_sd_ratio():
    s_p = torch.zeros(1000, 16)
    s_p[:, 3] = torch.linspace(-1, 1, 1000) + 0.5  # mean 0.5
    s_t = torch.zeros(1000, 16)
    ref_mean, ref_sd = torch.zeros(32), torch.ones(32)
    d = summary_drift(s_p, s_t, ref_mean, ref_sd)
    assert d["drift_shift_price"] == pytest.approx(0.5) and d["drift_shift_tab"] == pytest.approx(0.0)
    assert d["drift_ratio_price"] == pytest.approx(0.0, abs=1e-6)  # the 0-sd columns are the furthest from 1
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_two_branch.py -v`
Expected: FAIL with `ImportError: cannot import name 'TRAINED' from 'dsr.two_branch'`.

- [ ] **Step 3: Write the loss, `prepare` and the diagnostics**

Append to `src/dsr/two_branch.py`:

```python
TRAINED = {"1a": ("blocks", "one_d", "price_head"), "1b": ("tab_body", "tab_head"), "2": ("combined",),
           "3": ("one_d", "price_head", "tab_body", "tab_head", "combined")}  # the parts each step trains


def parts(net: TwoBranchNet) -> dict[str, nn.Module]:
    """The net's parts, as the steps train or freeze them."""
    return {"blocks": net.price.blocks, "one_d": net.price.one_d, "price_head": net.price.head,
            "tab_body": net.tabular.body, "tab_head": net.tabular.head, "combined": net.combined}


def prepare(net: TwoBranchNet, step: str) -> dict[str, list[nn.Parameter]]:
    """Make the parts `step` trains ("1a", "1b", "2" or "3"; TRAINED) trainable and in training mode, and freeze every
    other part in eval mode, so its BatchNorm statistics stay put. Returns the trained parts' weights by part. Call it at
    the start of every epoch: predicting puts the whole net in eval mode."""
    trained = TRAINED[step]
    for name, module in parts(net).items():
        module.train(name in trained)
        for w in module.parameters():
            w.requires_grad_(name in trained)
    return {name: list(parts(net)[name].parameters()) for name in trained}


def loss(pred: torch.Tensor, y: torch.Tensor, delta: float | None) -> torch.Tensor:
    """The Huber loss with threshold delta (in the target's scaled units): squared for errors below delta, linear
    beyond. Squared error when delta is None (step 1b's reference only)."""
    return F.mse_loss(pred, y) if delta is None else F.huber_loss(pred, y, delta=delta)


def step3_loss(out: dict, y: torch.Tensor, delta: float | None, lam: float = 0.5) -> torch.Tensor:
    """L_comb + lam / (1 - lam) * (L_price + L_tab): each branch body learns from lam * its own loss + (1 - lam) * the
    combined loss, times 1 / (1 - lam), which AdamW's steps ignore. lam = 0 is the control: the combined loss alone."""
    own = loss(out["price"], y, delta) + loss(out["tab"], y, delta)
    return loss(out["comb"], y, delta) + lam / (1 - lam) * own


def grad_share(out: dict, y: torch.Tensor, delta: float | None) -> dict:
    """The combined loss's gradient at the two summaries (each its norm per stock-day, averaged over the batch), and the
    price branch's share of the two. Only L_comb: the branches' own losses are left out."""
    g_p, g_t = torch.autograd.grad(loss(out["comb"], y, delta), [out["s_p"], out["s_t"]], retain_graph=True)
    p, t = g_p.norm(dim=1).mean().item(), g_t.norm(dim=1).mean().item()
    return {"grad_price": p, "grad_tab": t, "grad_share_price": p / (p + t) if p + t > 0 else float("nan")}


def weight_share(combined: CombinedMLP) -> dict:
    """The combined MLP's first linear layer, its weights on the 16 price inputs against the 16 tabular ones, each
    column scaled by the BatchNorm in front of it (its scale over its sd), as norms, and the price share."""
    bn, lin = combined.layers[0], combined.layers[1]
    w = (lin.weight * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    p, t = w[:, :SUMMARY].norm().item(), w[:, SUMMARY:].norm().item()
    return {"weight_price": p, "weight_tab": t, "weight_share_price": p / (p + t) if p + t > 0 else float("nan")}


def summary_drift(s_p: torch.Tensor, s_t: torch.Tensor, ref_mean: torch.Tensor, ref_sd: torch.Tensor) -> dict:
    """How far the 32 summary numbers moved from step 2's statistics (ref_mean, ref_sd): for each half, the largest mean
    shift in units of the step 2 sd, and the sd ratio (now over step 2) furthest from 1."""
    s = torch.cat([s_p, s_t], 1).detach()
    shift = ((s.mean(0) - ref_mean) / ref_sd).abs()
    ratio = s.std(0) / ref_sd
    out = {}
    for name, half in (("price", slice(0, SUMMARY)), ("tab", slice(SUMMARY, 2 * SUMMARY))):
        r = ratio[half]
        out[f"drift_shift_{name}"] = shift[half].max().item()
        out[f"drift_ratio_{name}"] = r[(r.clamp_min(1e-12).log().abs()).argmax()].item()
    return out
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_two_branch.py -v`
Expected: 13 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: every test passes.

---

### Task 3: The four step trainers

**Files:**
- Create: `src/dsr/two_branch_train.py`
- Test: `tests/test_two_branch_train.py`

**Interfaces:**
- Consumes: Task 1's `TwoBranchNet`, Task 2's `prepare`, `loss`, `step3_loss`, `grad_share`, `weight_share`, `summary_drift`.
- Produces (all in `dsr.two_branch_train`):
  - `batches(idx, size, shuffle, device) -> list[Tensor]`
  - `outputs(net, data, idx, batch=512) -> dict[str, Tensor]` (eval mode; keys `s_p`, `s_t`, `price`, `tab`, `comb`)
  - `price_predictions(net, data, idx, batch=512) -> np.ndarray`
  - `tabular_predictions(net, features, idx, batch=65536) -> Tensor`
  - `train_price(net, data, y, train, hold, score, p, delta, on_epoch=None) -> {"epochs", "score"}`; `p` keys `learning_rate`, `weight_decay`, `batch_size`, `max_epochs`, `patience`
  - `train_tabular(net, features, y, train, p, delta, on_epoch=None) -> None`; `p` keys `learning_rate`, `weight_decay`, `batch_size`, `epochs`
  - `train_combined(net, fit, stop, p, delta, on_epoch=None) -> {"epochs", "stop_loss"}`; `fit` and `stop` are `(s_p, s_t, y)` tensors; `p` keys `learning_rate`, `weight_decay`, `batch_size`, `max_epochs`, `patience`
  - `finetune(net, data, y, idx, p, delta, lam, on_log=None, on_epoch=None, log_every=100) -> None`; `p` keys `epochs`, `batch_size`, `learning_rate` (dict by part), `weight_decay` (dict by part)
  - A data object passed as `data` needs `device` (torch.device), `features` (n x 17 tensor) and `draw(idx) -> (img B x 1 x 96 x 444, tab B x 17)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_two_branch_train.py`:

```python
import numpy as np
import pytest

import dsr.models  # noqa: F401  lightgbm before torch: torch first crashes lightgbm on macOS (two OpenMP runtimes)
import torch  # noqa: E402

from dsr.cnn import W_HEIGHT, W_WIDTH  # noqa: E402
from dsr.two_branch import TwoBranchNet, prepare  # noqa: E402
from dsr.two_branch_train import (finetune, outputs, tabular_predictions, train_combined, train_price,  # noqa: E402
                                  train_tabular)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def _charts(n, seed):
    """n random charts (n x 1 x 96 x 444, pixels 0 or 255), made with NumPy: elementwise torch work on the CPU starts
    OpenMP threads, which crash after LightGBM on macOS."""
    rng = np.random.default_rng(seed)
    return torch.as_tensor((rng.random((n, 1, W_HEIGHT, W_WIDTH)) > 0.97).astype(np.float32) * 255, device=DEVICE)


def _normal(*shape, seed):
    return torch.as_tensor(np.random.default_rng(seed).normal(size=shape).astype(np.float32), device=DEVICE)


class _Data:
    """Random charts and features on the device, drawn as TwoBranchData draws them."""

    def __init__(self, n, seed=0):
        self.device = torch.device(DEVICE)
        self.images, self.features = _charts(n, seed), _normal(n, 17, seed=seed)

    def draw(self, idx):
        return self.images[idx], self.features[idx]


def test_train_price_stops_on_the_score_and_keeps_the_best_epoch():
    data, net = _Data(48), TwoBranchNet().to(DEVICE)
    y = _normal(48, seed=3)
    scores, kept = iter([0.1, 0.3, 0.2, 0.1, 0.5]), {}
    epochs = []

    def remember(e):
        epochs.append(e)
        if e["epoch"] == 2:
            kept.update({k: v.clone() for k, v in net.price.state_dict().items()})

    p = {"learning_rate": 1e-3, "weight_decay": 0.05, "batch_size": 16, "max_epochs": 5, "patience": 2}
    info = train_price(net, data, y, np.arange(32), np.arange(32, 48), lambda pred: next(scores), p, 1.0, remember)
    assert len(epochs) == 4 and info == {"epochs": 2, "score": 0.3}
    assert all(torch.equal(kept[k], v) for k, v in net.price.state_dict().items())


def test_train_tabular_learns_a_linear_signal():
    features = _normal(2048, 17, seed=4)
    y = features @ _normal(17, seed=5)
    y = y / y.std()
    net = TwoBranchNet().to(DEVICE)
    p = {"learning_rate": 1e-2, "weight_decay": 0.0, "batch_size": 64, "epochs": 5}
    train_tabular(net, features, y, np.arange(1536), p, 1.0)
    pred = tabular_predictions(net, features, np.arange(1536, 2048))
    assert np.corrcoef(pred.cpu().numpy(), y[1536:].cpu().numpy())[0, 1] > 0.7


def test_train_combined_stops_on_the_stop_rows_and_skips_a_last_batch_of_one():
    s = lambda n: (_normal(n, 16, seed=n), _normal(n, 16, seed=n + 1), _normal(n, seed=n + 2))
    net = TwoBranchNet().to(DEVICE)
    bn_before = net.combined.layers[0].running_mean.clone()
    p = {"learning_rate": 1e-3, "weight_decay": 0.035, "batch_size": 16, "max_epochs": 5, "patience": 1}
    info = train_combined(net, s(33), s(20), p, 1.0)  # 33 = 2 x 16 + 1: the last batch has one row
    assert 1 <= info["epochs"] <= 5 and np.isfinite(info["stop_loss"])
    assert not torch.equal(bn_before, net.combined.layers[0].running_mean)  # step 2 trains its BatchNorm


def test_finetune_trains_only_after_the_blocks_and_logs_the_balance():
    data, net = _Data(40, seed=6), TwoBranchNet().to(DEVICE)
    y = _normal(40, seed=6)
    blocks_before = {k: v.clone() for k, v in net.price.blocks.state_dict().items()}
    one_d_before = net.price.one_d.layers[4].weight.detach().clone()
    logs, ends = [], []
    p = {"epochs": 2, "batch_size": 16,
         "learning_rate": {"one_d": 1e-3, "price_head": 1e-3, "tab_body": 1e-3, "tab_head": 1e-3, "combined": 1e-3},
         "weight_decay": {"one_d": 0.05, "price_head": 0.05, "tab_body": 0.035, "tab_head": 0.035, "combined": 0.035}}
    finetune(net, data, y, np.arange(40), p, 1.0, 0.5, on_log=logs.append, on_epoch=ends.append, log_every=1)
    assert all(torch.equal(blocks_before[k], v) for k, v in net.price.blocks.state_dict().items())
    assert not torch.equal(one_d_before, net.price.one_d.layers[4].weight)
    assert ends == [1, 2] and len(logs) == 2 * 3  # 3 batches an epoch: 16, 16, 8
    assert {"grad_share_price", "weight_share_price", "drift_shift_price", "drift_ratio_tab"} <= set(logs[0])


def test_outputs_predict_in_eval_mode_and_prepare_puts_the_trained_parts_back():
    data, net = _Data(10, seed=7), TwoBranchNet().to(DEVICE)
    out = outputs(net, data, np.arange(10), batch=4)
    assert out["comb"].shape == (10,) and out["s_p"].shape == (10, 16) and not net.combined.training
    prepare(net, "3")
    assert net.combined.training and not net.price.blocks.training
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_two_branch_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dsr.two_branch_train'`.

- [ ] **Step 3: Write the trainers**

Create `src/dsr/two_branch_train.py`:

```python
"""Training the two-branch model (explanations/CNN/fusion.md): the four steps, the outputs they are scored on, the data
on the GPU, each fold's rows, and the writers of the scored outputs and of the balance diagnostics."""

from collections.abc import Callable

import numpy as np
import torch

from dsr.two_branch import TwoBranchNet, grad_share, loss, prepare, step3_loss, summary_drift, weight_share


def batches(idx, size: int, shuffle: bool, device) -> list[torch.Tensor]:
    """The positions idx in batches of `size` (the last one shorter), shuffled or in order, on the device."""
    idx = torch.as_tensor(np.asarray(idx), device=device)
    if shuffle:
        idx = idx[torch.randperm(len(idx), device=device)]
    return [idx[i:i + size] for i in range(0, len(idx), size)]


@torch.no_grad()
def outputs(net: TwoBranchNet, data, idx, batch: int = 512) -> dict[str, torch.Tensor]:
    """The net's outputs for the stock-days idx, in eval mode: the summaries s_p and s_t (n x 16) and the predictions
    price, tab and comb (n), in the target's scaled units."""
    net.eval()
    found = [net(*data.draw(b)) for b in batches(idx, batch, False, data.device)]
    return {k: torch.cat([f[k] for f in found]) for k in found[0]}


@torch.no_grad()
def price_predictions(net: TwoBranchNet, data, idx, batch: int = 512) -> np.ndarray:
    """The price head's predictions for the stock-days idx, in eval mode (the charts alone are drawn)."""
    net.eval()
    return torch.cat([net.price(data.draw(b)[0])[1] for b in batches(idx, batch, False, data.device)]).cpu().numpy()


@torch.no_grad()
def tabular_predictions(net: TwoBranchNet, features: torch.Tensor, idx, batch: int = 65536) -> torch.Tensor:
    """The tabular head's predictions for the rows idx of features, in eval mode."""
    net.eval()
    return torch.cat([net.tabular(features[b])[1] for b in batches(idx, batch, False, features.device)])


def train_price(net: TwoBranchNet, data, y: torch.Tensor, train, hold, score: Callable[[np.ndarray], float], p: dict,
                delta: float | None, on_epoch: Callable[[dict], None] | None = None) -> dict:
    """Step 1a: the price branch alone on the Huber loss of its head, AdamW, at most p["max_epochs"] epochs over the
    stock-days `train`, stopped after p["patience"] epochs without a higher score(the price predictions of `hold`) (the
    hold-out's rank IC) and returned to the best epoch's weights. y: every row's scaled target, on the data's device."""
    params = [w for ws in prepare(net, "1a").values() for w in ws]
    opt = torch.optim.AdamW(params, lr=p["learning_rate"], weight_decay=p["weight_decay"])
    best, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    for epoch in range(1, int(p["max_epochs"]) + 1):
        prepare(net, "1a")
        total = torch.zeros((), device=data.device)  # kept on the GPU: reading each batch's loss would stall it
        for b in batches(train, int(p["batch_size"]), True, data.device):
            batch_loss = loss(net.price(data.draw(b)[0])[1], y[b], delta)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.detach() * len(b)
        s = float(score(price_predictions(net, data, hold)))
        if on_epoch is not None:
            on_epoch({"epoch": epoch, "train_loss": float(total) / len(train), "holdout_score": s})
        if s > best:
            best, best_epoch, stale = s, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.price.state_dict().items()}
        else:
            stale += 1
            if stale >= int(p["patience"]):
                break
    net.price.load_state_dict(best_state)
    return {"epochs": best_epoch, "score": best}


def train_tabular(net: TwoBranchNet, features: torch.Tensor, y: torch.Tensor, train, p: dict, delta: float | None,
                  on_epoch: Callable[[dict], None] | None = None) -> None:
    """Step 1b: the tabular branch alone on the Huber loss of its head (squared error when delta is None), AdamW,
    p["epochs"] epochs over the rows `train` of features (no early stopping)."""
    params = [w for ws in prepare(net, "1b").values() for w in ws]
    opt = torch.optim.AdamW(params, lr=p["learning_rate"], weight_decay=p["weight_decay"])
    for epoch in range(1, int(p["epochs"]) + 1):
        prepare(net, "1b")
        total = torch.zeros((), device=features.device)
        for b in batches(train, int(p["batch_size"]), True, features.device):
            batch_loss = loss(net.tabular(features[b])[1], y[b], delta)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.detach() * len(b)
        if on_epoch is not None:
            on_epoch({"epoch": epoch, "train_loss": float(total) / len(train)})


def train_combined(net: TwoBranchNet, fit: tuple, stop: tuple, p: dict, delta: float | None,
                   on_epoch: Callable[[dict], None] | None = None) -> dict:
    """Step 2: the combined MLP alone on stored summaries, fit = (s_p, s_t, y) of the rows it fits and stop = the same
    of the rows it stops on, Huber loss, AdamW, at most p["max_epochs"] epochs, stopped after p["patience"] epochs
    without a lower loss on the stop rows and returned to the best epoch (its BatchNorm statistics with it). A last
    batch of one row is skipped: BatchNorm can't train on it. on_epoch also gets the weight share."""
    s_p, s_t, y = fit
    opt = torch.optim.AdamW(prepare(net, "2")["combined"], lr=p["learning_rate"], weight_decay=p["weight_decay"])
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, int(p["max_epochs"]) + 1):
        prepare(net, "2")
        total = torch.zeros((), device=y.device)
        for b in batches(np.arange(len(y)), int(p["batch_size"]), True, y.device):
            if len(b) < 2:
                continue
            batch_loss = loss(net.combined(s_p[b], s_t[b]), y[b], delta)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.detach() * len(b)
        net.combined.eval()
        with torch.no_grad():
            v = float(loss(net.combined(stop[0], stop[1]), stop[2], delta))
        if on_epoch is not None:
            on_epoch({"epoch": epoch, "train_loss": float(total) / len(y), "stop_loss": v,
                      **weight_share(net.combined)})
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.combined.state_dict().items()}
        else:
            stale += 1
            if stale >= int(p["patience"]):
                break
    net.combined.load_state_dict(best_state)
    return {"epochs": best_epoch, "stop_loss": best}


def finetune(net: TwoBranchNet, data, y: torch.Tensor, idx, p: dict, delta: float | None, lam: float,
             on_log: Callable[[dict], None] | None = None, on_epoch: Callable[[int], None] | None = None,
             log_every: int = 100) -> None:
    """Step 3: everything after the CNN blocks on step3_loss (delta, lam), AdamW with each part's learning rate
    p["learning_rate"][part] and weight decay p["weight_decay"][part], p["epochs"] epochs over the stock-days idx. The
    blocks stay frozen in eval mode; the combined MLP's BatchNorm keeps updating. Every log_every batches on_log gets
    the epoch, the batch, the gradient share, the weight share and the summaries' drift from the statistics the
    combined BatchNorm had at the start; on_epoch gets each finished epoch. A last batch of one row is skipped."""
    groups = prepare(net, "3")
    opt = torch.optim.AdamW([{"params": ws, "lr": p["learning_rate"][k], "weight_decay": p["weight_decay"][k]}
                             for k, ws in groups.items()])
    bn = net.combined.layers[0]
    ref_mean, ref_sd = bn.running_mean.clone(), torch.sqrt(bn.running_var + bn.eps)
    for epoch in range(1, int(p["epochs"]) + 1):
        prepare(net, "3")
        for i, b in enumerate(batches(idx, int(p["batch_size"]), True, data.device)):
            if len(b) < 2:
                continue
            out = net(*data.draw(b))
            log = None
            if on_log is not None and i % log_every == 0:
                log = {"epoch": epoch, "batch": i, **grad_share(out, y[b], delta), **weight_share(net.combined),
                       **summary_drift(out["s_p"], out["s_t"], ref_mean, ref_sd)}
            batch_loss = step3_loss(out, y[b], delta, lam)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            if log is not None:
                on_log(log)
        if on_epoch is not None:
            on_epoch(epoch)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_two_branch_train.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: every test passes.

---

### Task 4: The data, each fold's rows, and the output writers

**Files:**
- Modify: `src/dsr/two_branch_train.py` (append)
- Test: `tests/test_two_branch_train.py` (append)

**Interfaces:**
- Consumes: `dsr.walkforward.fold_rows`, `dsr.walkforward.last_months`, `dsr.walkforward.Fold`, `dsr.evaluation.summarize`.
- Produces (in `dsr.two_branch_train`):
  - `PRICE_EVERY = 9`, `HOLD_IC_EVERY = 5`, `STOP_MONTHS = 3`
  - `class TwoBranchData(windows, features, device)` with `device`, `features`, `draw(idx) -> (img, tab)`; raises `ValueError` on missing features
  - `fold_indices(keys, fold, target, hold_out_months=12) -> dict` with index arrays `train`, `train_price`, `hold`, `hold_ic`, `hold_fit`, `hold_stop`, `test` and the float `scale`; `keys` has columns `day`, `permno`, `day_pos`, `large_cap`, the target
  - `scored_frame(keys, idx, pred, scale, fold, target) -> DataFrame` (columns `day`, `permno`, `fold`, `pred`, target, `ret_next`, `large_cap`)
  - `write_fold(folder, fold, frame) -> None`, `fold_done(folder, fold) -> bool`, `collect(folder, target) -> dict` (the `oos_1b` summary)
  - `class Balance(path)` with `add(line: dict) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_two_branch_train.py`:

```python
import pandas as pd  # noqa: E402

from dsr.two_branch_train import (Balance, TwoBranchData, collect, fold_done, fold_indices, scored_frame,  # noqa: E402
                                  write_fold)
from dsr.walkforward import folds  # noqa: E402


def _keys():
    days = pd.bdate_range("2014-01-02", "2019-12-31")
    keys = pd.DataFrame([(d, p) for d in days for p in (1, 2, 3)], columns=["day", "permno"])
    keys["day_pos"] = keys.groupby("permno").cumcount()
    keys["large_cap"] = keys["permno"] != 3
    rng = np.random.default_rng(0)
    keys["target_demeaned"] = rng.normal(0, 0.02, len(keys))
    keys.loc[keys.sample(frac=0.05, random_state=0).index, "target_demeaned"] = np.nan
    keys["ret_next"] = keys["target_demeaned"]
    return keys


def test_fold_indices_split_the_fold_as_the_spec_says():
    keys = _keys()
    fold = folds(keys["day"].unique(), first="2019-01-01", every="year")[0]
    ix = fold_indices(keys, fold, "target_demeaned")
    k = lambda name: keys.iloc[ix[name]]
    has_target = lambda name: k(name)["target_demeaned"].notna().all()
    assert k("train")["day"].max() < pd.Timestamp("2018-01-01") and set(k("train")["permno"]) == {1, 2, 3}
    assert has_target("train") and has_target("hold") and has_target("train_price")
    assert set(ix["train_price"]) <= set(ix["train"])
    assert ((k("train_price")["day_pos"] + k("train_price")["permno"]) % 9 == 0).all()
    assert k("hold")["day"].min() >= pd.Timestamp("2018-01-01") and k("hold")["large_cap"].all()
    assert ((k("hold_ic")["day_pos"] + k("hold_ic")["permno"]) % 5 == 0).all() and set(ix["hold_ic"]) <= set(ix["hold"])
    assert sorted(np.r_[ix["hold_fit"], ix["hold_stop"]]) == sorted(ix["hold"])
    assert k("hold_stop")["day"].min() > k("hold_fit")["day"].max()
    assert k("hold_stop")["day"].min() > pd.Timestamp("2018-09-30")
    assert k("test")["day"].dt.year.eq(2019).all() and k("test")["large_cap"].all()
    assert k("test")["target_demeaned"].isna().any()  # the test year is predicted whatever its target
    large_train = keys.iloc[ix["train"]].query("large_cap")["target_demeaned"]
    assert ix["scale"] == pytest.approx(float(np.std(large_train.to_numpy(np.float32))), rel=1e-5)


def test_the_data_refuse_missing_features():
    with pytest.raises(ValueError, match="missing"):
        TwoBranchData(windows=None, features=np.array([[0.0, np.nan]]), device=DEVICE)


def test_a_fold_written_twice_is_collected_once(tmp_path):
    keys = _keys().dropna().reset_index(drop=True)
    idx = np.arange(0, 600)
    frame = scored_frame(keys, idx, torch.ones(600), 0.02, "2019", "target_demeaned")
    assert frame["pred"].eq(0.02).all() and list(frame.columns) == ["day", "permno", "fold", "pred",
                                                                     "target_demeaned", "ret_next", "large_cap"]
    assert not fold_done(tmp_path / "out", "2019")
    write_fold(tmp_path / "out", "2019", frame)
    write_fold(tmp_path / "out", "2019", frame)
    assert fold_done(tmp_path / "out", "2019")
    summary = collect(tmp_path / "out", "target_demeaned")
    assert len(pd.read_parquet(tmp_path / "out" / "predictions.parquet")) == 600 and "ic_mean" in summary
    assert (tmp_path / "out" / "summary.json").exists()


def test_the_balance_log_is_readable_while_it_grows(tmp_path):
    log = Balance(tmp_path / "balance.parquet")
    log.add({"fold": "2019", "step": "2", "weight_share_price": 0.4})
    log.add({"fold": "2019", "step": "3", "weight_share_price": 0.3})
    assert len(pd.read_parquet(tmp_path / "balance.parquet")) == 2
    assert len(Balance(tmp_path / "balance.parquet").rows) == 2  # a resumed run keeps the earlier lines
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_two_branch_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'Balance' from 'dsr.two_branch_train'`.

- [ ] **Step 3: Write the data, the fold rows and the writers**

In `src/dsr/two_branch_train.py`, replace the import block with:

```python
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dsr.evaluation import summarize
from dsr.two_branch import TwoBranchNet, grad_share, loss, prepare, step3_loss, summary_drift, weight_share
from dsr.walkforward import Fold, fold_rows, last_months

PRICE_EVERY, HOLD_IC_EVERY = 9, 5  # step 1a trains on 1 day in 9 of each stock and stops on 1 in 5 of the hold-out
STOP_MONTHS = 3  # step 2 stops on the hold-out year's last 3 months
```

and append:

```python
class TwoBranchData:
    """Every panel stock-day's chart and features on one device. windows: a dsr.cnn.HourWindows whose target i is row i
    of the features; draw(idx) gives the charts' grey channel (B x 1 x 96 x 444, pixels 0-255) and the features."""

    def __init__(self, windows, features, device):
        f = np.asarray(features, dtype=np.float32)
        if np.isnan(f).any():
            raise ValueError("the features have missing values: the model panel holds them as 0")
        self.windows, self.device = windows, torch.device(device)
        self.features = torch.as_tensor(f, device=self.device)

    def draw(self, idx):
        return self.windows.draw(idx)[:, :1], self.features[idx]


def fold_indices(keys: pd.DataFrame, fold: Fold, target: str, hold_out_months: int = 12) -> dict:
    """One fold's rows, as positions in keys (day, permno, day_pos, large_cap and the target):

    - train: every stock-day of the training years with a target (steps 1b);
    - train_price: those on one day in PRICE_EVERY of each stock, staggered ((day_pos + permno) % 9 == 0; step 1a);
    - hold: the hold-out year's large caps with a target (steps 2 and 3), hold_ic the one day in HOLD_IC_EVERY step
      1a stops on, hold_fit its first months and hold_stop its last STOP_MONTHS (step 2);
    - test: the predicted year's large caps, whatever their target;
    - scale: the sd of the target over the training years' large caps, which every head's target is divided by."""
    y = keys[target].to_numpy(np.float32)
    day, large = keys["day"].to_numpy(), keys["large_cap"].to_numpy(bool)
    train, hold, rows = fold_rows(day, y, large, fold, hold_out_months, holds_out=True)
    stagger = (keys["day_pos"] + keys["permno"]).to_numpy()
    hold = hold & large
    stop_days = last_months(pd.DatetimeIndex(np.unique(day[hold])), STOP_MONTHS)
    stopping = np.isin(day, stop_days.to_numpy().astype(day.dtype))
    return {"train": np.flatnonzero(train), "train_price": np.flatnonzero(train & (stagger % PRICE_EVERY == 0)),
            "hold": np.flatnonzero(hold), "hold_ic": np.flatnonzero(hold & (stagger % HOLD_IC_EVERY == 0)),
            "hold_fit": np.flatnonzero(hold & ~stopping), "hold_stop": np.flatnonzero(hold & stopping),
            "test": np.flatnonzero(rows & large), "scale": float(np.std(y[train & large])) or 1.0}


def scored_frame(keys: pd.DataFrame, idx, pred, scale: float, fold: str, target: str) -> pd.DataFrame:
    """The predictions of the stock-days idx (scaled units, a tensor or an array) scaled back, with their keys, target
    and return, as the walk-forwards write them."""
    k = keys.iloc[np.asarray(idx)]
    p = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    return pd.DataFrame({"day": k["day"].to_numpy(), "permno": k["permno"].to_numpy(), "fold": fold,
                         "pred": p.astype(np.float64) * scale, target: k[target].to_numpy(),
                         "ret_next": k["ret_next"].to_numpy(), "large_cap": k["large_cap"].to_numpy(bool)})


def write_fold(folder: Path, fold: str, frame: pd.DataFrame) -> None:
    """One fold's scored predictions to folder/folds/<fold>.parquet, replacing an earlier write of the same fold."""
    (folder / "folds").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "folds" / f"{fold}.parquet")


def fold_done(folder: Path, fold: str) -> bool:
    """Whether the fold is already written to folder: a stopped command skips it when started again."""
    return (folder / "folds" / f"{fold}.parquet").exists()


def collect(folder: Path, target: str) -> dict:
    """Every fold written to folder, as folder/predictions.parquet and folder/summary.json ({"oos_1b": the summary}),
    the other walk-forwards' layout, which scripts/models/05_backtest.py reads; returns the summary."""
    scored = pd.concat([pd.read_parquet(f) for f in sorted((folder / "folds").glob("*.parquet"))], ignore_index=True)
    scored.to_parquet(folder / "predictions.parquet")
    summary = summarize(scored, target)
    (folder / "summary.json").write_text(json.dumps({"oos_1b": summary}, indent=2))
    return summary


class Balance:
    """The balance diagnostics: each line printed and appended to `path`, a parquet rewritten after every line so it can
    be read while the run goes on; a resumed run keeps the lines already there."""

    def __init__(self, path: Path):
        self.path = path
        self.rows = pd.read_parquet(path).to_dict("records") if path.exists() else []

    def add(self, line: dict) -> None:
        self.rows.append(line)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(self.path)
        print("two-branch " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in line.items()),
              flush=True)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_two_branch_train.py -v`
Expected: 9 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: every test passes.

---

### Task 5: The walk-forward script

**Files:**
- Create: `scripts/models/10_two_branch_walkforward.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4; `dsr.cnn.HourWindows(prices, shares, colors, row_of, early, stock, day, device)`; `dsr.cnn.xavier`; `dsr.evaluation.daily_rank_ic`; `dsr.walkforward.feature_columns`, `folds`.
- Produces: the commands below, and under `--out` (default `data/walkforward/two_branch/`): one folder per scored output (`tab_s1_delta_<0.5|1|2|mse>`, `price_s1`, `tab_s1`, `comb_s2_<setting>`, `comb_s3`, `price_s3`, `tab_s3`), each with `folds/<fold>.parquet`, `predictions.parquet` and `summary.json`; `balance.parquet`; `choices.json` (`delta`, `delta_name`, `delta_ic`, `step2`, `step2_ic`); `checkpoints/<fold>/` with `tab_delta_<name>.pt`, `price.pt` (the price branch, its best epoch and hold-out IC), `summaries.pt` (the frozen branches' outputs on the hold-out and test years), `comb_<setting>.pt` (with the epoch cap it was trained under), `stage3.pt`. Rerunning `stage12` reuses `price.pt` and `summaries.pt`, so it never retrains or reruns the CNN; it runs only the step 2 settings whose outputs are missing.

There are no unit tests for the script (the repository does not test scripts); Step 3 below runs it end to end on a small sample.

- [ ] **Step 1: Write the script**

Create `scripts/models/10_two_branch_walkforward.py`:

````python
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Walk-forward of the two-branch model

    The model of `explanations/CNN/fusion.md`: the 21-day hourly chart (grey) through a densely linked CNN and the 1D block, the 17 tabular features through the tuned MLP's layer, and a combined MLP on the two 16-number summaries (`dsr.two_branch`, trained by `dsr.two_branch_train`). Yearly folds over 2019-2023, run in the spec's order:

    1. `delta`: step 1b (the tabular branch alone) in every fold with δ = 0.5, 1 and 2 and squared error; the δ with the highest `oos_1b/ic_mean` is kept (squared error is never chosen);
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
    from dsr.two_branch_train import (Balance, TwoBranchData, collect, finetune, fold_done, fold_indices, outputs,
                                      scored_frame, tabular_predictions, train_combined, train_price, train_tabular,
                                      write_fold)
    from dsr.walkforward import feature_columns, folds

    return (
        Balance, COMBINED, CombinedMLP, HourWindows, Path, TwoBranchData, TwoBranchNet, argparse, collect,
        daily_rank_ic, feature_columns, finetune, fold_done, fold_indices, folds, json, loss, mo, np, outputs, pd,
        prepare, scored_frame, step3_loss, summary_drift, sys, tabular_predictions, time, torch, train_combined,
        train_price, train_tabular, weight_share, write_fold, xavier,
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
    DELTAS = {"0.5": 0.5, "1": 1.0, "2": 2.0, "mse": None}  # step 1b's grid; squared error is reported, never chosen
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
    folds, json, load, loss, np, outputs, pd, prepare, scored_frame, step3_loss, summary_drift, tabular_predictions,
    time, torch, train_combined, train_price, train_tabular, weight_share, write_fold, xavier,
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
                train_tabular(net, data.features, y, ix["train"], p, delta)
                (out / "checkpoints" / fold.name).mkdir(parents=True, exist_ok=True)
                torch.save(net.tabular.state_dict(), out / "checkpoints" / fold.name / f"tab_delta_{name}.pt")
                pred = tabular_predictions(net, data.features, ix["test"])
                write_fold(folder, fold.name, scored_frame(keys, ix["test"], pred, ix["scale"], fold.name, TARGET))
                print(f"two-branch {fold.name} step 1b delta {name}: {time.perf_counter() - start:.0f} s", flush=True)
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
            if all(fold_done(out / f"comb_s2_{s}", fold.name) for s in STEP2):
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
                if i == 0 and info["score"] <= 0:
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
                if i == 0 and (end["corr_price"] < 0.05 or end["weight_share_price"] < 0.10):
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
            if fold_done(out / "comb_s3", fold.name):
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
                if i == 0 and not warned and e["weight_share_price"] < share_2 / 2:
                    warned.append(True)
                    print("two-branch WARNING SIGN 3: fine-tuning halves the chart's weight share "
                          f"({share_2:.3f} -> {e['weight_share_price']:.3f}); stop and look", flush=True)

            def on_epoch(epoch):
                s = outputs(net, data, sample)
                balance.add({"fold": fold.name, "step": "3 epoch end", "lam": lam, "epoch": epoch,
                             "corr_price": spearman(s["comb"], s["price"]), "corr_tab": spearman(s["comb"], s["tab"]),
                             **summary_drift(s["s_p"], s["s_t"], ref_mean, ref_sd), **weight_share(net.combined)})

            finetune(net, data, y, ix["hold"], p, delta, lam, on_log=on_log, on_epoch=on_epoch)
            (out / "checkpoints" / fold.name).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), out / "checkpoints" / fold.name / "stage3.pt")
            test_out = outputs(net, data, ix["test"])
            for name, head in (("comb_s3", "comb"), ("price_s3", "price"), ("tab_s3", "tab")):
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
````

- [ ] **Step 2: Check the script parses as a marimo notebook**

Run: `uv run marimo check scripts/models/10_two_branch_walkforward.py && uv run python -c "import ast; ast.parse(open('scripts/models/10_two_branch_walkforward.py').read())"`
Expected: no output from `marimo check`, and no error.

- [ ] **Step 3: Run it end to end on a small sample**

Run each, in order:

```bash
uv run python scripts/models/10_two_branch_walkforward.py delta --stocks 300 --folds 1 --quick --out /tmp/two_branch_check
uv run python scripts/models/10_two_branch_walkforward.py stage12 --stocks 300 --folds 1 --quick --out /tmp/two_branch_check
uv run python scripts/models/10_two_branch_walkforward.py choose --out /tmp/two_branch_check
uv run python scripts/models/10_two_branch_walkforward.py stage3 --stocks 300 --folds 1 --quick --out /tmp/two_branch_check
uv run python scripts/models/10_two_branch_walkforward.py stage3 --stocks 300 --folds 1 --quick --lam 0 --source /tmp/two_branch_check --out /tmp/two_branch_check_lambda0
```

Expected: each prints its result dictionary; `/tmp/two_branch_check/` holds `choices.json`, `balance.parquet`, `checkpoints/2019/` and the output folders `tab_s1_delta_*`, `price_s1`, `tab_s1`, `comb_s2_*` (6), `comb_s3`, `price_s3`, `tab_s3`, each with `predictions.parquet` and `summary.json`. Then run `stage12` once more with the same arguments: it prints its result at once, skipping the finished fold. Then check that step 2 reruns alone: `rm -r /tmp/two_branch_check/comb_s2_*/folds` and run `stage12 --stocks 300 --folds 1 --quick --max-epochs 3 --out /tmp/two_branch_check`; the log shows step 2 lines and no step 1a line, and it ends within a minute.

- [ ] **Step 4: Check the balance log has every measure**

Run: `uv run python -c "import pandas as pd; b = pd.read_parquet('/tmp/two_branch_check/balance.parquet'); print(b['step'].value_counts()); print(sorted(b.columns))"`
Expected: steps `1a`, `2`, `2 end`, `3`, `3 epoch end`; columns include `holdout_score`, `stop_loss`, `weight_share_price`, `corr_price`, `corr_tab`, `grad_share_price`, `drift_shift_price`, `drift_ratio_tab`.

---

### Task 6: The backtest reads the new outputs and the benchmarks

**Files:**
- Modify: `scripts/models/05_backtest.py` (the settings cell with `MODELS`)

**Interfaces:**
- Consumes: the output folders of Task 5 and `data/walkforward/lgbm_yearly`, `data/walkforward/lgbm_yearly_refit`.
- Produces: backtest rows for every model in `MODELS` whose `predictions.parquet` exists (the loop already skips missing ones).

- [ ] **Step 1: Add the models**

In `scripts/models/05_backtest.py`, replace

```python
    MODELS = ("linear", "mlp", "lgbm", "cnn_one_day")
```

with

```python
    MODELS = ("linear", "mlp", "lgbm", "lgbm_yearly", "lgbm_yearly_refit", "cnn_one_day",  # the benchmark: lgbm_yearly_refit
              "two_branch/comb_s3", "two_branch/tab_s3", "two_branch/price_s3", "two_branch/price_s1",
              "two_branch/tab_s1", "two_branch_lambda0/comb_s3")
```

and in the same cell, after it, add the chosen step 2 setting when the run has chosen one:

```python
    _choices = WALKFORWARD / "two_branch" / "choices.json"
    if _choices.exists():
        import json  # noqa: PLC0415
        MODELS = MODELS + (f"two_branch/comb_s2_{json.loads(_choices.read_text())['step2']}",)
```

(`WALKFORWARD` is defined in the same cell above `MODELS`; keep that order.)

- [ ] **Step 2: Check it runs**

Run: `uv run marimo check scripts/models/05_backtest.py && uv run python scripts/models/05_backtest.py`
Expected: no error (the two-branch folders don't exist yet and are skipped).

---

### Task 7: Remove the first fused model and update the docs

**Files:**
- Modify: `src/dsr/cnn.py` (delete `EMBED, REDUCE`, `FusionNet`, `ImagesWithFeatures`, and `fit`'s handling of them)
- Delete: `scripts/models/09_fusion_walkforward.py`
- Modify: `tests/test_cnn.py` (delete `test_the_fused_model_learns_from_the_image_and_the_features_and_its_control_from_the_features_alone`)
- Modify: `explanations/CNN/fusion.md` (Code section), `explanations/CNN/cnn.md` (Stopped section)

**Interfaces:**
- Consumes: nothing new.
- Produces: `dsr.cnn` without the fused model; `dsr.cnn.fit` as before the fused model.

- [ ] **Step 1: Delete the fused model's test**

In `tests/test_cnn.py`, delete the whole function `test_the_fused_model_learns_from_the_image_and_the_features_and_its_control_from_the_features_alone` (from its `def` line to the line `assert ChartCNN().head[2].in_features == 128 * 3 * WIDTH  # the plain CNN is unchanged`).

- [ ] **Step 2: Delete the fused model from `src/dsr/cnn.py`**

Delete everything from the line `# the chart and the tabular features learned together: the grey one-day image's CNN and the model panel's features` to the end of `class ImagesWithFeatures` (its `network` method's `return FusionNet(...)`), and change the module docstring's first line from

```python
"""The CNN on chart images (explanations/CNN/cnn.md; the fused model: fusion.md): each large-cap
```

to

```python
"""The CNN on chart images (explanations/CNN/cnn.md; the two-branch model built on it: dsr.two_branch): each large-cap
```

In `fit`, replace

```python
    mean = sd = None
    if getattr(images, "image", True):  # the images' channel means and sds (draw may give (image, features))
        with torch.no_grad():
            step = images.predict_batch
            first = lambda batch: batch[0] if isinstance(batch, tuple) else batch
            x = torch.cat([first(images.draw(sample[i:i + step])) for i in range(0, len(sample), step)])
            mean, sd = x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)).clamp_min(1e-6)
```

with

```python
    with torch.no_grad():  # the images' channel means and sds
        step = images.predict_batch
        x = torch.cat([images.draw(sample[i:i + step]) for i in range(0, len(sample), step)])
        mean, sd = x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)).clamp_min(1e-6)
```

- [ ] **Step 3: Delete the fused model's script**

Run: `rm scripts/models/09_fusion_walkforward.py`

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q && uv run python -c "import dsr.cnn as c; assert not hasattr(c, 'FusionNet') and not hasattr(c, 'ImagesWithFeatures')"`
Expected: every test passes; no error.

- [ ] **Step 5: Update the docs**

In `explanations/CNN/fusion.md`, section Code, replace the line starting `` - `src/dsr/cnn.py`: the two-branch network `` with:

```markdown
- `src/dsr/two_branch.py`: the network (the dense blocks with time dilation 1, 3 and 9, the 1D block, the heads, the combined MLP), the loss, which parts each step trains and the balance diagnostics; `src/dsr/two_branch_train.py`: the data on the GPU, each fold's rows, the four step trainers and the output writers. The speed check before the run (`10_two_branch_walkforward.py speed`) times the dilated blocks on the Mac's GPU.
```

and replace the line starting `` - `scripts/models/10_two_branch_walkforward.py`: `` with:

```markdown
- `scripts/models/10_two_branch_walkforward.py`: the walk-forward as commands in the spec's order (`delta`, `stage12`, `choose`, `stage3`; `speed` first), one folder per scored output under `data/walkforward/two_branch/` (`comb_s3` is the model), the balance diagnostics in `balance.parquet`, the choices in `choices.json`; the control with `stage3 --lam 0 --source data/walkforward/two_branch --out data/walkforward/two_branch_lambda0`, the backup with `--channels 8 --out data/walkforward/two_branch_c8`.
```

and replace the line starting `` - Removed: `FusionNet` `` with:

```markdown
- Removed: `FusionNet`, `ImagesWithFeatures` and `scripts/models/09_fusion_walkforward.py`, the earlier fused model this one replaces.
```

In `explanations/CNN/cnn.md`, section Stopped, replace `` **The first fused model** (`scripts/models/09_fusion_walkforward.py`, `fusion_grey`) `` with `` **The first fused model** (`fusion_grey`; its code was removed when the two-branch model replaced it) ``.

---

### Task 8: Speed check, then the run

**Files:** none (runs only). Logs go to `data/logs/two_branch/`.

- [ ] **Step 1: Time the network**

Run: `uv run python scripts/models/10_two_branch_walkforward.py speed`
Expected: a dictionary with images a second for steps 1a and 3 and the hours they imply. Report it to the user before going on. If one epoch of step 1a over the 5 folds takes more than about 3 hours (the spec's budget is about 9 hours for at most 5 epochs with early stopping), say so and stop for a decision: the dilated convolutions may be slow on the GPU.

- [ ] **Step 2: Step 1b and δ**

Run:

```bash
mkdir -p data/logs/two_branch
caffeinate -is uv run python scripts/models/10_two_branch_walkforward.py delta > data/logs/two_branch/delta.log 2>&1
```

Expected: minutes; the last line shows `oos_1b/ic_mean` for δ = 0.5, 1, 2 and squared error, and the chosen δ. Report the table to the user (a spread under about 0.003 means the choice was noise).

- [ ] **Step 3: Steps 1a and 2, watching the first fold**

Run in the background:

```bash
nohup caffeinate -is uv run python scripts/models/10_two_branch_walkforward.py stage12 > data/logs/two_branch/stage12.log 2>&1 &
```

After the first fold (about 2 hours), run `grep -E "WARNING|2019 steps" data/logs/two_branch/stage12.log`. If a warning sign printed, stop the run (`pkill -f "10_two_branch_walkforward.py stage12"`) and report to the user. If warning sign 1 printed, the spec's backup applies (8 channels, rerun once); the decision to start it is the user's.

- [ ] **Step 4: Choose step 2's setting**

Run: `uv run python scripts/models/10_two_branch_walkforward.py choose`
Expected: the table of the 6 settings' `oos_1b/ic_mean`, the chosen one, the spread, and any fold whose best epoch hit the cap of 40. If a fold hit the cap, raise it as the spec says before step 3: `rm -r data/walkforward/two_branch/comb_s2_*/folds`, then `uv run python scripts/models/10_two_branch_walkforward.py stage12 --max-epochs 80` (it reuses every fold's saved price branch and summaries and reruns step 2 alone, in minutes), then `choose` again. Then check the backup trigger: if the pooled `oos_1b/ic_mean` of `price_s1` (in `stage12`'s last line) is below 0.0038, stop and report it: the spec's backup (8 channels, rerun once) applies, and starting it is the user's decision.

- [ ] **Step 5: Step 3 and the control**

Run:

```bash
nohup caffeinate -is sh -c 'uv run python scripts/models/10_two_branch_walkforward.py stage3 > data/logs/two_branch/stage3.log 2>&1 && uv run python scripts/models/10_two_branch_walkforward.py stage3 --lam 0 --source data/walkforward/two_branch --out data/walkforward/two_branch_lambda0 > data/logs/two_branch/stage3_lambda0.log 2>&1' &
```

Watch `grep WARNING data/logs/two_branch/stage3.log` during the first fold.

- [ ] **Step 6: Backtest and report**

Run: `uv run python scripts/models/05_backtest.py` (or open it in marimo), then report to the user, for 2019-2023 on stocks of $1B or more: IC (t), decile spread (t) and Sharpe of `two_branch/comb_s3` against the success rule's two bars (the tabular head after step 3, `two_branch/tab_s3`; LightGBM refit through the hold-out year, IC 0.0118 and Sharpe 1.25), with the step 1 heads, step 2's output and the λ = 0 control alongside, and the paired daily differences with their t-statistics. State plainly whether the model is adopted under the spec's rule. Do not run the test years.

---

## Self-review notes

- **Spec coverage:** network and weights (Task 1); Huber loss, λ, frozen parts, BatchNorm modes, balance diagnostics (Task 2); steps 1a, 1b, 2 and 3 (Task 3); rows per step, sampling 1 in 9 and 1 in 5, the 9 + 3 month split, the target scale, outputs (Task 4); run order, δ grid, step 2's 6 settings, warning signs 1-3, the fixed 20,000 sample, the λ = 0 control, the 8-channel backup flag (Task 5); the benchmark and the scored outputs in the backtest (Task 6); the removed fused model and the docs (Task 7); speed check, run, the backup trigger, the success rule (Task 8).
- **Not in this plan, by the spec:** the test years (built and run only if the model is adopted).
