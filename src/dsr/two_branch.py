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


def head_losses(out: dict, y: torch.Tensor, delta: float | None) -> dict[str, torch.Tensor]:
    """The three heads' losses: L_price, L_tab and L_comb, by head."""
    return {head: loss(out[head], y, delta) for head in ("price", "tab", "comb")}


def combine_losses(losses: dict, lam: float = 0.5) -> torch.Tensor:
    """L_comb + lam / (1 - lam) * (L_price + L_tab): each branch body learns from lam * its own loss + (1 - lam) * the
    combined loss, times 1 / (1 - lam), which AdamW's steps ignore. lam = 0 is the control: the combined loss alone."""
    return losses["comb"] + lam / (1 - lam) * (losses["price"] + losses["tab"])


def step3_loss(out: dict, y: torch.Tensor, delta: float | None, lam: float = 0.5) -> torch.Tensor:
    """Step 3's loss (combine_losses) of the net's outputs."""
    return combine_losses(head_losses(out, y, delta), lam)


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
