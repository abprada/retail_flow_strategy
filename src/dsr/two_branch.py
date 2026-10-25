"""The two-branch model (explanations/CNN/fusion.md): the 21-day hourly chart through a densely linked CNN and the 1D
block, the 17 tabular features through the tuned MLP's layer, each ending in a 16-number summary and its own head, and
a combined MLP on the two summaries, each normalized as a branch; the loss, which parts each training step trains, and
the balance diagnostics."""

import torch
from torch import nn
from torch.nn import functional as F

from dsr.cnn import HEIGHT as DAY_HEIGHT
from dsr.cnn import WIDTH as DAY_WIDTH
from dsr.cnn import W_WIDTH, blocks_height, conv_blocks, xavier

FILTERS = (16, 16, 16)  # the three blocks' filters
TIME_DILATION = (1, 3, 9)  # their time dilations: a block 3 value sees 27 columns (9 bars), with no gaps
ROWS = 4  # the image rows left after the blocks: 96 -> 32 (stride 3) -> 16 -> 8 -> 4 (the pools)
BARS = W_WIDTH // 3  # 148: each bar is 3 columns
SUMMARY = 16  # each branch's summary
WIDTH = 16  # the 1D block's channels (32: the backup)
STRIDED = 3  # the 1D block's convolutions that halve time: 148 bars -> 74 -> 37 -> 19 steps
DENSE = 32  # the 1D block's dense layer between the flattened steps and the summary
DAY_FILTERS = (32, 64, 128)  # the one-day chart's CNN: the one-day CNN's blocks (dsr.cnn.conv_blocks)
DAY_DROPOUT = 0.5  # the one-day CNN's dropout before its readout
CHARTS = ("hours", "day", "day_cnn")  # the 21-day hourly chart; the one-day chart with the 1D block; the one-day CNN
DROPOUT = 0.54  # the tabular head's and the combined MLP's
N_FEATURES = 17
COMBINED = {"16": (16,), "16-8": (16, 8)}  # the combined MLP's hidden layers, the two step 2 tries
JOINT_TABULAR, JOINT_COMBINED = (16, 8), (8,)  # the joint run's tabular MLP (17 -> 16 -> 8 -> 1) and head (9 -> 8 -> 1)


def pool_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """x (B x C x H x W) average-pooled over its rows down to `rows` (H a multiple of rows), the columns untouched."""
    return F.avg_pool2d(x, (x.shape[2] // rows, 1))


class DenseBlocks(nn.Module):
    """Three blocks of a 5x3 convolution, BatchNorm, LeakyReLU and a 2x1 average pool (the rows only), densely linked:
    each block reads the image and every earlier block, average-pooled to its height. The first block moves 3 rows at
    a time; `dilation` spaces each block's 3 time taps. Returns the image and the three blocks, each average-pooled to
    ROWS rows, side by side: B x (1 + sum(filters)) x ROWS x W. Average pools, not max pools, so every pixel's gradient
    gets through."""

    def __init__(self, filters=FILTERS, dilation=TIME_DILATION):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i, (c, d) in enumerate(zip(filters, dilation)):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(1 + sum(filters[:i]), c, (5, 3), stride=(3, 1) if i == 0 else 1, padding=(2, d),
                          dilation=(1, d)),
                nn.BatchNorm2d(c), nn.LeakyReLU(), nn.AvgPool2d((2, 1))))
        self.channels = 1 + sum(filters)

    def forward(self, x):
        outs = [x]
        for block in self.blocks:
            rows = outs[-1].shape[2]
            outs.append(block(torch.cat([pool_rows(o, rows) for o in outs], 1)))
        return torch.cat([pool_rows(o, ROWS) for o in outs], 1)


def _norm_act(layer: nn.Module, n: int) -> list[nn.Module]:
    return [layer, nn.BatchNorm1d(n), nn.LeakyReLU()]


class OneDBlock(nn.Module):
    """The blocks' output to the 16-number summary (explanations/CNN/fusion.md, the 1D block): an average over each
    bar's 3 columns; the rows stacked as channels, so each bar is a column of c_in x ROWS values; a 1D convolution of
    one bar to `width` channels; STRIDED 1D convolutions of 3 steps that halve time; flatten; a dense layer of DENSE and
    the summary. Every layer after the pool is followed by BatchNorm and LeakyReLU. `rows` and `bars`: the CNN output's
    rows and the chart's bars (the hourly chart's by default)."""

    def __init__(self, c_in: int, width: int = WIDTH, rows: int = ROWS, bars: int = BARS):
        super().__init__()
        self.pool, self.rows = nn.AvgPool2d((1, 3)), nn.Flatten(1, 2)
        layers, steps = _norm_act(nn.Conv1d(c_in * rows, width, 1), width), bars
        for _ in range(STRIDED):
            layers += _norm_act(nn.Conv1d(width, width, 3, stride=2, padding=1), width)
            steps = (steps + 1) // 2
        self.time = nn.Sequential(*layers)
        self.dense = nn.Sequential(nn.Flatten(), *_norm_act(nn.Linear(width * steps, DENSE), DENSE),
                                   *_norm_act(nn.Linear(DENSE, SUMMARY), SUMMARY))

    def forward(self, x):
        return self.dense(self.time(self.rows(self.pool(x))))


class PriceBranch(nn.Module):
    """The chart (pixels 0-255) through the CNN: the summary s_p, and the price head's prediction. chart "hours": the
    21-day hourly chart (B x 1 x 96 x 444) divided by 255, the dense blocks and the 1D block; "day": the one-day chart
    (B x 1 x 64 x 120) divided by 255, the one-day CNN's blocks (DAY_FILTERS, max pools, the first convolution dilated)
    and the 1D block; "day_cnn": the one-day CNN itself (dsr.cnn.ChartCNN, grey): the chart standardized with the
    training images' mean and sd, blocks 1-2 (`blocks`), block 3 (`one_d`) and its readout (`head`: flatten, dropout,
    one weight per block output), whose prediction is also the summary (one number)."""

    def __init__(self, width: int = WIDTH, chart: str = "hours"):
        super().__init__()
        self.chart = chart
        if chart == "day_cnn":
            f = conv_blocks(DAY_FILTERS, 1)
            self.blocks, self.one_d = f[:8], f[8:]  # blocks 1-2 (4 layers each); block 3
            n = DAY_FILTERS[-1] * blocks_height(DAY_HEIGHT, len(DAY_FILTERS)) * DAY_WIDTH
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(DAY_DROPOUT), nn.Linear(n, 1))
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("sd", torch.ones(1))
            return
        if chart == "hours":
            self.blocks = DenseBlocks()
            self.one_d = OneDBlock(self.blocks.channels, width)
        elif chart == "day":
            self.blocks = conv_blocks(DAY_FILTERS, 1)
            self.one_d = OneDBlock(DAY_FILTERS[-1], width, blocks_height(DAY_HEIGHT, len(DAY_FILTERS)), DAY_WIDTH // 3)
        else:
            raise ValueError(f"chart is one of {CHARTS}, not {chart!r}")
        self.head = nn.Linear(SUMMARY, 1)

    def forward(self, img):
        if self.chart == "day_cnn":
            pred = self.head(self.one_d(self.blocks((img - self.mean) / self.sd))).squeeze(1)
            return pred[:, None], pred
        s = self.one_d(self.blocks(img / 255))
        return s, self.head(s).squeeze(1)


def copy_chart_cnn(price: PriceBranch, cnn: nn.Module) -> None:
    """Load a trained dsr.cnn.ChartCNN (grey, DAY_FILTERS) into a "day_cnn" price branch: its blocks, its readout and
    its pixels' mean and sd."""
    with torch.no_grad():
        price.blocks.load_state_dict(cnn.features[:8].state_dict())
        price.one_d.load_state_dict(cnn.features[8:].state_dict())
        price.head.load_state_dict(cnn.head.state_dict())
        price.mean.copy_(cnn.mean)
        price.sd.copy_(cnn.sd)


class TabularBranch(nn.Module):
    """The tuned MLP's layer (17 -> 16, LeakyReLU) giving the summary s_t, taken before the head's dropout, and the
    tabular head's prediction. `hidden`: the layers, each linear and LeakyReLU, the last one the summary: the first in
    `body`, the others in `deep` (none in the tuned MLP's 17 -> 16; (16, 8): the joint run's MLP, 17 -> 16 -> 8, its 8
    numbers the summary)."""

    def __init__(self, n_features: int = N_FEATURES, hidden=(SUMMARY,)):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_features, hidden[0]), nn.LeakyReLU())
        layers = []
        for c, h in zip(hidden, hidden[1:]):
            layers += [nn.Linear(c, h), nn.LeakyReLU()]
        self.deep = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(hidden[-1], 1))

    def forward(self, x):
        s = self.deep(self.body(x))
        return s, self.head(s).squeeze(1)


class BranchNorm(nn.Module):
    """One branch's summary, each number centred and the branch divided by one scale: the root mean of its numbers'
    variances. The branch as a whole gets sd 1, and a number that barely moves stays small next to the others. In
    training, the batch's mean and variances, and running averages kept as BatchNorm keeps them (0.9 x the old + 0.1 x
    the batch's); at prediction, the running averages. No learned scale: the linear layer after it has one."""

    def __init__(self, n: int = SUMMARY, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def scale(self) -> torch.Tensor:
        return torch.sqrt(self.running_var.mean() + self.eps)

    def forward(self, s):
        if not self.training:
            return (s - self.running_mean) / self.scale()
        mean, var = s.mean(0), s.var(0, unbiased=False)
        with torch.no_grad():
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        return (s - mean) / torch.sqrt(var.mean() + self.eps)


class CombinedMLP(nn.Module):
    """Each summary normalized as a branch (BranchNorm), the two side by side, the hidden layers (each linear,
    LeakyReLU and dropout) and one output: the model's prediction. n_price and n_tab: the price and tabular summaries'
    numbers (n_price 1 when it is the price head's prediction)."""

    def __init__(self, hidden=(16,), n_price: int = SUMMARY, n_tab: int = SUMMARY):
        super().__init__()
        self.norm_p, self.norm_t = BranchNorm(n_price), BranchNorm(n_tab)
        layers, c = [], n_price + n_tab
        for h in hidden:
            layers += [nn.Linear(c, h), nn.LeakyReLU(), nn.Dropout(DROPOUT)]
            c = h
        self.layers = nn.Sequential(*layers, nn.Linear(c, 1))

    def forward(self, s_p, s_t):
        return self.layers(torch.cat([self.norm_p(s_p), self.norm_t(s_t)], 1)).squeeze(1)


class TwoBranchNet(nn.Module):
    """The price branch, the tabular branch and the combined MLP. forward gives the summaries s_p and s_t and the three
    predictions: price, tab and comb (the model's). tab_hidden: the tabular branch's layers (TabularBranch)."""

    def __init__(self, width: int = WIDTH, hidden=(16,), n_features: int = N_FEATURES, chart: str = "hours",
                 tab_hidden=(SUMMARY,)):
        super().__init__()
        self.price = PriceBranch(width, chart)
        self.tabular = TabularBranch(n_features, tab_hidden)
        self.combined = CombinedMLP(hidden, 1 if chart == "day_cnn" else SUMMARY, tab_hidden[-1])
        xavier(self)

    def forward(self, img, tab):
        s_p, price = self.price(img)
        s_t, tab_pred = self.tabular(tab)
        return {"s_p": s_p, "s_t": s_t, "price": price, "tab": tab_pred, "comb": self.combined(s_p, s_t)}


TRAINED = {"1a": ("blocks", "one_d", "price_head"), "1b": ("tab_body", "tab_deep", "tab_head"), "2": ("combined",),
           "3": ("one_d", "price_head", "tab_body", "combined"),
           "joint": ("one_d", "price_head", "tab_deep", "combined")}  # the parts each step trains
FIXED = {"3": ("one_d", "price_head"),  # trained parts whose BatchNorm statistics stay as step 1a left them, and
         "joint": ("one_d", "price_head")}  # dropout off


def parts(net: TwoBranchNet) -> dict[str, nn.Module]:
    """The net's parts, as the steps train or freeze them: tab_body is the tabular branch's first layer, tab_deep the
    layers after it (none in the tuned MLP's 17 -> 16; the joint run's 16 -> 8)."""
    return {"blocks": net.price.blocks, "one_d": net.price.one_d, "price_head": net.price.head,
            "tab_body": net.tabular.body, "tab_deep": net.tabular.deep, "tab_head": net.tabular.head,
            "combined": net.combined}


def prepare(net: TwoBranchNet, step: str) -> dict[str, list[nn.Parameter]]:
    """Make the parts `step` trains ("1a", "1b", "2" or "3"; TRAINED) trainable and in training mode, and freeze every
    other part in eval mode, so its BatchNorm statistics stay put. A trained part in FIXED keeps its BatchNorms and
    dropout in eval mode: in step 3 the price branch's statistics stay those step 2's summaries were made with, and its
    prediction is as noisy as at prediction time. Returns the trained parts' weights by part. Call it at the start of
    every epoch: predicting puts the whole net in eval mode."""
    trained = TRAINED[step]
    for name, module in parts(net).items():
        module.train(name in trained)
        for w in module.parameters():
            w.requires_grad_(name in trained)
        if name in FIXED.get(step, ()):
            for m in module.modules():
                if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
                    m.eval()
    return {name: list(parts(net)[name].parameters()) for name in trained}


def loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared error, every head's loss (in the target's scaled units)."""
    return F.mse_loss(pred, y)


def head_losses(out: dict, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """The three heads' losses: L_price, L_tab and L_comb, by head."""
    return {head: loss(out[head], y) for head in ("price", "tab", "comb")}


def combine_losses(losses: dict, lam: float = 0.5) -> torch.Tensor:
    """L_comb + lam / (1 - lam) * L_price: the price branch learns from lam * its own loss + (1 - lam) * the combined
    loss, times 1 / (1 - lam), which AdamW's steps ignore; the tabular branch, which the combined MLP leans on anyway,
    from the combined loss alone. lam = 0 is the control: the combined loss alone."""
    return losses["comb"] + lam / (1 - lam) * losses["price"]


def joint_loss(comb: torch.Tensor, y: torch.Tensor, price: torch.Tensor, y_charted: torch.Tensor,
               lam: float) -> torch.Tensor:
    """The joint run's loss, by stock-day over the batch's stock-days: (1 - lam) times the combined head's squared error
    on every one, plus lam times the CNN's on those with a chart (price, y_charted). A stock-day with a chart gives the
    CNN lam * L_price + (1 - lam) * L_comb; the tabular MLP and the head learn from L_comb alone."""
    return ((1 - lam) * (comb - y).pow(2).sum() + lam * (price - y_charted).pow(2).sum()) / len(y)


def step3_loss(out: dict, y: torch.Tensor, lam: float = 0.5) -> torch.Tensor:
    """Step 3's loss (combine_losses) of the net's outputs."""
    return combine_losses(head_losses(out, y), lam)


def grad_share(out: dict, y: torch.Tensor) -> dict:
    """The combined loss's gradient at the two summaries (each its norm per stock-day, averaged over the batch), and the
    price branch's share of the two. Only L_comb: the branches' own losses are left out."""
    g_p, g_t = torch.autograd.grad(loss(out["comb"], y), [out["s_p"], out["s_t"]], retain_graph=True)
    p, t = g_p.norm(dim=1).mean().item(), g_t.norm(dim=1).mean().item()
    return {"grad_price": p, "grad_tab": t, "grad_share_price": p / (p + t) if p + t > 0 else float("nan")}


def weight_share(combined: CombinedMLP) -> dict:
    """The combined MLP's first linear layer, its weights on the 16 price inputs against the 16 tabular ones, each
    column scaled by its input's sd after the branch normalization (the number's sd over its branch's scale), so a
    column counts by how much it moves the layer; as norms, and the price share."""
    sd = torch.cat([torch.sqrt(n.running_var) / n.scale() for n in (combined.norm_p, combined.norm_t)])
    w = (combined.layers[0].weight * sd).detach()
    n_price = combined.norm_p.running_mean.numel()
    p, t = w[:, :n_price].norm().item(), w[:, n_price:].norm().item()
    return {"weight_price": p, "weight_tab": t, "weight_share_price": p / (p + t) if p + t > 0 else float("nan")}


def norm_reference(combined: CombinedMLP) -> tuple[torch.Tensor, torch.Tensor]:
    """The combined normalization's statistics as they stand (after step 2): the summary numbers' running means, and
    the two branches' scales."""
    return (torch.cat([combined.norm_p.running_mean, combined.norm_t.running_mean]).clone(),
            torch.stack([combined.norm_p.scale(), combined.norm_t.scale()]).detach().clone())


def summary_drift(s_p: torch.Tensor, s_t: torch.Tensor, ref_mean: torch.Tensor, ref_scale: torch.Tensor) -> dict:
    """How far each branch's summary moved from step 2's statistics (norm_reference), in units of the branch's step 2
    scale: the largest shift of a number's mean, and the branch's spread (the root mean of its numbers' variances) over
    that scale."""
    out, n_price = {}, s_p.shape[1]
    for k, (name, s, mean) in enumerate((("price", s_p, ref_mean[:n_price]), ("tab", s_t, ref_mean[n_price:]))):
        s = s.detach()
        out[f"drift_shift_{name}"] = ((s.mean(0) - mean).abs().max() / ref_scale[k]).item()
        out[f"drift_ratio_{name}"] = (torch.sqrt(s.var(0, unbiased=False).mean()) / ref_scale[k]).item()
    return out
