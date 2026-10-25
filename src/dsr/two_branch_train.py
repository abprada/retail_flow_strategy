"""Training the two-branch model (explanations/CNN/fusion.md): the four steps (squared error throughout), the outputs
they are scored on, the data on the GPU, each fold's rows, and the writers of the scored outputs and of the balance
diagnostics."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dsr.evaluation import summarize
from dsr.two_branch import (TwoBranchNet, combine_losses, grad_share, head_losses, joint_loss, loss, norm_reference,
                            prepare, summary_drift, weight_share)
from dsr.walkforward import Fold, fold_rows

PRICE_EVERY, HOLD_IC_EVERY = 1, 5  # step 1a trains on every day of each large cap, stops on 1 day in 5 of the hold-out


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
                on_epoch: Callable[[dict], None] | None = None) -> dict:
    """Step 1a: the price branch alone on the squared error of its head, AdamW, at most p["max_epochs"] epochs over the
    stock-days `train`, stopped after p["patience"] epochs without a higher score(the price predictions of `hold`) (the
    hold-out's rank IC) and returned to the best epoch's weights. y: every row's scaled target, on the data's device."""
    params = [w for ws in prepare(net, "1a").values() for w in ws]
    opt = torch.optim.AdamW(params, lr=p["learning_rate"], weight_decay=p["weight_decay"])
    best, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    for epoch in range(1, int(p["max_epochs"]) + 1):
        prepare(net, "1a")
        total = torch.zeros((), device=data.device)  # kept on the GPU: reading each batch's loss would stall it
        for b in batches(train, int(p["batch_size"]), True, data.device):
            batch_loss = loss(net.price(data.draw(b)[0])[1], y[b])
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


def train_tabular(net: TwoBranchNet, features: torch.Tensor, y: torch.Tensor, train, p: dict,
                  on_epoch: Callable[[dict], None] | None = None, stop=None) -> dict:
    """Step 1b: the tabular branch alone on the squared error of its head, AdamW, over the rows `train` of features:
    p["epochs"] epochs, or, with `stop` rows, at most p["max_epochs"], stopped after p["patience"] epochs without a
    lower loss on them and returned to the best epoch's weights."""
    params = [w for ws in prepare(net, "1b").values() for w in ws]
    opt = torch.optim.AdamW(params, lr=p["learning_rate"], weight_decay=p["weight_decay"])
    best, best_state, best_epoch, stale = float("inf"), None, int(p.get("epochs", 0)), 0
    for epoch in range(1, int(p["max_epochs"] if stop is not None else p["epochs"]) + 1):
        prepare(net, "1b")
        total = torch.zeros((), device=features.device)
        for b in batches(train, int(p["batch_size"]), True, features.device):
            batch_loss = loss(net.tabular(features[b])[1], y[b])
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.detach() * len(b)
        line = {"epoch": epoch, "train_loss": float(total) / len(train)}
        if stop is not None:
            stop_t = torch.as_tensor(np.asarray(stop), device=features.device)
            line["stop_loss"] = float(loss(tabular_predictions(net, features, stop), y[stop_t]))
        if on_epoch is not None:
            on_epoch(line)
        if stop is None:
            continue
        if line["stop_loss"] < best:
            best, best_epoch, stale = line["stop_loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.tabular.state_dict().items()}
        else:
            stale += 1
            if stale >= int(p["patience"]):
                break
    if best_state is not None:
        net.tabular.load_state_dict(best_state)
    return {"epochs": best_epoch, "stop_loss": best}


def train_combined(net: TwoBranchNet, fit: tuple, stop: tuple, p: dict,
                   on_epoch: Callable[[dict], None] | None = None) -> dict:
    """Step 2: the combined MLP alone on stored summaries, fit = (s_p, s_t, y) of the rows it fits and stop = the same
    of the rows it stops on, squared error, AdamW, at most p["max_epochs"] epochs, stopped after p["patience"] epochs
    without a lower loss on the stop rows and returned to the best epoch (its normalization's statistics with it). A
    last batch of one row is skipped: a batch's variance needs two. on_epoch also gets the weight share."""
    s_p, s_t, y = fit
    opt = torch.optim.AdamW(prepare(net, "2")["combined"], lr=p["learning_rate"], weight_decay=p["weight_decay"])
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, int(p["max_epochs"]) + 1):
        prepare(net, "2")
        total = torch.zeros((), device=y.device)
        for b in batches(np.arange(len(y)), int(p["batch_size"]), True, y.device):
            if len(b) < 2:
                continue
            batch_loss = loss(net.combined(s_p[b], s_t[b]), y[b])
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            total += batch_loss.detach() * len(b)
        net.combined.eval()
        with torch.no_grad():
            v = float(loss(net.combined(stop[0], stop[1]), stop[2]))
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


def finetune(net: TwoBranchNet, data, y: torch.Tensor, idx, p: dict, lam: float,
             on_log: Callable[[dict], None] | None = None, on_epoch: Callable[[dict], None] | None = None,
             log_every: int = 100, seed: int = 0, stop=None) -> dict:
    """Step 3: the parts step 3 trains (prepare) on step 3's loss (combine_losses; lam), AdamW with each part's learning
    rate p["learning_rate"][part] and weight decay p["weight_decay"][part], over the stock-days idx, its shuffling and
    dropout seeded by `seed`, so the λ = 0 control is paired with the run: p["epochs"] epochs, or, with `stop` rows, at
    most p["max_epochs"], stopped after p["patience"] epochs without a lower combined loss on them and returned to the
    best epoch. The frozen parts stay in eval mode; the combined normalization keeps updating. Every log_every batches
    on_log gets the epoch, the batch, the gradient share, the weight share and the summaries' drift from the statistics
    the combined normalization had at the start; on_epoch gets each finished epoch with its average L_price, L_tab and
    L_comb (and the stop rows' combined loss). A last batch of one row is skipped."""
    torch.manual_seed(seed)
    groups = prepare(net, "3")
    opt = torch.optim.AdamW([{"params": ws, "lr": p["learning_rate"][k], "weight_decay": p["weight_decay"][k]}
                             for k, ws in groups.items()])
    ref_mean, ref_scale = norm_reference(net.combined)
    best, best_state, best_epoch, stale = float("inf"), None, int(p.get("epochs", 0)), 0
    for epoch in range(1, int(p["max_epochs"] if stop is not None else p["epochs"]) + 1):
        prepare(net, "3")
        totals, n = torch.zeros(3, device=data.device), 0  # kept on the GPU: reading each batch's losses would stall it
        for i, b in enumerate(batches(idx, int(p["batch_size"]), True, data.device)):
            if len(b) < 2:
                continue
            out = net(*data.draw(b))
            log = None
            if on_log is not None and i % log_every == 0:
                log = {"epoch": epoch, "batch": i, **grad_share(out, y[b]), **weight_share(net.combined),
                       **summary_drift(out["s_p"], out["s_t"], ref_mean, ref_scale)}
            losses = head_losses(out, y[b])
            batch_loss = combine_losses(losses, lam)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            totals += torch.stack([losses["price"], losses["tab"], losses["comb"]]).detach() * len(b)
            n += len(b)
            if log is not None:
                on_log(log)
        avg = (totals / max(n, 1)).tolist()
        line = {"epoch": epoch, "loss_price": avg[0], "loss_tab": avg[1], "loss_comb": avg[2]}
        if stop is not None:
            stop_t = torch.as_tensor(np.asarray(stop), device=data.device)
            line["stop_loss"] = float(loss(outputs(net, data, stop)["comb"], y[stop_t]))
        if on_epoch is not None:
            on_epoch(line)
        if stop is None:
            continue
        if line["stop_loss"] < best:
            best, best_epoch, stale = line["stop_loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= int(p["patience"]):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"epochs": best_epoch, "stop_loss": best}


def train_joint(net: TwoBranchNet, data, y: torch.Tensor, idx, p: dict, lam: float,
                on_epoch: Callable[[dict], None] | None = None, seed: int = 0, stop=None,
                score: Callable[[np.ndarray], float] | None = None) -> dict:
    """The joint run: its parts (prepare "joint": block 3, the readout, the tabular MLP's layers after the first, and
    the head) trained together on joint_loss (lam), AdamW with each part's learning rate p["learning_rate"][part] and weight decay
    p["weight_decay"][part], over the stock-days idx, with a chart or not (data: a JointData). The head reads the CNN's
    prediction on the stock-days with a chart and 0 (no view) on the others (price_input); the CNN runs only when a
    batch has two charts or more (its readout's statistics need two). Shuffling and dropout seeded by `seed`, so the λs
    are paired. p["epochs"] epochs, or, with `stop` rows (all with a chart), at most p["max_epochs"], stopped after
    p["patience"] epochs without a lower combined loss on them and returned to the best epoch. on_epoch gets each
    epoch's average L_price (over the stock-days with a chart) and L_comb (over all), the stop rows' combined loss and,
    with `score`, score(their combined prediction) as stop_score."""
    torch.manual_seed(seed)
    groups = prepare(net, "joint")
    opt = torch.optim.AdamW([{"params": ws, "lr": p["learning_rate"][k], "weight_decay": p["weight_decay"][k]}
                             for k, ws in groups.items()])
    best, best_state, best_epoch, stale = float("inf"), None, int(p.get("epochs", 0)), 0
    for epoch in range(1, int(p["max_epochs"] if stop is not None else p["epochs"]) + 1):
        prepare(net, "joint")
        totals, n_all, n_charted = torch.zeros(2, device=data.device), 0, 0  # L_price, L_comb summed
        for b in batches(idx, int(p["batch_size"]), True, data.device):
            if len(b) < 2:
                continue
            has = data.chart_row[b] >= 0
            c = b[has]
            if len(c) < 2:
                has, c = torch.zeros_like(has), c[:0]
            s_p, price = net.price(data.charts(c)) if len(c) else (torch.zeros(0, 1, device=data.device), y[c])
            comb = net.combined(price_input(s_p, has), net.tabular(data.features[b])[0])
            batch_loss = joint_loss(comb, y[b], price, y[c], lam)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            totals += torch.stack([(price - y[c]).pow(2).sum(), (comb - y[b]).pow(2).sum()]).detach()
            n_all, n_charted = n_all + len(b), n_charted + len(c)
        avg = totals.tolist()
        line = {"epoch": epoch, "loss_price": avg[0] / max(n_charted, 1), "loss_comb": avg[1] / max(n_all, 1)}
        if stop is not None:
            stop_t = torch.as_tensor(np.asarray(stop), device=data.device)
            comb = outputs(net, data, stop)["comb"]
            line["stop_loss"] = float(loss(comb, y[stop_t]))
            if score is not None:
                line["stop_score"] = score(comb.cpu().numpy())
        if on_epoch is not None:
            on_epoch(line)
        if stop is None:
            continue
        if line["stop_loss"] < best:
            best, best_epoch, stale = line["stop_loss"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= int(p["patience"]):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"epochs": best_epoch, "stop_loss": best}


def price_input(s_p: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
    """The head's price input for a batch: the CNN's prediction s_p (one row for each stock-day with a chart, `has`)
    where there is one, 0 (no view: the target is demeaned) elsewhere."""
    x = torch.zeros(len(has), s_p.shape[1], device=s_p.device, dtype=s_p.dtype)
    x[has] = s_p
    return x


class JointData:
    """The joint run's data on one device: every stock-day's features, and the charts of those with one. images: a
    dsr.cnn.Images (anything that draws charts by row); chart_row: each stock-day's row in it, -1 for none.
    charts(idx) gives the grey channel of the stock-days idx, which must all have a chart; draw(idx), as
    TwoBranchData's, the charts and the features."""

    def __init__(self, images, chart_row, features, device):
        f = np.asarray(features, dtype=np.float32)
        if np.isnan(f).any():
            raise ValueError("the features have missing values: the model panel holds them as 0")
        self.images, self.device = images, torch.device(device)
        self.chart_row = torch.as_tensor(np.asarray(chart_row, dtype=np.int64), device=self.device)
        self.features = torch.as_tensor(f, device=self.device)

    def _on_device(self, idx):
        return idx if torch.is_tensor(idx) else torch.as_tensor(np.asarray(idx), device=self.device)

    def charts(self, idx):
        rows = self.chart_row[self._on_device(idx)]
        if bool((rows < 0).any()):
            raise ValueError("a stock-day without a chart")
        return self.images.draw(rows)[:, :1]

    def draw(self, idx):
        idx = self._on_device(idx)
        return self.charts(idx), self.features[idx]


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

    - train: every stock-day of the training years with a target (step 1b);
    - train_comb: the training years' large caps with a target (step 2 fits on them);
    - train_price: those on one day in PRICE_EVERY of each stock, staggered ((day_pos + permno) % PRICE_EVERY == 0;
      step 1a; PRICE_EVERY is 1, every day);
    - hold: the hold-out year's large caps with a target (step 2 stops on them, step 3 fits on them), hold_ic the one
      day in HOLD_IC_EVERY step 1a stops on;
    - test: the predicted year's large caps, whatever their target;
    - scale: the sd of the target over the training years' large caps, which every head's target is divided by."""
    y = keys[target].to_numpy(np.float32)
    day, large = keys["day"].to_numpy(), keys["large_cap"].to_numpy(bool)
    train, hold, rows = fold_rows(day, y, large, fold, hold_out_months, holds_out=True)
    stagger = (keys["day_pos"] + keys["permno"]).to_numpy()
    hold = hold & large
    return {"train": np.flatnonzero(train), "train_price": np.flatnonzero(train & large & (stagger % PRICE_EVERY == 0)),
            "train_comb": np.flatnonzero(train & large), "hold": np.flatnonzero(hold),
            "hold_ic": np.flatnonzero(hold & (stagger % HOLD_IC_EVERY == 0)),
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


def folds_done(folders: list[Path], fold: str) -> bool:
    """Whether every one of the folders has the fold: outputs written together are skipped together."""
    return all(fold_done(folder, fold) for folder in folders)


def same_folds(folders: list[Path]) -> list[str]:
    """The folds written to every one of the folders, which must be the same: outputs are compared only over the same
    years. Raises ValueError when they differ (a command stopped partway)."""
    found = {folder.name: sorted(f.stem for f in (folder / "folds").glob("*.parquet")) for folder in folders}
    if len({tuple(v) for v in found.values()}) > 1:
        raise ValueError(f"the outputs hold different folds, finish the command first: {found}")
    return next(iter(found.values()))


def learns_nothing(score: float) -> bool:
    """Warning sign 1: the price head's best hold-out rank IC is 0 or below (or missing)."""
    return not score > 0


def min_weight_share(n_price: int, n_tab: int = 16) -> float:
    """Warning sign 2's weight-share threshold: a fifth of the share equal weights on every input give (as norms,
    sqrt(n_price) against sqrt(n_tab)): 10% when both summaries have 16 numbers, 4% when the price summary is one."""
    return float(np.sqrt(n_price) / (np.sqrt(n_price) + np.sqrt(n_tab))) / 5


def ignores_chart(corr_price: float, weight_share_price: float, min_share: float = 0.10) -> bool:
    """Warning sign 2: the model's rank correlation with the price head is below 0.05, or the price branch's weight
    share below min_share (min_weight_share; either missing counts: a constant output has no correlation)."""
    return not (corr_price >= 0.05 and weight_share_price >= min_share)


def pushes_chart_out(weight_share_price: float, share_after_step_2: float) -> bool:
    """Warning sign 3: fine-tuning halves the price branch's weight share from where step 2 left it (or loses it)."""
    return not weight_share_price >= share_after_step_2 / 2
