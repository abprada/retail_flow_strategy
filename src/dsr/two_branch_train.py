"""Training the two-branch model (explanations/CNN/fusion.md): the four steps, the outputs they are scored on, the data
on the GPU, each fold's rows, and the writers of the scored outputs and of the balance diagnostics."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dsr.evaluation import summarize
from dsr.two_branch import (TwoBranchNet, combine_losses, grad_share, head_losses, loss, prepare, summary_drift,
                            weight_share)
from dsr.walkforward import Fold, fold_rows, last_months

PRICE_EVERY, HOLD_IC_EVERY = 9, 5  # step 1a trains on 1 day in 9 of each stock and stops on 1 in 5 of the hold-out
STOP_MONTHS = 3  # step 2 stops on the hold-out year's last 3 months


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
             on_log: Callable[[dict], None] | None = None, on_epoch: Callable[[dict], None] | None = None,
             log_every: int = 100, seed: int = 0) -> None:
    """Step 3: everything after the CNN blocks on step 3's loss (combine_losses; delta, lam), AdamW with each part's
    learning rate p["learning_rate"][part] and weight decay p["weight_decay"][part], p["epochs"] epochs over the
    stock-days idx, its shuffling and dropout seeded by `seed`, so the λ = 0 control is paired with the run. The blocks
    stay frozen in eval mode; the combined MLP's BatchNorm keeps updating. Every log_every batches on_log gets the
    epoch, the batch, the gradient share, the weight share and the summaries' drift from the statistics the combined
    BatchNorm had at the start; on_epoch gets each finished epoch with its average L_price, L_tab and L_comb. A last
    batch of one row is skipped."""
    torch.manual_seed(seed)
    groups = prepare(net, "3")
    opt = torch.optim.AdamW([{"params": ws, "lr": p["learning_rate"][k], "weight_decay": p["weight_decay"][k]}
                             for k, ws in groups.items()])
    bn = net.combined.layers[0]
    ref_mean, ref_sd = bn.running_mean.clone(), torch.sqrt(bn.running_var + bn.eps)
    for epoch in range(1, int(p["epochs"]) + 1):
        prepare(net, "3")
        totals, n = torch.zeros(3, device=data.device), 0  # kept on the GPU: reading each batch's losses would stall it
        for i, b in enumerate(batches(idx, int(p["batch_size"]), True, data.device)):
            if len(b) < 2:
                continue
            out = net(*data.draw(b))
            log = None
            if on_log is not None and i % log_every == 0:
                log = {"epoch": epoch, "batch": i, **grad_share(out, y[b], delta), **weight_share(net.combined),
                       **summary_drift(out["s_p"], out["s_t"], ref_mean, ref_sd)}
            losses = head_losses(out, y[b], delta)
            batch_loss = combine_losses(losses, lam)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            totals += torch.stack([losses["price"], losses["tab"], losses["comb"]]).detach() * len(b)
            n += len(b)
            if log is not None:
                on_log(log)
        if on_epoch is not None:
            avg = (totals / max(n, 1)).tolist()
            on_epoch({"epoch": epoch, "loss_price": avg[0], "loss_tab": avg[1], "loss_comb": avg[2]})


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


def ignores_chart(corr_price: float, weight_share_price: float) -> bool:
    """Warning sign 2: the model's rank correlation with the price head is below 0.05, or the price branch's weight
    share below 10% (either missing counts: a constant output has no correlation)."""
    return not (corr_price >= 0.05 and weight_share_price >= 0.10)


def pushes_chart_out(weight_share_price: float, share_after_step_2: float) -> bool:
    """Warning sign 3: fine-tuning halves the price branch's weight share from where step 2 left it (or loses it)."""
    return not weight_share_price >= share_after_step_2 / 2
