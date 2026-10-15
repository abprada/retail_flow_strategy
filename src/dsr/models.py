"""One fit function for each model family, with the same inputs, so every model sees the same rows.

fit(kind, params, x_train, y_train, x_valid, y_valid) returns the fitted model (Fitted); fit_predict(..., x_pred)
returns its predictions for x_pred and a dict of what the fit chose. x_valid/y_valid is the early-stopping hold-out
(the training window's last months): LightGBM and the MLP stop on it; the linear models ignore it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import torch
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

THREADS = 16  # of the M5 Pro's 18 cores
EARLY_STOP_TREES = 50  # LightGBM stops after this many trees without improving the hold-out
EARLY_STOP_EPOCHS = 3  # the MLP stops after this many epochs without improving the hold-out
ACTIVATIONS = {"relu": torch.nn.ReLU, "leaky_relu": torch.nn.LeakyReLU, "gelu": torch.nn.GELU}
MIN_UNITS = 4  # a decreasing network's narrowest layer (v3's best ended at 4: 32, 16, 8, 4)


@dataclass
class Fitted:
    """A fitted model: predict(x) gives its predictions (averaged over its seeds), info what the fit chose, and
    linear the linear model's coefficients and scaling (empty for the others)."""
    predict: Callable[[np.ndarray], np.ndarray]
    info: dict
    linear: dict = field(default_factory=dict)


def fit(kind: str, params: dict, x_train, y_train, x_valid, y_valid, device: str | None = None,
        on_epoch=None) -> Fitted:
    return {"linear": _linear, "lgbm": _lgbm, "mlp": _mlp}[kind](
        params, x_train, y_train, x_valid, y_valid, **({"device": device, "on_epoch": on_epoch} if kind == "mlp" else {}))


def fit_predict(kind: str, params: dict, x_train, y_train, x_valid, y_valid, x_pred, device: str | None = None,
                on_epoch=None):
    model = fit(kind, params, x_train, y_train, x_valid, y_valid, device=device, on_epoch=on_epoch)
    return model.predict(x_pred), model.info


def _linear(p, x, y, xv, yv):
    """An elastic net, whose ends are the other linear models: alpha 0 is OLS (fitted directly: scikit-learn's elastic
    net solver is unreliable at 0), l1_ratio 0 Ridge, l1_ratio 1 LASSO. Fit on features scaled to unit variance on the
    training rows and the target scaled to unit variance
    (scaled back for the predictions), so a penalty means the same whatever the target's units. The penalty follows
    scikit-learn's elastic net, (1 / 2n) * squared errors + alpha * (l1_ratio * |w|_1 + (1 - l1_ratio) / 2 * |w|_2^2):
    a Ridge with alpha * n on the sum of squared errors. `linear` keeps the coefficients on the scaled features and
    target, the scaling, and that Ridge penalty (0 for OLS; None for an L1 penalty, which has no such form)."""
    alpha, l1 = float(p["alpha"]), float(p["l1_ratio"])
    scale = float(np.std(y)) or 1.0
    # each column scaled by its training standard deviation, so the penalty treats every feature alike (the 0/1 flags
    # are not standardized in the panel); a constant column (early_close in a window without one) is left as it is
    sd = np.asarray(x).std(axis=0, dtype=np.float64)  # float64 sums without a float64 copy of x
    sd[sd == 0] = 1.0
    sd32 = sd.astype(np.float32)
    if alpha == 0:
        model, penalty = LinearRegression(), 0.0
    elif l1 == 0:
        model, penalty = Ridge(alpha=alpha * len(y)), alpha * len(y)
    else:
        model, penalty = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000), None
    model.fit(np.asarray(x) / sd32, np.asarray(y) / scale)
    return Fitted(lambda xp: model.predict(np.asarray(xp) / sd32) * scale,
                  {"nonzero": int(np.count_nonzero(model.coef_))},
                  {"coef": model.coef_, "sd": sd, "scale": scale, "penalty": penalty})


def _lgbm(p, x, y, xv, yv):
    """Gradient-boosted trees, squared error, up to max_trees, stopped early on the hold-out. Row and feature
    sampling make a fit depend on its seed, so predictions are averaged over `seeds` fits (seeds 0, 1, ...; seed 0
    alone is a sweep run's model)."""
    params = {
        "objective": "regression", "metric": "l2", "verbosity": -1, "num_threads": THREADS,
        "learning_rate": p["learning_rate"], "num_leaves": int(p["num_leaves"]),
        "min_data_in_leaf": int(p["min_data_in_leaf"]), "feature_fraction": p["feature_fraction"],
        "bagging_fraction": p["bagging_fraction"], "bagging_freq": 1, "lambda_l2": p["lambda_l2"],
        "linear_tree": bool(p["linear_tree"]),
    }
    boosters = []
    for seed in range(int(p.get("seeds", 1))):
        # a new Dataset per seed: the seed also sets how the Dataset bins the features
        train = lgb.Dataset(x, y, free_raw_data=True)
        valid = lgb.Dataset(xv, yv, reference=train)
        booster = lgb.train({**params, "seed": seed}, train, num_boost_round=int(p.get("max_trees", 2000)),
                            valid_sets=[valid], callbacks=[lgb.early_stopping(EARLY_STOP_TREES, verbose=False)])
        boosters.append(booster.free_dataset())  # predicting doesn't need the binned training rows
    return Fitted(lambda xp: np.mean([b.predict(xp, num_iteration=b.best_iteration) for b in boosters], axis=0),
                  {"trees": float(np.mean([b.best_iteration for b in boosters]))})


def _network(p, n_in):
    """`layers` hidden layers, each a linear layer, its activation (ReLU, LeakyReLU or GELU) and dropout, then one
    output. The hidden layers all have `width` units ("constant" shape) or halve layer by layer from `width`
    ("decreasing": 256, 128, 64, ...), not below MIN_UNITS. Runs before these settings existed used constant ReLU
    layers."""
    activation = ACTIVATIONS[p.get("activation", "relu")]
    decreasing = p.get("shape", "constant") == "decreasing"
    layers, width_in = [], n_in
    for k in range(int(p["layers"])):
        width = max(int(p["width"]) // 2 ** k, MIN_UNITS) if decreasing else int(p["width"])
        layers += [torch.nn.Linear(width_in, width), activation(), torch.nn.Dropout(p["dropout"])]
        width_in = width
    return torch.nn.Sequential(*layers, torch.nn.Linear(width_in, 1))


def _device(device):
    if device:
        return torch.device(device)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _mlp(p, x, y, xv, yv, device=None, on_epoch=None):
    """A feed-forward network (_network), AdamW, squared error, trained on the Mac's GPU (mps)
    with the whole window held on it, stopped early on the hold-out. The target is scaled by its training standard
    deviation and scaled back. Predictions are averaged over `seeds` networks. After each epoch, `on_epoch` (if given)
    gets the seed, the epoch, the training loss (the average of the epoch's batch losses) and the hold-out loss."""
    dev = _device(device)
    tensor = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    scale = float(np.std(y)) or 1.0
    xt, yt, xvt, yvt = tensor(x), tensor(y) / scale, tensor(xv), tensor(yv) / scale
    batch = int(p["batch_size"])
    nets, epochs = [], []
    for seed in range(int(p.get("seeds", 1))):
        torch.manual_seed(seed)
        net = _network(p, xt.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        best, best_state, best_epoch, stale = float("inf"), None, 0, 0
        for epoch in range(1, int(p.get("max_epochs", 20)) + 1):
            net.train()
            order = torch.randperm(len(xt), device=dev)
            total = torch.zeros((), device=dev)  # kept on the GPU: reading each batch's loss would stall it
            for i in range(0, len(xt), batch):
                idx = order[i:i + batch]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                total += loss.detach() * len(idx)
            net.eval()
            with torch.no_grad():
                v = float(torch.cat([torch.nn.functional.mse_loss(net(xvt[i:i + 65536]).squeeze(1), yvt[i:i + 65536],
                                                                   reduction="sum").reshape(1)
                                     for i in range(0, len(xvt), 65536)]).sum()) / len(xvt)
            if on_epoch is not None:
                on_epoch({"seed": seed, "epoch": epoch, "train_loss": float(total) / len(xt), "holdout_loss": v})
            if v < best:
                best, best_epoch, stale = v, epoch, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                stale += 1
                if stale >= EARLY_STOP_EPOCHS:
                    break
        net.load_state_dict(best_state)
        net.eval()
        nets.append(net)
        epochs.append(best_epoch)

    def predict(xp):
        xpt = tensor(xp)
        with torch.no_grad():
            preds = [torch.cat([net(xpt[i:i + 65536]).squeeze(1) for i in range(0, len(xpt), 65536)]).cpu().numpy()
                     for net in nets]
        return np.mean(preds, axis=0) * scale

    return Fitted(predict, {"epochs": float(np.mean(epochs))})
