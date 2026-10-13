import numpy as np
import pytest

from dsr.models import _network, fit, fit_predict


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype("float32")
    y = (0.5 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * rng.normal(size=n)).astype("float32")
    return x, y


@pytest.mark.parametrize("params", [
    {"alpha": 0.0, "l1_ratio": 0.5},  # OLS
    {"alpha": 1e-3, "l1_ratio": 0.0},  # Ridge
    {"alpha": 1e-3, "l1_ratio": 1.0},  # LASSO
    {"alpha": 1e-3, "l1_ratio": 0.5},  # elastic net
])
def test_elastic_net_and_its_ends_recover_a_linear_signal(params):
    x, y = _data()
    pred, info = fit_predict("linear", params, x[:3000], y[:3000], x[3000:3500], y[3000:3500], x[3500:])
    assert np.corrcoef(pred, y[3500:])[0, 1] > 0.95


def test_lightgbm_early_stops_and_predicts():
    x, y = _data()
    params = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 20, "feature_fraction": 1.0,
              "bagging_fraction": 1.0, "lambda_l2": 0.0, "linear_tree": False, "max_trees": 300}
    pred, info = fit_predict("lgbm", params, x[:3000], y[:3000], x[3000:3500], y[3000:3500], x[3500:])
    assert np.corrcoef(pred, y[3500:])[0, 1] > 0.9 and 0 < info["trees"] <= 300


def test_lightgbm_averages_seeds_and_seed_0_alone_is_reproducible():
    x, y = _data()
    params = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 20, "feature_fraction": 0.5,
              "bagging_fraction": 0.5, "lambda_l2": 0.0, "linear_tree": False, "max_trees": 300}
    args = (x[:3000], y[:3000], x[3000:3500], y[3000:3500], x[3500:])
    one, _ = fit_predict("lgbm", params, *args)
    again, _ = fit_predict("lgbm", params, *args)
    three, info = fit_predict("lgbm", {**params, "seeds": 3}, *args)
    assert np.array_equal(one, again)  # a fixed seed: the same model every time
    assert not np.allclose(one, three)  # row and feature sampling differ by seed, so the average differs
    assert np.corrcoef(three, y[3500:])[0, 1] > 0.9 and info["trees"] > 0


def test_mlp_averages_seeds_and_stops_early_on_cpu():
    x, y = _data()
    params = {"layers": 1, "width": 16, "dropout": 0.0, "learning_rate": 1e-2, "weight_decay": 0.0,
              "batch_size": 256, "max_epochs": 30, "seeds": 2}
    history = []
    pred, info = fit_predict("mlp", params, x[:3000], y[:3000], x[3000:3500], y[3000:3500], x[3500:], device="cpu",
                             on_epoch=history.append)
    assert np.corrcoef(pred, y[3500:])[0, 1] > 0.9 and 1 <= info["epochs"] <= 30
    # every epoch of both seeds is reported, and the losses fall as the network learns
    assert {h["seed"] for h in history} == {0, 1} and history[0]["epoch"] == 1
    assert min(h["holdout_loss"] for h in history) < history[0]["holdout_loss"]


def test_strong_penalties_shrink_the_model_and_the_target_scale_does_not_matter():
    x, y = _data()
    lasso, info = fit_predict("linear", {"alpha": 1.0, "l1_ratio": 1.0}, x[:3000], y[:3000], None, None, x[3000:])
    assert info["nonzero"] == 0 and np.allclose(lasso, lasso[0])  # alpha 1 on a unit-variance target: all zero
    ridge, _ = fit_predict("linear", {"alpha": 1e4, "l1_ratio": 0.0}, x[:3000], y[:3000], None, None, x[3000:])
    assert np.std(ridge) < 0.01 * np.std(y)
    # the same penalty on the target times 100 gives the same predictions times 100
    for l1 in (0.0, 0.5, 1.0):
        small, _ = fit_predict("linear", {"alpha": 1e-2, "l1_ratio": l1}, x[:3000], y[:3000], None, None, x[3000:])
        big, _ = fit_predict("linear", {"alpha": 1e-2, "l1_ratio": l1}, x[:3000], 100 * y[:3000], None, None, x[3000:])
        assert np.allclose(big, 100 * small, rtol=1e-3, atol=1e-4)


def test_the_linear_penalty_does_not_depend_on_a_feature_s_scale():
    # a 0/1 flag and a standardized feature should be penalized alike: rescaling a column changes nothing
    x, y = _data()
    x10 = x.copy()
    x10[:, 1] *= 10
    for l1 in (0.0, 0.5, 1.0):
        a, _ = fit_predict("linear", {"alpha": 1e-2, "l1_ratio": l1}, x[:3000], y[:3000], None, None, x[3000:])
        b, _ = fit_predict("linear", {"alpha": 1e-2, "l1_ratio": l1}, x10[:3000], y[:3000], None, None, x10[3000:])
        assert np.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_mlp_layers_constant_or_decreasing_with_the_chosen_activation():
    import torch  # after dsr.models: torch imported before lightgbm crashes lightgbm on macOS (two OpenMP runtimes)

    p = {"layers": 3, "width": 64, "dropout": 0.1}
    constant = _network({**p, "activation": "relu"}, 17)
    decreasing = _network({**p, "shape": "decreasing", "activation": "gelu"}, 17)
    sizes = lambda net: [(m.in_features, m.out_features) for m in net if isinstance(m, torch.nn.Linear)]
    assert sizes(constant) == [(17, 64), (64, 64), (64, 64), (64, 1)]
    assert sizes(decreasing) == [(17, 64), (64, 32), (32, 16), (16, 1)]
    assert sum(isinstance(m, torch.nn.GELU) for m in decreasing) == 3 and sum(isinstance(m, torch.nn.ReLU) for m in constant) == 3
    leaky = _network({**p, "activation": "leaky_relu"}, 17)
    assert sum(isinstance(m, torch.nn.LeakyReLU) for m in leaky) == 3


def test_fit_returns_the_model_fit_predict_uses():
    x, y = _data()
    args = (x[:3000], y[:3000], x[3000:3500], y[3000:3500])
    for kind, params in [("linear", {"alpha": 1e-3, "l1_ratio": 0.0}),
                         ("lgbm", {"learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 20, "feature_fraction": 1.0,
                                   "bagging_fraction": 1.0, "lambda_l2": 0.0, "linear_tree": True, "max_trees": 50,
                                   "seeds": 2}),
                         ("mlp", {"layers": 1, "width": 8, "dropout": 0.0, "learning_rate": 1e-2, "weight_decay": 0.0,
                                  "batch_size": 512, "max_epochs": 3})]:
        kw = {"device": "cpu"} if kind == "mlp" else {}
        model = fit(kind, params, *args, **kw)
        pred, info = fit_predict(kind, params, *args, x[3500:], **kw)
        assert np.array_equal(model.predict(x[3500:]), pred) and model.info == info
    # the linear model keeps its coefficients on the scaled features and target, and its Ridge penalty
    linear = fit("linear", {"alpha": 1e-3, "l1_ratio": 0.0}, *args).linear
    assert set(linear) == {"coef", "sd", "scale", "penalty"} and linear["penalty"] == 1e-3 * 3000
