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
    assert [e["epoch"] for e in ends] == [1, 2] and len(logs) == 2 * 3  # 3 batches an epoch: 16, 16, 8
    assert {"grad_share_price", "weight_share_price", "drift_shift_price", "drift_ratio_tab"} <= set(logs[0])


def test_outputs_predict_in_eval_mode_and_prepare_puts_the_trained_parts_back():
    data, net = _Data(10, seed=7), TwoBranchNet().to(DEVICE)
    out = outputs(net, data, np.arange(10), batch=4)
    assert out["comb"].shape == (10,) and out["s_p"].shape == (10, 16) and not net.combined.training
    prepare(net, "3")
    assert net.combined.training and not net.price.blocks.training


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


def _finetune_p():
    return {"epochs": 1, "batch_size": 16,
            "learning_rate": {"one_d": 1e-3, "price_head": 1e-3, "tab_body": 1e-3, "tab_head": 1e-3, "combined": 1e-3},
            "weight_decay": {"one_d": 0.05, "price_head": 0.05, "tab_body": 0.035, "tab_head": 0.035, "combined": 0.035}}


def test_finetune_reports_each_epoch_s_three_losses():
    data, net = _Data(40, seed=8), TwoBranchNet().to(DEVICE)
    ends = []
    finetune(net, data, _normal(40, seed=8), np.arange(40), _finetune_p(), 1.0, 0.5, on_epoch=ends.append)
    assert set(ends[0]) == {"epoch", "loss_price", "loss_tab", "loss_comb"}
    assert all(np.isfinite(ends[0][k]) and ends[0][k] > 0 for k in ("loss_price", "loss_tab", "loss_comb"))


def test_finetune_is_seeded():
    import copy  # noqa: PLC0415
    data, y = _Data(40, seed=9), _normal(40, seed=9)
    start = TwoBranchNet().to(DEVICE)
    nets = {s: copy.deepcopy(start) for s in ("a", "b", "c")}
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        finetune(nets[name], data, y, np.arange(40), _finetune_p(), 1.0, 0.5, seed=seed)
    w = {k: n.price.one_d.layers[4].weight.detach() for k, n in nets.items()}
    assert torch.allclose(w["a"], w["b"], atol=1e-6) and not torch.allclose(w["a"], w["c"], atol=1e-6)


from dsr.two_branch_train import folds_done, ignores_chart, learns_nothing, pushes_chart_out, same_folds  # noqa: E402


def test_a_fold_is_done_only_when_every_output_has_it(tmp_path):
    frame = pd.DataFrame({"x": [1]})
    write_fold(tmp_path / "comb_s3", "2019", frame)
    write_fold(tmp_path / "price_s3", "2019", frame)
    outs = [tmp_path / "comb_s3", tmp_path / "price_s3", tmp_path / "tab_s3"]
    assert not folds_done(outs, "2019")
    write_fold(tmp_path / "tab_s3", "2019", frame)
    assert folds_done(outs, "2019")


def test_outputs_are_compared_only_over_the_same_folds(tmp_path):
    frame = pd.DataFrame({"x": [1]})
    for fold in ("2019", "2020"):
        write_fold(tmp_path / "a", fold, frame)
    write_fold(tmp_path / "b", "2019", frame)
    with pytest.raises(ValueError, match="different folds"):
        same_folds([tmp_path / "a", tmp_path / "b"])
    write_fold(tmp_path / "b", "2020", frame)
    assert same_folds([tmp_path / "a", tmp_path / "b"]) == ["2019", "2020"]


def test_the_warning_signs_fire_on_nan():
    nan = float("nan")
    assert learns_nothing(0.0) and learns_nothing(nan) and not learns_nothing(0.001)
    assert ignores_chart(0.04, 0.5) and ignores_chart(0.3, 0.09) and ignores_chart(nan, 0.5)
    assert not ignores_chart(0.3, 0.5)
    assert pushes_chart_out(0.1, 0.3) and pushes_chart_out(nan, 0.3) and not pushes_chart_out(0.2, 0.3)
