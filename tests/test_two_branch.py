import numpy as np
import pytest

import dsr.models  # noqa: F401  lightgbm before torch: torch first crashes lightgbm on macOS (two OpenMP runtimes)
import torch  # noqa: E402

from dsr.cnn import W_HEIGHT, W_WIDTH  # noqa: E402
from dsr.two_branch import COMBINED, DenseBlocks, TwoBranchNet, prepare  # noqa: E402

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
    assert _weights(net) == 29_267
    assert _weights(net.price.blocks) == 12_384 and _weights(net.price.one_d) == 16_016
    assert _weights(net.tabular) == 305 and _weights(net.combined) == 545
    assert _weights(TwoBranchNet(hidden=COMBINED["16-8"])) == 29_395
    assert _weights(TwoBranchNet(width=32)) == 49_235  # the backup


def test_the_one_day_network_reads_the_one_day_chart_through_the_one_day_cnn_s_blocks():
    from dsr.cnn import HEIGHT, WIDTH  # noqa: PLC0415
    net = TwoBranchNet(chart="day")
    assert _weights(net) == 167_475
    assert _weights(net.price.blocks) == 154_752 and _weights(net.price.one_d) == 11_856
    assert [b.out_channels for b in net.price.blocks if isinstance(b, torch.nn.Conv2d)] == [32, 64, 128]
    net = net.to(DEVICE).eval()
    out = net(torch.zeros(2, 1, HEIGHT, WIDTH, device=DEVICE), torch.zeros(2, 17, device=DEVICE))
    assert out["s_p"].shape == (2, 16) and out["comb"].shape == (2,)
    x = net.price.blocks(torch.zeros(1, 1, HEIGHT, WIDTH, device=DEVICE))
    assert x.shape == (1, 128, 3, WIDTH)  # 3 rows, the 40 bars' 120 columns
    assert net.price.one_d.time(net.price.one_d.rows(net.price.one_d.pool(x))).shape == (1, 16, 5)  # 40 -> 20 -> 10 -> 5


def test_the_day_cnn_network_is_the_one_day_cnn_with_the_tabular_branch_and_a_head_on_its_prediction():
    net = TwoBranchNet(chart="day_cnn")
    assert _weights(net) == 201_443
    assert _weights(net.price.blocks) == 31_488 and _weights(net.price.one_d) == 123_264  # blocks 1-2; block 3
    assert _weights(net.price.head) == 46_081 and _weights(net.combined) == 305  # the readout; 17 -> 16 -> 1
    from dsr.cnn import HEIGHT, WIDTH  # noqa: PLC0415
    out = net.to(DEVICE).eval()(torch.zeros(2, 1, HEIGHT, WIDTH, device=DEVICE), torch.zeros(2, 17, device=DEVICE))
    assert out["s_p"].shape == (2, 1) and torch.equal(out["s_p"][:, 0], out["price"])  # the summary is the prediction


def test_the_day_cnn_branch_predicts_as_the_chart_cnn_it_was_copied_from():
    from dsr.cnn import ChartCNN  # noqa: PLC0415
    from dsr.two_branch import copy_chart_cnn  # noqa: PLC0415
    torch.manual_seed(3)
    cnn = ChartCNN((32, 64, 128), 0.5, mean=[40.0], sd=[90.0], channels=1).to(DEVICE).eval()
    net = TwoBranchNet(chart="day_cnn").to(DEVICE).eval()
    copy_chart_cnn(net.price, cnn)
    img = _charts(3, seed=11)[:, :, :64, :120]
    with torch.no_grad():
        assert torch.allclose(net.price(img)[1], cnn(img), atol=1e-5)


def test_step_3_trains_block_3_and_the_readout_without_dropout_or_new_statistics():
    net = TwoBranchNet(chart="day_cnn")
    prepare(net, "3")
    assert not any(w.requires_grad for w in net.price.blocks.parameters()) and not net.price.blocks.training
    assert all(w.requires_grad for w in net.price.one_d.parameters())
    assert all(w.requires_grad for w in net.price.head.parameters())
    stats_and_dropout = [m for m in net.price.modules()
                         if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout))]
    assert stats_and_dropout and not any(m.training for m in stats_and_dropout)


def test_step_3_freezes_the_one_day_cnn_s_blocks():
    net = TwoBranchNet(chart="day")
    prepare(net, "3")
    assert not net.price.blocks.training and not any(w.requires_grad for w in net.price.blocks.parameters())
    assert net.price.one_d.dense[1].weight.requires_grad


def test_the_network_gives_two_summaries_and_three_predictions():
    net = TwoBranchNet().to(DEVICE).eval()
    out = net(torch.zeros(2, 1, W_HEIGHT, W_WIDTH, device=DEVICE), torch.zeros(2, 17, device=DEVICE))
    assert out["s_p"].shape == (2, 16) and out["s_t"].shape == (2, 16)
    assert out["price"].shape == out["tab"].shape == out["comb"].shape == (2,)


def test_the_dense_blocks_put_the_image_and_the_three_blocks_side_by_side():
    blocks = DenseBlocks().to(DEVICE).eval()
    assert blocks(torch.zeros(2, 1, W_HEIGHT, W_WIDTH, device=DEVICE)).shape == (2, 49, 4, W_WIDTH)
    assert [b[0].in_channels for b in blocks.blocks] == [1, 17, 33]
    assert [b[0].dilation for b in blocks.blocks] == [(1, 1), (1, 3), (1, 9)]


def test_one_block_3_value_reads_27_columns_without_gaps():
    blocks = DenseBlocks().to(DEVICE).eval()
    with torch.no_grad():
        for conv in (b[0] for b in blocks.blocks):
            conv.weight.fill_(1.0)
            conv.bias.zero_()
    x = torch.zeros(1, 1, W_HEIGHT, W_WIDTH, device=DEVICE, requires_grad=True)
    blocks(x)[0, -16:, :, 222].sum().backward()  # the last 16 channels are block 3
    read = (x.grad[0, 0].abs().sum(0) > 0).nonzero().flatten().cpu().numpy()
    assert read.min() == 209 and read.max() == 235 and len(read) == 27


def test_the_1d_block_halves_time_three_times_and_every_bar_reaches_the_summary():
    one_d = TwoBranchNet().price.one_d.to(DEVICE).eval()
    x = _normal(1, 49, 4, W_WIDTH, seed=5).requires_grad_(True)
    assert one_d.time(one_d.rows(one_d.pool(x))).shape == (1, 16, 19)  # 148 bars -> 74 -> 37 -> 19 steps
    s = one_d(x)
    assert s.shape == (1, 16)
    s.sum().backward()
    by_bar = x.grad[0].abs().sum((0, 1)).reshape(-1, 3).sum(1)
    assert (by_bar > 0).all()  # the oldest bar and today's last both count


def test_every_pixel_of_a_chart_reaches_the_price_prediction():
    price = TwoBranchNet().price.to(DEVICE).eval()
    x = _charts(2, seed=10).requires_grad_(True)
    price(x)[1].sum().backward()
    assert (x.grad != 0).all()  # average pooling everywhere: no pool drops a pixel's gradient


def test_every_convolution_and_linear_layer_starts_with_zero_biases():
    layers = [m for m in TwoBranchNet().modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear))]
    assert any(isinstance(m, torch.nn.Conv1d) for m in layers)
    assert all(m.bias.abs().max().item() == 0 for m in layers)  # Xavier-uniform weights, zero biases


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


from dsr.two_branch import (TRAINED, BranchNorm, grad_share, loss, norm_reference, parts, prepare,  # noqa: E402
                            step3_loss, summary_drift, weight_share)


def test_the_loss_is_squared_error():
    pred, y = torch.tensor([0.5, 3.0]), torch.zeros(2)
    assert loss(pred, y).item() == pytest.approx((0.5 ** 2 + 3.0 ** 2) / 2)


def test_step_3_s_loss_is_the_combined_and_price_losses_alike_at_one_half_and_the_combined_loss_at_zero():
    y = torch.zeros(3)
    out = {"price": torch.ones(3), "tab": 2 * torch.ones(3), "comb": 3 * torch.ones(3)}
    full = loss(out["comb"], y) + loss(out["price"], y)  # no tabular loss: the tabular branch learns from L_comb
    assert step3_loss(out, y, 0.5).item() == pytest.approx(full.item())
    assert step3_loss(out, y, 0.0).item() == pytest.approx(loss(out["comb"], y).item())


def test_prepare_trains_only_the_step_s_parts_and_freezes_the_rest_in_eval_mode():
    net = TwoBranchNet()
    for step, trained in TRAINED.items():
        prepare(net, step)
        for name, module in parts(net).items():
            assert module.training == (name in trained), (step, name)
            assert all(w.requires_grad == (name in trained) for w in module.parameters()), (step, name)
    assert TRAINED["3"] == ("one_d", "price_head", "tab_body", "combined")  # the tabular head has no loss in step 3


def test_step_3_leaves_the_blocks_and_their_batchnorm_alone_and_updates_the_combined_normalization():
    img, tab, y = _charts(8, seed=1), _normal(8, 17, seed=1), _normal(8, seed=2)
    net = TwoBranchNet().to(DEVICE)
    blocks_before = {k: v.clone() for k, v in net.price.blocks.state_dict().items()}
    one_d_before = net.price.one_d.dense[1].weight.detach().clone()
    norm_before = net.combined.norm_p.running_mean.clone()
    groups = prepare(net, "3")
    opt = torch.optim.AdamW([w for ws in groups.values() for w in ws], lr=1e-2)
    step3_loss(net(img, tab), y).backward()
    opt.step()
    assert all(torch.equal(blocks_before[k], v) for k, v in net.price.blocks.state_dict().items())
    assert not torch.equal(one_d_before, net.price.one_d.dense[1].weight)
    assert not torch.equal(norm_before, net.combined.norm_p.running_mean)


def test_step_3_trains_the_1d_block_but_keeps_its_batchnorm_statistics():
    img, tab, y = _charts(8, seed=7), _normal(8, 17, seed=7), _normal(8, seed=8)
    net = TwoBranchNet().to(DEVICE)
    bns = [m for m in net.price.one_d.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    stats_before = [m.running_mean.clone() for m in bns]
    groups = prepare(net, "3")
    assert not any(m.training for m in bns) and net.price.one_d.dense[1].training
    opt = torch.optim.AdamW([w for ws in groups.values() for w in ws], lr=1e-2)
    step3_loss(net(img, tab), y).backward()
    opt.step()
    assert all(torch.equal(a, m.running_mean) for a, m in zip(stats_before, bns))  # step 2's summaries stay valid
    prepare(net, "1a")
    assert all(m.training for m in bns)  # step 1a sets them


def test_the_gradient_share_uses_the_combined_loss_only():
    img, tab, y = _charts(8, seed=3), _normal(8, 17, seed=3), _normal(8, seed=4)
    net = TwoBranchNet().to(DEVICE)
    prepare(net, "3")
    net.eval()  # no dropout, so both passes see the same network
    first = grad_share(net(img, tab), y)
    with torch.no_grad():
        net.price.head.weight.mul_(100.0)  # the price head changes L_price, not L_comb
    second = grad_share(net(img, tab), y)
    assert first["grad_share_price"] == pytest.approx(second["grad_share_price"])
    assert 0 < first["grad_share_price"] < 1


def test_branch_norm_divides_a_branch_by_one_scale_so_quiet_numbers_stay_quiet():
    rng = np.random.default_rng(6)
    sd = np.r_[np.full(8, 2.0), np.full(8, 0.01)]  # half the numbers barely move
    s = torch.as_tensor((rng.normal(size=(4000, 16)) * sd + 3.0).astype(np.float32), device=DEVICE)
    norm = BranchNorm().to(DEVICE)
    out = norm(s)  # training mode: the batch's statistics
    assert out.mean(0).abs().max().item() < 1e-4
    out_sd = out.std(0, unbiased=False)
    assert out_sd.pow(2).mean().item() == pytest.approx(1.0, rel=1e-3)  # the branch as a whole at sd 1
    assert (out_sd[:8] / out_sd[8:].mean()).mean().item() == pytest.approx(200, rel=0.05)  # the ratio kept
    assert norm.running_mean.mean().item() == pytest.approx(0.1 * 3.0, rel=0.01)  # 0.9 x 0 + 0.1 x the batch's
    norm.eval()
    for w in norm.buffers():
        w.zero_()
    norm.running_var.fill_(4.0)
    assert torch.allclose(norm(s), s / 2, atol=1e-4)  # prediction: the stored statistics


def test_the_weight_share_counts_each_input_by_its_spread_after_normalization():
    net = TwoBranchNet()
    lin = net.combined.layers[0]
    with torch.no_grad():
        lin.weight.fill_(1.0)
        lin.weight[:, 16:] = 2.0  # the tabular inputs' weights are twice the price ones
        net.combined.norm_p.running_var[:8] = 4.0
        net.combined.norm_p.running_var[8:] = 0.0  # half the price numbers dead: the rest carry the branch
        net.combined.norm_t.running_var.fill_(1.0)
    assert weight_share(net.combined)["weight_share_price"] == pytest.approx(1 / 3, rel=1e-4)


def test_the_summary_drift_is_the_shift_and_the_spread_in_units_of_each_branch_s_step_2_scale():
    s_p = torch.zeros(1000, 16)
    s_p[:, 3] = torch.linspace(-1, 1, 1000) + 1.0  # mean 1
    s_t = torch.zeros(1000, 16)
    ref_mean, ref_scale = torch.zeros(32), torch.tensor([2.0, 1.0])
    d = summary_drift(s_p, s_t, ref_mean, ref_scale)
    assert d["drift_shift_price"] == pytest.approx(0.5) and d["drift_shift_tab"] == pytest.approx(0.0)
    spread = float(np.sqrt(np.var(np.linspace(-1, 1, 1000)) / 16))
    assert d["drift_ratio_price"] == pytest.approx(spread / 2, rel=1e-4) and d["drift_ratio_tab"] == 0.0


def test_the_reference_is_the_combined_normalization_s_running_means_and_scales():
    net = TwoBranchNet()
    with torch.no_grad():
        net.combined.norm_p.running_var.fill_(4.0)
        net.combined.norm_t.running_mean.fill_(1.0)
    mean, scale = norm_reference(net.combined)
    assert mean.shape == (32,) and mean[16:].eq(1.0).all() and mean[:16].eq(0.0).all()
    assert scale.tolist() == pytest.approx([2.0, 1.0], rel=1e-4)


from dsr.two_branch import joint_loss  # noqa: E402


def test_the_joint_network_gives_the_tabular_mlp_s_eight_numbers_and_the_cnn_s_prediction_to_a_nine_input_head():
    from dsr.cnn import HEIGHT, WIDTH  # noqa: PLC0415
    net = TwoBranchNet(hidden=(8,), chart="day_cnn", tab_hidden=(16, 8))
    assert _weights(net.tabular.body) == 17 * 16 + 16 and _weights(net.tabular.deep) == 16 * 8 + 8  # 17 -> 16 -> 8
    assert _weights(net.combined) == (9 * 8 + 8) + (8 + 1)  # 9 -> 8 -> 1
    out = net.to(DEVICE).eval()(torch.zeros(2, 1, HEIGHT, WIDTH, device=DEVICE), torch.zeros(2, 17, device=DEVICE))
    assert out["s_t"].shape == (2, 8) and out["s_p"].shape == (2, 1) and out["comb"].shape == (2,)


def test_step_1b_trains_both_layers_of_a_two_layer_tabular_mlp():
    net = TwoBranchNet(hidden=(8,), chart="day_cnn", tab_hidden=(16, 8))
    prepare(net, "1b")
    assert all(w.requires_grad for w in net.tabular.parameters()) and net.tabular.training


def test_the_joint_run_freezes_the_mlp_s_first_layer_and_trains_its_second_with_the_cnn_and_the_head():
    net = TwoBranchNet(hidden=(8,), chart="day_cnn", tab_hidden=(16, 8))
    prepare(net, "joint")
    assert not any(w.requires_grad for w in net.tabular.body[0].parameters())  # 17 -> 16, as step 1b left it
    assert all(w.requires_grad for w in net.tabular.deep.parameters())  # 16 -> 8
    assert not any(w.requires_grad for w in net.tabular.head.parameters())  # its own output: no loss
    assert all(w.requires_grad for w in net.combined.parameters())
    assert all(w.requires_grad for w in [*net.price.one_d.parameters(), *net.price.head.parameters()])
    assert not any(w.requires_grad for w in net.price.blocks.parameters())
    stats_and_dropout = [m for m in net.price.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout))]
    assert stats_and_dropout and not any(m.training for m in stats_and_dropout)  # as step 3: the CNN's kept as is


def test_the_joint_loss_weighs_each_stock_day_s_combined_and_cnn_errors_by_lambda():
    comb, y = torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.zeros(4)  # four stock-days, the last two with a chart
    price, y_charted = torch.tensor([5.0, 6.0]), torch.zeros(2)
    lam = 0.25
    expected = ((1 - lam) * (1 + 4 + 9 + 16) + lam * (25 + 36)) / 4  # per stock-day, over the batch's four
    assert joint_loss(comb, y, price, y_charted, lam).item() == pytest.approx(expected)
    assert joint_loss(comb, y, price[:0], y_charted[:0], lam).item() == pytest.approx((1 - lam) * 30 / 4)
