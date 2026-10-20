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
