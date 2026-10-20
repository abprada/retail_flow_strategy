import numpy as np
import pandas as pd
import pytest

import dsr.models  # noqa: F401  lightgbm before torch: torch first crashes lightgbm on macOS (two OpenMP runtimes)
import torch  # noqa: E402

from dsr.cnn import (COLORS, FLOW_COLUMNS, HEIGHT, HISTORY_DAYS, LAST10, NIGHT, NO_ROW, N_SLOTS, PRICE_ROWS,
                     VOLUME_ROWS, WIDTH, ChartCNN, Images, RetailHistory, day_table, drawing, fit, predict, render,
                     standardize_by_slot)

# the GPU when there is one: CPU convolutions after LightGBM crash on macOS (two OpenMP runtimes)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _blank(n=1):
    return (np.full((n, N_SLOTS, 4), NO_ROW, np.uint8), np.zeros((n, N_SLOTS), np.uint8),
            np.full((n, N_SLOTS, len(COLORS)), 128, np.uint8))


def test_render_draws_the_ticks_the_range_the_volume_and_their_colors():
    rows, volume, colors = _blank()
    rows[0, 2] = [10, 5, 20, 15]  # slot 2 (the first session bar): columns 6, 7, 8
    volume[0, 2] = 6
    colors[0, 2] = [200, 100, 50, 30, 60, 90]
    img = render(*(torch.as_tensor(a) for a in (rows, volume, colors))).numpy()
    assert img.shape == (1, 4, HEIGHT, WIDTH)
    grey = img[0, 0]
    assert grey[10, 6] == 255 and grey[15, 8] == 255 and grey[:, 7][5:21].min() == 255
    assert grey[HEIGHT - 6:, 7].min() == 255 and grey[HEIGHT - 7, 7] == 0  # 6 volume rows from the bottom
    assert (grey > 0).sum() == 1 + 16 + 1 + 6
    assert img[0, 1, 10, 6] == 200 and img[0, 2, 10, 6] == 100 and img[0, 3, 10, 6] == 50  # price colors
    assert img[0, 1, HEIGHT - 1, 7] == 30 and img[0, 3, HEIGHT - 1, 7] == 90  # volume colors
    assert img[0, 1:, grey == 0].max() == 0  # no color off the chart


def test_drawing_scales_prices_to_the_stock_s_range_and_caps_the_bars_before_the_session():
    idx = pd.MultiIndex.from_tuples([(7, 2), (7, 3), (7, LAST10), (7, NIGHT)], names=["permno", "slot"])
    table = pd.DataFrame({"open": [10, 11, 10.2, 10.2], "high": [11, 12, 10.5, 10.2], "low": [9, 10, 10, 8],
                          "close": [10.5, 11.5, 10.2, 8], "shares": [100, 50, 400, 10]}, index=idx, dtype=float)
    pixels = pd.DataFrame({f"{c}_px": [200] * 4 for c in COLORS}, index=idx, dtype=np.uint8)
    d = drawing(table, pixels, [7])
    assert d["rows"][0, 3, 1] == 0 and d["rows"][0, NIGHT, 2] == PRICE_ROWS - 1  # the top is 12, the bottom 8
    assert d["rows"][0, 2, 0] == round((12 - 10) / 4 * (PRICE_ROWS - 1))
    assert list(d["volume"][0, :4]) == [VOLUME_ROWS, 1, VOLUME_ROWS, 6]  # the last ten minutes capped; the night >= 1
    assert (d["rows"][0, 4:] == NO_ROW).all() and (d["volume"][0, 4:] == 0).all() and (d["colors"][0, 4:] == 128).all()


def test_standardize_by_slot_maps_the_mean_to_128_missing_to_128_and_each_slot_alone():
    idx = pd.MultiIndex.from_product([range(5), [2, 3]], names=["permno", "slot"])
    df = pd.DataFrame({"x": [1.0, 101, 2, 102, 3, 103, 4, 104, np.nan, np.nan]}, index=idx)  # slot 3 = slot 2 + 100
    px = standardize_by_slot(df, ["x"])["x_px"]
    assert px.loc[(4, 2)] == 128 and px.loc[(4, 3)] == 128
    assert (px.xs(2, level="slot").to_numpy() == px.xs(3, level="slot").to_numpy()).all()
    assert px.loc[(0, 2)] < 128 < px.loc[(3, 2)]


def _bars(rows):
    out = pd.DataFrame(rows, columns=["permno", "piece", "bar", "close", "shares", "mid_buy_trades"])
    for c in FLOW_COLUMNS:
        if c not in out:
            out[c] = 1
    return out


def test_day_table_puts_the_night_between_the_last_ten_minutes_close_and_the_last_trade_before_the_open():
    bars = _bars([(1, "prev_last10", 0, 10.0, 500, 3), (1, "prev_evening", 0, 9.0, 20, 1), (1, "pre_open", 0, 9.5, 30, 2),
                  (1, "session", 0, 9.8, 100, 5), (2, "prev_evening", 0, 20.0, 10, 0)])
    prices = pd.DataFrame({"permno": [1, 1, 2], "piece": ["prev_last10", "session", "prev_last10"], "bar": [0, 0, 0],
                           "px_open": [10.1, 9.6, 21.0], "px_high": [10.2, 9.9, 21.5], "px_low": [9.9, 9.5, 20.5],
                           "px_close": [10.0, 9.8, 21.2]})
    t = day_table(bars, prices)
    night = t.loc[(1, NIGHT)]
    assert (night["open"], night["close"], night["high"], night["low"]) == (10.0, 9.5, 10.0, 9.5)
    assert night["shares"] == 50 and night["mid_buy_trades"] == 3  # the evening and the pre-open summed
    assert t.loc[(1, 2), "open"] == 9.6 and t.loc[(1, LAST10), "high"] == 10.2  # round-lot prices
    assert t.loc[(2, NIGHT), "close"] == 20.0 and t.loc[(2, NIGHT), "high"] == 21.2  # no pre-open: the evening's close


def test_retail_history_counts_days_without_trades_as_zero_and_skips_slots_a_day_did_not_have():
    h = RetailHistory([1, 2])
    trades = lambda d: pd.Series(d, index=pd.MultiIndex.from_tuples(list(d), names=["permno", "slot"]), dtype=float)
    h.add({1, 2}, N_SLOTS, trades({(1, 2): 4.0, (1, 30): 6.0}))
    h.add({1}, 22, trades({(1, 2): 2.0}))  # an early close: slots 22 and up did not exist; stock 2 not in the panel
    mean, days = h.mean(pd.MultiIndex.from_tuples([(1, 2), (2, 2), (1, 30), (3, 2)], names=["permno", "slot"]))
    assert list(days) == [2, 1, 1, 0] and mean[0] == 3.0 and mean[1] == 0.0 and mean[2] == 6.0 and np.isnan(mean[3])
    for _ in range(HISTORY_DAYS):
        h.add({1}, N_SLOTS, trades({}))
    assert h.mean(pd.MultiIndex.from_tuples([(1, 2)], names=["permno", "slot"]))[0][0] == 0.0  # the old days left


def test_the_network_takes_a_batch_of_images_and_gives_one_number_each():
    net = ChartCNN().to(DEVICE)
    assert net(torch.zeros(3, 4, HEIGHT, WIDTH, device=DEVICE)).shape == (3,)
    assert net.head[2].in_features == 128 * 3 * WIDTH


def test_fit_and_predict_run_end_to_end_on_a_few_images():
    rng = np.random.default_rng(0)
    n = 64
    rows = np.sort(rng.integers(0, PRICE_ROWS, (n, N_SLOTS, 4)), axis=2).astype(np.uint8)
    volume = rng.integers(0, VOLUME_ROWS + 1, (n, N_SLOTS)).astype(np.uint8)
    colors = rng.integers(0, 256, (n, N_SLOTS, len(COLORS))).astype(np.uint8)
    y = rng.normal(0, 0.01, n).astype(np.float32)
    images = Images(rows, volume, colors, DEVICE)
    epochs = []
    p = {"filters": (4, 4, 4), "dropout": 0.5, "learning_rate": 1e-3, "weight_decay": 0.05, "batch_size": 16,
         "max_epochs": 2, "patience": 2, "norm_sample": 32}
    net, scale, info = fit(images, y, np.arange(48), np.arange(48, 64), p, on_epoch=epochs.append)
    assert len(epochs) == 2 and info["epochs"] in (1, 2) and scale == pytest.approx(float(np.std(y[:48])))
    assert predict(net, images, np.arange(10)).shape == (10,)


from dsr.cnn import (CONTEXT_DAYS, DAY_SLOTS, HOUR_SLOTS, LAST_FULL, NIGHT_SLOT, W_BARS, W_HEIGHT, W_PRICE_ROWS,  # noqa: E402
                     W_WIDTH, WINDOW_DAYS, HourWindows, hour_table, moving_average, render_window)


def _session(n_bars, permno=1):
    bar = np.arange(n_bars)
    out = pd.DataFrame({"permno": permno, "bar": bar, "open": 10.0 + bar, "high": 10.5 + bar, "low": 9.5 + bar,
                        "close": 10.2 + bar})
    for c in FLOW_COLUMNS:
        out[c] = 1.0
    return out


def test_hour_table_keeps_the_cutoff_hour_and_adds_the_last_ten_minutes_only_to_the_full_last_bar():
    ten = _session(1).assign(open=60.0, high=61.0, low=59.0, close=60.5)
    night = _bars([(1, "prev_evening", 0, 7.0, 5, 1), (1, "pre_open", 0, 8.0, 5, 1)])
    t = hour_table(_session(38), ten, night, pd.Series({1: 9.0}), 38)
    assert t.loc[(1, 0), "open"] == 10.0 and t.loc[(1, 0), "high"] == 15.5 and t.loc[(1, 0), "shares"] == 6
    assert t.loc[(1, 6), "close"] == 47.2 and t.loc[(1, 6), "shares"] == 2  # bars 36-37: 15:30 to the cutoff
    assert t.loc[(1, LAST_FULL), "close"] == 60.5 and t.loc[(1, LAST_FULL), "high"] == 61.0
    assert t.loc[(1, LAST_FULL), "shares"] == 3 and t.loc[(1, LAST_FULL), "open"] == 46.0
    night_bar = t.loc[(1, NIGHT_SLOT)]
    assert (night_bar["open"], night_bar["close"], night_bar["high"], night_bar["low"]) == (9.0, 8.0, 9.0, 8.0)
    early = hour_table(_session(20), ten, night, pd.Series({1: 9.0}), 20)  # an early close: the last hour is slot 3
    assert early.loc[(1, 3), "shares"] == 2 and early.loc[(1, LAST_FULL), "shares"] == 3
    assert (1, 4) not in early.index


def test_moving_average_skips_missing_closes_and_needs_enough_of_them():
    closes = torch.tensor([[1.0, 2.0, float("nan"), 3.0, 4.0, 5.0]])
    ma = moving_average(closes, hours=3, at_least=2).numpy()[0]
    assert np.isnan(ma[0]) and ma[1] == 1.5 and ma[2] == 1.5 and ma[3] == 2.5 and ma[5] == 4.0


def _context(b=1):
    """CONTEXT_DAYS days of stored slots: day k's prices are around 100 + k; the last ten minutes jump to 200."""
    prices = np.full((b, CONTEXT_DAYS, DAY_SLOTS, 4), np.nan, np.float32)
    for k in range(CONTEXT_DAYS):
        prices[:, k, :HOUR_SLOTS] = [100 + k, 100.5 + k, 99.5 + k, 100 + k]
        prices[:, k, LAST_FULL] = [100 + k, 200, 99.5 + k, 200]
    prices[:, -1, NIGHT_SLOT] = [124.0, 124.0, 120.0, 120.0]
    shares = np.ones((b, CONTEXT_DAYS, DAY_SLOTS), np.float32)
    colors = np.full((b, CONTEXT_DAYS, DAY_SLOTS, len(COLORS)), 200, np.uint8)
    return prices, shares, colors


def test_render_window_draws_past_days_with_their_last_ten_minutes_and_the_decision_day_without():
    prices, shares, colors = (torch.as_tensor(a, device=DEVICE) for a in _context())
    early = torch.zeros(1, CONTEXT_DAYS, dtype=torch.bool, device=DEVICE)
    img = render_window(prices, shares, colors, early).cpu().numpy()[0]
    assert img.shape == (4, W_HEIGHT, W_WIDTH) and W_WIDTH == 3 * W_BARS
    grey = img[0, :W_PRICE_ROWS]
    assert grey[0, 3 * (HOUR_SLOTS - 1) + 2] == 255  # a past day's last bar closes at 200, the top row
    today_last = 3 * (W_BARS - 1)  # the decision day's last bar: no jump to 200
    assert grey[0, today_last:today_last + 3].max() == 0
    night = 3 * (WINDOW_DAYS - 1) * HOUR_SLOTS  # the night bar, just before the decision day's first bar
    assert grey[:, night + 1].max() == 255 and img[1, :W_PRICE_ROWS, night + 1].max() == 200
    assert (img[1:][:, img[0] == 0] == 0).all()  # no color off the chart
    prices[0, -1, NIGHT_SLOT] = torch.tensor([124.0, 1000.0, 120.0, 1000.0])  # a stray print in the pre-open
    stray = render_window(prices, shares, colors, early).cpu().numpy()[0]
    assert (stray[0, :, :night] == img[0, :, :night]).all()  # the scale ignores it
    assert stray[0, 0, night + 2] == 255  # its close clipped at the top row


def test_hour_windows_gather_each_target_s_days_and_leave_missing_days_empty():
    prices, shares, colors = _context()
    row_of = np.arange(CONTEXT_DAYS)[None, :]
    w = HourWindows(prices[0], shares[0], colors[0], row_of, np.zeros(CONTEXT_DAYS, bool), [0, 0],
                    [CONTEXT_DAYS - 1, 2], DEVICE)
    img = w.draw(torch.tensor([0, 1], device=DEVICE)).cpu().numpy()
    expected = render_window(*(torch.as_tensor(a, device=DEVICE) for a in (prices, shares, colors)),
                             torch.zeros(1, CONTEXT_DAYS, dtype=torch.bool, device=DEVICE)).cpu().numpy()[0]
    assert (img[0] == expected).all()  # the last day: every gathered day exists
    first_drawn = 3 * HOUR_SLOTS * (WINDOW_DAYS - 3)  # day 2: only days 0-2 exist, the last three of its window
    assert img[1, 0, :, :first_drawn].max() == 0 and img[1, 0, :, first_drawn:].max() == 255


from dsr.cnn import D_CONTEXT, D_DAYS, D_HEIGHT, D_PRICE_ROWS, D_SLOTS, D_WIDTH, DayWindows, daily_table, render_days  # noqa: E402,I001


def test_daily_table_has_the_day_to_the_cutoff_and_to_the_close():
    ten = _session(1).assign(open=60.0, high=61.0, low=59.0, close=60.5)
    t = daily_table(_session(38), ten)
    assert t.loc[(1, 0), "open"] == 10.0 and t.loc[(1, 0), "close"] == 47.2 and t.loc[(1, 0), "shares"] == 38
    assert t.loc[(1, 1), "close"] == 60.5 and t.loc[(1, 1), "high"] == 61.0 and t.loc[(1, 1), "shares"] == 39


def test_render_days_draws_past_days_to_the_close_and_the_decision_day_to_the_cutoff():
    prices = np.full((1, D_CONTEXT, D_SLOTS, 4), np.nan, np.float32)
    for k in range(D_CONTEXT):
        prices[0, k, 0] = [100 + k, 101 + k, 99 + k, 100 + k]
        prices[0, k, 1] = [100 + k, 200, 99 + k, 200]  # the close jumps to 200 in the last ten minutes
    shares = np.ones((1, D_CONTEXT, D_SLOTS), np.float32)
    colors = np.full((1, D_CONTEXT, D_SLOTS, len(COLORS)), 200, np.uint8)
    img = render_days(*(torch.as_tensor(a, device=DEVICE) for a in (prices, shares, colors))).cpu().numpy()[0]
    assert img.shape == (4, D_HEIGHT, D_WIDTH) and D_WIDTH == 3 * D_DAYS
    grey = img[0, :D_PRICE_ROWS]
    assert grey[0, 2] == 255 and grey[0, D_WIDTH - 3:].max() == 0  # past days close at the top; the decision day not
    w = DayWindows(prices[0], shares[0], colors[0], np.arange(D_CONTEXT)[None, :], np.zeros(D_CONTEXT, bool), [0],
                   [D_CONTEXT - 1], DEVICE)
    assert (w.draw(torch.tensor([0], device=DEVICE)).cpu().numpy()[0] == img).all()


def test_grey_images_draw_the_grey_channel_alone_and_the_network_takes_one_channel():
    from dsr.cnn import GreyImages  # noqa: PLC0415
    rows, volume, colors = _blank(2)
    rows[:, 2] = [10, 5, 20, 15]
    full, grey = Images(rows, volume, colors, DEVICE), GreyImages(rows, volume, colors, DEVICE)
    idx = torch.tensor([0, 1], device=DEVICE)
    assert grey.draw(idx).shape == (2, 1, HEIGHT, WIDTH)
    assert (grey.draw(idx) == full.draw(idx)[:, :1]).all()
    net = ChartCNN(channels=1).to(DEVICE)
    assert net(grey.draw(idx)).shape == (2,) and net.features[0].in_channels == 1


def test_fit_runs_on_the_grey_channel_alone():
    from dsr.cnn import GreyImages  # noqa: PLC0415
    rng = np.random.default_rng(1)
    n = 48
    rows = np.sort(rng.integers(0, PRICE_ROWS, (n, N_SLOTS, 4)), axis=2).astype(np.uint8)
    volume = rng.integers(0, VOLUME_ROWS + 1, (n, N_SLOTS)).astype(np.uint8)
    colors = rng.integers(0, 256, (n, N_SLOTS, len(COLORS))).astype(np.uint8)
    y = rng.normal(0, 0.01, n).astype(np.float32)
    p = {"filters": (4, 4, 4), "dropout": 0.5, "learning_rate": 1e-3, "weight_decay": 0.05, "batch_size": 16,
         "max_epochs": 1, "patience": 1, "norm_sample": 16}
    net, _, _ = fit(GreyImages(rows, volume, colors, DEVICE), y, np.arange(32), np.arange(32, 48), p)
    assert net.features[0].in_channels == 1 and net.mean.shape == (1,)
