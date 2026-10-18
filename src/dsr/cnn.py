"""The CNN on chart images (explanations/CNN/features.md, architecture.md, training_plan.md): each large-cap
stock-day's drawing data (where each bar's prices and volume fall in the image, and its six color values), the images
drawn from it on the GPU, the network and its training."""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

# the image: the previous day's last ten minutes, the night, then the session's ten-minute bars, 3 pixels each
LAST10, NIGHT = 0, 1  # slots of the previous day's last ten minutes and of the night (its evening, today's pre-open)
FIRST_SESSION_SLOT = 2  # session bar b is slot b + 2
N_SLOTS = 40  # 2 + the 38 session bars of a normal day; on early closes (20 bars) the rest stay empty
NIGHT_PIECES = ("prev_evening", "pre_open")
PRICES = ["open", "high", "low", "close"]
PRICE_ROWS, VOLUME_ROWS = 51, 12
HEIGHT, WIDTH = PRICE_ROWS + 1 + VOLUME_ROWS, 3 * N_SLOTS  # 64 x 120
NO_ROW = 255  # a price that is not drawn

# the colors (features.md)
PRIOR_TRADES = 2  # the uniform prior Beta(1, 1) on the share of retail trades that are buys
HISTORY_DAYS, HISTORY_MIN = 20, 15  # abnormal retail activity: the same slot on the previous 20 days, at least 15
Z_CLIP = 3.0  # -3 standard deviations is pixel 0, +3 is 255, the cross-sectional mean 128
FLOW_COLUMNS = ["shares", "dollars", "mid_buy_trades", "mid_sell_trades", "mid_buy_dollars", "mid_sell_dollars",
                "off_trades", "off_odd_trades", "exch_qr_buy_shares", "exch_qr_sell_shares"]
PRICE_FEATURES = ["retail_imbalance", "size_tilt", "exchange_imbalance"]  # red, green, blue on price bars
VOLUME_FEATURES = ["retail_share", "abnormal_activity", "odd_lot_share"]  # red, green, blue on volume bars
COLORS = PRICE_FEATURES + VOLUME_FEATURES


def retail(buy_trades, sell_trades):
    """Signed retail trades and their imbalance: the posterior mean of 2p - 1, p the share of retail trades that are
    buys, under the prior Beta(1, 1): (buys - sells) / (buys + sells + 2). Never missing; 0 without trades."""
    signed = buy_trades + sell_trades
    return signed, (buy_trades - sell_trades) / (signed + PRIOR_TRADES)


def flow(b: pd.DataFrame) -> pd.DataFrame:
    """Each bar's raw color values from its counts (FLOW_COLUMNS): retail trades and imbalance (midpoint rule, in
    trades), the size tilt (the imbalance in dollars, shrunk by the same n / (n + 2), minus the one in trades), retail
    share of dollars, exchange order-flow imbalance (quote rule, in shares) and odd-lot share of the off-exchange
    trades."""
    trades, imbalance = retail(b["mid_buy_trades"], b["mid_sell_trades"])
    dollars = b["mid_buy_dollars"] + b["mid_sell_dollars"]
    dollar_imbalance = ((b["mid_buy_dollars"] - b["mid_sell_dollars"]) / dollars.where(dollars > 0)
                        * trades / (trades + PRIOR_TRADES)).fillna(0.0)
    exchange = b["exch_qr_buy_shares"] + b["exch_qr_sell_shares"]
    return pd.DataFrame({
        "retail_trades": trades, "retail_imbalance": imbalance, "size_tilt": dollar_imbalance - imbalance,
        "retail_share": dollars / b["dollars"].where(b["dollars"] > 0),
        "exchange_imbalance": (b["exch_qr_buy_shares"] - b["exch_qr_sell_shares"]) / exchange.where(exchange > 0),
        "odd_lot_share": b["off_odd_trades"] / b["off_trades"].where(b["off_trades"] > 0),
    }, index=b.index)


def slot_of(piece, bar) -> np.ndarray:
    """The image slot of decision-day bars rows: prev_last10 -> LAST10, the evening and pre-open -> NIGHT, session bar
    b -> b + 2."""
    piece = np.asarray(piece, dtype=object)
    return np.where(piece == "prev_last10", LAST10,
                    np.where(np.isin(piece, NIGHT_PIECES), NIGHT, np.asarray(bar) + FIRST_SESSION_SLOT))


def day_table(bars: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One decision day's bars by (permno, slot): the counts summed over each slot's rows (FLOW_COLUMNS; the night sums
    the evening and the pre-open) and the prices drawn. bars: the day's bars.parquet rows (permno, piece, bar, close,
    FLOW_COLUMNS); prices: its prices.parquet rows (permno, piece, bar, px_open, px_high, px_low, px_close). The session
    bars and the last ten minutes take their round-lot prices; the night opens at the last ten minutes' round-lot close,
    closes at the last trade before 09:30 (the pre-open's, else the evening's; any size), and its high and low are the
    larger and the smaller of the two."""
    b = bars.assign(slot=slot_of(bars["piece"], bars["bar"]))
    counts = b.groupby(["permno", "slot"])[FLOW_COLUMNS].sum().astype(float)
    px = (prices.assign(slot=slot_of(prices["piece"], prices["bar"]))
          .set_index(["permno", "slot"])[[f"px_{p}" for p in PRICES]]
          .rename(columns=lambda c: c.removeprefix("px_")).astype(float))
    piece = np.asarray(b["piece"], dtype=object)
    late = {p: b.loc[piece == p].set_index("permno")["close"].astype(float) for p in NIGHT_PIECES}
    stocks = pd.Index(sorted(set(late["prev_evening"].index) | set(late["pre_open"].index)), name="permno")
    start = px.loc[px.index.get_level_values("slot") == LAST10, "close"].droplevel("slot")
    night = pd.DataFrame({"open": start.reindex(stocks),
                          "close": late["pre_open"].reindex(stocks).fillna(late["prev_evening"].reindex(stocks))})
    night["high"], night["low"] = night[["open", "close"]].max(axis=1), night[["open", "close"]].min(axis=1)
    night.index = pd.MultiIndex.from_arrays([stocks, np.full(len(stocks), NIGHT)], names=["permno", "slot"])
    return counts.join(pd.concat([px, night[PRICES]]), how="outer")


def standardize_by_slot(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each column's pixel value (column_px, uint8) across the rows of each slot (df indexed by permno and slot, one
    day): winsorized at the 1st and 99th percentiles, standardized, -Z_CLIP to 0, the mean to 128, +Z_CLIP to 255;
    missing to 128."""
    slot = df.index.get_level_values("slot")
    g = df[columns].groupby(slot)
    lo, hi = g.quantile(0.01).reindex(slot).to_numpy(), g.quantile(0.99).reindex(slot).to_numpy()
    w = pd.DataFrame(np.clip(df[columns].to_numpy(float), lo, hi), index=df.index, columns=columns)
    by = w.groupby(slot)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((w - by.transform("mean")) / by.transform("std")).to_numpy(float)
        px = np.where(np.isnan(z), 128, np.round((np.clip(z, -Z_CLIP, Z_CLIP) + Z_CLIP) / (2 * Z_CLIP) * 255))
    return pd.DataFrame(px.astype(np.uint8), index=df.index, columns=[f"{c}_px" for c in columns])


class RetailHistory:
    """Each stock's retail trades in each slot over the last HISTORY_DAYS days added. A day the stock was in the panel
    counts with 0 trades in a slot it did not trade in; a slot the day did not have (on early closes) and a day the
    stock was not in the panel do not count."""

    def __init__(self, permnos):
        self.pos = pd.Series(np.arange(len(permnos)), index=pd.Index(sorted(permnos)))
        self.trades = np.zeros((HISTORY_DAYS, len(permnos), N_SLOTS), np.float32)
        self.counted = np.zeros((HISTORY_DAYS, len(permnos), N_SLOTS), bool)
        self.days = 0

    def mean(self, index: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray]:
        """The mean retail trades and the number of days counted, for each (permno, slot) of `index`."""
        rows = self.pos.reindex(index.get_level_values("permno")).to_numpy()
        slots = index.get_level_values("slot").to_numpy()
        known = ~np.isnan(rows)
        r = np.where(known, rows, 0).astype(int)
        total, days = self.trades.sum(axis=0)[r, slots], self.counted.sum(axis=0)[r, slots]
        days = np.where(known, days, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(days > 0, total / days, np.nan), days

    def add(self, members, n_slots: int, trades: pd.Series):
        """One more day: `members` the stocks in the panel that day, `n_slots` its slots (2 + its session bars), and
        `trades` its retail trades by (permno, slot) (missing: 0)."""
        k = self.days % HISTORY_DAYS
        self.trades[k], self.counted[k] = 0, False
        rows = self.pos.reindex(list(members)).dropna().astype(int).to_numpy()
        self.counted[k][np.ix_(rows, np.arange(n_slots))] = True
        t = trades[trades.index.get_level_values("permno").isin(list(members))]
        r = self.pos.reindex(t.index.get_level_values("permno")).to_numpy(dtype=float)
        slots = np.asarray(t.index.get_level_values("slot"), dtype=int)
        ok = ~np.isnan(r) & (slots < n_slots)
        self.trades[k][r[ok].astype(int), slots[ok]] = t.to_numpy(dtype=float)[ok]
        self.days += 1


def day_retail_trades(bars: pd.DataFrame) -> pd.Series:
    """Signed retail trades (midpoint rule) by (permno, slot) from a day's bars rows (permno, piece, bar,
    mid_buy_trades, mid_sell_trades)."""
    b = bars.assign(slot=slot_of(bars["piece"], bars["bar"]))
    s = b.groupby(["permno", "slot"])[["mid_buy_trades", "mid_sell_trades"]].sum().astype(float)
    return retail(s["mid_buy_trades"], s["mid_sell_trades"])[0]


def pixel_values(table: pd.DataFrame, history: RetailHistory) -> pd.DataFrame:
    """The six color values of each bar that traded (table: day_table of one day's large caps), as pixels
    (standardize_by_slot), abnormal retail activity against the history."""
    present = table[table["shares"] > 0]
    f = flow(present)
    mean, days = history.mean(present.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        activity = np.log((1 + f["retail_trades"].to_numpy()) / (1 + mean))
    f["abnormal_activity"] = np.where(days >= HISTORY_MIN, activity, np.nan)
    return standardize_by_slot(f, COLORS)


def drawing(table: pd.DataFrame, pixels: pd.DataFrame, stocks) -> dict[str, np.ndarray]:
    """Each stock's drawing data, in the order of `stocks`: `rows` (n, N_SLOTS, 4) the image rows of each bar's open,
    high, low and close (0 the top; NO_ROW not drawn), the price scale running from the stock's highest high to its
    lowest low over the slots; `volume` (n, N_SLOTS) the rows each volume bar fills from the bottom, scaled to the
    session's tallest bar, with the two bars before the session capped at it and at least one row for a bar that
    traded; `colors` (n, N_SLOTS, 6) the color pixels (COLORS order; 128 when missing)."""
    stocks = pd.Index(stocks, name="permno")
    idx = pd.MultiIndex.from_product([stocks, range(N_SLOTS)], names=["permno", "slot"])
    t = table.reindex(idx)
    n = len(stocks)
    p = t[PRICES].to_numpy(float).reshape(n, N_SLOTS, 4)
    with warnings.catch_warnings():  # a stock with no price or no session volume: all-NaN slices
        warnings.simplefilter("ignore", RuntimeWarning)
        top, bottom = np.nanmax(p[..., 1], axis=1), np.nanmin(p[..., 2], axis=1)
        span = np.where(top > bottom, top - bottom, 1.0)
        rows = np.clip(np.round((top[:, None, None] - p) / span[:, None, None] * (PRICE_ROWS - 1)), 0, PRICE_ROWS - 1)
        v = t["shares"].to_numpy(float).reshape(n, N_SLOTS)
        session_max = np.nanmax(v[:, FIRST_SESSION_SLOT:], axis=1)
    v = np.where(np.arange(N_SLOTS) < FIRST_SESSION_SLOT, np.fmin(v, session_max[:, None]), v)
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(v > 0, np.maximum(1, np.round(v / session_max[:, None] * VOLUME_ROWS)), 0)
    c = pixels.reindex(idx)[[f"{k}_px" for k in COLORS]].to_numpy(float).reshape(n, N_SLOTS, len(COLORS))
    return {"rows": np.where(np.isnan(rows), NO_ROW, rows).astype(np.uint8),
            "volume": np.clip(np.nan_to_num(filled), 0, VOLUME_ROWS).astype(np.uint8),
            "colors": np.where(np.isnan(c), 128, c).astype(np.uint8)}


def render(rows: torch.Tensor, volume: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Draw a batch of images (float, B x 4 x HEIGHT x WIDTH, 0-255) from drawing data (uint8 tensors on one device:
    rows B x N_SLOTS x 4, volume B x N_SLOTS, colors B x N_SLOTS x 6). Grey: the open as a tick on the bar's left
    column, the high-low range in its middle column, the close as a tick on its right column, 255; one empty row; the
    volume bar in the middle column. Red, green, blue: the bar's price colors on its price pixels and its volume colors
    on its volume pixels; 0 elsewhere."""
    dev, b = rows.device, rows.shape[0]
    r = rows.long()
    pr = torch.arange(PRICE_ROWS, device=dev).view(1, PRICE_ROWS, 1)
    left, right = pr == r[:, None, :, 0], pr == r[:, None, :, 3]
    middle = (pr >= r[:, None, :, 1]) & (pr <= r[:, None, :, 2])
    price = torch.stack([left, middle, right], dim=-1).reshape(b, PRICE_ROWS, WIDTH)
    vr = torch.arange(VOLUME_ROWS, device=dev).view(1, VOLUME_ROWS, 1)
    bar = vr >= (VOLUME_ROWS - volume.long())[:, None, :]
    blank = torch.zeros_like(bar)
    vol = torch.stack([blank, bar, blank], dim=-1).reshape(b, VOLUME_ROWS, WIDTH)
    chart = torch.cat([price, torch.zeros(b, 1, WIDTH, dtype=torch.bool, device=dev), vol], dim=1)  # B x H x W
    c = colors.float().repeat_interleave(3, dim=1)  # B x W x 6
    in_price = (torch.arange(HEIGHT, device=dev) < PRICE_ROWS).view(1, HEIGHT, 1, 1)
    per_row = torch.where(in_price, c[:, None, :, :3], c[:, None, :, 3:])  # B x H x W x 3
    color = (per_row * chart[..., None]).permute(0, 3, 1, 2)
    return torch.cat([chart[:, None].float() * 255, color], dim=1)


class ChartCNN(torch.nn.Module):
    """Jiang, Kelly and Xiu's 20-day block, adapted (architecture.md): blocks of a 5x3 convolution, BatchNorm, LeakyReLU
    and a 2x1 max-pool (the price height only, never time), the first with vertical stride 3 and dilation 2; then
    flatten, dropout and one linear output. Inputs are standardized by channel with `mean` and `sd`."""

    def __init__(self, filters=(32, 64, 128), dropout=0.5, mean=None, sd=None):
        super().__init__()
        layers, c_in = [], 4
        for i, c in enumerate(filters):
            first = i == 0
            layers += [torch.nn.Conv2d(c_in, c, (5, 3), stride=(3, 1) if first else 1, dilation=(2, 1) if first else 1,
                                       padding=(12, 1) if first else (2, 1)),
                       torch.nn.BatchNorm2d(c), torch.nn.LeakyReLU(), torch.nn.MaxPool2d((2, 1))]
            c_in = c
        self.features = torch.nn.Sequential(*layers)
        h = (HEIGHT + 2 * 12 - 2 * (5 - 1) - 1) // 3 + 1  # the first convolution's output height; each pool halves it
        for _ in filters:
            h //= 2
        n = c_in * h * WIDTH  # no dry run on the CPU: after LightGBM, a CPU convolution crashes on macOS (OpenMP)
        self.head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(dropout), torch.nn.Linear(n, 1))
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.register_buffer("mean", torch.zeros(4) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(4) if sd is None else torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, x):
        x = (x - self.mean.view(1, 4, 1, 1)) / self.sd.view(1, 4, 1, 1)
        return self.head(self.features(x)).squeeze(1)


class Images:
    """The drawing data of every stock-day on one device (rows, volume, colors as in `drawing`, stacked)."""

    def __init__(self, rows, volume, colors, device):
        self.rows, self.volume, self.colors = (torch.as_tensor(a, device=device) for a in (rows, volume, colors))
        self.device = torch.device(device)

    def draw(self, idx: torch.Tensor) -> torch.Tensor:
        return render(self.rows[idx], self.volume[idx], self.colors[idx])


def predict(net: ChartCNN, images: Images, idx, batch: int = 4096) -> np.ndarray:
    """The network's predictions (in the target's units scaled by the training sd) for the stock-days `idx`."""
    net.eval()
    idx = torch.as_tensor(idx, device=images.device)
    with torch.no_grad():
        return torch.cat([net(images.draw(idx[i:i + batch])) for i in range(0, len(idx), batch)]).cpu().numpy()


def fit(images: Images, y: np.ndarray, train: np.ndarray, hold: np.ndarray, p: dict, seed: int = 0,
        on_epoch: Callable[[dict], None] | None = None) -> tuple[ChartCNN, float, dict]:
    """Train a ChartCNN on the stock-days `train` (positions in `images` and `y`), squared error on y scaled by its
    training sd, AdamW, stopped early on the hold-out `hold` (its squared error; patience p["patience"] epochs, at most
    p["max_epochs"]), keeping the best epoch's weights. The inputs are standardized by channel with the mean and sd of
    a sample of training images. After each epoch, `on_epoch` gets the epoch, the training loss, the hold-out loss and
    the hold-out predictions. Returns the network, the target's scale and what the fit chose."""
    dev = images.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(np.std(y[train])) or 1.0
    yt = torch.as_tensor(np.nan_to_num(y / scale).astype(np.float32), device=dev)
    sample = torch.as_tensor(np.sort(rng.choice(train, min(p.get("norm_sample", 20_000), len(train)), replace=False)),
                             device=dev)
    with torch.no_grad():
        x = torch.cat([images.draw(sample[i:i + 2048]) for i in range(0, len(sample), 2048)])
        mean, sd = x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)).clamp_min(1e-6)
    net = ChartCNN(tuple(p["filters"]), p["dropout"], mean, sd).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    train_t, hold_np = torch.as_tensor(train, device=dev), np.asarray(hold)
    y_hold = yt[torch.as_tensor(hold_np, device=dev)]
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, int(p["max_epochs"]) + 1):
        net.train()
        order = train_t[torch.randperm(len(train_t), device=dev)]
        total = torch.zeros((), device=dev)
        for i in range(0, len(order), int(p["batch_size"])):
            idx = order[i:i + int(p["batch_size"])]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(images.draw(idx)), yt[idx])
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
        pred = predict(net, images, hold_np)
        v = float(((torch.as_tensor(pred, device=dev) - y_hold) ** 2).mean())
        if on_epoch is not None:
            on_epoch({"epoch": epoch, "train_loss": float(total) / len(train_t), "holdout_loss": v,
                      "holdout_pred": pred * scale})
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= int(p["patience"]):
                break
    net.load_state_dict(best_state)
    return net, scale, {"epochs": best_epoch, "holdout_loss": best}
