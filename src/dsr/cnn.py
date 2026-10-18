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


def standardize_by_slot(df: pd.DataFrame, columns: list[str], reference=None) -> pd.DataFrame:
    """Each column's pixel value (column_px, uint8) across the rows of each slot (df indexed by permno and slot, one
    day): winsorized at the 1st and 99th percentiles, standardized, -Z_CLIP to 0, the mean to 128, +Z_CLIP to 255;
    missing to 128. The percentiles, mean and standard deviation are those of the `reference` rows (a boolean mask;
    all rows when None), applied to every row."""
    slot = df.index.get_level_values("slot")
    ref = np.ones(len(df), bool) if reference is None else np.asarray(reference, bool)
    values = df[columns].to_numpy(float)
    g = pd.DataFrame(values[ref], columns=columns).groupby(slot[ref])
    lo, hi = g.quantile(0.01).reindex(slot).to_numpy(), g.quantile(0.99).reindex(slot).to_numpy()
    w = np.clip(values, lo, hi)
    gw = pd.DataFrame(w[ref], columns=columns).groupby(slot[ref])
    mean, sd = gw.mean().reindex(slot).to_numpy(), gw.std().reindex(slot).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (w - mean) / sd
        px = np.where(np.isnan(z), 128, np.round((np.clip(z, -Z_CLIP, Z_CLIP) + Z_CLIP) / (2 * Z_CLIP) * 255))
    return pd.DataFrame(px.astype(np.uint8), index=df.index, columns=[f"{c}_px" for c in columns])


class RetailHistory:
    """Each stock's retail trades in each slot over the last HISTORY_DAYS days added. A day the stock was in the panel
    counts with 0 trades in a slot it did not trade in; a slot the day did not have (on early closes) and a day the
    stock was not in the panel do not count."""

    def __init__(self, permnos, n_slots: int = N_SLOTS):
        self.pos = pd.Series(np.arange(len(permnos)), index=pd.Index(sorted(permnos)))
        self.trades = np.zeros((HISTORY_DAYS, len(permnos), n_slots), np.float32)
        self.counted = np.zeros((HISTORY_DAYS, len(permnos), n_slots), bool)
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

    def add(self, members, n_slots, trades: pd.Series):
        """One more day: `members` the stocks in the panel that day, `n_slots` the slots it had (a number n: slots 0 to
        n - 1; or a list of slots), and `trades` its retail trades by (permno, slot) (missing: 0)."""
        k = self.days % HISTORY_DAYS
        self.trades[k], self.counted[k] = 0, False
        had = np.arange(n_slots) if np.isscalar(n_slots) else np.asarray(n_slots, dtype=int)
        rows = self.pos.reindex(list(members)).dropna().astype(int).to_numpy()
        self.counted[k][np.ix_(rows, had)] = True
        t = trades[trades.index.get_level_values("permno").isin(list(members))]
        r = self.pos.reindex(t.index.get_level_values("permno")).to_numpy(dtype=float)
        slots = np.asarray(t.index.get_level_values("slot"), dtype=int)
        ok = ~np.isnan(r) & np.isin(slots, had)
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

    def __init__(self, filters=(32, 64, 128), dropout=0.5, mean=None, sd=None, height=HEIGHT, width=WIDTH):
        super().__init__()
        layers, c_in = [], 4
        for i, c in enumerate(filters):
            first = i == 0
            layers += [torch.nn.Conv2d(c_in, c, (5, 3), stride=(3, 1) if first else 1, dilation=(2, 1) if first else 1,
                                       padding=(12, 1) if first else (2, 1)),
                       torch.nn.BatchNorm2d(c), torch.nn.LeakyReLU(), torch.nn.MaxPool2d((2, 1))]
            c_in = c
        self.features = torch.nn.Sequential(*layers)
        h = (height + 2 * 12 - 2 * (5 - 1) - 1) // 3 + 1  # the first convolution's output height; each pool halves it
        for _ in filters:
            h //= 2
        n = c_in * h * width  # no dry run on the CPU: after LightGBM, a CPU convolution crashes on macOS (OpenMP)
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

    height, width, predict_batch = HEIGHT, WIDTH, 4096

    def __init__(self, rows, volume, colors, device):
        self.rows, self.volume, self.colors = (torch.as_tensor(a, device=device) for a in (rows, volume, colors))
        self.device = torch.device(device)

    def draw(self, idx: torch.Tensor) -> torch.Tensor:
        return render(self.rows[idx], self.volume[idx], self.colors[idx])


def predict(net: ChartCNN, images, idx, batch: int | None = None) -> np.ndarray:
    """The network's predictions (in the target's units scaled by the training sd) for the stock-days `idx`, in
    batches of the images' `predict_batch` unless given."""
    batch = batch or images.predict_batch
    net.eval()
    idx = torch.as_tensor(idx, device=images.device)
    with torch.no_grad():
        return torch.cat([net(images.draw(idx[i:i + batch])) for i in range(0, len(idx), batch)]).cpu().numpy()


def fit(images, y: np.ndarray, train: np.ndarray, hold: np.ndarray, p: dict, seed: int = 0,
        on_epoch: Callable[[dict], None] | None = None,
        score: Callable[[np.ndarray], float] | None = None) -> tuple[ChartCNN, float, dict]:
    """Train a ChartCNN on the stock-days `train` (positions in `images`, an Images or HourWindows, and `y`), squared
    error on y scaled by its training sd, AdamW, stopped early on the hold-out `hold` (patience p["patience"] epochs, at
    most p["max_epochs"]), keeping the best epoch's weights: the lowest hold-out squared error, or the highest
    `score(hold-out predictions)` when given (e.g. the hold-out's rank IC). The inputs are standardized by channel
    with the mean and sd of a sample of training images. After each epoch, `on_epoch` gets the epoch, the training
    loss, the hold-out loss and the hold-out predictions. Returns the network, the target's scale and what the fit
    chose."""
    dev = images.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(np.std(y[train])) or 1.0
    yt = torch.as_tensor(np.nan_to_num(y / scale).astype(np.float32), device=dev)
    sample = torch.as_tensor(np.sort(rng.choice(train, min(p.get("norm_sample", 20_000), len(train)), replace=False)),
                             device=dev)
    with torch.no_grad():
        step = images.predict_batch
        x = torch.cat([images.draw(sample[i:i + step]) for i in range(0, len(sample), step)])
        mean, sd = x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3)).clamp_min(1e-6)
    net = ChartCNN(tuple(p["filters"]), p["dropout"], mean, sd, images.height, images.width).to(dev)
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
        if score is not None:
            v = -float(score(pred * scale))  # the best epoch has the lowest -score
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= int(p["patience"]):
                break
    net.load_state_dict(best_state)
    return net, scale, {"epochs": best_epoch, "holdout_best": -best if score is not None else best}


# the multi-day image (explanations/CNN/multiday_design.md): 21 trading days of hourly bars and the decision day's night
HOUR_SLOTS = 7  # 09:30, 10:30, ..., 14:30, then 15:30 to the cutoff (on early closes: 09:30 to 11:30, then 12:30)
LAST_FULL, NIGHT_SLOT = 7, 8  # stored slots: the day's last bar with its last ten minutes (drawn for the past days);
DAY_SLOTS = 9  # the night before the day (drawn for the decision day only)
WINDOW_DAYS, MA_HOURS, MA_MIN_HOURS = 21, 35, 15  # the moving average: the 35 hourly closes up to each bar, at least 15
CONTEXT_DAYS = WINDOW_DAYS + -(-MA_HOURS // HOUR_SLOTS)  # the days gathered: the window and the moving average's start
W_PRICE_ROWS, W_VOLUME_ROWS = 76, 19
W_HEIGHT = W_PRICE_ROWS + 1 + W_VOLUME_ROWS  # 96
W_BARS = WINDOW_DAYS * HOUR_SLOTS + 1  # 148: 20 past days, the decision day's night, the decision day
W_WIDTH = 3 * W_BARS  # 444
VOLUME_QUANTILE = 0.95  # the volume scale: this quantile of the window's bars (taller bars are capped)


def last_hour_slot(n_session_bars: int) -> int:
    """The hour slot of a day's last session bar: 6 on a normal day (15:30-15:50), 3 on an early close (12:30-12:50)."""
    return (n_session_bars - 1) // 6


def hour_table(session: pd.DataFrame, ten: pd.DataFrame | None, night: pd.DataFrame, start: pd.Series,
               n_session_bars: int) -> pd.DataFrame:
    """One day's stored slots by (permno, slot), from its ten-minute bars: the counts summed (FLOW_COLUMNS) and the
    first open, highest high, lowest low and last close. session: the day's session rows (permno, bar, FLOW_COLUMNS,
    open, high, low, close: round-lot prices, NaN without round lots); ten: its 15:50-16:00, the next decision day's
    prev_last10 rows (same columns; None when unknown). Slots 0-6: the hours, the last ending at the cutoff
    (`hour_slot`); LAST_FULL: the day's last bar with its last ten minutes; NIGHT_SLOT: the night before the day (night:
    the day's prev_evening and pre_open bars rows, permno, piece, close and FLOW_COLUMNS), opening at `start` (each
    stock's last round-lot price of the day before, in the day's terms) and closing at the last trade before 09:30,
    its high and low the larger and smaller of the two."""
    last = last_hour_slot(n_session_bars)
    hours = session.assign(slot=np.minimum(session["bar"].to_numpy() // 6, HOUR_SLOTS - 1), order=session["bar"])
    full = [hours[hours["slot"] == last].assign(slot=LAST_FULL)]
    if ten is not None and len(ten):
        full.append(ten.assign(slot=LAST_FULL, order=99))
    rows = pd.concat([hours, *full], ignore_index=True).sort_values(["permno", "slot", "order"])
    rows[PRICES] = rows[PRICES].astype(float)
    g = rows.groupby(["permno", "slot"])
    out = g[FLOW_COLUMNS].sum().astype(float)
    out["open"], out["high"], out["low"], out["close"] = (g["open"].first(), g["high"].max(), g["low"].min(),
                                                          g["close"].last())
    piece = np.asarray(night["piece"], dtype=object)
    late = {p: night.loc[piece == p].set_index("permno")["close"].astype(float) for p in NIGHT_PIECES}
    stocks = pd.Index(sorted(set(late["prev_evening"].index) | set(late["pre_open"].index)), name="permno")
    nb = pd.DataFrame({"open": start.reindex(stocks).astype(float),
                       "close": late["pre_open"].reindex(stocks).fillna(late["prev_evening"].reindex(stocks))})
    nb["high"], nb["low"] = nb[["open", "close"]].max(axis=1), nb[["open", "close"]].min(axis=1)
    nb = nb.join(night.groupby("permno")[FLOW_COLUMNS].sum().astype(float))
    nb.index = pd.MultiIndex.from_arrays([stocks, np.full(len(stocks), NIGHT_SLOT)], names=["permno", "slot"])
    return pd.concat([out, nb[out.columns]]).sort_index()


def hour_slots_of(n_session_bars: int) -> list[int]:
    """The stored slots a day has: its hours (0 to its last), LAST_FULL and NIGHT_SLOT."""
    return [*range(last_hour_slot(n_session_bars) + 1), LAST_FULL, NIGHT_SLOT]


def hour_retail_trades(table: pd.DataFrame) -> pd.Series:
    """Signed retail trades (midpoint rule) of each stored slot of a day's hour_table."""
    return retail(table["mid_buy_trades"], table["mid_sell_trades"])[0]


def hour_pixels(table: pd.DataFrame, history: RetailHistory, reference) -> pd.DataFrame:
    """The six color pixels of each stored slot that traded (table: a day's hour_table), abnormal retail activity
    against the history of the same slot, standardized by slot with the percentiles, mean and sd of the `reference`
    stocks (that day's large caps, a set of PERMNOs) applied to every stock."""
    present = table[table["shares"] > 0]
    f = flow(present)
    mean, days = history.mean(present.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        activity = np.log((1 + f["retail_trades"].to_numpy()) / (1 + mean))
    f["abnormal_activity"] = np.where(days >= HISTORY_MIN, activity, np.nan)
    return standardize_by_slot(f, COLORS, present.index.get_level_values("permno").isin(list(reference)))


def hour_arrays(table: pd.DataFrame, pixels: pd.DataFrame, stocks) -> dict[str, np.ndarray]:
    """Each stock's stored slots, in the order of `stocks`: `prices` (n, DAY_SLOTS, 4: open, high, low, close; NaN
    without a price), `shares` (n, DAY_SLOTS) and `colors` (n, DAY_SLOTS, 6; 128 when missing)."""
    idx = pd.MultiIndex.from_product([pd.Index(stocks, name="permno"), range(DAY_SLOTS)], names=["permno", "slot"])
    t, n = table.reindex(idx), len(stocks)
    c = pixels.reindex(idx)[[f"{k}_px" for k in COLORS]].to_numpy(float).reshape(n, DAY_SLOTS, len(COLORS))
    return {"prices": t[PRICES].to_numpy(np.float32).reshape(n, DAY_SLOTS, 4),
            "shares": np.nan_to_num(t["shares"].to_numpy(np.float32)).reshape(n, DAY_SLOTS),
            "colors": np.where(np.isnan(c), 128, c).astype(np.uint8)}


def moving_average(closes: torch.Tensor, hours: int = MA_HOURS, at_least: int = MA_MIN_HOURS) -> torch.Tensor:
    """The mean of the last `hours` hourly closes up to each bar (B x T, NaN where a bar has no close), skipping
    missing closes; NaN with fewer than `at_least` of them."""
    ok = ~torch.isnan(closes)
    pad = lambda x: torch.nn.functional.pad(torch.cumsum(x, dim=1), (1, 0))
    total, count = pad(torch.where(ok, closes, torch.zeros_like(closes))), pad(ok.float())
    t = closes.shape[1]
    lo = torch.clamp(torch.arange(1, t + 1, device=closes.device) - hours, min=0)
    hi = torch.arange(1, t + 1, device=closes.device)
    window_sum, window_count = total[:, hi] - total[:, lo], count[:, hi] - count[:, lo]
    return torch.where(window_count >= at_least, window_sum / window_count.clamp_min(1), torch.full_like(closes, float("nan")))


def render_window(prices: torch.Tensor, shares: torch.Tensor, colors: torch.Tensor, early: torch.Tensor) -> torch.Tensor:
    """Draw a batch of multi-day images (float, B x 4 x W_HEIGHT x W_WIDTH, 0-255) from CONTEXT_DAYS days of stored
    slots, oldest first (prices B x D x DAY_SLOTS x 4 with NaN, shares B x D x DAY_SLOTS, colors B x D x DAY_SLOTS x 6,
    early B x D: the day was an early close). The window is the last WINDOW_DAYS days' hourly bars, each past day's
    last bar with its last ten minutes, the decision day's night before its first bar; the moving average runs over the
    hourly closes of every gathered day. Prices are scaled to the window's highest high or average and its lowest, the
    night left out (it is clipped at the image's edge: its close is any trade, and a stray print would squeeze the window);
    volume to the VOLUME_QUANTILE of its bars, taller bars capped. Grey: the bars (ticks for open and close, the range
    in the middle column), the moving average as a 3-pixel segment per bar, the volume bars; red, green, blue: each
    bar's price colors on its price pixels and its volume colors on its volume pixels."""
    dev, b, d = prices.device, prices.shape[0], prices.shape[1]
    base = torch.arange(HOUR_SLOTS, device=dev).view(1, 1, HOUR_SLOTS).expand(b, d, HOUR_SLOTS)
    last = torch.where(early, 3, 6).view(b, d, 1)
    past = (torch.arange(d, device=dev) < d - 1).view(1, d, 1)
    slot = torch.where(past & (base == last), LAST_FULL, base)  # past days: the last bar with its last ten minutes
    hourly_p = torch.gather(prices, 2, slot[..., None].expand(b, d, HOUR_SLOTS, 4))
    hourly_v = torch.gather(shares, 2, slot)
    hourly_c = torch.gather(colors, 2, slot[..., None].expand(b, d, HOUR_SLOTS, colors.shape[-1]))
    ma = moving_average(hourly_p[..., 3].reshape(b, d * HOUR_SLOTS))[:, (d - WINDOW_DAYS) * HOUR_SLOTS:]
    first = d - WINDOW_DAYS
    seq = lambda h, night: torch.cat([h[:, first:d - 1].flatten(1, 2), night[:, None], h[:, d - 1]], dim=1)
    p = seq(hourly_p, prices[:, d - 1, NIGHT_SLOT])  # B x W_BARS x 4
    v = seq(hourly_v, shares[:, d - 1, NIGHT_SLOT])
    c = seq(hourly_c, colors[:, d - 1, NIGHT_SLOT])
    n_past = (WINDOW_DAYS - 1) * HOUR_SLOTS
    ma = torch.cat([ma[:, :n_past], torch.full((b, 1), float("nan"), device=dev), ma[:, n_past:]], dim=1)
    nan = torch.isnan
    hourly = torch.ones(W_BARS, dtype=torch.bool, device=dev)
    hourly[n_past] = False  # the night is left out of the scale: its close is any trade, and a stray print would
    hi, lo = p[:, hourly, 1], p[:, hourly, 2]  # squeeze the window; it is clipped at the image's edge instead
    top = torch.maximum(torch.where(nan(hi), -torch.inf, hi).amax(1), torch.where(nan(ma), -torch.inf, ma).amax(1))
    bottom = torch.minimum(torch.where(nan(lo), torch.inf, lo).amin(1), torch.where(nan(ma), torch.inf, ma).amin(1))
    span = torch.where(top > bottom, top - bottom, torch.ones_like(top))

    def row(x):  # a price's row, 0 at the top; NaN draws nothing
        shape = (b,) + (1,) * (x.dim() - 1)
        return torch.round((top.view(shape) - x) / span.view(shape) * (W_PRICE_ROWS - 1))

    r = row(p).clamp(0, W_PRICE_ROWS - 1)  # B x W_BARS x 4; only the night can fall outside
    pr = torch.arange(W_PRICE_ROWS, device=dev).view(1, W_PRICE_ROWS, 1).float()
    left, right = pr == r[:, None, :, 0], pr == r[:, None, :, 3]
    middle = (pr >= r[:, None, :, 1]) & (pr <= r[:, None, :, 2])
    price = torch.stack([left, middle, right], dim=-1).reshape(b, W_PRICE_ROWS, W_WIDTH)
    line = (pr == row(ma)[:, None, :]).repeat_interleave(3, dim=2)  # B x W_PRICE_ROWS x W_WIDTH
    vv = torch.where(v > 0, v, torch.full_like(v, float("nan")))
    scale = torch.nanquantile(vv, VOLUME_QUANTILE, dim=1)
    filled = torch.where(v > 0, torch.clamp(torch.round(v / scale[:, None] * W_VOLUME_ROWS), 1, W_VOLUME_ROWS),
                         torch.zeros_like(v))
    filled = torch.nan_to_num(filled)
    vr = torch.arange(W_VOLUME_ROWS, device=dev).view(1, W_VOLUME_ROWS, 1)
    bar = vr >= (W_VOLUME_ROWS - filled)[:, None, :]
    blank = torch.zeros_like(bar)
    vol = torch.stack([blank, bar, blank], dim=-1).reshape(b, W_VOLUME_ROWS, W_WIDTH)
    gap = torch.zeros(b, 1, W_WIDTH, dtype=torch.bool, device=dev)
    chart = torch.cat([price, gap, vol], dim=1)  # the bars: colored
    grey = chart | torch.cat([line, gap, torch.zeros_like(vol)], dim=1)  # with the moving average: grey only
    cc = c.float().repeat_interleave(3, dim=1)  # B x W_WIDTH x 6
    in_price = (torch.arange(W_HEIGHT, device=dev) < W_PRICE_ROWS).view(1, W_HEIGHT, 1, 1)
    per_row = torch.where(in_price, cc[:, None, :, :3], cc[:, None, :, 3:])
    color = (per_row * chart[..., None]).permute(0, 3, 1, 2)
    return torch.cat([grey[:, None].float() * 255, color], dim=1)


class HourWindows:
    """The multi-day images' data on one device: every stock-day's stored slots (`prices`, `shares`, `colors` as in
    hour_arrays, stacked), the row of each (stock, day) (`row_of`, stocks x days, -1 for none), each day's early-close
    flag, and each target stock-day's stock and day (positions in `row_of`). draw(idx) gathers each target's
    CONTEXT_DAYS days up to its day and renders them."""

    height, width, predict_batch = W_HEIGHT, W_WIDTH, 512  # a 96 x 444 image's activations: about 20 MB

    def __init__(self, prices, shares, colors, row_of, early, stock, day, device):
        t = lambda a: torch.as_tensor(np.asarray(a), device=device)
        self.prices, self.shares, self.colors = t(prices), t(shares), t(colors)
        self.row_of, self.early, self.stock, self.day = t(row_of), t(early), t(stock), t(day)
        self.device = torch.device(device)

    def draw(self, idx: torch.Tensor) -> torch.Tensor:
        s, d = self.stock[idx], self.day[idx]
        days = d[:, None] - torch.arange(CONTEXT_DAYS - 1, -1, -1, device=self.device)  # oldest first
        rows = torch.where(days >= 0, self.row_of[s[:, None], days.clamp_min(0)], torch.full_like(days, -1))
        has = (rows >= 0)[..., None]
        r = rows.clamp_min(0)
        prices = torch.where(has[..., None], self.prices[r], torch.full_like(self.prices[r], float("nan")))
        shares = torch.where(has, self.shares[r], torch.zeros_like(self.shares[r]))
        colors = torch.where(has[..., None], self.colors[r], torch.full_like(self.colors[r], 128))
        return render_window(prices, shares, colors, self.early[days.clamp_min(0)])
