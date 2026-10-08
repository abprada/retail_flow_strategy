import pandas as pd

from dsr.taq import check_day


def _day():
    """One symbol: in the regular session's first bar, two off-exchange BJZZ buys (z code 9) at a one-cent spread,
    signed buys by the midpoint method too (q code 12), and one exchange trade at the ask."""
    cells = pd.DataFrame([
        {"symbol": "X", "bucket": 1, "venue": 1, "z": 9, "q": 12, "spread": 1, "size": 1, "trades": 2, "shares": 60, "dollars": 600.48},
        {"symbol": "X", "bucket": 1, "venue": 0, "z": 0, "q": 14, "spread": 1, "size": 2, "trades": 1, "shares": 100, "dollars": 1001.0},
    ])
    zero = {f"{r}_{v}": 0 for r in ["bjzz_sell", "mid_sell"] for v in ["trades", "shares", "dollars"]}
    bars = pd.DataFrame([{
        "symbol": "X", "bucket": 1, "bar": 0, "trades": 3, "shares": 160, "dollars": 1601.48,
        "bjzz_buy_trades": 2, "bjzz_buy_shares": 60, "bjzz_buy_dollars": 600.48,
        "mid_buy_trades": 2, "mid_buy_shares": 60, "mid_buy_dollars": 600.48, **zero,
        "off_trades": 2, "off_shares": 60, "off_dollars": 600.48,
    }])
    daily = pd.DataFrame({"symbol": ["X"], "spread_bp": [9.99]})
    audit = pd.DataFrame([
        {"section": "step", "item": "raw", "venue": 0, "kept": True, "trades": 2, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "raw", "venue": 1, "kept": True, "trades": 3, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "sale_condition", "venue": 0, "kept": False, "trades": 1, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "records", "venue": 1, "kept": False, "trades": 1, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "kept", "venue": 0, "kept": True, "trades": 1, "shares": 0, "dollars": 0.0},
        {"section": "step", "item": "kept", "venue": 1, "kept": True, "trades": 2, "shares": 0, "dollars": 0.0},
    ])
    return daily, cells, bars, audit


def test_check_day_passes_on_consistent_tables():
    checks = check_day(*_day(), regular_bars=38)
    assert checks["passed"].all(), checks[~checks["passed"]]


def test_check_day_catches_a_retail_column_that_does_not_match_the_cells():
    daily, cells, bars, audit = _day()
    bars.loc[0, "bjzz_buy_trades"] = 3
    checks = check_day(daily, cells, bars, audit, regular_bars=38).set_index("check")["passed"]
    assert not checks["bars' bjzz_buy columns equal the matching cells"]
    assert checks.drop("bars' bjzz_buy columns equal the matching cells").all()


def test_check_day_catches_dollars_off_by_more_than_a_cent_and_bars_out_of_range():
    daily, cells, bars, audit = _day()
    bars.loc[0, "dollars"] += 0.02
    bars.loc[0, "bar"] = 20
    checks = check_day(daily, cells, bars, audit, regular_bars=20).set_index("check")["passed"]
    assert not checks["cells and bars have the same trades, shares and dollars per symbol and bucket"]
    assert not checks["regular bars are numbered 0 to 19, other buckets 0"]


def test_check_day_catches_audit_steps_that_do_not_add_up():
    daily, cells, bars, audit = _day()
    audit.loc[audit["item"].eq("records"), "trades"] = 2
    checks = check_day(daily, cells, bars, audit, regular_bars=38).set_index("check")["passed"]
    assert not checks["audit: raw minus the removed steps equals kept"]
    assert checks["audit: kept equals the trades in cells"]
