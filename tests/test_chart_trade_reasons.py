from dashboard.chart_data import (
    attach_trade_reasons,
    display_reason,
    normalize_reason,
    reason_matches_action,
)


def test_close_uses_round_trip_reason():
    orders = [{"time": 100, "action": "close_long"}]
    attach_trade_reasons(orders, [], [(100, "close_tp")])
    assert orders[0]["reason_code"] == "close_tp"
    assert orders[0]["reason"] == "take profit"


def test_open_uses_snapshot_action():
    orders = [{"time": 200, "action": "open_long"}]
    attach_trade_reasons(orders, [(195, "rebound_open")], [])
    assert orders[0]["reason_code"] == "rebound_open"
    assert "legacy" in orders[0]["reason"]


def test_close_prefers_round_trip_over_snapshot():
    orders = [{"time": 300, "action": "close_short"}]
    attach_trade_reasons(
        orders,
        [(299, "open_short")],
        [(300, "activity_ping")],
    )
    assert orders[0]["reason_code"] == "activity_ping"
    assert orders[0]["reason"] == "activity ping (done)"


def test_open_ignores_close_time_snapshot():
    orders = [{"time": 67232, "action": "open_short"}]
    attach_trade_reasons(
        orders,
        [(67230, "time"), (67231, "fade_short")],
        [],
    )
    assert orders[0]["reason_code"] == "open_short"
    assert orders[0]["reason"] == "much above average"


def test_legacy_time_normalized():
    assert normalize_reason("time") == "close_time"
    assert display_reason("time") == "held too long"


def test_open_short_rejects_open_long_reason():
    orders = [{"time": 700, "action": "open_short"}]
    attach_trade_reasons(orders, [(699, "open_long")], [])
    assert orders[0]["reason_code"] == "open_short"
    assert orders[0]["reason"] == "much above average"


def test_display_reason_open_long():
    assert display_reason("open_long") == "much cheap vs average"
