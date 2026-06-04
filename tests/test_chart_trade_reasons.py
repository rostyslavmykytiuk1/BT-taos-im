from dashboard.chart_data import attach_trade_reasons, normalize_reason, reason_matches_action


def test_close_uses_round_trip_reason():
    orders = [{"time": 100, "action": "close_long"}]
    attach_trade_reasons(orders, [], [(100, "close_tp")])
    assert orders[0]["reason"] == "close_tp"


def test_open_uses_snapshot_action():
    orders = [{"time": 200, "action": "open_long"}]
    attach_trade_reasons(orders, [(195, "rebound_open")], [])
    assert orders[0]["reason"] == "rebound_open"


def test_close_prefers_round_trip_over_snapshot():
    orders = [{"time": 300, "action": "close_short"}]
    attach_trade_reasons(
        orders,
        [(299, "open_short")],
        [(300, "activity_ping")],
    )
    assert orders[0]["reason"] == "activity_ping"


def test_open_ignores_close_time_snapshot():
    orders = [{"time": 67232, "action": "open_short"}]
    attach_trade_reasons(
        orders,
        [(67230, "time"), (67231, "fade_short")],
        [],
    )
    assert orders[0]["reason"] == "open_short"


def test_legacy_time_normalized():
    assert normalize_reason("time") == "close_time"
    assert normalize_reason("fade_long") == "open_long"


def test_close_legacy_snapshot():
    orders = [{"time": 400, "action": "close_short"}]
    attach_trade_reasons(orders, [(399, "fade_long_recover")], [])
    assert orders[0]["reason"] == "fade_long_recover"


def test_open_fallback_to_action():
    orders = [{"time": 500, "action": "open_short"}]
    attach_trade_reasons(orders, [(499, "time")], [])
    assert orders[0]["reason"] == "open_short"


def test_open_short_rejects_ping_open_reason():
    orders = [{"time": 600, "action": "open_short"}]
    attach_trade_reasons(orders, [(599, "ping_open")], [])
    assert orders[0]["reason"] == "open_short"


def test_open_short_rejects_open_long_reason():
    orders = [{"time": 700, "action": "open_short"}]
    attach_trade_reasons(orders, [(699, "open_long")], [])
    assert orders[0]["reason"] == "open_short"


def test_reason_matches_action():
    assert not reason_matches_action("ping_open", "open_short")
    assert not reason_matches_action("open_long", "open_short")
    assert reason_matches_action("ping_open", "close_short")
