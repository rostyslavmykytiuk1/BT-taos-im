# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Data loading and transforms for the miner telemetry dashboard."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taos.im.telemetry.paths import data_root, slug_validator, telemetry_root
from taos.im.utils import duration_from_timestamp, timestamp_from_duration

_EPS = 1e-9

# Match MeanReversionAgent rolling fair-value window (trade-print average).
CHART_AVERAGE_WINDOW_S = 300.0
CHART_AVERAGE_MIN_SAMPLES = 8

# Snapshot actions that do not describe a taker fill intent.
_SNAPSHOT_SKIP_ACTIONS = frozenset({
    "manage", "flat", "hold", "warmup", "pause", "falling", "cap",
    "rebound_hold", "ping_hold", "rebound_wait",
})

_CLOSE_ACTIONS = frozenset({"close_long", "close_short"})

# Snapshot actions that may justify an open fill (current + legacy agent builds).
_OPEN_SNAPSHOT_ACTIONS = frozenset({
    "open_long", "open_short", "rebound_open", "rebound_add", "ping_open",
    "fade_long", "fade_short",
})

# Snapshot actions that may justify a close fill when no round_trip row exists.
_CLOSE_SNAPSHOT_ACTIONS = frozenset({
    "close_tp", "close_sl", "close_time", "time",
    "rebound_stop", "rebound_time", "activity_ping",
    "fade_long_recover", "recover", "sl", "tp",
})

# Normalize legacy telemetry labels for display.
_REASON_LABELS: dict[str, str] = {
    "time": "close_time",
    "sl": "close_sl",
    "tp": "close_tp",
    "fade_long": "open_long",
    "fade_short": "open_short",
    # Legacy stretch/recover modes — not the same as current rebound_open.
    "fade_long_recover": "fade_long_recover",
    "recover": "recover",
}


def normalize_reason(reason: str) -> str:
    if not reason:
        return ""
    if reason.startswith("rebound_exit_"):
        return reason
    return _REASON_LABELS.get(reason, reason)


# Plain-language labels for the dashboard "reason" / "why" column.
REASON_DISPLAY: dict[str, str] = {
    "open_long": "much cheap vs average",
    "open_short": "much above average",
    "close_tp": "take profit",
    "close_sl": "stop loss",
    "close_time": "held too long",
    "ping_open": "activity ping (start)",
    "activity_ping": "activity ping (done)",
    "ping_hold": "activity ping (waiting)",
    "ping_followup_wait": "activity ping (waiting leg 2)",
    "rebound_open": "after sharp drop (legacy)",
    "rebound_stop": "rebound stop (legacy)",
    "rebound_time": "rebound max hold (legacy)",
    "rebound_add": "rebound add size (legacy)",
    "fade_long_recover": "legacy recover mode",
    "recover": "legacy recover mode",
    "fill": "fill",
    "pause": "paused after stop",
    "cap": "volume cap",
    "falling": "price still falling",
    "flat": "no trade",
    "warmup": "warming up",
    "manage": "holding",
}


def display_reason(code: str) -> str:
    """Map agent reason code to a short human-readable label."""
    code = normalize_reason((code or "").strip())
    if not code:
        return ""
    if code in REASON_DISPLAY:
        return REASON_DISPLAY[code]
    if code.startswith("rebound_exit_"):
        return "rebound scale-out (legacy)"
    return code.replace("_", " ")


# Reasons that must not be glued onto a fill with this action (from trades.csv).
_REASON_INCOMPATIBLE: dict[str, frozenset[str]] = {
    "open_long": frozenset({
        "open_short", "fade_short", "activity_ping", "close_tp", "close_sl", "close_time",
    }),
    "open_short": frozenset({
        "ping_open", "open_long", "fade_long", "rebound_open", "close_tp", "close_sl", "close_time",
        "activity_ping",
    }),
    "close_long": frozenset({
        "ping_open", "open_long", "open_short", "fade_long", "fade_short", "rebound_open",
    }),
    "close_short": frozenset({
        "open_long", "open_short", "fade_long", "fade_short", "rebound_open",
    }),
}


def reason_matches_action(reason: str, action: str) -> bool:
    """True when a telemetry reason can describe this fill action."""
    if not reason or not action:
        return not reason
    reason = normalize_reason(reason)
    bad = _REASON_INCOMPATIBLE.get(action)
    if bad and reason in bad:
        return False
    if action == "open_short" and reason.startswith("open_") and reason != "open_short":
        return False
    if action == "open_long" and reason == "ping_open":
        return True
    if action == "close_short" and reason == "ping_open":
        return True  # ping starts with a buy that covers part of a short
    if action.startswith("close_") and reason.startswith("open_"):
        return False
    if action.startswith("open_") and reason.startswith("close_"):
        return False
    return True


def fnum(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def is_buy_side(side_raw: Any) -> bool:
    return str(side_raw) in ("0", "BUY", "buy")


def sim_seconds_from_duration(duration: str) -> int:
    return timestamp_from_duration(duration.strip()) // 1_000_000_000


def sim_seconds_from_ts_ns(ts_ns: int) -> int:
    return int(ts_ns) // 1_000_000_000


def sim_time_label(sim_sec: int) -> str:
    if sim_sec <= 0:
        return ""
    h, rem = divmod(int(sim_sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def trade_sort_key(row: dict[str, Any]) -> tuple:
    ts = row.get("timestamp") or ""
    try:
        trade_id = int(row.get("tradeId") or 0)
    except (TypeError, ValueError):
        trade_id = 0
    return (ts, trade_id)


def find_trades_csv(uid: int, validator_slug: str, sim_id: str) -> Path | None:
    root = data_root() / str(uid)
    if not root.is_dir():
        return None
    for val_dir in root.iterdir():
        if not val_dir.is_dir():
            continue
        if val_dir.name != validator_slug and slug_validator(val_dir.name) != validator_slug:
            continue
        candidate = val_dir / sim_id / "trades.csv"
        if candidate.is_file():
            return candidate
    for val_dir in root.iterdir():
        candidate = val_dir / sim_id / "trades.csv"
        if candidate.is_file():
            return candidate
    return None


def telemetry_db(uid: int, validator_slug: str, sim_id: str) -> Path:
    return telemetry_root() / str(uid) / validator_slug / sim_id / "telemetry.sqlite"


def quote_volume_from_trades_csv(path: Path, uid: int, book: int) -> float:
    uid_s = str(uid)
    total = 0.0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("bookId", -1)) != book:
                continue
            if str(row.get("takerAgentId", "")) != uid_s and str(row.get("makerAgentId", "")) != uid_s:
                continue
            total += fnum(row.get("quantity")) * fnum(row.get("price"))
    return total


def _set_bucket(buckets: dict[int, float], bucket: int, value: float) -> None:
    buckets[bucket] = value


def _forward_fill(buckets: dict[int, float]) -> dict[int, float]:
    if not buckets:
        return buckets
    keys = sorted(buckets)
    out: dict[int, float] = {}
    last = buckets[keys[0]]
    for k in range(keys[0], keys[-1] + 1):
        if k in buckets:
            last = buckets[k]
        out[k] = last
    return out


def _last_print_buckets(path: Path, book: int, resolution: int) -> dict[int, float]:
    buckets: dict[int, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("bookId", -1)) != book:
                continue
            price = fnum(row.get("price"))
            if price <= 0:
                continue
            try:
                bucket = sim_seconds_from_duration(row.get("timestamp", "")) // resolution
            except ValueError:
                continue
            _set_bucket(buckets, bucket, price)
    return buckets


def _mid_buckets_from_db(conn: sqlite3.Connection, book: int, resolution: int) -> dict[int, float]:
    buckets: dict[int, float] = {}
    rows = conn.execute(
        """
        SELECT ts_ns, mid FROM snapshots
        WHERE book_id = ? AND mid IS NOT NULL
        ORDER BY ts_ns ASC
        """,
        (book,),
    ).fetchall()
    for row in rows:
        bucket = sim_seconds_from_ts_ns(int(row["ts_ns"])) // resolution
        _set_bucket(buckets, bucket, float(row["mid"]))
    return buckets


def _collect_mid_buckets(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    resolution: int,
    connect_db,
) -> dict[int, float]:
    buckets: dict[int, float] = {}
    trades_path = find_trades_csv(uid, validator_slug, sim_id)
    if trades_path:
        buckets.update(_last_print_buckets(trades_path, book, resolution))
    db_path = telemetry_db(uid, validator_slug, sim_id)
    if db_path.is_file():
        conn = connect_db(db_path)
        try:
            buckets.update(_mid_buckets_from_db(conn, book, resolution))
        finally:
            conn.close()
    return buckets


def _rolling_average_of_mid(
    ordered: list[tuple[int, float]],
    resolution: int,
    *,
    window_s: float = CHART_AVERAGE_WINDOW_S,
    min_samples: int = CHART_AVERAGE_MIN_SAMPLES,
) -> list[dict[str, Any]]:
    """Rolling mean of the chart mid line (bid+ask)/2 over the last ``window_s``."""
    if not ordered:
        return []
    window_pts = max(1, int(window_s) // max(1, resolution))
    out: list[dict[str, Any]] = []
    last_avg: float | None = None
    for i, (bucket, price) in enumerate(ordered):
        start = max(0, i - window_pts + 1)
        window = [p for _, p in ordered[start : i + 1]]
        if len(window) >= min_samples:
            last_avg = sum(window) / len(window)
            out.append({"time": int(bucket) * resolution, "value": round(last_avg, 6)})
        elif last_avg is not None:
            out.append({"time": int(bucket) * resolution, "value": round(last_avg, 6)})
    return out


def chart_time_window(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    resolution: int,
    limit: int,
    connect_db,
) -> tuple[int, int, list[tuple[int, float]]]:
    """Last `limit` buckets (forward-filled) — shared window for mid line and markers."""
    buckets = _collect_mid_buckets(uid, validator_slug, sim_id, book, resolution, connect_db)
    if not buckets:
        return 0, 0, []
    ordered = sorted(_forward_fill(buckets).items())[-limit:]
    if not ordered:
        return 0, 0, []
    t_min = ordered[0][0] * resolution
    t_max = ordered[-1][0] * resolution
    return t_min, t_max, ordered


def _signal_series_in_window(
    conn,
    book: int,
    resolution: int,
    ordered: list[tuple[int, float]],
    column: str,
) -> list[dict[str, Any]]:
    """Forward-fill a snapshot signal column onto the chart bucket grid."""
    if not ordered:
        return []
    try:
        rows = conn.execute(
            f"""
            SELECT ts_ns, {column} FROM snapshots
            WHERE book_id = ? AND {column} IS NOT NULL
            ORDER BY ts_ns ASC
            """,
            (book,),
        ).fetchall()
    except Exception:
        return []
    by_bucket: dict[int, float] = {}
    for row in rows:
        bucket = sim_seconds_from_ts_ns(int(row["ts_ns"])) // resolution
        by_bucket[bucket] = float(row[column])
    out: list[dict[str, Any]] = []
    last: float | None = None
    for bucket, _ in ordered:
        if bucket in by_bucket:
            last = by_bucket[bucket]
        if last is not None:
            out.append({"time": int(bucket) * resolution, "value": round(last, 6)})
    return out


def build_mid_series(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    resolution: int,
    limit: int,
    connect_db,
) -> dict[str, Any]:
    """Mid line, 5m rolling average, and optional Kalman level / slope from telemetry."""
    t_min, t_max, ordered = chart_time_window(
        uid, validator_slug, sim_id, book, resolution, limit, connect_db
    )
    if not ordered:
        return {
            "mid": [],
            "average": [],
            "kalman_level": [],
            "slope_bps": [],
            "origin": 0,
            "range": [0, 0],
        }

    average = _rolling_average_of_mid(ordered, resolution)
    kalman_level: list[dict[str, Any]] = []
    slope_bps: list[dict[str, Any]] = []
    db_path = telemetry_db(uid, validator_slug, sim_id)
    if db_path.is_file():
        conn = connect_db(db_path)
        try:
            kalman_level = _signal_series_in_window(conn, book, resolution, ordered, "signal_level")
            slope_bps = _signal_series_in_window(conn, book, resolution, ordered, "signal_trend_bps")
        finally:
            conn.close()

    return {
        "mid": [{"time": b * resolution, "value": p} for b, p in ordered],
        "average": average,
        "kalman_level": kalman_level,
        "slope_bps": slope_bps,
        "origin": t_min,
        "range": [t_min, t_max],
    }


@dataclass
class TakerOrder:
    order_id: str
    timestamp: str
    time: int | None
    time_label: str
    book_id: str
    action: str
    side: str
    price: float
    quantity: float
    fills: int
    pos_before: float
    pos_after: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "orderId": self.order_id,
            "timestamp": self.timestamp,
            "time": self.time,
            "time_label": self.time_label,
            "bookId": self.book_id,
            "action": self.action,
            "side": self.side,
            "price": round(self.price, 6),
            "quantity": round(self.quantity, 6),
            "fills": self.fills,
            "pos_before": round(self.pos_before, 4),
            "pos_after": round(self.pos_after, 4),
            "reason": self.reason,
        }


class PositionTracker:
    """Track signed BASE position while classifying taker fills."""

    def __init__(self) -> None:
        self.qty = 0.0

    def _record(
        self,
        out: list[TakerOrder],
        *,
        meta: dict[str, Any],
        is_buy: bool,
        qty: float,
        vwap: float,
        n_fills: int,
        action: str,
        pos_before: float,
        pos_after: float,
    ) -> None:
        try:
            t = sim_seconds_from_duration(meta.get("timestamp", ""))
        except ValueError:
            t = None
        out.append(
            TakerOrder(
                order_id=meta["order_id"],
                timestamp=meta.get("timestamp", ""),
                time=t,
                time_label=sim_time_label(t) if t is not None else "",
                book_id=meta.get("bookId", ""),
                action=action,
                side="buy" if is_buy else "sell",
                price=vwap,
                quantity=qty,
                fills=n_fills,
                pos_before=pos_before,
                pos_after=pos_after,
            )
        )

    def apply_taker(
        self,
        out: list[TakerOrder],
        *,
        meta: dict[str, Any],
        is_buy: bool,
        qty: float,
        vwap: float,
        n_fills: int,
    ) -> None:
        pos_before = self.qty
        remaining = qty

        if is_buy:
            if self.qty < -_EPS:
                close_qty = min(remaining, abs(self.qty))
                if close_qty > _EPS:
                    after = pos_before + close_qty
                    self._record(
                        out,
                        meta=meta,
                        is_buy=True,
                        qty=close_qty,
                        vwap=vwap,
                        n_fills=n_fills,
                        action="close_short",
                        pos_before=pos_before,
                        pos_after=after,
                    )
                    self.qty = after
                    pos_before = self.qty
                    remaining -= close_qty
                if remaining > _EPS:
                    after = pos_before + remaining
                    self._record(
                        out,
                        meta=meta,
                        is_buy=True,
                        qty=remaining,
                        vwap=vwap,
                        n_fills=n_fills,
                        action="open_long",
                        pos_before=pos_before,
                        pos_after=after,
                    )
                    self.qty = after
            else:
                after = pos_before + remaining
                self._record(
                    out,
                    meta=meta,
                    is_buy=True,
                    qty=remaining,
                    vwap=vwap,
                    n_fills=n_fills,
                    action="open_long",
                    pos_before=pos_before,
                    pos_after=after,
                )
                self.qty = after
        else:
            if self.qty > _EPS:
                close_qty = min(remaining, self.qty)
                if close_qty > _EPS:
                    after = pos_before - close_qty
                    self._record(
                        out,
                        meta=meta,
                        is_buy=False,
                        qty=close_qty,
                        vwap=vwap,
                        n_fills=n_fills,
                        action="close_long",
                        pos_before=pos_before,
                        pos_after=after,
                    )
                    self.qty = after
                    pos_before = self.qty
                    remaining -= close_qty
                if remaining > _EPS:
                    after = pos_before - remaining
                    self._record(
                        out,
                        meta=meta,
                        is_buy=False,
                        qty=remaining,
                        vwap=vwap,
                        n_fills=n_fills,
                        action="open_short",
                        pos_before=pos_before,
                        pos_after=after,
                    )
                    self.qty = after
            else:
                after = pos_before - remaining
                self._record(
                    out,
                    meta=meta,
                    is_buy=False,
                    qty=remaining,
                    vwap=vwap,
                    n_fills=n_fills,
                    action="open_short",
                    pos_before=pos_before,
                    pos_after=after,
                )
                self.qty = after


def aggregate_taker_orders(rows: list[dict[str, Any]], uid: int) -> list[dict[str, Any]]:
    uid_s = str(uid)
    groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        if str(row.get("takerAgentId", "")) != uid_s:
            continue
        oid = (row.get("takerOrderId") or "").strip()
        if not oid:
            oid = f"{row.get('timestamp')}|{row.get('bookId')}|{row.get('side')}"
        groups.setdefault(oid, []).append(row)

    events: list[tuple[int, str, list[dict[str, Any]]]] = []
    for oid, fills in groups.items():
        fills.sort(key=trade_sort_key)
        try:
            t = sim_seconds_from_duration(fills[0].get("timestamp", ""))
        except ValueError:
            t = 0
        events.append((t, oid, fills))
    events.sort()

    tracker = PositionTracker()
    orders: list[TakerOrder] = []
    for _, oid, fills in events:
        first = fills[0]
        qty = sum(fnum(f.get("quantity")) for f in fills)
        notional = sum(fnum(f.get("quantity")) * fnum(f.get("price")) for f in fills)
        meta = {"order_id": oid, "timestamp": first.get("timestamp", ""), "bookId": first.get("bookId", "")}
        tracker.apply_taker(
            orders,
            meta=meta,
            is_buy=is_buy_side(first.get("side")),
            qty=qty,
            vwap=notional / qty if qty > 0 else 0.0,
            n_fills=len(fills),
        )
    return [o.to_dict() for o in orders]


def _load_reason_events(
    connect_db,
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """(snapshot time, action) and (round-trip close time, reason) from telemetry DB."""
    path = telemetry_db(uid, validator_slug, sim_id)
    if not path.is_file():
        return [], []
    conn = connect_db(path)
    try:
        snap_rows = conn.execute(
            """
            SELECT ts_ns, action FROM snapshots
            WHERE book_id = ? AND action IS NOT NULL AND action != ''
            ORDER BY ts_ns
            """,
            (book,),
        ).fetchall()
        rt_rows = conn.execute(
            """
            SELECT ts_close_ns, reason FROM round_trips
            WHERE book_id = ? AND reason IS NOT NULL AND reason != ''
            ORDER BY ts_close_ns
            """,
            (book,),
        ).fetchall()
    finally:
        conn.close()
    snaps = [(sim_seconds_from_ts_ns(int(r[0])), str(r[1])) for r in snap_rows]
    rts = [(sim_seconds_from_ts_ns(int(r[0])), str(r[1])) for r in rt_rows]
    return snaps, rts


def _snapshot_action_ok(action: str, *, for_open: bool) -> bool:
    if not action or action in _SNAPSHOT_SKIP_ACTIONS:
        return False
    allowed = _OPEN_SNAPSHOT_ACTIONS if for_open else _CLOSE_SNAPSHOT_ACTIONS
    if action in allowed:
        return True
    if for_open:
        return action.startswith("open_") or action in ("ping_open", "rebound_open", "rebound_add")
    return (
        action.startswith("close_")
        or action.startswith("rebound_exit_")
        or action in ("rebound_stop", "rebound_time", "activity_ping", "time")
    )


def _pick_snapshot_reason(
    snaps: list[tuple[int, str]],
    t: int,
    *,
    for_open: bool,
    max_delta: int,
) -> str:
    allowed = [e for e in snaps if _snapshot_action_ok(e[1], for_open=for_open)]
    if not allowed:
        return ""
    before = [e for e in allowed if e[0] <= t]
    if before:
        ts, action = before[-1]
        if t - ts <= max_delta:
            return action
    nearest = min(allowed, key=lambda e: abs(e[0] - t))
    if abs(nearest[0] - t) <= max_delta:
        return nearest[1]
    return ""


def attach_trade_reasons(
    orders: list[dict[str, Any]],
    snap_events: list[tuple[int, str]],
    rt_events: list[tuple[int, str]],
    *,
    max_delta: int = 60,
) -> None:
    """Fill ``reason`` on taker orders from telemetry round-trips and snapshots."""
    snaps = [(t, a) for t, a in snap_events if a]
    rts = [(t, r) for t, r in rt_events if r]
    rt_used = [False] * len(rts)

    for order in orders:
        t = order.get("time")
        if t is None:
            order["reason"] = ""
            continue

        is_close = order.get("action") in _CLOSE_ACTIONS
        reason = ""

        if is_close:
            best_i = None
            best_d = max_delta + 1
            for i, (rt_t, rt_r) in enumerate(rts):
                if rt_used[i]:
                    continue
                d = abs(rt_t - int(t))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None:
                rt_used[best_i] = True
                reason = rts[best_i][1]
            if not reason:
                reason = _pick_snapshot_reason(snaps, int(t), for_open=False, max_delta=max_delta)
        else:
            reason = _pick_snapshot_reason(snaps, int(t), for_open=True, max_delta=max_delta)

        reason = normalize_reason(reason)
        action = str(order.get("action") or "")
        if not reason_matches_action(reason, action):
            reason = action
        if not reason:
            reason = action
        order["reason_code"] = reason
        order["reason"] = display_reason(reason)


def load_trades_for_api(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    limit: int,
    connect_db,
    *,
    t_min: int | None = None,
    t_max: int | None = None,
    chart_limit: int = 5000,
    resolution: int = 1,
) -> dict[str, Any]:
    path = find_trades_csv(uid, validator_slug, sim_id)
    if path is None:
        return {"orders": [], "origin": 0, "range": [0, 0]}

    if t_min is None or t_max is None:
        t_min, t_max, _ = chart_time_window(
            uid, validator_slug, sim_id, book, resolution, chart_limit, connect_db
        )

    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if book >= 0 and int(row.get("bookId", -1)) != book:
                continue
            rows.append(dict(row))
    rows.sort(key=trade_sort_key)

    orders = [o for o in aggregate_taker_orders(rows, uid) if o.get("time") is not None]
    if t_max > t_min:
        orders = [o for o in orders if t_min <= int(o["time"]) <= t_max]

    snap_events, rt_events = _load_reason_events(connect_db, uid, validator_slug, sim_id, book)
    attach_trade_reasons(orders, snap_events, rt_events)

    window = orders[-limit:]
    for i, order in enumerate(window, start=1):
        order["seq"] = i
    return {"orders": list(reversed(window)), "origin": t_min, "range": [t_min, t_max]}


def format_hold_s(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = float(seconds)
    if s >= 3600:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m}m {sec:.1f}s"
    if s >= 60:
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec:.1f}s"
    return f"{s:.1f}s"


def visible_sim_range(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    resolution: int,
    connect_db,
    limit: int = 5000,
) -> tuple[int, int]:
    t_min, t_max, _ = chart_time_window(
        uid, validator_slug, sim_id, book, resolution, limit, connect_db
    )
    return t_min, t_max


def format_round_trip(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ts_ns = int(out.pop("ts_close_ns", 0) or 0)
    out.pop("id", None)
    time_sec = ts_ns // 1_000_000_000 if ts_ns else 0
    out["time_sec"] = time_sec
    out["closed_at"] = sim_time_label(time_sec) if time_sec else ""
    hold = out.pop("hold_s", None)
    out["hold"] = format_hold_s(hold) if hold is not None else ""
    if out.get("reason") is not None:
        code = normalize_reason(str(out["reason"]))
        out["reason_code"] = code
        out["reason"] = display_reason(code)
    for key in ("qty", "entry_avg", "exit_avg", "realized_pnl"):
        if out.get(key) is not None:
            out[key] = round(float(out[key]), 4)
    return out
