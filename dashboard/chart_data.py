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


def chart_origin(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    resolution: int,
    connect_db,
) -> int:
    """Earliest simulation second on the chart (absolute, not bucket-relative)."""
    buckets = _collect_mid_buckets(uid, validator_slug, sim_id, book, resolution, connect_db)
    if not buckets:
        return 0
    return min(buckets) * resolution


def build_mid_series(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    resolution: int,
    limit: int,
    connect_db,
) -> dict[str, Any]:
    """Mid line: market prints backfill, telemetry mid overwrites; times are sim seconds."""
    buckets = _collect_mid_buckets(uid, validator_slug, sim_id, book, resolution, connect_db)
    if not buckets:
        return {"mid": [], "origin": 0}
    origin = chart_origin(uid, validator_slug, sim_id, book, resolution, connect_db)
    ordered = sorted(_forward_fill(buckets).items())[-limit:]
    return {
        "mid": [{"time": b * resolution, "value": p} for b, p in ordered],
        "origin": origin,
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


def load_trades_for_api(
    uid: int,
    validator_slug: str,
    sim_id: str,
    book: int,
    limit: int,
    connect_db,
) -> dict[str, Any]:
    path = find_trades_csv(uid, validator_slug, sim_id)
    if path is None:
        return {"orders": []}

    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if book >= 0 and int(row.get("bookId", -1)) != book:
                continue
            rows.append(dict(row))
    rows.sort(key=trade_sort_key)

    orders = [o for o in aggregate_taker_orders(rows, uid) if o.get("time") is not None]
    origin = chart_origin(uid, validator_slug, sim_id, max(book, 0), 1, connect_db)

    window = orders[-limit:]
    for i, order in enumerate(window, start=1):
        order["seq"] = i
    return {"orders": list(reversed(window)), "origin": origin}


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
) -> tuple[int, int]:
    buckets = _collect_mid_buckets(uid, validator_slug, sim_id, book, resolution, connect_db)
    if not buckets:
        return 0, 0
    keys = sorted(_forward_fill(buckets))
    return keys[0] * resolution, keys[-1] * resolution


def format_round_trip(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ts_ns = int(out.pop("ts_close_ns", 0) or 0)
    out.pop("id", None)
    time_sec = ts_ns // 1_000_000_000 if ts_ns else 0
    out["time_sec"] = time_sec
    out["closed_at"] = sim_time_label(time_sec) if time_sec else ""
    hold = out.pop("hold_s", None)
    out["hold"] = format_hold_s(hold) if hold is not None else ""
    for key in ("qty", "entry_avg", "exit_avg", "realized_pnl"):
        if out.get(key) is not None:
            out[key] = round(float(out[key]), 4)
    return out
