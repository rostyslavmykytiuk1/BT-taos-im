"""
Dry-run for the FINAL PureMakerAgent (tight-cut). NO network / validator / deploy. Unit-tests the
behaviors the final spec promises, calling agent methods directly with mock state:

  A. QUOTE INSIDE-ON-WIDE (improve = tick if spread>2·tick): wide spread -> 1 tick inside; tight
     spread -> at the touch (never crossing).
  B. IDLE FLAT books: a flat+stale book does NOTHING (return False, no order) — no force-seed.
  C. ACTIVITY backstop (held): a held lot past the deadline IOC-closes (deep safety).
  D. TIGHT STOP: a held long underwater past the vol-band (10-14bps) IOC-cuts; a barely-underwater
     fresh lot is HELD (return False).
  E. GIVEUP: a held lot older than EXIT_GIVEUP_S (150s) IOC-cuts even if not underwater.
  F. STOP BAND clamps to [10,14]: noise=0 -> 10bps floor; huge noise -> 14bps cap.
  G. REDUCE walks to BREAKEVEN: an aged long's passive reduce price never drops below entry.
  H. REPRICE keep-band = tick/2 (cushion removed): keep an at-price quote, cancel/reprice on a tick move.
  I. END-TO-END: open long -> price drops ~13bps -> managed-exit cuts -> RT registers with a BOUNDED
     loss (a tight cut, NOT a 50bps never-cut catastrophe).
  J. NO dead seed machinery remains (constants / attrs removed).

Run:  python3 tests/puremaker_v4_dryrun.py
"""

import importlib.util, sys
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("pmv4", REPO / "agents" / "PureMakerAgent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
V4, Inv, OD, TIF = mod.PureMakerAgent, mod._Inv, mod.OrderDirection, mod.TimeInForce
PASS, FAIL = [], []


def check(c, m):
    (PASS if c else FAIL).append(m)
    print(("  ok  " if c else " FAIL ") + m)


class Resp:
    def __init__(self): self.orders, self.cancels = [], []
    def limit_order(self, **kw): self.orders.append(kw)
    def cancel_orders(self, book_id, ids): self.cancels.append((book_id, list(ids)))


def acct(base_free=0.0, quote_free=1e7, orders=None, fee=0.000466):
    return NS(base_balance=NS(free=base_free, reserved=0.0),
              quote_balance=NS(free=quote_free, reserved=0.0),
              orders=orders or [], fees=NS(maker_fee_rate=fee), quote_loan=0.0)


def agent():
    a = object.__new__(V4)
    a.uid = 9999
    a.quote_lot, a.exch_min = 0.26, 0.25
    a._price_decimals, a._volume_decimals = 2, 4
    a._tick, a._flat_eps = 0.01, 0.5e-4
    a.tp_bps_base = mod.TP_BPS_BASE
    a.books_state, a.inv = {}, {}
    a.rt_window_ns = int(mod.RT_WINDOW_S * 1e9)
    a.volume_assessment_ns = int(86400 * 1e9)
    a.quote_expiry_ns = int(mod.QUOTE_EXPIRY_S * 1e9)
    a.exit_walk_start_ns = int(mod.EXIT_WALK_START_S * 1e9)
    a.exit_giveup_ns = int(mod.EXIT_GIVEUP_S * 1e9)
    a.reentry_cooldown_ns = int(mod.REENTRY_COOLDOWN_S * 1e9)
    a.activity_deadline_ns = int(mod.ACTIVITY_DEADLINE_S * 1e9)
    a.kappa_rt_history_ns = int(mod.KAPPA_RT_HISTORY_S * 1e9)
    a.kappa_min_lookback_ns = int(mod.KAPPA_MIN_LOOKBACK_S * 1e9)
    return a


def bstate(): return mod._BookState()


print("\n[A] quote INSIDE-ON-WIDE (improve = tick if spread>2·tick)")
a = agent()
# wide spread (5 ticks) -> step 1 tick inside
des = a._desired_quotes("v", 7, acct(base_free=10.0), Inv(), 0.0, 300.00, 300.05, 300.025, 0.000466, 1e12, 10**12)
b, s = des.get(OD.BUY), des.get(OD.SELL)
check(b and abs(b[0] - 300.01) < 1e-9, f"wide: BUY 1 tick inside best_bid -> 300.01 (got {b})")
check(s and abs(s[0] - 300.04) < 1e-9, f"wide: SELL 1 tick inside best_ask -> 300.04 (got {s})")
# tight spread (1 tick) -> at the touch, never crossing
des2 = a._desired_quotes("v", 8, acct(base_free=10.0), Inv(), 0.0, 300.00, 300.01, 300.005, 0.000466, 1e12, 10**12)
b2, s2 = des2.get(OD.BUY), des2.get(OD.SELL)
check(b2 and abs(b2[0] - 300.00) < 1e-9, f"tight: BUY at touch 300.00 (got {b2})")
check(s2 and abs(s2[0] - 300.01) < 1e-9, f"tight: SELL at touch 300.01 (got {s2})")
check(b2 and s2 and b2[0] < s2[0], "tight: bid < ask (never crosses)")

print("\n[B] IDLE flat books — flat+stale does NOTHING (no force-seed)")
a = agent(); r = Resp()
ok = a._activity_close(r, 5, acct(base_free=0.0), Inv(), 0.0, 300.00, 300.20, 4)
check(ok is False and not r.orders and not r.cancels, "flat -> return False, 0 orders, 0 cancels")

print("\n[C] ACTIVITY backstop — HELD lot IOC-closes (deep safety)")
a = agent(); r = Resp(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
ok = a._activity_close(r, 5, acct(base_free=0.26), inv, 0.26, 300.00, 300.20, 4)
sells = [x for x in r.orders if x.get("direction") == OD.SELL]
check(ok is True and len(sells) == 1 and sells[0].get("timeInForce") == TIF.IOC, "held -> 1 IOC SELL")

print("\n[D] TIGHT STOP — underwater past the 10-14bps band cuts; barely-underwater holds")
# underwater ~13bps (entry 300.00, best_bid 299.60) -> stop fires
a = agent(); r = Resp(); inv = Inv(); inv.longs.append((10**12, 0.26, 300.00, 0.0)); st = bstate()
now = 10**12 + int(5e9)  # 5s old (not aged)
ok = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 299.60, 299.62, 4, now, st)
sells = [x for x in r.orders if x.get("direction") == OD.SELL]
check(ok is True and len(sells) == 1 and sells[0].get("timeInForce") == TIF.IOC, "13bps underwater -> IOC-cut")
# barely underwater ~1bps, fresh -> HELD
a = agent(); r = Resp(); inv = Inv(); inv.longs.append((10**12, 0.26, 300.00, 0.0)); st = bstate()
ok = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 299.97, 299.99, 4, 10**12 + int(5e9), st)
check(ok is False and not r.orders, "1bps underwater + fresh -> HELD (no cut)")

print("\n[E] GIVEUP — lot older than 150s cuts even if not underwater")
a = agent(); r = Resp(); inv = Inv(); inv.longs.append((0, 0.26, 300.00, 0.0)); st = bstate()
now = int(200e9)  # 200s old > 150s giveup
ok = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 300.00, 300.02, 4, now, st)
check(ok is True and any(x.get("direction") == OD.SELL for x in r.orders), "aged>150s -> IOC-cut (age)")

print("\n[F] STOP BAND clamps to [10,14]bps")
a = agent()
st0 = bstate(); st0.noise_bps = 0.0
check(abs(a._stop_bps(st0) - 10.0) < 1e-9, f"noise=0 -> 10bps floor (got {a._stop_bps(st0)})")
st1 = bstate(); st1.noise_bps = 100.0   # 6*100 huge -> cap
check(abs(a._stop_bps(st1) - 14.0) < 1e-9, f"huge noise -> 14bps cap (got {a._stop_bps(st1)})")

print("\n[G] REDUCE walks to BREAKEVEN — aged long reduce never drops below entry")
a = agent()
px0 = 300.00
# aged (w=1): reduce should be >= entry (breakeven sell), never at/below the touch loss
px_aged = a._reduce_price(True, px0, a.exit_giveup_ns, 299.50, 0.0018, 0.0010, 10.0, 2)
check(px_aged >= px0 - 1e-9, f"aged long reduce {px_aged} >= entry {px0} (never sells at a loss passively)")

print("\n[H] REPRICE keep-band = tick/2 (cushion REMOVED -> V1 behavior)")
a = agent()
r = Resp()  # AT the desired price (0 < tick/2) -> KEEP
a._reconcile_quotes(r, acct(base_free=10, orders=[NS(side=0, price=300.01, quantity=0.26, id=1)]),
                    9, {OD.BUY: (300.01, 0.26)})
check(r.cancels == [] and r.orders == [], "resting quote AT desired price -> KEPT")
r2 = Resp()  # 1 tick away (> tick/2) -> CANCEL + reprice (no cushion)
a._reconcile_quotes(r2, acct(base_free=10, orders=[NS(side=0, price=300.00, quantity=0.26, id=2)]),
                    9, {OD.BUY: (300.01, 0.26)})
check(bool(r2.cancels) and r2.cancels[0][1] == [2], "resting quote 1 tick from desired -> CANCELLED (no cushion)")

print("\n[I] END-TO-END — open long -> ~13bps drop -> tight cut -> RT registers, loss BOUNDED")
a = agent(); val, bk = "v", 5
a._apply_fill(val, bk, True, 0.26, 300.00, 0.0, 10**12)            # open long @300.00
st = a._bstate(val, bk)
check(a._long_qty(a._inv(val, bk)) >= 0.25 and st.last_rt_ns == 0, "opened long, no RT yet")
r = Resp()
now = 10**12 + int(5e9)
ok = a._managed_exit(r, bk, acct(base_free=0.26), a._inv(val, bk), 0.26, 299.60, 299.62, 4, now, st)
cut = [x for x in r.orders if x.get("direction") == OD.SELL]
check(ok and cut, "managed-exit fired the cut")
a._apply_fill(val, bk, False, 0.26, cut[0]["price"], 0.0, now)     # cut fills at the IOC price
check(st.last_rt_ns == now and len(st.rt_events) == 1, "ROUND-TRIP registered")
loss_bps = st.rt_events[0][1] / (0.26 * 300.00) * 1e4
check(-40 < loss_bps < 0, f"realized loss is BOUNDED/tight ({loss_bps:.1f}bps, not a 50bps catastrophe)")

print("\n[J] dead seed machinery removed")
check(not any(hasattr(mod, n) for n in ("SEED_COOLDOWN_S", "SEED_CROSS_BPS", "SEED_MAX_PER_STEP")),
      "SEED_* module constants gone")
check(not hasattr(bstate(), "last_seed_ns"), "_BookState.last_seed_ns gone")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
sys.exit(1 if FAIL else 0)
