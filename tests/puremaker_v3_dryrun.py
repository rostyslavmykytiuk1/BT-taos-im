"""
Dry-run for PureMakerV3Agent's three changes — NO network / validator / deploy. Unit-tests the
behaviors the plan promises, by calling the agent methods directly with mock state:

  A. ACTIVITY FLOOR (flat): a flat+stale book FORCES a tiny marketable IOC BUY (~quote_lot) — the
     seed leg of the guaranteed round-trip. (Was: return False / do nothing.)
  B. ACTIVITY FLOOR (held): a held lot IOC-closes (the close leg) → round-trip completes next step.
  C. QUOTE AT TOUCH (improve=0): flat two-sided quotes land exactly at best_bid / best_ask (never
     stepped inside, never crossing).
  D. CUSHIONED REPRICE (REPRICE_KEEP_TICKS=1.5): a resting quote within 1.5 ticks of desired is KEPT
     (no cancel/repost); beyond 1.5 ticks it is cancelled.

Run:  python3 tests/puremaker_v3_dryrun.py
"""

import importlib.util, sys
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("pmv3", REPO / "agents" / "PureMakerV3Agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
V3, Inv, OD, TIF = mod.PureMakerV3Agent, mod._Inv, mod.OrderDirection, mod.TimeInForce
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
    a = object.__new__(V3)
    a.uid = 9999
    a.quote_lot, a.exch_min = 0.26, 0.25
    a._price_decimals, a._volume_decimals = 2, 4
    a._tick, a._flat_eps = 0.01, 0.5e-4
    a.tp_bps_base = 8.0
    a.books_state, a.inv = {}, {}
    a.rt_window_ns = int(570 * 1e9)
    a.volume_assessment_ns = int(86400 * 1e9)
    a.quote_expiry_ns = int(12 * 1e9)   # set in initialize() on the real agent
    a.seed_cooldown_ns = int(mod.SEED_COOLDOWN_S * 1e9)
    a.kappa_rt_history_ns = int(mod.KAPPA_RT_HISTORY_S * 1e9)
    a.kappa_min_lookback_ns = int(mod.KAPPA_MIN_LOOKBACK_S * 1e9)
    a._seeds_this_step = 0
    return a


def bstate():
    return mod._BookState()


print("\n[A] activity floor — FLAT+stale forces a seed IOC BUY (~quote_lot, marketable)")
a = agent(); r = Resp()
ok = a._activity_close(r, 5, acct(base_free=0.0), Inv(), 0.0, 300.00, 300.20, 4, 10**12, bstate())
check(ok is True, "returns True (no longer declines when flat)")
check(len(r.orders) == 1, f"submits exactly 1 order ({len(r.orders)})")
o = r.orders[0] if r.orders else {}
check(o.get("direction") == OD.BUY, "it's a BUY (seed a long)")
check(o.get("timeInForce") == TIF.IOC, "it's IOC (marketable taker — guaranteed fill)")
check(abs(o.get("quantity", 0) - 0.26) < 1e-9, f"qty == quote_lot 0.26 (not exch_min) ({o.get('quantity')})")
check(o.get("price", 0) >= 300.20, f"price crosses the ask (marketable): {o.get('price')} >= 300.20")
check(o.get("postOnly") is not True, "NOT post_only (must be allowed to take)")

print("\n[B] activity floor — HELD long IOC-closes (the close leg → RT completes)")
a = agent(); r = Resp(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
ok = a._activity_close(r, 5, acct(base_free=0.26), inv, 0.26, 300.00, 300.20, 4, 10**12, bstate())
sells = [x for x in r.orders if x.get("direction") == OD.SELL]
check(ok is True and len(sells) == 1, "held long → submits 1 IOC SELL")
check(sells and sells[0].get("timeInForce") == TIF.IOC, "the close is IOC (completes the round-trip)")

print("\n[E] seed cooldown — no re-seed while a seed's fill is in flight")
a = agent(); st = bstate()
r1 = Resp(); ok1 = a._activity_close(r1, 5, acct(), Inv(), 0.0, 300.00, 300.20, 4, 10**12, st)
r2 = Resp(); ok2 = a._activity_close(r2, 5, acct(), Inv(), 0.0, 300.00, 300.20, 4, 10**12 + int(5e9), st)
check(ok1 is True and len(r1.orders) == 1, "1st flat call seeds")
check(ok2 is False and len(r2.orders) == 0, "2nd call 5s later (within 15s cooldown) does NOT re-seed")
r3 = Resp(); ok3 = a._activity_close(r3, 5, acct(), Inv(), 0.0, 300.00, 300.20, 4, 10**12 + int(35e9), st)
check(ok3 is True and len(r3.orders) == 1, "3rd call 35s later (past cooldown) re-seeds")

print("\n[C] quote AT the touch (improve=0): flat quotes at best_bid / best_ask, never crossing")
a = agent()
des = a._desired_quotes("v", 7, acct(base_free=10.0), Inv(), 0.0, 300.00, 300.20, 300.10, 0.000466, 1e12, 10**12)
b = des.get(OD.BUY); s = des.get(OD.SELL)
check(b is not None and abs(b[0] - 300.00) < 1e-9, f"BUY at best_bid 300.00 (not inside): {b}")
check(s is not None and abs(s[0] - 300.20) < 1e-9, f"SELL at best_ask 300.20 (not inside): {s}")
check(b and s and b[0] < s[0], "bid < ask (never crosses)")

print("\n[D] cushioned reprice (REPRICE_KEEP_TICKS=1.5)")
a = agent()
# within 1.5 ticks (0.01 away) -> KEEP
r = Resp()
a._reconcile_quotes(r, acct(base_free=10, orders=[NS(side=0, price=300.00, quantity=0.26, id=1)]),
                    9, {OD.BUY: (300.01, 0.26)})
check(r.cancels == [] and r.orders == [], "resting quote 1 tick from desired is KEPT (no churn)")
# beyond 1.5 ticks (0.03 away) -> CANCEL + repost
r2 = Resp()
a._reconcile_quotes(r2, acct(base_free=10, orders=[NS(side=0, price=300.00, quantity=0.26, id=2)]),
                    9, {OD.BUY: (300.03, 0.26)})
check(r2.cancels and r2.cancels[0][1] == [2], "resting quote 3 ticks from desired is CANCELLED (repriced)")

print("\n[F] AF1 fix — seed CANCELS our own resting ask first, then crosses WIDE (external)")
a = agent()
resting = [NS(side=1, price=300.20, quantity=0.26, id=7)]   # our resting SELL at best_ask (side 1)
r = Resp()
ok = a._activity_close(r, 5, acct(orders=resting), Inv(), 0.0, 300.00, 300.20, 4, 10**12, bstate())
check(ok is True, "seeds")
check(bool(r.cancels) and 7 in r.cancels[0][1], "cancels our own resting ask FIRST (no self-cancel-then-unfilled)")
seed = [o for o in r.orders if o.get("direction") == OD.BUY]
exp = round(300.20 * (1 + mod.SEED_CROSS_BPS / 1e4), 2)
check(seed and abs(seed[0]["price"] - exp) < 1e-9, f"crosses at SEED_CROSS_BPS={mod.SEED_CROSS_BPS:.0f}bps wide (={exp}), not the 5bps touch-slip")

print("\n[G] end-to-end — seed → fill → close → fill REGISTERS a round-trip within 600s (the guarantee)")
a = agent(); val, bk = "v", 5
st = a._bstate(val, bk)
r = Resp()
a._activity_close(r, bk, acct(), a._inv(val, bk), 0.0, 300.00, 300.20, 4, 10**12, st)            # seed
a._apply_fill(val, bk, True, 0.26, 300.74, 0.0, 10**12)                                          # seed BUY fills
check(a._long_qty(a._inv(val, bk)) >= 0.25, "seed fill opened a long")
check(st.last_rt_ns == 0, "no RT yet (only the open leg filled)")
r2 = Resp()
a._activity_close(r2, bk, acct(base_free=0.26), a._inv(val, bk), a._net_qty(a._inv(val, bk)),
                  300.00, 300.20, 4, 10**12 + int(5e9), st)                                       # held → close
check(any(o.get("direction") == OD.SELL for o in r2.orders), "held → IOC SELL (close leg)")
a._apply_fill(val, bk, False, 0.26, 300.00, 0.0, 10**12 + int(5e9))                               # close SELL fills
check(st.last_rt_ns == 10**12 + int(5e9), "ROUND-TRIP REGISTERED (last_rt_ns = close ts)")
check(len(st.rt_events) == 1, "exactly 1 round-trip recorded")
check((st.last_rt_ns - 10**12) / 1e9 < 600, "round-trip completed inside the 600s activity window")

print("\n[H] timing budget (AF2): deadline < 600s grace, with retry room")
check(mod.ACTIVITY_DEADLINE_S < 600, f"deadline {mod.ACTIVITY_DEADLINE_S}s < 600s grace")
retries = (600 - mod.ACTIVITY_DEADLINE_S) / mod.SEED_COOLDOWN_S
check(retries >= 4, f"~{retries:.0f} seed retries fit before the window closes (>=4)")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
sys.exit(1 if FAIL else 0)

