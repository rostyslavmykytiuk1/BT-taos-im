"""
Dry-run for StableMakerV2Agent — NO network / validator / deploy. Unit-tests Patch 1 (and re-verifies the
inherited StableMaker logic is unbroken) by calling the agent methods directly with mock state:

  A. _net_edge_bps / B. _compute_gate_min_edge / C. _gate_ok — UNCHANGED edge gate (re-verified).
  D. _managed_exit (PATCH 1):
       - aged + UNDERWATER (0<uw<stop)          → HELD, no order.
       - aged + BREAKEVEN-OR-BETTER (uw<=0)     → HELD here (returns False, NO IOC); the passive reduce banks it.
       - STOPPED (uw>=stop)                      → IOC-cut with escalating concession (the only loss-realiser).
       - not aged, not stopped                   → HELD.
  E. NO THRASH — a held book still gets its reduce quote regardless of the gate.
  F. never-hold-forever — the 510s activity backstop force-closes a held underwater lot.
  G. COLLAPSE-PROOF regime sweep — at a maker-pays fee, >= MIN_ACTIVE_BOOKS still quote.
  H/I. _tune_gc / _refresh_book_kappa gating — UNCHANGED.
  J. PATCH 1 — _risk_trim passive-first: breach sheds the excess as a MAKER post at the far touch for
     RISK_TRIM_PASSIVE_STEPS, then escalates to the IOC drain; clears state when under cap.
  K. PATCH 1 — skew guard DROPPED (was buggy): a sub-exch_min lean must still be quoted (grown to a
     closeable lot), NOT stranded; a truly-flat book is two-sided.
  L. PATCH 1 — aged-breakeven lot is banked PASSIVELY by _desired_quotes (post_only reduce at breakeven).

Run:  python3 tests/stablemaker_v2_dryrun.py
"""

import importlib.util, sys
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("smv2", REPO / "agents" / "StableMakerV2Agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SM, Inv, OD, TIF = mod.StableMakerV2Agent, mod._Inv, mod.OrderDirection, mod.TimeInForce
ADV, MINB = mod.ADVERSE_SEL_BPS, mod.MIN_ACTIVE_BOOKS
PSTEPS = mod.RISK_TRIM_PASSIVE_STEPS
INVLOTS = mod.MAX_INVENTORY_LOTS
PASS, FAIL = [], []


def check(c, m):
    (PASS if c else FAIL).append(m)
    print(("  ok  " if c else " FAIL ") + m)


class Resp:
    def __init__(self): self.orders, self.cancels = [], []
    def limit_order(self, **kw): self.orders.append(kw)
    def cancel_orders(self, book_id, ids): self.cancels.append((book_id, list(ids)))


def acct(base_free=0.0, quote_free=1e7, orders=None, fee=0.00006):
    return NS(base_balance=NS(free=base_free, reserved=0.0),
              quote_balance=NS(free=quote_free, reserved=0.0),
              orders=orders or [], fees=NS(maker_fee_rate=fee), quote_loan=0.0)


def agent():
    a = object.__new__(SM)
    a.uid = 9999
    a.quote_lot, a.exch_min = 0.26, 0.25
    a._price_decimals, a._volume_decimals = 2, 4
    a._tick, a._flat_eps = 0.01, 0.5e-4
    a.tp_bps_base = 8.0
    a.books_state, a.inv = {}, {}
    a.accounts = {}
    a.rt_window_ns = int(570 * 1e9)
    a.volume_assessment_ns = int(86400 * 1e9)
    a.quote_expiry_ns = int(12 * 1e9)
    a.exit_walk_start_ns = int(mod.EXIT_WALK_START_S * 1e9)
    a.exit_giveup_ns = int(mod.EXIT_GIVEUP_S * 1e9)
    a.reentry_cooldown_ns = int(mod.REENTRY_COOLDOWN_S * 1e9)
    a.activity_deadline_ns = int(mod.ACTIVITY_DEADLINE_S * 1e9)
    a.kappa_rt_history_ns = int(mod.KAPPA_RT_HISTORY_S * 1e9)
    a.kappa_min_lookback_ns = int(mod.KAPPA_MIN_LOOKBACK_S * 1e9)
    return a


def bstate():
    return mod._BookState()


def raw_book(half_bps, mid=300.10):          # exact (unrounded) book for a given half-spread
    h = mid * half_bps / 1e4
    return mid - h, mid + h, mid


def mock_state(books):                       # books: {id: (bid, ask)}
    bk = {i: NS(bids=[NS(price=b)], asks=[NS(price=a)]) for i, (b, a) in books.items()}
    return NS(books=bk)


def is_ioc(o):  return o.get("timeInForce") == TIF.IOC
def is_post(o): return o.get("postOnly") is True and o.get("timeInForce") == TIF.GTT


print(f"\n[A] _net_edge_bps = FULL_spread − 2*fee − {ADV}bps  (unchanged edge economics)")
a = agent()
bb, ba, mid = raw_book(9.5)                   # full spread = 19bps
ne = a._net_edge_bps(bb, ba, mid, 0.00006)
check(abs(ne - (19.0 - 2 * 0.6 - ADV)) < 0.05, f"19 full, 0.6 fee → {ne:.2f} ≈ {19.0-1.2-ADV:.2f} (+EV)")
bbk, bak, midk = raw_book((2 * 4.25 + ADV) / 2)
nek = a._net_edge_bps(bbk, bak, midk, 0.000425)
check(abs(nek) < 0.05, f"calibration: full_spread 11bps @ fee 4.25 → ne {nek:+.2f} ≈ 0")
check(a._net_edge_bps(bb, ba, mid, 0.00110) < 0, "19 full, 11 fee → −EV")
check(a._net_edge_bps(bb, ba, mid, None) is None, "unknown fee → None (excluded)")
check(a._net_edge_bps(300.20, 300.00, 300.10, 0.00006) is None, "crossed/locked book → None")

print(f"\n[B] _compute_gate_min_edge — best-K floor (MIN_ACTIVE_BOOKS={MINB}, unchanged)")
a = agent()
wide = {i: raw_book(9.5)[:2] for i in range(128)}
a.accounts = {i: acct(fee=0.00006) for i in range(128)}
check(abs(a._compute_gate_min_edge(mock_state(wide))) < 1e-9, "128 +EV books → threshold 0.0")
varied = {i: raw_book(12.0 - i * 0.05)[:2] for i in range(128)}
a.accounts = {i: acct(fee=0.00090) for i in range(128)}
thr2 = a._compute_gate_min_edge(mock_state(varied))
passing = sum(1 for i in range(128) if a._gate_ok(*varied[i], 0.5 * sum(varied[i]), 0.00090, thr2))
check(thr2 < 0 and passing == MINB, f"maker-pays → selects EXACTLY best {MINB} (collapse-proof)")
a.accounts = {i: acct(fee=0.00006) for i in range(50)}
check(a._compute_gate_min_edge(mock_state({i: raw_book(9.5)[:2] for i in range(50)})) == float("-inf"),
      "only 50 books → -inf (admit all)")

print("\n[C] _gate_ok — quote iff net_edge >= floor (unchanged)")
a = agent()
bb, ba, mid = raw_book(9.5)
check(a._gate_ok(bb, ba, mid, 0.00006, 0.0) is True, "+EV vs floor 0 → quote")
check(a._gate_ok(bb, ba, mid, 0.00090, 0.0) is False, "−EV vs floor 0 → idle")
check(a._gate_ok(bb, ba, mid, 0.00090, -12.0) is True, "−EV vs deep negative floor → quote (best-K)")

print("\n[D] _managed_exit — PATCH 1: stop is the ONLY IOC; aged-breakeven is HELD (passive)")
NOW = int(200e9)                              # > giveup (180s)
# D1: aged + UNDERWATER (uw ~3bps, < stop) → HELD, no order
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
held = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 300.00, 300.20, 4, NOW, st)
check(held is False and not r.orders, "aged + underwater (3bps) → HELD, no order")
# D2 (CHANGED): aged + BREAKEVEN-OR-BETTER → HELD here too (NO IOC); the passive reduce banks it (see L)
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
held = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 300.20, 300.40, 4, NOW, st)
check(held is False and not r.orders, "PATCH1: aged + breakeven-or-better → HELD, NO IOC at the touch (was IOC)")
# D3: STOPPED (uw ~20bps >= 18) → cut WITH concession (the only loss-realiser, still IOC)
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
ok = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 299.50, 299.70, 4, NOW, st)
sell = [o for o in r.orders if o.get("direction") == OD.SELL]
check(ok is True and sell and is_ioc(sell[0]), "stopped (20bps) → CUTS via IOC (catastrophe stop unchanged)")
check(sell and sell[0]["price"] < 299.50, f"cut concedes below the bid: {sell and sell[0]['price']} < 299.50")
# D4: not aged, not stopped → HELD
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((int(195e9), 0.26, 300.10, 0.0))
r = Resp()
held = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 300.05, 300.25, 4, NOW, st)
check(held is False and not r.orders, "young + slightly underwater → HELD (revert window)")
# D5: SHORT mirror — stopped short cuts via IOC BUY
a = agent(); st = bstate(); inv = Inv(); inv.shorts.append((0, 0.26, 300.10, 0.0))
r = Resp()
ok = a._managed_exit(r, 5, acct(base_free=0.0), inv, -0.26, 300.50, 300.70, 4, NOW, st)
buy = [o for o in r.orders if o.get("direction") == OD.BUY]
check(ok is True and buy and is_ioc(buy[0]) and buy[0]["price"] > 300.70, "short stopped → IOC BUY above the ask")

print("\n[E] NO THRASH — held book still reduces even when the gate would idle it (unchanged)")
a = agent()
inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
bbT, baT, midT = raw_book(1.0)
des = a._desired_quotes("v", 7, acct(base_free=0.26), inv, 0.26, bbT, baT, midT, 0.00090, 1e12, int(10e9), -1.0)
check(OD.SELL in des and OD.BUY not in des, "held long → reduce-only SELL")

print("\n[F] never-hold-forever — 510s activity backstop force-closes a held underwater lot (unchanged)")
a = agent()
inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
ok = a._activity_close(r, 5, acct(base_free=0.26), inv, 0.26, 300.00, 300.20, 4)
check(ok is True and any(o.get("direction") == OD.SELL for o in r.orders), "held underwater → force-closed")

print("\n[G] regime sweep — never self-collapses below MIN_ACTIVE_BOOKS (unchanged)")
a = agent()
for fee_bps in [0.6, 4.0, 7.0, 9.0, 12.0]:
    bks = {i: raw_book(12.0 - i * 0.05)[:2] for i in range(128)}
    a.accounts = {i: acct(fee=fee_bps / 1e4) for i in range(128)}
    thr = a._compute_gate_min_edge(mock_state(bks))
    n = sum(1 for i in range(128) if a._gate_ok(*bks[i], 0.5 * sum(bks[i]), fee_bps / 1e4, thr))
    print(f"     fee +{fee_bps:>4.1f}bps → {n:>3}/128 quote")
    check(n >= MINB, f"  fee {fee_bps}bps: {n} >= {MINB}")

print("\n[H] _tune_gc — behaviour-neutral GC tuning (unchanged)")
import gc as _gc
a = agent(); a._tune_gc()
check(a.history_len == 0 and _gc.get_threshold()[0] == 50000, "history_len=0 and gen0 threshold raised")

print("\n[I] _refresh_book_kappa gated to MAIN_VALIDATOR (unchanged)")
a = agent()
base = int(1e9); span = int((mod.KAPPA_MIN_LOOKBACK_S + 600) * 1e9)
rts = [(base, 0.02), (base + span // 2, -0.01), (base + span, 0.03)]
for val in ("not-main", mod.MAIN_VALIDATOR):
    s = a._bstate(val, 5); s.rt_events = list(rts); s.kappa3 = None
    a._refresh_book_kappa(val, 5, base + span)
check(a._bstate("not-main", 5).kappa3 is None and a._bstate(mod.MAIN_VALIDATOR, 5).kappa3 is not None,
      "non-main skipped; MAIN computed")

print(f"\n[J] PATCH 1 — _risk_trim passive-first ({PSTEPS} passive steps → IOC), cap={INVLOTS}lot")
a = agent(); st = bstate()
bb, ba, mid = 300.00, 300.20, 300.10
acc = acct(base_free=1.0)                      # long 0.50 = breaches the 0.26 cap by ~0.24
seen = []
for step in range(PSTEPS + 2):
    r = Resp()
    fired = a._risk_trim(r, 5, acc, 0.50, mid, bb, ba, 4, st)
    sell = [o for o in r.orders if o.get("direction") == OD.SELL]
    seen.append((fired, sell[0] if sell else None))
for i in range(PSTEPS):
    o = seen[i][1]
    check(seen[i][0] and o and is_post(o) and abs(o["price"] - ba) < 1e-9,
          f"  step {i+1}: passive post_only SELL at the ask ({o and o['price']})")
o = seen[PSTEPS][1]
check(seen[PSTEPS][0] and o and is_ioc(o) and o["price"] < mid,
      f"  step {PSTEPS+1}: escalates to IOC SELL below mid ({o and o['price']})")
# under-cap → returns False and resets the breach counter
r = Resp()
check(a._risk_trim(r, 5, acct(base_free=1.0), 0.20, mid, bb, ba, 4, st) is False and st.trim_breach_count == 0,
      "  under cap (0.20 < 0.26) → no trim, breach counter reset")
# SHORT breach mirror — first step posts a passive BUY at the bid
a = agent(); st = bstate(); r = Resp()
fired = a._risk_trim(r, 5, acct(base_free=0.0), -0.50, mid, bb, ba, 4, st)
buy = [o for o in r.orders if o.get("direction") == OD.BUY]
check(fired and buy and is_post(buy[0]) and abs(buy[0]["price"] - bb) < 1e-9, "  short breach → passive BUY at the bid")
# COUNTER STABILITY (fix 2): a breached step that can't submit (no base) must NOT advance the escalation.
a = agent(); st = bstate()
for _ in range(PSTEPS + 2):
    a._risk_trim(Resp(), 5, acct(base_free=0.0), 0.50, mid, bb, ba, 4, st)   # long breach, zero base to sell
check(st.trim_breach_count == 0, "  breached but un-submittable (0 base) → counter stays 0 (no false IOC escalation)")

print(f"\n[K] PATCH 1 — skew guard DROPPED: a sub-exch_min lean is NOT stranded (the fixed dead-zone)")
a = agent()
bb, ba, mid = raw_book(9.5)
GMIN = -1e9
# truly flat → two-sided
des = a._desired_quotes("v", 7, acct(base_free=1.0), Inv(), 0.0, bb, ba, mid, 0.00006, 1e12, int(10e9), GMIN)
check(OD.BUY in des and OD.SELL in des, "flat (net=0) → quotes BOTH sides")
# sub-exch_min long dust with REALISTIC free_base (= the dust): must still get a quote (grow-to-closeable),
# NOT the empty {} the skew guard produced. We model the held dust as base_balance.free = net.
for d in (0.10, 0.24, 0.249):          # spans the old [0.234, 0.25) dead-zone
    inv = Inv(); inv.longs.append((0, d, 300.10, 0.0))
    des = a._desired_quotes("v", 7, acct(base_free=d), inv, d, bb, ba, mid, 0.00006, 1e12, int(10e9), GMIN)
    check(len(des) > 0, f"  long dust net={d} (free_base={d}) → quotes something (no strand): {list('B' if k==OD.BUY else 'S' for k in des)}")

print("\n[L] PATCH 1 — aged-breakeven lot is banked PASSIVELY by _desired_quotes (the close path)")
a = agent()
inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))   # ts=0, aged at NOW
bb, ba, mid = 300.18, 300.30, 300.24                     # bid >= entry → breakeven-or-better
des = a._desired_quotes("v", 7, acct(base_free=0.26), inv, 0.26, bb, ba, mid, 0.00006, 1e12, NOW, -1.0)
check(OD.SELL in des, "aged breakeven long → a passive SELL reduce is posted (banks the RT as a maker)")
px = des.get(OD.SELL, (None,))[0]
check(px is not None and px >= 300.10 - 1e-9, f"  reduce rests at >= breakeven ({px}), never gives away the spread")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
sys.exit(1 if FAIL else 0)
