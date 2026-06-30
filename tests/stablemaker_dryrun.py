"""
Dry-run for StableMakerAgent — NO network / validator / deploy. Unit-tests the post-review changes by
calling the agent methods directly with mock state:

  A. _net_edge_bps — TRUE round-trip economics: half_spread − 2*fee − ADVERSE_SEL_BPS; None on bad book / unknown fee.
  B. _compute_gate_min_edge — best-K floor: all-+EV → threshold 0 (all pass); fee-spike → admits exactly the
     best MIN_ACTIVE_BOOKS (never fewer) so the median can NEVER collapse; <80 rankable → -inf (all pass).
  C. _gate_ok — flat book quotes iff net_edge >= the floor.
  D. BREAKEVEN-TIMER (the core fix) in _managed_exit:
       - aged + UNDERWATER (0<uw<stop)  → HELD, no order (kills the manufactured 180s loss tail).
       - aged + BREAKEVEN-OR-BETTER (uw<=0) → closed at the touch, slip 0 (banks the RT, no concession).
       - STOPPED (uw>=stop)              → cut with escalating concession (the only loss-realiser).
       - not aged, not stopped           → HELD.
  E. NO THRASH — a held book still gets its reduce quote regardless of the gate.
  F. never-hold-forever — the 510s activity backstop force-closes a held underwater lot (the hard max-hold).
  G. COLLAPSE-PROOF regime sweep — at a maker-pays fee, >= MIN_ACTIVE_BOOKS still quote (no self-collapse).

Run:  python3 tests/stablemaker_dryrun.py
"""

import importlib.util, sys
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("sm", REPO / "agents" / "StableMakerAgent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
SM, Inv, OD, TIF = mod.StableMakerAgent, mod._Inv, mod.OrderDirection, mod.TimeInForce
ADV, MINB = mod.ADVERSE_SEL_BPS, mod.MIN_ACTIVE_BOOKS
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


print(f"\n[A] _net_edge_bps = FULL_spread − 2*fee − {ADV}bps  (true RT economics; agent banks ~full spread)")
a = agent()
bb, ba, mid = raw_book(9.5)                   # full spread = 19bps
ne = a._net_edge_bps(bb, ba, mid, 0.00006)   # fee 0.6bps
check(abs(ne - (19.0 - 2 * 0.6 - ADV)) < 0.05, f"19 full, 0.6 fee → {ne:.2f} ≈ {19.0-1.2-ADV:.2f} (+EV)")
# CALIBRATION: ne=0 must land at the measured breakeven full_spread = 2*fee + 2.5 (at fee 4.25 ⇒ 11bps)
bbk, bak, midk = raw_book((2 * 4.25 + ADV) / 2)   # full_spread = 2*4.25+2.5 = 11bps
nek = a._net_edge_bps(bbk, bak, midk, 0.000425)
check(abs(nek) < 0.05, f"calibration: full_spread 11bps @ fee 4.25 → ne {nek:+.2f} ≈ 0 (matches +10.99→+0.02 data)")
ne2 = a._net_edge_bps(bb, ba, mid, 0.00110)  # fee 11bps (deep maker-pays)
check(ne2 < 0, f"19 full, 11 fee → {ne2:.2f} (−EV)")
check(a._net_edge_bps(bb, ba, mid, None) is None, "unknown fee → None (excluded, fail-safe)")
check(a._net_edge_bps(300.20, 300.00, 300.10, 0.00006) is None, "crossed/locked book → None")

print(f"\n[B] _compute_gate_min_edge — best-K floor (MIN_ACTIVE_BOOKS={MINB})")
a = agent()
# 128 books all wide+cheap → all +EV → threshold 0.0
wide = {i: raw_book(9.5)[:2] for i in range(128)}
a.accounts = {i: acct(fee=0.00006) for i in range(128)}
thr = a._compute_gate_min_edge(mock_state(wide))
check(abs(thr - 0.0) < 1e-9, f"128 +EV books → threshold {thr:.3f} == 0.0 (every +EV book passes)")
# maker-pays + VARIED spreads (half 12→5.65bps) at fee 9bps → mostly −EV. Floor must select EXACTLY the best 80.
varied = {i: raw_book(12.0 - i * 0.05)[:2] for i in range(128)}
a.accounts = {i: acct(fee=0.00090) for i in range(128)}
thr2 = a._compute_gate_min_edge(mock_state(varied))
passing = sum(1 for i in range(128)
              if a._gate_ok(*varied[i], 0.5 * sum(varied[i]), 0.00090, thr2))
check(thr2 < 0, f"maker-pays (mostly −EV) → threshold {thr2:.3f} < 0 (admits least-bad books)")
check(passing == MINB, f"selects EXACTLY the best {MINB} (idles the worst {128-MINB}) — collapse-proof + selective")
# fewer than 80 rankable books → -inf (admit all)
a.accounts = {i: acct(fee=0.00006) for i in range(50)}
thr3 = a._compute_gate_min_edge(mock_state({i: raw_book(9.5)[:2] for i in range(50)}))
check(thr3 == float("-inf"), "only 50 books → threshold -inf (can't reach floor; admit all)")

print("\n[C] _gate_ok — quote iff net_edge >= floor")
a = agent()
bb, ba, mid = raw_book(9.5)
check(a._gate_ok(bb, ba, mid, 0.00006, 0.0) is True, "+EV book vs floor 0 → quote")
check(a._gate_ok(bb, ba, mid, 0.00090, 0.0) is False, "−EV book vs floor 0 → idle")
check(a._gate_ok(bb, ba, mid, 0.00090, -12.0) is True, "−EV book vs a deeper negative floor (fee-spike) → quote (best-K)")

print("\n[D] BREAKEVEN-TIMER in _managed_exit (the core fix)")
NOW = int(200e9)                              # > giveup (180s)
# D1: aged + UNDERWATER (uw ~3bps, < 18 stop) → HELD, no order
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
held = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 300.00, 300.20, 4, NOW, st)
check(held is False and not r.orders, "aged + underwater (3bps) → HELD, NO cut (kills the 180s loss tail)")
# D2: aged + BREAKEVEN-OR-BETTER (best_bid 300.20 >= entry 300.10) → close at the touch, slip 0
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
ok = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 300.20, 300.40, 4, NOW, st)
sell = [o for o in r.orders if o.get("direction") == OD.SELL]
check(ok is True and len(sell) == 1, "aged + breakeven-or-better → closes (banks the RT)")
check(sell and abs(sell[0]["price"] - 300.20) < 1e-9, f"closes AT the touch (slip 0): px={sell and sell[0]['price']} == best_bid 300.20")
check(sell and sell[0].get("timeInForce") == TIF.IOC, "the close is IOC")
# D3: STOPPED (uw ~20bps >= 18) → cut WITH concession (the only loss-realiser)
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
ok = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 299.50, 299.70, 4, NOW, st)
sell = [o for o in r.orders if o.get("direction") == OD.SELL]
check(ok is True and sell, "stopped (20bps underwater) → CUTS (catastrophe stop)")
check(sell and sell[0]["price"] < 299.50, f"cut concedes below the bid (slip): px={sell and sell[0]['price']} < 299.50")
# D4: not aged, not stopped → HELD
a = agent(); st = bstate(); inv = Inv(); inv.longs.append((int(195e9), 0.26, 300.10, 0.0))
r = Resp()
held = a._managed_exit(r, 5, acct(base_free=0.26), inv, 0.26, 300.05, 300.25, 4, NOW, st)
check(held is False and not r.orders, "young + slightly underwater → HELD (revert window)")

print("\n[E] NO THRASH — held book still reduces even when the gate would idle it")
a = agent()
inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
bbT, baT, midT = raw_book(1.0)                 # tight spread, would fail the gate
des = a._desired_quotes("v", 7, acct(base_free=0.26), inv, 0.26, bbT, baT, midT, 0.00090, 1e12, int(10e9), -1.0)
check(OD.SELL in des and OD.BUY not in des, "held long → reduce-only SELL (gate does NOT eject a position)")

print("\n[F] never-hold-forever — 510s activity backstop force-closes a held underwater lot")
a = agent()
inv = Inv(); inv.longs.append((0, 0.26, 300.10, 0.0))
r = Resp()
ok = a._activity_close(r, 5, acct(base_free=0.26), inv, 0.26, 300.00, 300.20, 4)  # underwater
check(ok is True and any(o.get("direction") == OD.SELL for o in r.orders),
      "held underwater lot past the window → force-closed (hard max-hold; stop+backstop = never forever)")

print("\n[G] regime sweep — books passing the gate across the fee curve (VARIED half-spreads 12→5.65bps, 128 books)")
a = agent()
for fee_bps in [0.6, 4.0, 7.0, 9.0, 12.0]:
    bks = {i: raw_book(12.0 - i * 0.05)[:2] for i in range(128)}   # varied edges so the floor truly selects
    a.accounts = {i: acct(fee=fee_bps / 1e4) for i in range(128)}
    thr = a._compute_gate_min_edge(mock_state(bks))
    n = sum(1 for i in range(128) if a._gate_ok(*bks[i], 0.5 * sum(bks[i]), fee_bps / 1e4, thr))
    n_pos = sum(1 for i in range(128)
                if (a._net_edge_bps(*bks[i], 0.5 * sum(bks[i]), fee_bps / 1e4) or -9) >= 0)
    label = "all +EV → full breadth" if n_pos >= 128 else (
        f"{n_pos} +EV; floor holds the line at {MINB}" if n == MINB else f"{n_pos} +EV books")
    print(f"     fee +{fee_bps:>4.1f}bps → {n:>3}/128 quote  ({label})")
    check(n >= MINB, f"  fee {fee_bps}bps: {n} >= {MINB} (never self-collapses)")

print("\n[H] _tune_gc — behaviour-neutral GC tuning (axon-timeout fix)")
import gc as _gc
a = agent()
a._tune_gc()
check(a.history_len == 0, "history_len set to 0 (framework skips the 128-book deep-copy retention)")
check(_gc.get_threshold()[0] == 50000, f"gen0 GC threshold raised to 50000 ({_gc.get_threshold()})")

print("\n[I] _refresh_book_kappa gated to MAIN_VALIDATOR (skips the O(B²) scan off-main)")
a = agent()
base = int(1e9)
span = int((mod.KAPPA_MIN_LOOKBACK_S + 600) * 1e9)   # > 90min lookback, 3 nonzero obs → real kappa on MAIN
rts = [(base, 0.02), (base + span // 2, -0.01), (base + span, 0.03)]
for val in ("not-main", mod.MAIN_VALIDATOR):
    st = a._bstate(val, 5)
    st.rt_events = list(rts)
    st.kappa3 = None
    a._refresh_book_kappa(val, 5, base + span)
check(a._bstate("not-main", 5).kappa3 is None, "non-main validator → skipped (kappa3 stays None, no wasted scan)")
check(a._bstate(mod.MAIN_VALIDATOR, 5).kappa3 is not None,
      f"MAIN validator → computed ({a._bstate(mod.MAIN_VALIDATOR, 5).kappa3:+.4f}) — proves the gate suppresses off-main")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
sys.exit(1 if FAIL else 0)
