"""
Dry-run harness for TakerScalperV3Agent — NO network/validator/deploy.

Verifies the V3 FEE-BASED rework vs the live mechanics:
  * FEE-ONLY sleep with hysteresis: a book sleeps when its taker fee is POSITIVE (pays to take) and
    wakes only once the fee is a rebate beyond SLEEP_WAKE_FEE_BPS (fee < -2bps); the (-2bps,0] band
    holds state. NO spread in the sleep decision.
  * budget cap: when more books pay than free slots, the WORST (highest-fee) sleep; once MAX_SLEEP
    full, no more books sleep (no overflow, no displacement).
  * FEE-ONLY open gate: a rebated book opens EVEN with a wide spread / negative est_pnl (spread is not
    gated — it only feeds the logged expected PnL).
  * no_sleep flag disables sleep entirely (every book stays awake — the uid184 null hypothesis).
  * tighter stop: SL fires at -2.5 bps; -1.5 bps does not.
  * core RT mechanics: open -> fill -> close -> fill produces a clean net-PnL RT, no crash.

Run:  python3 tests/takerscalper_v3_dryrun.py
"""
import importlib.util, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("ts2_mod", REPO / "agents" / "TakerScalperV3Agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TS2 = mod.TakerScalperV3Agent
from taos.im.protocol.models import OrderDirection  # noqa: E402

VAL = mod.MAIN_VALIDATOR
SIM = "dry"
VD, NS = 4, mod._NS
UID = 777
PASS, FAIL = [], []


def check(c, m):
    (PASS if c else FAIL).append(m)
    print(("  PASS " if c else "  FAIL ") + m)


def ns(**k):
    return types.SimpleNamespace(**k)


def simcfg():
    return ns(simulation_id=SIM, volumeDecimals=VD, miner_wealth=1e6, book_count=8, publish_interval=NS)


def acct(rate, quote=1e6, base=100.0):
    return ns(base_balance=ns(free=base, reserved=0.0), quote_balance=ns(free=quote, reserved=0.0),
              fees=ns(taker_fee_rate=rate, maker_fee_rate=0.0), quote_loan=0.0, orders=[])


def book(bid, ask, q=5.0):
    return ns(bids=[ns(price=bid, quantity=q)], asks=[ns(price=ask, quantity=q)])


def book_spread(spread_bps, mid=300.0, q=5.0):
    """A book centered at `mid` with the given spread in bps (used to prove the spread does NOT gate)."""
    half = mid * spread_bps / 1e4 / 2.0
    return book(mid - half, mid + half, q)


def state(ts, books):
    return ns(dendrite=ns(hotkey=VAL), timestamp=ts, config=simcfg(), books=books)


def instrs(resp):
    out = []
    for i in resp.instructions:
        t = getattr(i, "type", "")
        if "MARKET" in t or "LIMIT" in t:
            out.append((getattr(i, "direction", None), getattr(i, "quantity", None), getattr(i, "bookId", None)))
    return out


def mkagent(accounts, no_sleep=False):
    # ParseKwargs coerces "1" -> float 1.0, so pass a FLOAT here to mirror production exactly.
    a = TS2(UID, ns(lazy_load=1, no_sleep=1.0 if no_sleep else 0.0))
    a._active_validator = VAL
    a._sim_id[VAL] = SIM
    a.simulation_config = simcfg()
    a.accounts = accounts
    a._step_ts_ns = 10 * NS
    a._agent_start_ns[VAL] = 10 * NS
    a._sync_order_size(VD)
    return a


def fill(a, book_id, direction, qty, price, rate, ts):
    ev = ns(bookId=book_id, takerAgentId=UID, makerAgentId=999, side=direction, quantity=qty,
            price=price, takerFee=rate * price * qty, timestamp=ts)
    a.onTrade(ev, validator=VAL)


def run():
    # 0,1,2 = rebate 4bps tight; 3 = rebate 3bps on a WIDE spread (est_pnl<0); 4,5 = PAY 2bps.
    REB = -0.0004
    accounts = {
        0: acct(REB), 1: acct(REB), 2: acct(REB),
        3: acct(-0.0003), 4: acct(0.0002), 5: acct(0.0002),
    }
    books = {
        0: book(299.99, 300.01), 1: book(349.99, 350.01), 2: book(259.99, 260.01),
        3: book_spread(20.0),                        # rebate but very wide spread -> est_pnl < 0
        4: book(299.99, 300.01), 5: book(349.99, 350.01),
    }
    a = mkagent(accounts)

    print("\n=== T1-T3: fee-based sleep + FEE-ONLY gate (spread does not gate or sleep) ===")
    resp = a.respond(state(10 * NS, books))
    orders = instrs(resp)
    opened_books = sorted({b for _, _, b in orders})
    awake = sorted(b for b in accounts if not a._bstate(VAL, b).sleeping)
    asleep = sorted(b for b in accounts if a._bstate(VAL, b).sleeping)
    print(f"  awake={awake}  asleep={asleep}  opened={opened_books}")
    check(asleep == [4, 5], f"only PAYING books (fee>0) sleep (got {asleep})")
    check(awake == [0, 1, 2, 3], f"all rebated books awake, incl. the wide-spread one (got {awake})")
    check(set(opened_books) == {0, 1, 2}, f"opens only +EV books (got {opened_books})")
    est3 = a._estimate_rt_pnl(-0.0003, books[3], a.min_order_size)
    check(est3 < 0 and 3 not in opened_books and not a._bstate(VAL, 3).sleeping,
          f"book 3 AWAKE (rebate) but NOT opened: est_pnl={est3:+.3f}<0 -> spread-aware open gate skips it")
    check(4 not in opened_books and 5 not in opened_books, "paying books sleep, never open")

    print("\n=== T_GATE: OPEN gate IS spread-aware (est_pnl>0); fee alone is not enough ===")
    ga = mkagent({0: acct(-0.0004), 1: acct(-0.0004)})  # both rebate 4bps (fee passes), differ only by spread
    gr = ga.respond(state(10 * NS, {0: book_spread(2.0), 1: book_spread(20.0)}))  # 0: est_pnl>0, 1: est_pnl<0
    gop = sorted({b for _, _, b in instrs(gr)})
    check(0 in gop and 1 not in gop, f"tight-spread rebate book opens, wide-spread one does NOT (got {gop})")

    print("\n=== T_SLEEP_FEE: enter at fee>0, wake at fee<-2bps, (-2,0] band holds state ===")
    sa = mkagent({0: acct(0.0001)})                                  # fee +1bps -> pays
    sa._update_sleep_states(VAL, 10 * NS)
    check(sa._bstate(VAL, 0).sleeping, "fee +1bps (pays) -> sleeps")
    sa.accounts[0].fees.taker_fee_rate = -0.0001                     # fee -1bps, inside (-2,0] band
    sa._update_sleep_states(VAL, 11 * NS)
    check(sa._bstate(VAL, 0).sleeping, "fee -1bps (band) -> STAYS asleep (hysteresis, not < -2)")
    sa.accounts[0].fees.taker_fee_rate = -0.0003                     # fee -3bps, rebate > 2
    sa._update_sleep_states(VAL, 12 * NS)
    check(not sa._bstate(VAL, 0).sleeping, "fee -3bps (rebate>2) -> WAKES")
    sa.accounts[0].fees.taker_fee_rate = -0.0001                     # awake, fee -1bps band
    sa._update_sleep_states(VAL, 13 * NS)
    check(not sa._bstate(VAL, 0).sleeping, "awake book at fee -1bps (band) -> stays awake")
    sa.accounts[0].fees.taker_fee_rate = 0.00005                     # awake book turns to a payer
    sa._update_sleep_states(VAL, 14 * NS)
    check(sa._bstate(VAL, 0).sleeping, "awake book turns fee>0 -> sleeps")

    print("\n=== T_BUDGET: > MAX_SLEEP payers -> worst-fee 40 sleep; once full, no overflow ===")
    N = mod.MAX_SLEEP + 5
    ba = mkagent({i: acct((i + 1) * 1e-5) for i in range(N)})         # all fee>0, increasing: book0 lowest .. bookN-1 highest
    ba._update_sleep_states(VAL, 10 * NS)
    nap = sorted(b for b in range(N) if ba._bstate(VAL, b).sleeping)
    check(len(nap) == mod.MAX_SLEEP, f"sleepers capped at MAX_SLEEP={mod.MAX_SLEEP} (got {len(nap)})")
    check(nap == list(range(5, N)), "the WORST-fee 40 sleep; the 5 least-bad payers stay awake")
    ba._update_sleep_states(VAL, 11 * NS)                            # re-pass with budget full
    nap2 = sorted(b for b in range(N) if ba._bstate(VAL, b).sleeping)
    check(nap2 == nap, "budget full -> the 5 awake payers DO NOT sleep (no overflow, no displacement)")

    print("\n=== T_BUDGET_NULLFEE: sleepers with fees=None still count -> cap never exceeded (review blocker) ===")
    M = mod.MAX_SLEEP + 5
    ca = mkagent({i: acct((i + 1) * 1e-5) for i in range(M)})        # all fee>0
    ca._update_sleep_states(VAL, 10 * NS)                            # worst 40 sleep (5..M-1); 0..4 awake payers
    for b in range(5, 13):                                           # 8 sleeping books lose their fee this pass
        ca.accounts[b].fees = None
    ca._update_sleep_states(VAL, 11 * NS)
    total_sleeping = sum(1 for s in ca.books_state.get(VAL, {}).values() if s.sleeping)
    check(total_sleeping <= mod.MAX_SLEEP, f"sleeping never exceeds MAX_SLEEP with None-fee sleepers (got {total_sleeping})")

    print("\n=== T_NOSLEEP: no_sleep flag disables sleeping entirely ===")
    nsa = mkagent({i: acct(0.0005) for i in range(5)}, no_sleep=True)  # all PAY heavily
    check(nsa.no_sleep is True, "no_sleep parsed from config float 1.0 (as ParseKwargs produces)")
    nsa._update_sleep_states(VAL, 10 * NS)
    check(not any(nsa._bstate(VAL, b).sleeping for b in range(5)), "no_sleep: no book sleeps even at fee +5bps")
    def _nsflag(v):
        return TS2(UID, ns(lazy_load=1, no_sleep=v)).no_sleep
    check(_nsflag(1.0) and _nsflag("1") and _nsflag("true") and _nsflag("yes"), "no_sleep truthy: 1.0 / '1' / 'true' / 'yes'")
    check(not _nsflag(0.0) and not _nsflag("0") and not _nsflag(0), "no_sleep falsy: 0.0 / '0' / 0")

    print("\n=== T4: open -> fill -> hold ===")
    for b in (0, 1, 2):
        d = next(dr for dr, _, bb in orders if bb == b)
        px = 300.01 if d == OrderDirection.BUY else 299.99
        if b == 1: px = 350.01 if d == OrderDirection.BUY else 349.99
        if b == 2: px = 260.01 if d == OrderDirection.BUY else 259.99
        fill(a, b, d, 0.3, px, REB, 10 * NS)
    held = sum(1 for b in (0, 1, 2) if abs(a._book_positions(VAL)[b].qty) > 1e-6)
    check(held == 3, f"3 positions held after fills ({held})")

    print("\n=== T5: tighter stop — SL at -2bps (not -1.5) ===")
    pos0 = a._book_positions(VAL)[0]
    longp = pos0.qty > 0
    entry = pos0.avg
    px_15 = entry * (1 - 1.5e-4) if longp else entry * (1 + 1.5e-4)
    a._step_ts_ns = 12 * NS
    r2 = a.respond(state(12 * NS, {**books, 0: book(px_15 - 0.005, px_15 + 0.005)}))
    check(not any(b == 0 for _, _, b in instrs(r2)), "book 0 at -1.5bps does NOT stop out (SL=2.0)")
    px_25 = entry * (1 - 2.5e-4) if longp else entry * (1 + 2.5e-4)
    ts3 = 12 * NS + NS // 2
    a._step_ts_ns = ts3
    r3 = a.respond(state(ts3, {**books, 0: book(px_25 - 0.005, px_25 + 0.005)}))
    close_orders = [b for _, _, b in instrs(r3) if b == 0]
    check(len(close_orders) == 1, f"book 0 at -2.5bps stops out (close order emitted: {len(close_orders)})")

    print("\n=== T6: close fill -> RT recorded with net_pnl ===")
    cd = next(dr for dr, _, b in instrs(r3) if b == 0)
    fill(a, 0, cd, 0.3, px_25, REB, ts3)
    st0 = a._bstate(VAL, 0)
    flat = abs(a._book_positions(VAL)[0].qty) < 1e-6
    check(flat, "book 0 flat after close fill")
    check(len(st0.rt_events) == 1, f"one RT recorded for book 0 ({len(st0.rt_events)})")

    print("\n=== T7: constants (fee-based rework) ===")
    check(mod.MAX_GROSS_SL_BPS == 2.0, f"SL 2.0 ({mod.MAX_GROSS_SL_BPS})")
    check(mod.MIN_GROSS_TP_BPS == 2.5, f"TP 2.5 ({mod.MIN_GROSS_TP_BPS})")
    check(mod.MAX_GROSS_SL_BPS <= mod.MIN_GROSS_TP_BPS, "skew NON-NEGATIVE (SL <= TP)")
    check(mod.MAX_HOLD_S == 3.0, f"hold 3.0 ({mod.MAX_HOLD_S})")
    check(not hasattr(mod, "USE_KAPPA_OPEN_GATE") and not hasattr(TS2, "_kappa_open_ok"),
          "kappa-projection open-gate fully removed (gate is provably fee-only)")
    check(mod.KAPPA_MIN_REBATE_BPS == 2.0, f"fee-gate floor: open only at rebate >= 2bps ({mod.KAPPA_MIN_REBATE_BPS})")
    check(mod.SLEEP_WAKE_FEE_BPS == -2.0, f"fee wake bar -2bps ({mod.SLEEP_WAKE_FEE_BPS})")
    check(mod.MAX_SLEEP == 40, f"MAX_SLEEP budget cap = 40 ({mod.MAX_SLEEP})")
    for dead in ("OPEN_MIN_EDGE_BPS", "SLEEP_ENTER_BPS", "SLEEP_WAKE_BPS", "SLEEP_ANTIFLAP_S", "SLEEP_DWELL_S", "SLEEP_CAP"):
        check(not hasattr(mod, dead), f"dead constant removed: {dead}")
    check(not hasattr(TS2, "_estimate_rt_bps"), "spread-edge helper _estimate_rt_bps removed (spread no longer gates)")


if __name__ == "__main__":
    run()
    print(f"\n==== RESULT: {len(PASS)} passed, {len(FAIL)} failed ====")
    if FAIL:
        for m in FAIL:
            print("  FAILED:", m)
        sys.exit(1)
    print("ALL ASSERTIONS PASSED")
