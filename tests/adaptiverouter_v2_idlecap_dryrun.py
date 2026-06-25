"""
Regression dry-run for AdaptiveRouterV2Agent's IDLE HARD CAP — NO network/validator/deploy.

Guards the bug that slipped past three code reviews (2026-06-25): the idle hard cap was enforced on
the ROUTING path but the PnL-BACKOFF path was EXEMPT, so in a SUSTAINED bleeding regime cumulative
backoffs drove idle far past MAX_IDLE_BOOKS (live: 109/128) — past the 48 free-drop cliff that craters
the kappa median (the very thing the cap exists to prevent). The reviews missed it because they only
exercised a SINGLE-STEP routing scenario; none ran a sustained bleeding regime to watch idle accumulate.

This test runs that sustained bleeding regime and asserts idle stays <= MAX_IDLE_BOOKS across ALL paths.

Run:  python3 tests/adaptiverouter_v2_idlecap_dryrun.py
"""
import importlib.util, sys, types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("arv2", REPO / "agents" / "AdaptiveRouterV2Agent.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
A, NS, VAL, CAP = m.AdaptiveRouterV2Agent, m._NS, m.MAIN_VALIDATOR, m.MAX_IDLE_BOOKS
VD, PD = 4, 2
PASS, FAIL = [], []


def check(c, msg):
    (PASS if c else FAIL).append(msg)
    print(("  PASS " if c else "  FAIL ") + msg)


def ns(**k): return types.SimpleNamespace(**k)
def simcfg(n): return ns(simulation_id="dry", priceDecimals=PD, volumeDecimals=VD, miner_wealth=1e6, book_count=n, publish_interval=NS)
def acct(tk, mk): return ns(base_balance=ns(free=100.0, reserved=0.0), quote_balance=ns(free=1e6, reserved=0.0),
                            fees=ns(taker_fee_rate=tk, maker_fee_rate=mk), quote_loan=0.0, orders=[])
def book(spread_bps, mid=300.0, q=5.0):
    half = mid * spread_bps / 1e4 / 2.0
    return ns(bids=[ns(price=mid - half, quantity=q)], asks=[ns(price=mid + half, quantity=q)])
def state(ts, books, n): return ns(dendrite=ns(hotkey=VAL), timestamp=ts, config=simcfg(n), books=books)
def idle_count(a): return sum(1 for x in a.books_state[VAL] if a.books_state[VAL][x].mode == "idle")


def run_regime(n, tk, mk, spread_bps, bleeding, cap_on, steps=6):
    """n flat books; if bleeding, seed each with negative RTs so the PnL-backoff fires every step."""
    a = A(7, ns(lazy_load=1, idle_hard_cap=1.0 if cap_on else 0.0))
    a.simulation_config = simcfg(n); a._active_validator = VAL; a._sim_id[VAL] = "dry"; a._sync_precision(PD, VD)
    a.accounts = {i: acct(tk, mk) for i in range(n)}
    bks = {i: book(spread_bps) for i in range(n)}
    t = 10_000 * NS
    peak = 0
    for _ in range(steps):
        if bleeding:
            for i in range(n):
                a._bstate(VAL, i).rt_events = [(t - 50 * NS, -1.0)] * 6   # 6 losses in the 600s window
        a._step_ts_ns[VAL] = t
        a.respond(state(t, bks, n))
        peak = max(peak, idle_count(a))
        t += 30 * NS
    return peak


print(f"=== idle hard cap = {CAP} ===")

# 1) THE REGRESSION: sustained bleeding, all books maker-favorable (would route MAKER) but losing ->
#    PnL-backoff wants idle on all of them. With the cap ON, idle must NOT exceed CAP.
peak_on = run_regime(60, 0.0005, 0.0001, 40.0, bleeding=True, cap_on=True)
check(peak_on <= CAP, f"sustained backoff (cap ON): peak idle {peak_on} <= {CAP}")

# 2) NEGATIVE CONTROL: same regime with the cap OFF reproduces the overflow (proves the test bites).
peak_off = run_regime(60, 0.0005, 0.0001, 40.0, bleeding=True, cap_on=False)
check(peak_off > CAP, f"sustained backoff (cap OFF) reproduces overflow: peak idle {peak_off} > {CAP}")

# 3) ROUTING path still capped: many genuinely both-bad books (no rebate, no maker edge) -> idle, capped.
peak_route = run_regime(60, 0.0005, 0.0009, 2.0, bleeding=False, cap_on=True)
check(peak_route <= CAP, f"both-bad routing (cap ON): peak idle {peak_route} <= {CAP}")

# 4) Under the cap, idle is NOT forced up: few both-bad books stay all-idle (no spurious redirect).
peak_small = run_regime(20, 0.0005, 0.0009, 2.0, bleeding=False, cap_on=True)
check(peak_small == 20, f"under-cap both-bad: all {peak_small} idle (no over-trim)")

print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILED'} ({len(PASS)} passed)")
sys.exit(1 if FAIL else 0)
