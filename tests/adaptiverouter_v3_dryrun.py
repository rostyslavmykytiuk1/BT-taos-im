"""
Dry-run for AdaptiveRouterV3Agent — NO network / validator / deploy. Unit-tests the V3 additions over
V2 by calling the agent methods directly with mock state:

  [1] DETECTOR (_update_char): per-step mids -> SMOOTH / CHOP / DIRECTIONAL via Kaufman Efficiency Ratio
      + EWMA vol. Monotonic ramp (ER~1, net>=8bps) = DIRECTIONAL; big oscillation (ER~0) = CHOP; tiny
      noise = SMOOTH. EWMA vol_var grows with amplitude.
  [2] CHAR GATE (_route): a DIRECTIONAL book routes to PTAKER when a cross is ~+EV (rt_cost<=3bps) else
      IDLE; a SMOOTH book falls through to the normal fee routing (never forced to ptaker/idle by the gate).
  [3] PATIENT TAKER (_PatientTakerMode): flat+cheap -> market open on the bias side; held+old -> scale-out
      ONE clip near-touch IOC; held+young+bias-with -> accumulate ONE clip; deep underwater -> catastrophe
      close (whole side); within the per-book tick -> no action (throttle).
  [4] MAKER A-S + two-sided (_desired_quotes): holding -> continuous TWO-SIDED (reduce leg floored at
      breakeven + an Avellaneda-Stoikov-skewed ADD leg); the ADD leg is OFF on a DIRECTIONAL book and once
      soft inventory is reached; flat SMOOTH -> both sides at touch-inside, no skew (V2-equivalent).
  [5] REGIME-ADAPTIVE STOP (_managed_exit): a held lot cuts at 6bps on a DIRECTIONAL book (fast-small) but
      holds to 15bps never-cut on SMOOTH/CHOP.

Run:  python3 tests/adaptiverouter_v3_dryrun.py
"""

import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("arv3", REPO / "agents" / "AdaptiveRouterV3Agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

V3 = mod.AdaptiveRouterV3Agent
Inv = mod._Inv
OD = mod.OrderDirection
TIF = mod.TimeInForce
NS_PER_S = mod._NS
BUY, SELL = OD.BUY, OD.SELL
PASS, FAIL = [], []


def check(c, m):
    (PASS if c else FAIL).append(m)
    print(("  ok  " if c else " FAIL ") + m)


class Resp:
    def __init__(self):
        self.market, self.limit, self.cancels = [], [], []

    def market_order(self, **kw):
        self.market.append(kw)

    def limit_order(self, **kw):
        self.limit.append(kw)

    def cancel_orders(self, book_id, ids):
        self.cancels.append((book_id, list(ids)))


def acct(base_free=1e6, quote_free=1e9, maker_fee=0.0001, taker_fee=-0.0002, orders=None):
    return NS(
        base_balance=NS(free=base_free, reserved=0.0),
        quote_balance=NS(free=quote_free, reserved=0.0),
        fees=NS(maker_fee_rate=maker_fee, taker_fee_rate=taker_fee),
        quote_loan=0.0, orders=orders or [],
    )


def book_bias(buy=True, mid=300.0, spread_bps=2.0):
    """Orderbook whose microprice leans BUY (heavy bid) or SELL (heavy ask)."""
    half = mid * spread_bps / 1e4 / 2.0
    bid_q, ask_q = (10.0, 1.0) if buy else (1.0, 10.0)
    return NS(bids=[NS(price=mid - half, quantity=bid_q)],
              asks=[NS(price=mid + half, quantity=ask_q)])


def agent():
    a = object.__new__(V3)
    a.uid = 4242
    a.clip = mod.TARGET_CLIP            # 0.26
    a.exch_min = mod.EXCHANGE_MIN_ORDER_SIZE   # 0.25
    a._flat_eps = 0.5e-4
    a._price_decimals = 2
    a._volume_decimals = 4
    a._tick = 0.01
    a.inv, a.books_state, a._rt_log = {}, {}, {}
    a.rt_window_ns = int(570 * NS_PER_S)
    a.volume_assessment_ns = int(86400 * NS_PER_S)
    a.activity_deadline_ns = int(1500 * NS_PER_S)
    a.mk_quote_expiry_ns = int(12 * NS_PER_S)
    a.mk_walk_start_ns = int(20 * NS_PER_S)
    a.mk_giveup_ns = int(120 * NS_PER_S)
    a.mk_reentry_cooldown_ns = int(30 * NS_PER_S)
    a.mk_streak_cooldown_ns = int(600 * NS_PER_S)
    a.char_min_dwell_ns = int(mod.CHAR_MIN_DWELL_S * NS_PER_S)
    a.char_window_ns = int(mod.CHAR_WINDOW_S * NS_PER_S)
    a.char_sample_gap_ns = int(mod.CHAR_SAMPLE_GAP_S * NS_PER_S)
    a.pt_tick_ns = int(mod.PT_TICK_S * NS_PER_S)
    a.pt_min_hold_ns = int(mod.PT_MIN_HOLD_S * NS_PER_S)
    return a


GAP = int(mod.CHAR_SAMPLE_GAP_S * NS_PER_S)   # detector sub-sample cadence; space test mids by this so each records


def feed(a, st, prices, t0=GAP):
    """Feed a price series to the detector, one per sub-sample gap (so each mid is recorded)."""
    t = t0
    for px in prices:
        a._update_char(st, px, t)
        t += GAP
    return t


def st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER, spread_ema_bps=1.0, vol_var=0.0, now=0):
    st = a._bstate("test", 1)
    st.char = char
    st.mode = mode
    st.spread_ema_bps = spread_ema_bps
    st.vol_var = vol_var
    st.last_rt_ns = now            # recent RT so activity never fires inside these unit tests
    st.seen_ns = now
    return st


def long_inv(px, qty, ts):
    inv = Inv()
    inv.longs.append((ts, qty, px, 0.0))
    return inv


def short_inv(px, qty, ts):
    inv = Inv()
    inv.shorts.append((ts, qty, px, 0.0))
    return inv


# =====================================================================================
print("\n[1] DETECTOR — _update_char classifies SMOOTH / CHOP / DIRECTIONAL")

NW = mod.CHAR_MIN_SAMPLES + 3               # enough sub-samples to fill + classify
base = 300.0

# 1a. DIRECTIONAL: monotonic ramp +24bps over the window (ER ~ 1.0, net >= 6bps, range >= 2.5x spread)
a = agent()
st = st_of(a)
st.spread_ema_bps = 0.5
feed(a, st, [base * (1.0 + 24e-4 * i / NW) for i in range(NW)])
check(st.char == mod.CHAR_DIRECTIONAL, f"monotonic ramp -> DIRECTIONAL (got {st.char})")
check(st.vol_var > 0.0, "EWMA vol_var populated on a moving book")

# 1b. CHOP: large zig-zag, near-zero net (ER ~ 0), big range
a = agent()
st = st_of(a)
st.spread_ema_bps = 0.5
feed(a, st, [base * (1.0 + (10e-4 if i % 2 == 0 else -10e-4)) for i in range(NW + 2)])
check(st.char == mod.CHAR_CHOP, f"big oscillation, ~0 net -> CHOP (got {st.char})")

# 1c. SMOOTH: tiny sub-spread noise
a = agent()
st = st_of(a)
st.spread_ema_bps = 5.0
feed(a, st, [base * (1.0 + (0.3e-4 if i % 2 == 0 else -0.3e-4)) for i in range(NW + 2)])
check(st.char == mod.CHAR_SMOOTH, f"tiny noise -> SMOOTH (got {st.char})")

# 1d. Warmup stays SMOOTH (fewer than CHAR_MIN_SAMPLES recorded)
a = agent()
st = st_of(a)
feed(a, st, [base * (1.0 + 12e-4 * i) for i in range(mod.CHAR_MIN_SAMPLES - 2)])
check(st.char == mod.CHAR_SMOOTH, "warmup (< CHAR_MIN_SAMPLES samples) stays SMOOTH home")

# 1e. Hysteresis: once DIRECTIONAL, a single reverting tick inside the dwell does NOT immediately flip
a = agent()
st = st_of(a)
st.spread_ema_bps = 0.5
t = feed(a, st, [base * (1.0 + 24e-4 * i / NW) for i in range(NW)])
assert st.char == mod.CHAR_DIRECTIONAL
a._update_char(st, base, t)   # one revert tick, still within min-dwell
check(st.char == mod.CHAR_DIRECTIONAL, "DIRECTIONAL holds through a single revert within min-dwell")

# 1f. ENTRY CONFIRMATION: a single is_dir sample must NOT latch DIRECTIONAL (review #8 spike guard)
a = agent()
st = st_of(a)
st.spread_ema_bps = 0.5
t = feed(a, st, [base * (1.0 + (0.3e-4 if i % 2 == 0 else -0.3e-4)) for i in range(mod.CHAR_MIN_SAMPLES + 1)])
a._update_char(st, base * (1.0 + 12e-4), t); t += GAP          # first directional sample
check(st.char != mod.CHAR_DIRECTIONAL and st.dir_streak == 1, "1 is_dir sample -> NOT latched (dir_streak=1)")
a._update_char(st, base * (1.0 + 24e-4), t)                    # second consecutive directional sample
check(st.char == mod.CHAR_DIRECTIONAL and st.dir_streak >= 2, "2nd consecutive is_dir sample -> latches DIRECTIONAL")


# =====================================================================================
print("\n[2] CHAR GATE — _route diverts DIRECTIONAL, leaves SMOOTH on the fee router")

# 2a. DIRECTIONAL + cheap cross (rebate covers spread) -> PTAKER
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER, spread_ema_bps=1.0)
ac = acct(taker_fee=-0.0002)   # rebate 2bps ; rt_cost = 2*1 - 2*2 = -2 <= 3
want = a._route(st, ac, 299.97, 300.03, 300.0)
check(want == mod.MODE_PTAKER, f"DIRECTIONAL + cheap cross -> PTAKER (got {want})")

# 2b. DIRECTIONAL + expensive cross (wide spread, no rebate) -> IDLE
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER, spread_ema_bps=5.0)
ac = acct(taker_fee=0.0)       # rt_cost = 2*5 - 0 = 10 > 3
want = a._route(st, ac, 299.85, 300.15, 300.0)
check(want == mod.MODE_IDLE, f"DIRECTIONAL + expensive cross -> IDLE (got {want})")

# 2c. SMOOTH is NOT forced by the gate (returns a normal fee-routed mode, not idle-by-gate)
a = agent()
st = st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER, spread_ema_bps=5.0)
ac = acct(taker_fee=0.0, maker_fee=0.0001)   # wide spread -> maker edge positive
want = a._route(st, ac, 299.85, 300.15, 300.0)
check(want in (mod.MODE_MAKER, mod.MODE_TAKER, mod.MODE_IDLE) and want == mod.MODE_MAKER,
      f"SMOOTH wide-spread falls through to maker (got {want})")

# 2d. flat PTAKER on a STILL-directional book is re-gated (NOT fee-routed back to maker): cheap -> stay
#     PTAKER, expensive -> IDLE. Never falls through to a passive maker lot in a trend.
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER, spread_ema_bps=1.0)
want = a._route(st, acct(taker_fee=-0.0002), 299.97, 300.03, 300.0)
check(want == mod.MODE_PTAKER, f"flat PTAKER + directional + cheap -> stays PTAKER (got {want})")
a2 = agent()
st2 = st_of(a2, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER, spread_ema_bps=5.0)
want2 = a2._route(st2, acct(taker_fee=0.0), 299.85, 300.15, 300.0)
check(want2 == mod.MODE_IDLE, f"flat PTAKER + directional + expensive -> IDLE not maker (got {want2})")

# 2e. CLIFF (idle overflow): a directional book with an expensive cross falls through to a RESTING MAKER
#     (cliff relaxes maker-edge->0; the maker scale-out-cuts at 6bps) NOT idle, so a market-wide trend can't
#     blow the 48-book idle budget (review #2)
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER, spread_ema_bps=5.0)
want_noclip = a._route(st, acct(taker_fee=0.0, maker_fee=0.0001), 299.85, 300.15, 300.0)   # not at cliff -> IDLE
st2 = st_of(agent(), char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER, spread_ema_bps=5.0)
want_cliff = a._route(st2, acct(taker_fee=0.0, maker_fee=0.0001), 299.85, 300.15, 300.0, cliff=True)
check(want_noclip == mod.MODE_IDLE and want_cliff == mod.MODE_MAKER,
      f"directional+expensive: IDLE normally ({want_noclip}), resting MAKER at cliff ({want_cliff})")


# =====================================================================================
print("\n[3] PATIENT TAKER — open / scale-out / accumulate / catastrophe / throttle")
pt = mod._PatientTakerMode()
NOW = 10_000 * NS_PER_S

# 3a. flat + cheap -> market OPEN on the bias side (BUY when microprice leans up)
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER)
st.pt_last_act_ns = 0
r = Resp()
bk = book_bias(buy=True)
pt.step(a, r, "test", 1, bk, acct(), st, Inv(), 0.0, 299.97, 300.03, 300.0, 4, 1e15, NOW)
check(len(r.market) == 1 and r.market[0]["direction"] == BUY, "flat -> 1 market BUY (bias-up open)")
check(st.pt_last_act_ns == NOW, "pt_last_act_ns stamped after the open")

# 3b. throttle: within the per-book tick -> NO action
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER)
st.pt_last_act_ns = NOW - a.pt_tick_ns // 2     # acted recently
r = Resp()
pt.step(a, r, "test", 1, book_bias(True), acct(), st, Inv(), 0.0, 299.97, 300.03, 300.0, 4, 1e15, NOW)
check(not r.market and not r.limit, "within pt_tick -> throttled (no orders)")

# 3c. held LONG, OLD lot -> scale-out ONE clip near-touch IOC SELL
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER)
st.pt_last_act_ns = 0
held = 0.52   # 2 clips
inv = long_inv(300.0, held, NOW - 3 * a.pt_min_hold_ns)   # old
r = Resp()
pt.step(a, r, "test", 1, book_bias(True), acct(), st, inv, held, 299.99, 300.03, 300.0, 4, 1e15, NOW)
check(len(r.limit) == 1 and r.limit[0]["direction"] == SELL and r.limit[0].get("timeInForce") == TIF.IOC,
      "held+old long -> IOC SELL reduce")
check(abs(r.limit[0]["quantity"] - a.clip) < 1e-9, f"reduce is ONE clip ({r.limit[0]['quantity']})")

# 3d. held LONG, YOUNG + bias-with + IN PROFIT -> press (pyramid a winner) ONE clip market BUY
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER)
st.pt_last_act_ns = 0
inv = long_inv(300.0, a.clip, NOW - a.pt_min_hold_ns // 4)   # young, 1 lot, entry 300.0
r = Resp()   # price ABOVE entry -> in profit -> press
pt.step(a, r, "test", 1, book_bias(buy=True, mid=300.20), acct(), st, inv, a.clip, 300.18, 300.22, 300.20, 4, 1e15, NOW)
check(len(r.market) == 1 and r.market[0]["direction"] == BUY, "held+young+bias-up+PROFIT -> press market BUY")

# 3d2. held LONG, YOUNG + bias-with but UNDERWATER -> do NOT average down (hold, no order)
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER)
st.pt_last_act_ns = 0
inv = long_inv(300.0, a.clip, NOW - a.pt_min_hold_ns // 4)   # young, 1 lot, entry 300.0
r = Resp()   # price BELOW entry (3bps uw, < catastrophe), young, bias-up -> neither reduce nor press
pt.step(a, r, "test", 1, book_bias(buy=True, mid=299.91), acct(), st, inv, a.clip, 299.90, 299.92, 299.91, 4, 1e15, NOW)
check(not r.market and not r.limit, "held+young+bias-up+UNDERWATER -> hold (no average-down press)")

# 3e. held LONG, YOUNG but bias AGAINST -> reduce (bleed WITH the move), not accumulate
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER)
st.pt_last_act_ns = 0
inv = long_inv(300.0, a.clip, NOW - a.pt_min_hold_ns // 4)   # young, 1 lot
r = Resp()
pt.step(a, r, "test", 1, book_bias(buy=False), acct(), st, inv, a.clip, 299.99, 300.03, 300.0, 4, 1e15, NOW)
check(not r.market and len(r.limit) == 1 and r.limit[0]["direction"] == SELL,
      "held+young+bias-DOWN long -> reduce (no accumulate)")

# 3f. deep underwater -> CATASTROPHE close the WHOLE side, one IOC
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_PTAKER)
st.pt_last_act_ns = 0
held = a.clip            # 1 lot -> catastrophe threshold = 30bps
bb = 300.0 * (1 - 40e-4)   # 40bps underwater
inv = long_inv(300.0, held, NOW - a.pt_min_hold_ns // 4)   # young, so only catastrophe (not age) fires
r = Resp()
pt.step(a, r, "test", 1, book_bias(True), acct(), st, inv, held, bb, bb + 0.02, bb + 0.01, 4, 1e15, NOW)
check(len(r.limit) == 1 and r.limit[0]["direction"] == SELL
      and abs(r.limit[0]["quantity"] - held) < 1e-9,
      "deep underwater -> catastrophe IOC SELL of the WHOLE side")


# =====================================================================================
print("\n[4] MAKER — continuous two-sided + Avellaneda-Stoikov inventory skew")
mk = mod._MakerMode()
NOW = 5_000 * NS_PER_S


def bid_inside_of(a, bb, ba):
    spread = ba - bb
    improve = a._tick if spread > 2 * a._tick else 0.0
    bi = round(bb + improve, a._price_decimals)
    ai = round(ba - improve, a._price_decimals)
    return (bb, ba) if bi >= ai else (bi, ai)


# 4a. flat SMOOTH -> both sides, NO skew (V2-equivalent)
a = agent()
st = st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER, vol_var=0.0)
d = mk._desired_quotes(a, "test", 1, acct(), Inv(), 0.0, 299.90, 300.10, 300.0, 1e15, NOW, 4, st)
check(BUY in d and SELL in d, "flat SMOOTH -> quotes BOTH sides")
bi, ai = bid_inside_of(a, 299.90, 300.10)
check(abs(d[BUY][0] - bi) < 1e-9 and abs(d[SELL][0] - ai) < 1e-9, "flat -> at touch-inside (no skew)")

# 4b. holding LONG, SMOOTH, inv<soft -> SELL reduce (>= entry) + skewed ADD BUY (< bid_inside)
a = agent()
st = st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER, vol_var=(8e-4) ** 2)   # ~8bps/step vol
inv = long_inv(300.0, a.clip, NOW - a.mk_walk_start_ns // 2)   # 1 lot, young
d = mk._desired_quotes(a, "test", 1, acct(), inv, a.clip, 299.90, 300.10, 300.0, 1e15, NOW, 4, st)
check(SELL in d and d[SELL][0] >= 300.0, f"holding long -> SELL reduce floored at/above entry ({d.get(SELL)})")
check(BUY in d, "holding long SMOOTH inv<soft -> continuous ADD (BUY) leg present")
bi, _ = bid_inside_of(a, 299.90, 300.10)
check(BUY in d and d[BUY][0] < bi, f"A-S skew: ADD BUY below bid_inside {bi} (got {d.get(BUY)})")

# 4c. holding LONG, DIRECTIONAL -> reduce ONLY, no ADD into the trend
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER, vol_var=(8e-4) ** 2)
inv = long_inv(300.0, a.clip, NOW - a.mk_walk_start_ns // 2)
d = mk._desired_quotes(a, "test", 1, acct(), inv, a.clip, 299.90, 300.10, 300.0, 1e15, NOW, 4, st)
check(SELL in d and BUY not in d, "holding long DIRECTIONAL -> reduce SELL only, no ADD")

# 4d. holding LONG over soft inventory (2 lots >= 1.5) -> no ADD even when SMOOTH
a = agent()
st = st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER, vol_var=(8e-4) ** 2)
held = 2 * a.clip
inv = long_inv(300.0, held, NOW - a.mk_walk_start_ns // 2)
d = mk._desired_quotes(a, "test", 1, acct(), inv, held, 299.90, 300.10, 300.0, 1e15, NOW, 4, st)
check(SELL in d and BUY not in d, "holding long >= soft inventory -> reduce only, no ADD")

# 4e. holding SHORT, SMOOTH -> BUY reduce (<= entry) + skewed ADD SELL (> ask_inside)
a = agent()
st = st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER, vol_var=(8e-4) ** 2)
inv = short_inv(300.0, a.clip, NOW - a.mk_walk_start_ns // 2)
d = mk._desired_quotes(a, "test", 1, acct(), inv, -a.clip, 299.90, 300.10, 300.0, 1e15, NOW, 4, st)
check(BUY in d and d[BUY][0] <= 300.0, f"holding short -> BUY reduce floored at/below entry ({d.get(BUY)})")
_, ai = bid_inside_of(a, 299.90, 300.10)
check(SELL in d and d[SELL][0] > ai, f"A-S skew: ADD SELL above ask_inside {ai} (got {d.get(SELL)})")

# 4f. flat DIRECTIONAL -> do NOT open passive liquidity into a trend
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER, vol_var=0.0)
d = mk._desired_quotes(a, "test", 1, acct(), Inv(), 0.0, 299.90, 300.10, 300.0, 1e15, NOW, 4, st)
check(not d, "flat DIRECTIONAL maker -> no fresh quotes (don't post into a trend)")


# =====================================================================================
print("\n[5] PURE NEVER-CUT STOP — _managed_exit realizes a loss ONLY at 15bps, WHOLE-SIDE, ALL regimes")
NOW = 7_000 * NS_PER_S

# 5a. long 8bps underwater, SMOOTH -> NO cut (8 < 15 never-cut), holds for revert
a = agent()
st = st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER)
bb = 300.0 * (1 - 8e-4)
inv = long_inv(300.0, a.clip, NOW - a.mk_walk_start_ns)
r = Resp()
cut = mk._managed_exit(a, r, "test", 1, acct(), inv, a.clip, bb, bb + 0.02, 4)
check(cut is False and not r.limit, "long 8bps uw + SMOOTH -> HELD (never-cut 15bps)")

# 5b. long 8bps underwater, DIRECTIONAL -> STILL HELD (no more 6bps tighten; never-cut regardless of char)
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER)
bb = 300.0 * (1 - 8e-4)
inv = long_inv(300.0, a.clip, NOW - a.mk_walk_start_ns)
r = Resp()
cut = mk._managed_exit(a, r, "test", 1, acct(), inv, a.clip, bb, bb + 0.02, 4)
check(cut is False and not r.limit, "long 8bps uw + DIRECTIONAL -> HELD (never-cut, no regime tighten)")

# 5c. long 8bps underwater, CHOP -> STILL HELD (8 < 15, regardless of char)
a = agent()
st = st_of(a, char=mod.CHAR_CHOP, mode=mod.MODE_MAKER)
bb = 300.0 * (1 - 8e-4)
inv = long_inv(300.0, a.clip, NOW - a.mk_walk_start_ns)
r = Resp()
cut = mk._managed_exit(a, r, "test", 1, acct(), inv, a.clip, bb, bb + 0.02, 4)
check(cut is False and not r.limit, "long 8bps uw + CHOP -> HELD (never-cut 15bps)")

# 5d. long 16bps underwater, DIRECTIONAL -> CUT at 15bps WHOLE side (the one stop fires regardless of char)
a = agent()
st = st_of(a, char=mod.CHAR_DIRECTIONAL, mode=mod.MODE_MAKER)
held = 3 * a.clip
bb = 300.0 * (1 - 16e-4)
inv = long_inv(300.0, held, NOW - a.mk_walk_start_ns)
r = Resp()
cut = mk._managed_exit(a, r, "test", 1, acct(), inv, held, bb, bb + 0.02, 4)
check(cut is True and len(r.limit) == 1 and r.limit[0]["direction"] == SELL
      and r.limit[0].get("timeInForce") == TIF.IOC and abs(r.limit[0]["quantity"] - held) < 1e-9,
      f"long 16bps uw -> 15bps WHOLE-side IOC SELL regardless of char ({r.limit[0]['quantity'] if r.limit else None})")

# 5e. short 16bps underwater, SMOOTH -> CUT at 15bps WHOLE side (IOC BUY)
a = agent()
st = st_of(a, char=mod.CHAR_SMOOTH, mode=mod.MODE_MAKER)
ba = 300.0 * (1 + 16e-4)
inv = short_inv(300.0, held, NOW - a.mk_walk_start_ns)
r = Resp()
cut = mk._managed_exit(a, r, "test", 1, acct(), inv, -held, ba - 0.02, ba, 4)
check(cut is True and len(r.limit) == 1 and r.limit[0]["direction"] == BUY
      and abs(r.limit[0]["quantity"] - held) < 1e-9,
      f"short 16bps uw -> 15bps WHOLE-side IOC BUY ({r.limit[0]['quantity'] if r.limit else None})")


# =====================================================================================
print("\n[6] PERF — vol_log running sum (O(1) rolled volume, prune drops aged front)")
a = agent()
a.volume_assessment_ns = int(100 * NS_PER_S)   # 100s window for the test
a._record_trade_volume("test", 1, 1.0, 300.0, 10 * NS_PER_S)   # vol 300 @ t=10s
a._record_trade_volume("test", 1, 2.0, 300.0, 50 * NS_PER_S)   # vol 600 @ t=50s
st = a._bstate("test", 1)
check(abs(st.vol_sum - 900.0) < 1e-6, f"running sum tracks appends ({st.vol_sum})")
rolled = a._rolled_quote_volume("test", 1, 60 * NS_PER_S)       # both still in window
check(abs(rolled - 900.0) < 1e-6, f"rolled volume = sum in-window ({rolled})")
rolled = a._rolled_quote_volume("test", 1, 120 * NS_PER_S)      # t=120: cutoff=20s -> first entry (t=10) drops
check(abs(rolled - 600.0) < 1e-6 and len(st.vol_log) == 1, f"aged front pruned, sum decremented ({rolled})")
rolled = a._rolled_quote_volume("test", 1, 200 * NS_PER_S)      # cutoff=100s -> all drop
check(abs(rolled - 0.0) < 1e-6 and st.vol_sum == 0.0, f"window empty -> sum reset to 0 ({rolled})")


# =====================================================================================
print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    for m in FAIL:
        print("  FAIL " + m)
    sys.exit(1)
print("ALL GREEN")
