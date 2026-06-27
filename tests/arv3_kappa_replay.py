"""
§10.1 offline KAPPA REPLAY — the go/no-go number the reviews asked for.

Question the detector-classification test (arv3_detector_realcsv.py) CANNOT answer: does V3's regime-adaptive
CUT policy (6bps fast-small scale-out on a DIRECTIONAL book; clip-scale-out on CHOP) actually RAISE the
per-book Sortino-3 kappa the validator scores, versus V2 (never-cut, whole-side 15bps stop)? This adjudicates
the open "cut-rate guard" debate: if V3 kappa <= V2 on the residual false-positives, the 6bps cut is hurting
and needs the guard; if V3 > V2, the cut is net-positive as-is.

Method: run a textbook inventory market-maker on the REAL winning-maker price paths (dashboard_data CSVs).
BOTH policies share the identical fill model / spread / inventory cap — the ONLY difference is the cut logic
(V2 vs V3, the latter driven by the REAL _update_char detector). Each produces a per-(ts,book) realized-PnL
stream, scored by the REAL validator kappa_3 (taos/im/utils/kappa.py). We compare the median per-book kappa.

HONEST LIMITS: the fill model is a simplified mid-cross MM (no order-book depth / queue / other agents), and
fills come from a fill-price proxy of the mid. So treat the ABSOLUTE kappa as indicative, and the RELATIVE
V2-vs-V3 delta (identical environment, only the cut differs) as the signal. The true go/no-go remains the live
A/B canary; this is the offline pre-check.

Run:  python3 tests/arv3_kappa_replay.py
"""
import csv
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("arv3", REPO / "agents" / "AdaptiveRouterV3Agent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from taos.im.utils.kappa import kappa_3   # noqa: E402  (real validator scorer)
NS = mod._NS

DIR_STOP_BPS = 6.0         # the experimental tighter-directional stop (REMOVED from the shipped agent after this
                           # replay showed it loses to pure never-cut-15 in every regime — defined here so this
                           # decision artifact stays reproducible)
CLIP = 1.0                 # 1 lot (kappa is scale-free; keep it 1 for clarity)
MAX_LOTS = 2              # mirrors MK_MAX_INVENTORY_LOTS hard cap
MAKER_FEE_BPS = 7.0       # per leg (live maker-pays regime ~+5.5-9bps); same for both policies
STEP_S = 1.0             # true publish cadence (1 sim-second)
KCFG = dict(tau=0.0, lookback=10800, norm_min=-1.0, norm_max=1.0, min_lookback=0,
            min_realized_observations=mod.RT_MAX and 3, grace_period=0, deregistered_uids=[])


def parse_t(s):
    hh, mm, ss = s.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def load_books(path):
    books = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if not r.get("time") or not r.get("price") or not r.get("book"):
                continue
            try:
                t = parse_t(r["time"]); px = float(r["price"]); b = int(r["book"])
            except (ValueError, AttributeError):
                continue
            if px > 0:
                books.setdefault(b, []).append((t, px))
    for b in books:
        books[b].sort()
    return books


def resample(series, step_s):
    if len(series) < 2:
        return []
    t0, t1 = series[0][0], series[-1][0]
    if t1 <= t0:
        return []
    out, j, n, t = [], 0, len(series), series[0][0]
    while t <= t1:
        while j + 1 < n and series[j + 1][0] <= t:
            j += 1
        out.append(series[j][1])
        t += step_s
    return out


def half_spread_frac(series):
    rets = [abs(series[i][1] - series[i - 1][1]) / series[i - 1][1]
            for i in range(1, len(series)) if series[i - 1][1] > 0]
    if not rets:
        return mod.SPREAD_FLOOR_BPS / 2 / 1e4
    rets.sort()
    return max(rets[len(rets) // 2] / 2.0, mod.SPREAD_FLOOR_BPS / 2 / 1e4)


def sim_book(mids, h, policy, book_id):
    """Inventory MM on a mid path. policy in {'v2','v3'}. Returns {ts_ns: realized_pnl} (closes only)."""
    a = object.__new__(mod.AdaptiveRouterV3Agent)
    a.char_min_dwell_ns = int(mod.CHAR_MIN_DWELL_S * NS)
    a.char_window_ns = int(mod.CHAR_WINDOW_S * NS)
    a.char_sample_gap_ns = int(mod.CHAR_SAMPLE_GAP_S * NS)
    st = mod._BookState()
    st.spread_ema_bps = h * 1e4   # half-spread in bps (drives the detector rvs denominator)
    fee = MAKER_FEE_BPS / 1e4
    longs = []   # FIFO list of entry prices (each 1 clip)
    shorts = []
    realized = {}
    now = 0
    prev_px = mids[0]

    def record(pnl):
        realized[now] = realized.get(now, 0.0) + pnl

    for px in mids:
        if policy in ("v3", "scaleout", "tighten"):
            a._update_char(st, px, now)
        bid = prev_px * (1.0 - h)     # quotes posted last step (around the PREVIOUS mid)
        ask = prev_px * (1.0 + h)
        prev_px = px
        # ---- passive fills: the NEW mid crosses last step's resting quote ----
        # ask lifted (price rose through our ask): reduce a long at breakeven+ (never-cut), else open a short
        if px >= ask:
            if longs and ask >= longs[0]:                 # reduce oldest long at >= entry
                entry = longs.pop(0)
                record((ask - entry) - fee * (entry + ask))
            elif not longs and len(shorts) < MAX_LOTS:    # open short
                shorts.append(ask)
        # bid hit (price fell through our bid): reduce a short at breakeven-, else open a long
        if px <= bid:
            if shorts and bid <= shorts[0]:               # reduce oldest short at <= entry
                entry = shorts.pop(0)
                record((entry - bid) - fee * (entry + bid))
            elif not shorts and len(longs) < MAX_LOTS:    # open long
                longs.append(bid)
        # ---- cut policy (the ONLY difference between variants) ----
        # policy: 'v2'        = never-cut 15bps WHOLE side (no detector)
        #         'v3'        = 6bps tighten on DIRECTIONAL + scale-out one clip when char!=SMOOTH
        #         'scaleout'  = 15bps always, but scale-out one clip when char!=SMOOTH (NO 6bps tighten)
        #         'tighten'   = 6bps tighten on DIRECTIONAL, WHOLE side (NO scale-out)
        use_det = policy in ("v3", "scaleout", "tighten")
        if use_det:
            pass  # _update_char already called above for these
        directional = (policy in ("v3", "tighten") and st.char == mod.CHAR_DIRECTIONAL)
        scale_out = (policy in ("v3", "scaleout") and st.char != mod.CHAR_SMOOTH)
        stop = (DIR_STOP_BPS if directional else mod.MK_STOP_LOSS_BPS) / 1e4
        if longs:
            uw = (longs[0] - px) / longs[0]
            if uw >= stop:
                n_cut = 1 if scale_out else len(longs)
                for _ in range(n_cut):
                    if not longs:
                        break
                    entry = longs.pop(0)
                    record((px - entry) - fee * (entry + px))   # realize the loss at market
        if shorts:
            uw = (px - shorts[0]) / shorts[0]
            if uw >= stop:
                n_cut = 1 if scale_out else len(shorts)
                for _ in range(n_cut):
                    if not shorts:
                        break
                    entry = shorts.pop(0)
                    record((entry - px) - fee * (entry + px))
        now += int(STEP_S * NS)
    return realized


POLICIES = ["v2", "v3", "scaleout", "tighten"]


def run(label, path):
    books = load_books(REPO / "dashboard_data" / path)
    streams = {p: {} for p in POLICIES}
    bmax = 0
    for b, series in books.items():
        mids = resample(series, STEP_S)
        if len(mids) < 200:
            continue
        h = half_spread_frac(series)
        bmax = max(bmax, b)
        for p in POLICIES:
            for ts, pnl in sim_book(mids, h, p, b).items():
                streams[p].setdefault(ts, {})[b] = pnl
    bc = bmax + 1
    res = {}
    for i, p in enumerate(POLICIES):
        k = kappa_3(10 + i, streams[p], book_count=bc, **KCFG)
        res[p] = k.get("median") if k else None
    base = res["v2"]
    parts = []
    for p in POLICIES:
        m = res[p]
        d = "" if (p == "v2" or m is None or base is None) else f" (Δ{m-base:+.4f})"
        parts.append(f"{p}={m:+.4f}{d}" if m is not None else f"{p}=n/a")
    print(f"  {label}: " + "  ".join(parts))


print("=== §10.1 KAPPA REPLAY — decompose V3's held-lot cut policy (real kappa_3, identical fills) ===")
print("   v2=never-cut15-whole | v3=6bps-dir+scaleout | scaleout=15bps-1clip-only | tighten=6bps-dir-whole\n")
print(" WINNING-MAKER (mean-reverting) price paths — never-cut SHOULD win here:")
run("uid60 ", "60_maker_38.127.44.98.csv")
run("uid84 ", "84_maker_38.127.44.98.csv")
run("uid145", "145_maker_38.127.44.98.csv")
print("\n TREND-HEAVY (directional) price paths — the 6bps tighten SHOULD pay off here if anywhere:")
run("uid184", "184_taker_38.127.44.98.csv")
run("uid215", "215_taker_38.127.44.98.csv")
print("\n NOTE: relative deltas vs v2 are the signal; absolute kappa is indicative (simplified fill model).")
print(" Maker set = mean-reverting (never-cut's home); taker set = trend-heavy (tighten's home).")
print(" Decision rule: keep the 6bps tighten only if tighten >~ v2 on the trend set AND ~= v2 on the maker set.")
