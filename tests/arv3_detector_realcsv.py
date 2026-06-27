"""
§10.2 validation (the check the external feedback rightly asked for): replay the REAL winning-agent trade
CSVs through the ACTUAL V3 detector (_update_char) and report how each book CLASSIFIES.

Central question: does the detector flag the WINNING MAKERS' books (uid60/84/145 — the must-NOT-divert books)
as DIRECTIONAL? If a large fraction of their book-time is DIRECTIONAL, the router would wrongly divert the
exact books where the winners make money (the regression the plan must avoid). If they classify mostly
SMOOTH/CHOP, the detector is sound and keeps making there.

Caveats (this is a proxy, not the live mid feed): the CSV is a FILL stream (prices ~ mid ± half-spread bounce,
sampled at irregular fill times). We forward-fill to a uniform time grid to mimic the agent's per-publish mid
sampling, and estimate each book's half-spread from the fill-to-fill bounce. Bid-ask bounce in fills ADDS path
length => LOWERS the Efficiency Ratio => makes the proxy if anything UNDER-state DIRECTIONAL vs true mids; read
the numbers as an upper-confidence that winners are not over-diverted, not a precise rate.

Run:  python3 tests/arv3_detector_realcsv.py
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
NS = mod._NS


def agent():
    a = object.__new__(mod.AdaptiveRouterV3Agent)
    a.char_min_dwell_ns = int(mod.CHAR_MIN_DWELL_S * NS)
    return a


def parse_t(s):
    # "20:10:17.095" -> seconds of day (float)
    hh, mm, ss = s.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def load_book_series(path):
    """book_id -> list[(t_seconds, price)] sorted by time."""
    books = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if not r.get("time") or not r.get("price") or not r.get("book"):
                continue
            try:
                t = parse_t(r["time"]); px = float(r["price"]); b = r["book"]
            except (ValueError, KeyError, AttributeError):
                continue
            if px > 0:
                books.setdefault(b, []).append((t, px))
    for b in books:
        books[b].sort()
    return books


def est_half_spread_bps(series):
    """Estimate full-spread from median |consecutive fill return|; half it. Floored at SPREAD_FLOOR/2."""
    if len(series) < 3:
        return mod.SPREAD_FLOOR_BPS / 2.0
    rets = []
    for i in range(1, len(series)):
        p0, p1 = series[i - 1][1], series[i][1]
        if p0 > 0:
            rets.append(abs(p1 - p0) / p0 * 1e4)
    if not rets:
        return mod.SPREAD_FLOOR_BPS / 2.0
    rets.sort()
    full = rets[len(rets) // 2]            # median consecutive-fill move ~ full spread (bounce)
    return max(full / 2.0, mod.SPREAD_FLOOR_BPS / 2.0)


def resample(series, step_s):
    """Forward-fill onto a uniform step_s grid -> list of mids (one per step)."""
    if len(series) < 2:
        return []
    t0, t1 = series[0][0], series[-1][0]
    if t1 <= t0:
        return []
    mids, j, n = [], 0, len(series)
    t = t0
    while t <= t1:
        while j + 1 < n and series[j + 1][0] <= t:
            j += 1
        mids.append(series[j][1])
        t += step_s
    return mids


def classify_book(series, step_s, half_spread_override=None):
    """Replay at the publish cadence (step_s); the detector sub-samples internally (CHAR_SAMPLE_GAP_S) and uses
    a CHAR_WINDOW_S time window. Return (n_scored_steps, frac_dir, frac_chop, frac_smooth)."""
    mids = resample(series, step_s)
    if len(mids) <= mod.CHAR_MIN_SAMPLES + 2:
        return None
    a = agent()
    a.char_window_ns = int(mod.CHAR_WINDOW_S * NS)
    a.char_sample_gap_ns = int(mod.CHAR_SAMPLE_GAP_S * NS)
    st = mod._BookState()
    st.spread_ema_bps = (est_half_spread_bps(series) if half_spread_override is None else half_spread_override)
    counts = {mod.CHAR_SMOOTH: 0, mod.CHAR_CHOP: 0, mod.CHAR_DIRECTIONAL: 0}
    now = 0
    scored = 0
    for px in mids:
        a._update_char(st, px, now)
        now += int(step_s * NS)
        if len(st.mid_hist) >= mod.CHAR_MIN_SAMPLES:   # only count once the window can classify
            counts[st.char] += 1
            scored += 1
    if scored == 0:
        return None
    return scored, counts[mod.CHAR_DIRECTIONAL] / scored, counts[mod.CHAR_CHOP] / scored, counts[mod.CHAR_SMOOTH] / scored


def run_file(label, path, step_s):
    p = REPO / "dashboard_data" / path
    if not p.exists():
        print(f"  (missing {path})")
        return
    books = load_book_series(str(p))
    rows = []
    for b, series in books.items():
        r = classify_book(series, step_s)
        if r:
            rows.append((b, *r))
    if not rows:
        print(f"  {label}: no scorable books")
        return
    nb = len(rows)
    # book-time-weighted aggregate fractions
    tot = sum(r[1] for r in rows)
    wdir = sum(r[1] * r[2] for r in rows) / tot
    wchop = sum(r[1] * r[3] for r in rows) / tot
    wsmooth = sum(r[1] * r[4] for r in rows) / tot
    # books whose MAJORITY of time is DIRECTIONAL (these would be diverted away from maker)
    maj_dir = sum(1 for r in rows if r[2] > 0.5)
    any_dir = sum(1 for r in rows if r[2] > 0.10)
    print(f"  {label}: {nb} books | book-time DIR={wdir*100:.1f}% CHOP={wchop*100:.1f}% SMOOTH={wsmooth*100:.1f}% "
          f"| books>50%DIR={maj_dir} books>10%DIR={any_dir}")
    # worst offenders
    rows.sort(key=lambda r: -r[2])
    worst = ", ".join(f"bk{r[0]}={r[2]*100:.0f}%" for r in rows[:5])
    print(f"      most-directional books: {worst}")


for step in (1.0, 2.0):    # TRUE publish cadence (1 sim-s); detector sub-samples to CHAR_SAMPLE_GAP_S internally
    print(f"\n=== publish step = {step:.0f}s | detector: sample/{mod.CHAR_SAMPLE_GAP_S:.0f}s span {mod.CHAR_WINDOW_S:.0f}s ===")
    print(" WINNING MAKERS (must classify SMOOTH/CHOP, NOT directional):")
    run_file("uid60 maker ", "60_maker_38.127.44.98.csv", step)
    run_file("uid84 maker ", "84_maker_38.127.44.98.csv", step)
    run_file("uid145 maker", "145_maker_38.127.44.98.csv", step)
    print(" TAKERS (directional response is fine here):")
    run_file("uid62 taker ", "62_57.129.75.161.csv", step)
    run_file("uid184 taker", "184_taker_38.127.44.98.csv", step)
