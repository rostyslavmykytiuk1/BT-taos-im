#!/usr/bin/env python3
"""
SN-79 market regime monitor — TAKER vs MAKER.

Classifies the CURRENT favored mode from the LIVE validator leaderboard (what is actually
winning on kappa) plus the dynamic fee regime, persists the verdict, and ALERTS when the
regime FLIPS (with hysteresis so it does not flap on noise).

  python3 tools/market_monitor.py --once            # one reading: print verdict, update state, alert if changed
  python3 tools/market_monitor.py --interval 600    # loop every 10 min (run under pm2/cron)

Signal (per reading, main validator only):
  * pull every miner's kappa + maker/taker daily volume
  * take the TOP_N miners by kappa (kappa = 79% of score → the elite set IS the regime)
  * compare median kappa of the maker-leaning elite vs the taker-leaning elite
  * verdict = MAKER / TAKER (one side beats the other by >KAPPA_MARGIN) else MIXED
  * also report market maker-share (MTR) and the dynamic fee for each side at that MTR

Alert on a CONFIRMED change (CONFIRM consecutive readings agreeing on the new regime):
  1) prints a banner, 2) appends tools/regime_history.log, 3) writes tools/REGIME_ALERT.txt,
  4) if env WASH_REGIME_WEBHOOK is set, POSTs JSON {text: ...} to it (Telegram/Discord/Slack/etc).

ALSO alert when the MAIN validator's simulation REFRESHES — its sim_id (stamped on every
metric line) changes, i.e. a NEW simulation started (kappa history, positions and agent state
all reset on changeover). The refresh alert BUNDLES the current market status (verdict + MTR +
fees + elite kappa) so you see the regime at the moment of the reset. NOTE: the leaderboard
kappa is cold for ~60-90 min after a refresh, so the bundled verdict is provisional.
"""
import argparse, json, os, re, statistics, sys, time, urllib.request
from datetime import datetime

METRICS_URL = "http://84.32.70.8:9001/metrics/miner"
MAIN_VAL = "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF"
TOOLS = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(TOOLS, "market_regime_state.json")
HISTLOG = os.path.join(TOOLS, "regime_history.log")
ALERTFILE = os.path.join(TOOLS, "REGIME_ALERT.txt")
WEBHOOK_FILE = "/root/.sn79_regime_webhook"   # secret, OUTSIDE the repo (no git leak); env WASH_REGIME_WEBHOOK overrides
SIM_XML = os.path.join(TOOLS, "..", "simulate", "trading", "run", "config", "simulation_0.xml")

TOP_N = 20            # size of the "elite set" whose lean defines the regime
KAPPA_MARGIN = 1.15   # a side must beat the other's median kappa by 15% to own the regime
CONFIRM = 2           # consecutive agreeing readings required before declaring a flip

RECO = {
    "MAKER": "PureMakerAgent (post liquidity; favor books with taker flow / per-book MTR < 0.4 for rebate+spread)",
    "TAKER": "TakerScalperAgent (final) — cross + scalp; favor maker-heavy books / MTR > 0.4 for taker rebate",
    "MIXED": "no clear edge — hold current mix; AdaptiveRouter (routes per book) is the safe default",
}


def fee_params():
    p = dict(makerFee=0.0, takerFee=0.00023, maxMakerRate=0.015, maxTakerRate=0.015,
             targetMTR=0.4, shapeMakerFee=2.0, shapeMakerRebate=2.0)
    try:
        txt = open(SIM_XML).read()
        m = re.search(r'<FeePolicy\s+type="dynamic".*?/>', txt, re.S)
        blk = m.group(0) if m else txt
        for k in list(p):
            mm = re.search(rf'{k}="([-\d.]+)"', blk)
            if mm:
                p[k] = float(mm.group(1))
    except OSError:
        pass
    return p


def fee_at(x, p):
    """Return (maker_bps, taker_bps) at book maker-share x. +=pay, -=rebate."""
    t = p["targetMTR"]
    coefLHS = -(p["maxMakerRate"] / (t ** p["shapeMakerFee"]))
    coefRHS = (p["maxTakerRate"] / ((1 - t) ** p["shapeMakerRebate"]))
    if abs(t - x) < 1e-9:
        c = 0.0
    elif t > x:
        c = coefLHS * (t - x) ** p["shapeMakerFee"]
    else:
        c = coefRHS * (x - t) ** p["shapeMakerRebate"]
    return (p["makerFee"] + c) * 1e4, (p["takerFee"] - c) * 1e4


def fetch():
    """Return (metrics_by_uid, main_validator_sim_id). The sim_id is now CAPTURED (was discarded)
    from the MAIN_VAL metric lines so run_once() can detect a simulation REFRESH (sim_id change)."""
    raw = urllib.request.urlopen(METRICS_URL, timeout=25).read().decode()
    pat = re.compile(r'miner_gauges\{agent_id="(?P<uid>\d+)",miner_gauge_name="(?P<gn>[^"]+)",'
                     r'netuid="79",sim_id="(?P<sim>[^"]+)",wallet="(?P<wal>[^"]+)"\}\s+(?P<val>[-\d.eE+]+)')
    rows, sim_seen = [], {}
    for m in pat.finditer(raw):
        if m.group("wal") != MAIN_VAL:
            continue
        sim_seen[m.group("sim")] = sim_seen.get(m.group("sim"), 0) + 1
        rows.append((int(m.group("uid")), m.group("gn"), m.group("sim"), m.group("val")))
    # the main validator's CURRENT sim = the sim_id stamped on the most of its metric lines
    # (during a clean changeover every line carries the new id; majority is robust to any straggler)
    sim_id = max(sim_seen, key=sim_seen.get) if sim_seen else None
    # build the metrics snapshot from ONLY the current sim's lines, so the bundled market status is a
    # clean current-sim reading even if a stale-sim straggler is momentarily present at the boundary.
    d = {}
    for uid, gn, sim, val in rows:
        if sim != sim_id:
            continue
        if gn in ("kappa", "average_daily_maker_volume", "average_daily_taker_volume"):
            d.setdefault(uid, {})[gn] = float(val)
    return d, sim_id


def classify(d, p):
    rows = []
    for u, x in d.items():
        k = x.get("kappa")
        if k is None:
            continue
        mk = x.get("average_daily_maker_volume", 0.0)
        tk = x.get("average_daily_taker_volume", 0.0)
        tot = mk + tk
        rows.append((k, u, "maker" if mk > tk else "taker", (mk / tot if tot else 0.0)))
    rows.sort(reverse=True)
    top = rows[:TOP_N]
    top10 = rows[:10]
    mk_k = [r[0] for r in top if r[2] == "maker"]
    tk_k = [r[0] for r in top if r[2] == "taker"]
    mk_med = statistics.median(mk_k) if mk_k else 0.0
    tk_med = statistics.median(tk_k) if tk_k else 0.0
    if mk_k and mk_med > tk_med * KAPPA_MARGIN:
        verdict = "MAKER"
    elif tk_k and tk_med > mk_med * KAPPA_MARGIN:
        verdict = "TAKER"
    else:
        verdict = "MIXED"
    allmk = sum(x.get("average_daily_maker_volume", 0.0) for x in d.values())
    alltk = sum(x.get("average_daily_taker_volume", 0.0) for x in d.values())
    mtr = allmk / (allmk + alltk) if (allmk + alltk) else 0.0
    mk_bps, tk_bps = fee_at(mtr, p)
    return verdict, dict(top5=top[:5], mk_top=len(mk_k), tk_top=len(tk_k),
                         top10_mk=sum(1 for r in top10 if r[2] == "maker"),
                         top10_tk=sum(1 for r in top10 if r[2] == "taker"),
                         mk_med=mk_med, tk_med=tk_med, mtr=mtr, mk_bps=mk_bps, tk_bps=tk_bps,
                         miners=len(rows))


def strength_of(verdict, info):
    """Conviction of the verdict → STRONG / MODERATE / WEAK, from the winning side's kappa edge
    ratio over the loser, gated by the winner actually being positive. Returns (label, ratio).
      ratio = winning median kappa / losing median kappa.
      >=2.0 STRONG · >=1.5 MODERATE · >1.15 WEAK (just past the flip threshold).
      If the winning side isn't even positive, it's a WEAK(flat) edge over a worse loser."""
    if verdict == "MAKER":
        win, lose = info["mk_med"], info["tk_med"]
    elif verdict == "TAKER":
        win, lose = info["tk_med"], info["mk_med"]
    else:
        return "no-edge", 1.0
    ratio = (win / lose) if lose > 0 else (999.0 if win > 0 else 1.0)
    if win <= 0:
        return "WEAK(flat)", ratio
    if ratio >= 2.0:
        return "STRONG", ratio
    if ratio >= 1.5:
        return "MODERATE", ratio
    return "WEAK", ratio


def load_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {"regime": None, "pending": None, "pending_count": 0, "since": None, "sim_id": None}


def save_state(s):
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)


def _webhook():
    h = os.environ.get("WASH_REGIME_WEBHOOK")
    if h:
        return h.strip()
    try:
        return open(WEBHOOK_FILE).read().strip() or None
    except OSError:
        return None


def post_discord(text):
    hook = _webhook()
    if not hook:
        print("  (no webhook configured — skipping Discord)")
        return
    try:
        req = urllib.request.Request(hook, data=json.dumps({"content": text, "text": text}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "sn79-market-monitor/1.0"})  # Discord/CF 403s the default UA
        urllib.request.urlopen(req, timeout=15)
        print("  (Discord alert sent)")
    except Exception as e:
        print(f"  (webhook failed: {e})")


def notify(ts, old, new, info, from_side=None):
    # local logging fires on EVERY confirmed change (incl. via MIXED)
    strength, ratio = strength_of(new, info)
    rtxt = f"{ratio:.2f}x" if ratio < 900 else "much"
    msg = (f"[SN-79 REGIME CHANGE] {old} -> {new} [{strength}] @ {ts}\n"
           f"  recommend: {RECO[new]}\n"
           f"  edge: {new} {rtxt} the other (maker k{info['mk_med']:.4f} / taker k{info['tk_med']:.4f})\n"
           f"  FEES @ MTR {info['mtr']:.3f}: maker {info['mk_bps']:+.1f}bps  taker {info['tk_bps']:+.1f}bps  (+=pay, -=rebate)")
    banner = "\n" + "!" * 64 + "\n" + msg + "\n" + "!" * 64 + "\n"
    print(banner)
    sys.stdout.flush()
    with open(HISTLOG, "a") as f:
        f.write(f"{ts}\tCHANGE\t{old}->{new}\t{strength}\tMTR={info['mtr']:.3f}\tmk_med={info['mk_med']:.4f}\ttk_med={info['tk_med']:.4f}\n")
    with open(ALERTFILE, "w") as f:
        f.write(banner)
    # Discord fires ONLY on a firm TAKER<->MAKER side flip (from_side set); MIXED transitions are silent
    if from_side is not None:
        post_discord(
            f"🔄 **SN-79 regime flip: {from_side} → {new}**  [{strength}]  ({ts})\n"
            f"Recommend: **{RECO[new]}**\n"
            f"edge: {new} {rtxt} the other — maker k{info['mk_med']:.3f} / taker k{info['tk_med']:.3f}  |  MTR {info['mtr']:.3f}\n"
            f"**FEES: maker {info['mk_bps']:+.1f}bps, taker {info['tk_bps']:+.1f}bps**  (+ = pay, - = rebate)")


def notify_sim_refresh(ts, old_sim, new_sim, verdict, info, p):
    """Fired when the MAIN validator starts a NEW simulation (its sim_id changed). Bundles the
    current market status so the regime is visible at the moment of the reset. Same sinks as
    notify(): banner + regime_history.log + REGIME_ALERT.txt + Discord (always — a refresh is rare
    and operationally important: kappa/positions reset, agents cold-start)."""
    strength, ratio = strength_of(verdict, info)
    rtxt = f"{ratio:.2f}x" if ratio < 900 else "much"
    edge = "" if verdict == "MIXED" else f", {verdict.lower()} {rtxt} the other"
    msg = (f"[SN-79 SIM REFRESH] main validator started a NEW simulation @ {ts}\n"
           f"  sim_id: {old_sim} -> {new_sim}\n"
           f"  market now: {verdict} [{strength}]  (MTR={info['mtr']:.3f}, target={p['targetMTR']}{edge})\n"
           f"  elite kappa: maker_med={info['mk_med']:.4f} ({info['mk_top']}) "
           f"taker_med={info['tk_med']:.4f} ({info['tk_top']})\n"
           f"  FEES @ MTR {info['mtr']:.3f}: maker {info['mk_bps']:+.1f}bps  taker {info['tk_bps']:+.1f}bps  (+=pay, -=rebate)\n"
           f"  recommend: {RECO[verdict]}\n"
           f"  NOTE: kappa/positions reset on refresh — the leaderboard verdict is COLD ~60-90 min.")
    banner = "\n" + "~" * 64 + "\n" + msg + "\n" + "~" * 64 + "\n"
    print(banner)
    sys.stdout.flush()
    with open(HISTLOG, "a") as f:
        f.write(f"{ts}\tSIM_REFRESH\t{old_sim}->{new_sim}\tverdict={verdict}\tMTR={info['mtr']:.3f}\n")
    with open(ALERTFILE, "w") as f:
        f.write(banner)
    post_discord(
        f"🆕 **SN-79 sim refresh — main validator started a new simulation**  ({ts})\n"
        f"`{old_sim}` → `{new_sim}`\n"
        f"Market now: **{verdict}** [{strength}]  |  MTR {info['mtr']:.3f}\n"
        f"**FEES: maker {info['mk_bps']:+.1f}bps / taker {info['tk_bps']:+.1f}bps**  (+ = pay, - = rebate)\n"
        f"elite kappa — maker {info['mk_med']:.3f} / taker {info['tk_med']:.3f}\n"
        f"Recommend: **{RECO[verdict]}**\n"
        f"_kappa/positions reset on refresh — verdict cold ~60-90 min._")


def run_once():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p = fee_params()
    try:
        d, sim_id = fetch()
    except Exception as e:
        print(f"[{ts}] fetch error: {e}")
        return
    verdict, info = classify(d, p)
    strength, ratio = strength_of(verdict, info)
    rtxt = f"{ratio:.2f}x" if ratio < 900 else "much"
    edge = "" if verdict == "MIXED" else f"  ({verdict.lower()} {rtxt} the other side)"
    print(f"\n{'='*64}\n[{ts}]  FAVORED: {verdict}  [{strength}]{edge}")
    print(f"  main-val sim_id: {sim_id}")
    print(f"{'='*64}")
    print(f"  FEES @ MTR {info['mtr']:.3f}:  maker {info['mk_bps']:+.1f}bps   taker {info['tk_bps']:+.1f}bps   "
          f"(+ = pay,  - = rebate;  crossover at MTR {p['targetMTR']})")
    print(f"  elite (top {TOP_N}): {info['mk_top']} maker k{info['mk_med']:.4f}  vs  {info['tk_top']} taker k{info['tk_med']:.4f}"
          f"   |  top10: {info['top10_mk']}M / {info['top10_tk']}T")
    print(f"  top 5: " + ", ".join(f"uid{u}({lean[0].upper()},k{k:.3f})" for k, u, lean, _ in info['top5']))
    print(f"  recommend: {RECO[verdict]}")

    s = load_state()
    if s.get("last_firm") is None and s.get("regime") in ("MAKER", "TAKER"):
        s["last_firm"] = s["regime"]   # seed the last firm side from existing state

    # --- MAIN-validator simulation refresh: alert (with market status) when the sim_id changes.
    # Independent of the regime flip below. First run just records the id (no prior to compare).
    prev_sim = s.get("sim_id")
    if sim_id and sim_id != prev_sim:
        if prev_sim is not None:
            notify_sim_refresh(ts, prev_sim, sim_id, verdict, info, p)
        else:
            print(f"  [sim_id initialized: {sim_id}]")
        s["sim_id"] = sim_id

    if s["regime"] is None:
        s["regime"] = verdict; s["since"] = ts; s["pending"] = None; s["pending_count"] = 0
        if verdict in ("MAKER", "TAKER"):
            s["last_firm"] = verdict
        print(f"  [state initialized to {verdict}]")
    elif verdict == s["regime"]:
        s["pending"] = None; s["pending_count"] = 0
        print(f"  [stable: {verdict} since {s['since']}]")
    else:
        if verdict == s.get("pending"):
            s["pending_count"] += 1
        else:
            s["pending"] = verdict; s["pending_count"] = 1
        print(f"  [candidate flip -> {verdict}: {s['pending_count']}/{CONFIRM} confirmations]")
        if s["pending_count"] >= CONFIRM:
            firm_flip = verdict in ("MAKER", "TAKER") and verdict != s.get("last_firm")
            notify(ts, s["regime"], verdict, info, from_side=(s.get("last_firm") if firm_flip else None))
            if firm_flip:
                s["last_firm"] = verdict       # Discord pinged: TAKER<->MAKER side actually changed
            s["regime"] = verdict; s["since"] = ts; s["pending"] = None; s["pending_count"] = 0
    save_state(s)
    with open(HISTLOG, "a") as f:
        f.write(f"{ts}\t{verdict}\tMTR={info['mtr']:.3f}\tmk_med={info['mk_med']:.4f}\ttk_med={info['tk_med']:.4f}\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    a = ap.parse_args()
    if a.once:
        run_once(); return
    print(f"Market regime monitor — every {a.interval}s. Verdict TAKER/MAKER/MIXED; alerts on confirmed flip.")
    while True:
        run_once()
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
