#!/usr/bin/env python3
"""Empirical validation of FEE-CHURN (and the other modes) for an AdaptiveRouter miner.
Run when the regime turns VOLATILE (fee-churn only activates on volatile, no-rebate books).
Usage: check_feechurn.py [miner-N]   (default miner-1)

Verdict logic for fee-churn: it PAYS the fee, so it must capture a favorable price move >
round-trip cost. If avgNet is clearly negative it's a drag -> lengthen its hold (if _bias has
edge) or disable it. If positive, _bias is predictive on volatile books -> keep/tune.
"""
import re, subprocess, sys, statistics
ansi = re.compile(r'\x1b\[[0-9;]*m'); N = r'([-+]?[0-9.]+)'
rt = re.compile(r'RT\] book=\d+ mode=(\w+).*?hold_s=' + N + r'.*?net=' + N)
miner = sys.argv[1] if len(sys.argv) > 1 else 'miner-1'
log = f'/root/.pm2/logs/{miner}-out.log'
out = subprocess.run(['tail', '-n', '120000', log], capture_output=True, text=True).stdout
modes = {}
for ln in out.splitlines():
    if 'RT]' not in ln: continue
    m = rt.search(ansi.sub('', ln))
    if not m: continue
    modes.setdefault(m.group(1), []).append((float(m.group(2)), float(m.group(3))))
print(f"=== {miner}: per-mode RT outcomes (last 120k log lines) ===")
print(f"{'mode':10s}{'RTs':>6}{'avgNet':>10}{'win%':>6}{'medHold':>9}{'sumNet':>9}")
for mode in ['taker', 'feechurn', 'maker']:
    rows = modes.get(mode, [])
    if not rows:
        print(f"{mode:10s}{'0':>6}   (inactive)")
        continue
    holds = [h for h, _ in rows]; nets = [n for _, n in rows]
    print(f"{mode:10s}{len(rows):>6}{statistics.mean(nets):>+10.4f}{100*sum(1 for x in nets if x>0)/len(nets):>5.0f}%{statistics.median(holds):>9.1f}{sum(nets):>+9.2f}")
fc = modes.get('feechurn', [])
print("\n--- FEE-CHURN verdict ---")
if len(fc) < 20:
    print(f"  only {len(fc)} feechurn RTs — not enough to judge (regime not volatile-no-rebate yet). Re-run later.")
else:
    avg = statistics.mean(n for _, n in fc)
    if avg > 0.002:   print(f"  +EV (avgNet {avg:+.4f}) -> _bias is predictive on volatile books; KEEP, consider longer hold to capture more.")
    elif avg > -0.002: print(f"  MARGINAL (avgNet {avg:+.4f}) -> backoff-bounded but low value; lengthen hold or leave dormant.")
    else:              print(f"  -EV DRAG (avgNet {avg:+.4f}) -> _bias not predictive enough; LENGTHEN hold or DISABLE fee-churn (drop 'feechurn' from ALLOWED_MODES).")
