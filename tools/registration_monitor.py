#!/usr/bin/env python3
"""
SN-79 competitor registration monitor.

Watches the netuid-79 metagraph for NEW miner registrations at a target axon IP
(default 38.127.44.98 — the ~44-miner operator we track), and:

  1) the moment a new (uid, hotkey) appears at the IP -> Discord alert (uid + reg time)
  2) ~CLASSIFY_DELAY_S (default 5 min) later -> read the validator's live `trades`
     feed, work out whether that uid is trading MAKER or TAKER, and Discord-alert
     the verdict (uid, time, mode).

Mode is inferred from the validator trades metric: every trade tags maker_agent_id
and taker_agent_id, so a uid's maker-fill vs taker-fill mix tells us its mode.

Reuses the SAME Discord webhook as market_monitor.py (env WASH_REGIME_WEBHOOK, else
/root/.sn79_regime_webhook).

Usage:
  python3 tools/registration_monitor.py --once                 # single sweep (seeds on first run)
  python3 tools/registration_monitor.py --interval 120         # loop (run under pm2)
  python3 tools/registration_monitor.py --ip 38.127.44.98      # override target IP
  python3 tools/registration_monitor.py --reclassify           # re-run mode check for all known uids now
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

import bittensor as bt

NETUID = 79
ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"
TRADES_URL = "http://84.32.70.8:9001/metrics/trades"
TARGET_IP = "38.127.44.98"          # the tracked operator (override with --ip)
BLOCK_SECONDS = 12.0                  # finney block time, for block -> wall-clock estimate

CLASSIFY_DELAY_S = 300                # wait this long after detection before first mode check
CLASSIFY_MAX_WAIT_S = 1800            # give up classifying after this long -> report mode=UNKNOWN
CLASSIFY_MIN_FILLS = 4                # need at least this many observed fills to call a mode
CLASSIFY_POLL_BUDGET_S = 60           # per-attempt: keep polling the (flaky) trades feed up to this long
CLASSIFY_POLL_OK = 6                  # ...or until this many successful polls

TOOLS = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(TOOLS, "registration_monitor_state.json")
HISTLOG = os.path.join(TOOLS, "registration_history.log")
WEBHOOK_FILE = "/root/.sn79_regime_webhook"

_LINE = re.compile(r'^trades\{([^}]*)\}\s+([-+0-9.eEnaN]+)\s*$')
_LABEL = re.compile(r'(\w+)="([^"]*)"')


# --------------------------------------------------------------- discord
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
        req = urllib.request.Request(
            hook, data=json.dumps({"content": text, "text": text}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "sn79-reg-monitor/1.0"})
        urllib.request.urlopen(req, timeout=15)
        print("  (Discord alert sent)")
    except Exception as e:
        print(f"  (webhook failed: {e})")


# --------------------------------------------------------------- state
def load_state():
    try:
        s = json.load(open(STATE))
    except (OSError, ValueError):
        s = {}
    s.setdefault("known", {})       # uid(str) -> {hotkey, reg_block}
    s.setdefault("pending", [])     # list of {uid, hotkey, reg_block, reg_ts, detected_ts, classify_after}
    s.setdefault("seeded", False)
    return s


def save_state(s):
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)


# --------------------------------------------------------------- chain
def fetch_metagraph():
    """Return (rows, current_block) where rows = {uid: {hotkey, ip, reg_block}} for ALL uids at TARGET_IP."""
    sub = bt.Subtensor(network=ENDPOINT)
    meta = sub.metagraph(NETUID)
    cur = sub.get_current_block()
    n = len(meta.uids)
    reg = getattr(meta, "block_at_registration", None)
    rows = {}
    for uid in range(n):
        try:
            ip = meta.axons[uid].ip
        except Exception:
            ip = None
        if ip != TARGET_IP:
            continue
        try:
            rb = int(reg[uid]) if reg is not None else None
        except Exception:
            rb = None
        rows[uid] = {"hotkey": meta.hotkeys[uid], "ip": ip, "reg_block": rb}
    return rows, cur


def block_to_ts(reg_block, cur_block):
    """Estimate registration wall-clock (UTC) from block delta."""
    if reg_block is None:
        return None
    delta_s = (cur_block - reg_block) * BLOCK_SECONDS
    return datetime.now(timezone.utc) - timedelta(seconds=delta_s)


# --------------------------------------------------------------- trades / mode
def _parse_trades(text):
    grouped = {}
    for line in text.splitlines():
        if not line.startswith("trades{"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group(1)))
        grouped.setdefault((labels.get("book_id"), labels.get("slot")), {})[
            labels.get("trade_gauge_name", "")] = m.group(2)
    return grouped


def classify_mode(uid):
    """Poll the (flaky) trades feed, accumulate this uid's maker vs taker fills, return
    (mode, maker_fills, taker_fills). mode in MAKER/TAKER/MIXED/UNKNOWN."""
    maker = taker = 0
    seen_trades = set()
    ok = 0
    start = time.time()
    while ok < CLASSIFY_POLL_OK and (time.time() - start) < CLASSIFY_POLL_BUDGET_S:
        try:
            text = urllib.request.urlopen(TRADES_URL, timeout=9).read().decode("utf-8", "replace")
        except Exception:
            time.sleep(1.5)
            continue
        ok += 1
        for _, g in _parse_trades(text).items():
            tid = g.get("trade_id")
            if tid is None or tid in seen_trades:
                continue
            try:
                mk = int(float(g.get("maker_agent_id")))
                tk = int(float(g.get("taker_agent_id")))
            except (TypeError, ValueError):
                continue
            seen_trades.add(tid)
            if mk == uid:
                maker += 1
            if tk == uid:
                taker += 1
        time.sleep(2)

    tot = maker + taker
    if tot < CLASSIFY_MIN_FILLS:
        return "UNKNOWN", maker, taker
    frac_t = taker / tot
    if frac_t >= 0.75:
        mode = "TAKER"
    elif frac_t <= 0.25:
        mode = "MAKER"
    else:
        mode = f"MIXED({frac_t*100:.0f}% taker)"
    return mode, maker, taker


# --------------------------------------------------------------- alerts
def alert_new(uid, hotkey, reg_ts):
    ts = reg_ts.strftime("%Y-%m-%d %H:%M:%S UTC") if reg_ts else "unknown"
    print(f"  NEW REGISTRATION: uid={uid} hotkey={hotkey[:12]} reg={ts}")
    post_discord(
        f"🆕 **SN-79 new miner @ {TARGET_IP}**\n"
        f"uid **{uid}**  ·  registered {ts}\n"
        f"hotkey `{hotkey[:16]}…`\n"
        f"_mode check in ~{CLASSIFY_DELAY_S//60} min…_")
    with open(HISTLOG, "a") as f:
        f.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\tNEW\tuid={uid}\thotkey={hotkey}\treg={ts}\n")


def alert_mode(uid, reg_ts, mode, maker, taker):
    ts = reg_ts.strftime("%Y-%m-%d %H:%M:%S UTC") if reg_ts else "unknown"
    icon = {"MAKER": "🟦", "TAKER": "🟥", "UNKNOWN": "⬜"}.get(mode.split("(")[0], "🟨")
    print(f"  MODE: uid={uid} -> {mode} (maker={maker} taker={taker})")
    post_discord(
        f"{icon} **SN-79 uid {uid} @ {TARGET_IP} → {mode}**\n"
        f"registered {ts}\n"
        f"observed fills: maker {maker} / taker {taker}")
    with open(HISTLOG, "a") as f:
        f.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\tMODE\tuid={uid}\tmode={mode}\tmaker={maker}\ttaker={taker}\treg={ts}\n")


# --------------------------------------------------------------- main sweep
def run_once(reclassify=False):
    now = time.time()
    s = load_state()
    try:
        rows, cur = fetch_metagraph()
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] metagraph fetch error: {e}")
        return
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC] {TARGET_IP}: {len(rows)} miners "
          f"(known={len(s['known'])}, pending_mode={len(s['pending'])}, block={cur})")

    # ----- first run: seed without alerting (don't spam all existing miners)
    if not s["seeded"]:
        for uid, info in rows.items():
            s["known"][str(uid)] = {"hotkey": info["hotkey"], "reg_block": info["reg_block"]}
        s["seeded"] = True
        save_state(s)
        print(f"  seeded {len(rows)} existing miners (no alerts). Future registrations will alert.")
        return

    # ----- detect new registrations (new uid at IP, or same uid with a NEW hotkey = re-registration)
    for uid, info in rows.items():
        k = str(uid)
        prev = s["known"].get(k)
        is_new = (prev is None) or (prev.get("hotkey") != info["hotkey"])
        if not is_new:
            continue
        reg_ts = block_to_ts(info["reg_block"], cur)
        alert_new(uid, info["hotkey"], reg_ts)
        s["known"][k] = {"hotkey": info["hotkey"], "reg_block": info["reg_block"]}
        s["pending"].append({
            "uid": uid, "hotkey": info["hotkey"], "reg_block": info["reg_block"],
            "reg_ts": reg_ts.isoformat() if reg_ts else None,
            "detected_ts": now, "classify_after": now + CLASSIFY_DELAY_S,
        })

    # ----- forced reclassify of everything currently known (manual flag)
    if reclassify:
        for uid in list(rows):
            info = rows[uid]
            mode, mk, tk = classify_mode(uid)
            alert_mode(uid, block_to_ts(info["reg_block"], cur), mode, mk, tk)
        save_state(s)
        return

    # ----- process pending mode-classifications whose delay has elapsed
    still_pending = []
    for p in s["pending"]:
        if now < p["classify_after"]:
            still_pending.append(p)
            continue
        reg_ts = datetime.fromisoformat(p["reg_ts"]) if p.get("reg_ts") else None
        mode, mk, tk = classify_mode(p["uid"])
        age = now - p["detected_ts"]
        if mode == "UNKNOWN" and age < CLASSIFY_MAX_WAIT_S:
            # not enough fills yet — retry on a later sweep
            p["classify_after"] = now + max(60, CLASSIFY_DELAY_S // 2)
            print(f"  uid={p['uid']} mode still UNKNOWN ({mk}+{tk} fills); retrying later")
            still_pending.append(p)
            continue
        alert_mode(p["uid"], reg_ts, mode, mk, tk)
    s["pending"] = still_pending
    save_state(s)


def main():
    global TARGET_IP
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=120, help="sweep seconds")
    ap.add_argument("--ip", default=TARGET_IP, help="target axon IP to watch")
    ap.add_argument("--reclassify", action="store_true", help="re-run mode check for all current miners now, then exit")
    a = ap.parse_args()

    TARGET_IP = a.ip

    if a.reclassify:
        run_once(reclassify=True)
        return
    if a.once:
        run_once()
        return
    print(f"Registration monitor — watching {TARGET_IP} on netuid {NETUID} every {a.interval}s. "
          f"Mode check {CLASSIFY_DELAY_S//60} min after each new registration.")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"sweep error: {e}")
        sys.stdout.flush()
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
