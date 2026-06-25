"""
Dry-run harness for the channel-based WashTradeAgent — NO network, NO validator, NO deploy.

Drives a winner + a sink through real side-channel files (temp dir) and a fill-simulator that
reproduces the cycle:
  winner fires marketable ENTRY (filled by a stranger at the touch) →
  winner rests EXIT at the ACTUAL fill price ± gap + publishes (side, price, remaining-held qty)
    to the channel WHILE STILL HOLDING (no seq, no ack) →
  sink reads the channel → fires a marketable IOC DYNAMICALLY sized to sweep the WHOLE book depth
    through the winner's exit price (so it reaches the winner past any strangers), bounded only by
    its owned balance → winner↔sink cross-UID fill → winner RT completes, flips long/short → repeat.

Asserts: channel round-trips (no seq); winner completes RTs with POSITIVE net; direction ALTERNATES;
the sink's fired qty SWEEPS THROUGH the strangers (qty >= stranger depth, i.e. the old 0.8 cap is
gone); winner↔sink are the exit counterparties (cross-UID, self_vol=0 in reality); winner takes NO
loans (short entry only sells owned base; leverage=0).

Run:  python3 tests/wash_dryrun.py
"""

import importlib.util, sys, types, tempfile, os, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("wash_mod", REPO / "agents" / "WashTradeAgent.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
WashTradeAgent = mod.WashTradeAgent
from taos.im.protocol.models import OrderDirection, TimeInForce  # noqa: E402

VAL = mod.MAIN_VALIDATOR
SIM = "dry"
PDP, VDP, NS = 2, 4, mod._NS
WIN, SINK, STRANGER = 1001, 1002, 9999
CHAN = tempfile.mkdtemp()
MAKER_FEE, TAKER_FEE = 0.000417, -0.000187   # live regime: maker pays, taker rebated
STRANGER_DEPTH = 1.0                          # 1.0-lot stranger sitting between touch and winner exit
PASS, FAIL = [], []


def check(c, m):
    (PASS if c else FAIL).append(m)
    print(("  PASS " if c else "  FAIL ") + m)


def ns(**k):
    return types.SimpleNamespace(**k)


def cfg(role, partner):
    return ns(lazy_load=1, wash_role=role, wash_partner_uid=partner, wash_channel_dir=CHAN,
              wash_books="0", wash_lot_min=0.30, wash_lot_max=0.30, wash_gap_min_s=0, wash_gap_max_s=0,
              wash_margin_bps=2.0, wash_activity_s=999999, wash_giveup_s=999999, wash_entry_slip_bps=20,
              wash_sink_flatten_lots=2.0, wash_summary_s=999999, wash_debug=0)


def simcfg():
    return ns(simulation_id=SIM, priceDecimals=PDP, volumeDecimals=VDP, miner_wealth=1e6,
              book_count=1, publish_interval=NS)


def acct(quote=1e6, base=100.0, orders=None):
    return ns(base_balance=ns(free=base, reserved=0.0), quote_balance=ns(free=quote, reserved=0.0),
              fees=ns(maker_fee_rate=MAKER_FEE, taker_fee_rate=TAKER_FEE, volume_traded=0.0),
              quote_loan=0.0, orders=orders or [])


def book(bid=299.5, ask=300.5):
    return ns(bids=[ns(price=bid, quantity=STRANGER_DEPTH)], asks=[ns(price=ask, quantity=STRANGER_DEPTH)])


def state(ts, uid):
    return ns(dendrite=ns(hotkey=VAL), timestamp=ts, config=simcfg(),
              books={0: book()}, accounts={uid: {0: acct()}})


def setup(agent):
    agent._active_validator = VAL
    agent._sim_id[VAL] = SIM
    agent._sync_precision(PDP, VDP)


def fill(agent, uid, *, is_taker, is_buy, qty, price, counterparty, ts):
    """Deliver a TradeEvent to one agent (duck-typed; agent only reads these fields)."""
    side = OrderDirection.BUY if is_buy else OrderDirection.SELL
    if is_taker:
        ev = ns(bookId=0, takerAgentId=uid, makerAgentId=counterparty, side=side,
                takerFee=TAKER_FEE * price * qty, makerFee=MAKER_FEE * price * qty, price=price,
                quantity=qty, timestamp=ts)
    else:
        ev = ns(bookId=0, takerAgentId=counterparty, makerAgentId=uid,
                side=(OrderDirection.SELL if is_buy else OrderDirection.BUY),  # maker tag is opposite the trade-initiation side
                takerFee=TAKER_FEE * price * qty, makerFee=MAKER_FEE * price * qty, price=price,
                quantity=qty, timestamp=ts)
    agent.onTrade(ev, validator=VAL)


def instrs(resp):
    out = []
    for i in resp.instructions:
        t = getattr(i, "type", "")
        if t == "PLACE_ORDER_LIMIT":
            out.append(("L", i.direction, i.quantity, i.price, i.timeInForce))
        elif t == "CANCEL_ORDERS":
            out.append(("C", None, None, None, None))
    return out


def winner_step(w, ts):
    """Run the winner one tick; simulate entry fills (stranger at the touch). Returns events."""
    w._active_validator = VAL; w._step_ts_ns[VAL] = ts
    resp = w.respond(state(ts, WIN))
    evs = []
    for kind, direction, qty, price, tif in instrs(resp):
        if kind != "L":
            continue
        if tif == TimeInForce.IOC:
            is_buy = direction == OrderDirection.BUY
            px = 300.5 if is_buy else 299.5   # touch
            fill(w, WIN, is_taker=True, is_buy=is_buy, qty=qty, price=px, counterparty=STRANGER, ts=ts)
            evs.append(("entry", "buy" if is_buy else "sell", px))
    return evs


def sink_fills_winner_exit(s, w, ts):
    """Run the sink one tick; if it emits a fill IOC, settle the winner's exit winner↔sink.
    Returns (channel, settled, sink_fired_qty). sink_fired_qty proves the DYNAMIC, UNCAPPED
    sizing — it must exceed the stranger depth to reach the winner's resting order beyond it."""
    s._active_validator = VAL; s._step_ts_ns[VAL] = ts
    chan = s._read_channel(VAL, SIM)
    resp = s.respond(state(ts, SINK))
    settled = None
    sink_q = 0.0
    for kind, direction, qty, price, tif in instrs(resp):
        if kind == "L" and tif == TimeInForce.IOC:
            sink_buys = direction == OrderDirection.BUY
            exit_px = price
            sink_q = qty
            wqty = 0.30   # the winner's exit (cross-UID) settles fully against the sink; strangers swept beyond
            fill(s, SINK, is_taker=True, is_buy=sink_buys, qty=wqty, price=exit_px, counterparty=WIN, ts=ts)
            fill(w, WIN, is_taker=False, is_buy=(not sink_buys), qty=wqty, price=exit_px, counterparty=SINK, ts=ts)
            settled = (sink_buys, exit_px)
            break
    return chan, settled, sink_q


# ---------------------------------------------------------------- tests
def run():
    w = WashTradeAgent(WIN, cfg("winner", -1))     # winner is partner-agnostic
    s = WashTradeAgent(SINK, cfg("sink", WIN))      # sink knows the winner's uid
    setup(w); setup(s)

    print(f"\n=== channel-based wash (dynamic uncapped sink): {N_RT} round trips ===")
    ts = 10 * NS
    dirs, exits_seen, partner_exits, sink_qtys = [], 0, 0, []
    for rt in range(N_RT):
        # tick 1: winner fires entry → filled by stranger → winner holds, locks exit_px
        e = winner_step(w, ts); ts += 2 * NS
        if e and e[0][0] == "entry":
            dirs.append(e[0][1])
        # tick 2: winner posts exit + publishes channel (side, price, held qty) while holding
        winner_step(w, ts); ts += 2 * NS
        published = mod.WashTradeAgent._safe_read(w._chan_path(VAL, WIN)).get("books", {})
        if "0" in published:
            exits_seen += 1
        # tick 3: sink reads channel and sweeps through to the winner's exit
        chan, settled, sink_q = sink_fills_winner_exit(s, w, ts); ts += 2 * NS
        if settled is not None:
            partner_exits += 1
            sink_qtys.append(sink_q)

    wst = w.books_state[VAL][0]
    sst = s.books_state[VAL][0]
    w_net = sum(x.net_pnl for x in w.books_state[VAL].values())
    s_realized = sum(x.net_pnl for x in s.books_state[VAL].values())
    w_held = abs(w._net_qty(w._inv(VAL, 0)))
    s_held = abs(s._net_qty(s._inv(VAL, 0)))
    min_sink_q = min(sink_qtys) if sink_qtys else 0.0
    print(f"  winner: rts={wst.rts} net={w_net:+.4f} held={w_held:.3f} dirs={dirs[:6]}...")
    print(f"  sink:   rts={sst.rts} realized_net={s_realized:+.4f} held={s_held:.3f} "
          f"partner_fills={sst.partner_fills} stranger_fills={sst.stranger_fills}")
    print(f"  channel exits published={exits_seen}/{N_RT}  sink-filled={partner_exits}/{N_RT}  "
          f"sink_fired_qty(min)={min_sink_q:.3f} (stranger_depth={STRANGER_DEPTH})")

    check(exits_seen >= N_RT - 1, f"winner published its exit to the channel ({exits_seen}/{N_RT})")
    check(partner_exits >= N_RT - 1, f"sink read the channel and reached the winner's exit ({partner_exits}/{N_RT})")
    check(min_sink_q >= STRANGER_DEPTH, f"sink sized DYNAMICALLY through strangers, no cap (min {min_sink_q:.3f} >= {STRANGER_DEPTH})")
    check(wst.rts >= N_RT - 1, f"winner completed round-trips ({wst.rts})")
    check(w_net > 0, f"WINNER net POSITIVE ({w_net:+.4f})")
    check(len(set(dirs[:4])) == 2 and dirs[0] != dirs[1], f"winner ALTERNATES direction ({dirs[:4]})")
    check(sst.partner_fills >= N_RT - 1, f"sink's exit fills were with the WINNER (partner) ({sst.partner_fills})")
    check(s_held <= 2 * 0.30 + 1e-6, f"sink inventory bounded ({s_held:.3f})")
    check(w_held < 0.30 + 1e-6, f"winner ends ~flat (no stranded/leveraged inventory) ({w_held:.3f})")


def fill_through_check():
    """The sink ALWAYS sweeps through ALL strangers to reach the winner's order — no cap, no skip,
    however deep the wall. Bounded only by balance. (Winner win-rate = 100%; sink loss irrelevant.)"""
    print("\n=== fill-through: sweep ALL strangers to reach the winner (no cap, no skip) ===")

    def sink_order_qty(stranger_depth, exit_px=300.63):
        s = WashTradeAgent(SINK, cfg("sink", WIN)); setup(s)
        chan_path = os.path.join(CHAN, f"wash_{WIN}_{VAL[:8]}.json")
        json.dump({"sim": SIM, "ts": 10 * NS,
                   "books": {"0": {"s": int(OrderDirection.SELL), "p": exit_px, "q": 0.30}}},
                  open(chan_path, "w"))
        bk = ns(bids=[ns(price=299.5, quantity=1.0)], asks=[ns(price=300.5, quantity=stranger_depth)])
        stt = ns(dendrite=ns(hotkey=VAL), timestamp=10 * NS, config=simcfg(),
                 books={0: bk}, accounts={SINK: {0: acct()}})
        s._active_validator = VAL; s._step_ts_ns[VAL] = 10 * NS
        resp = s.respond(stt)
        for kind, direction, qty, price, tif in instrs(resp):
            if kind == "L" and tif == TimeInForce.IOC:
                return qty
        return 0.0

    small = sink_order_qty(1.0)     # 1.0 stranger wall + 0.30 winner -> ~1.30
    big = sink_order_qty(50.0)      # 50-lot wall + 0.30 winner -> ~50.30 (sweeps the WHOLE wall, no skip)
    print(f"  wall=1.0 -> sink fires {small:.3f} ; wall=50 -> sink fires {big:.3f}")
    check(small >= 1.0, f"small wall: sink sweeps through to the winner ({small:.3f})")
    check(big >= 50.0, f"BIG wall: sink STILL sweeps the whole wall to reach the winner — no skip ({big:.3f})")


N_RT = 10
if __name__ == "__main__":
    run()
    fill_through_check()
    print(f"\n==== RESULT: {len(PASS)} passed, {len(FAIL)} failed ====")
    if FAIL:
        for m in FAIL:
            print("  FAILED:", m)
        sys.exit(1)
    print("ALL ASSERTIONS PASSED")
