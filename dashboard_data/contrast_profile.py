import csv
from collections import defaultdict, deque
from statistics import median

def parse_time(t):
    h, m, rest = t.split(":")
    s = float(rest)
    return int(h)*3600 + int(m)*60 + s

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "book": r["book"],
                "t": parse_time(r["time"]),
                "side": r["side"],
                "price": float(r["price"]),
                "vol": float(r["volume"]),
                "role": r["role"].upper(),
                "fee": float(r["fee"]),
            })
    return rows

def fifo_book(trades):
    # signed lot queue; positive=long lots, negative=short lots
    trades = sorted(trades, key=lambda x: x["t"])
    lots = deque()  # each: [signed_qty, price, t]
    realized = 0.0
    wins = 0
    closes = 0
    holds = []
    for tr in trades:
        sgn = 1.0 if tr["side"] == "BUY" else -1.0
        qty = tr["vol"]
        # match against opposite-sign lots
        while qty > 1e-12 and lots and (lots[0][0] * sgn) < 0:
            lot = lots[0]
            avail = abs(lot[0])
            m = min(avail, qty)
            entry = lot[1]
            exit_ = tr["price"]
            # lot sign: if lot is long (+), this close is a sell -> pnl = m*(exit-entry)
            # if lot short (-), this close is a buy -> pnl = m*(entry-exit)
            if lot[0] > 0:
                pnl = m * (exit_ - entry)
            else:
                pnl = m * (entry - exit_)
            realized += pnl
            closes += 1
            if pnl > 0:
                wins += 1
            holds.append(tr["t"] - lot[2])
            if lot[0] > 0:
                lot[0] -= m
            else:
                lot[0] += m
            qty -= m
            if abs(lot[0]) < 1e-12:
                lots.popleft()
        if qty > 1e-12:
            lots.append([sgn*qty, tr["price"], tr["t"]])
    return realized, wins, closes, holds

def profile(path, name):
    rows = load(path)
    books = defaultdict(list)
    for r in rows:
        books[r["book"]].append(r)
    n_books = len(books)
    n_taker = sum(1 for r in rows if r["role"] == "TAKER")
    n_maker = sum(1 for r in rows if r["role"] == "MAKER")
    n_buy = sum(1 for r in rows if r["side"] == "BUY")
    n_sell = sum(1 for r in rows if r["side"] == "SELL")
    fees = [r["fee"] for r in rows]
    fee_sum = sum(fees)
    n_neg_fee = sum(1 for f in fees if f < 0)
    n_pos_fee = sum(1 for f in fees if f > 0)
    n_zero_fee = sum(1 for f in fees if f == 0)
    # time span
    tmin = min(r["t"] for r in rows)
    tmax = max(r["t"] for r in rows)
    span = (tmax - tmin)
    # per-book FIFO
    book_pnls = []
    all_holds = []
    total_wins = 0
    total_closes = 0
    rts_per_book = []
    per_book_wr = []
    book_net = []  # net pnl - fees(signed). fee>0 paid -> subtract; fee<0 rebate -> adds. net = realized - sum(fee)
    for b, tr in books.items():
        realized, wins, closes, holds = fifo_book(tr)
        bfee = sum(x["fee"] for x in tr)
        net = realized - bfee  # subtract signed fee: pos fee reduces, neg fee (rebate) adds
        book_pnls.append(realized)
        book_net.append(net)
        all_holds.extend(holds)
        total_wins += wins
        total_closes += closes
        rts_per_book.append(closes)
        if closes > 0:
            per_book_wr.append(wins/closes)
    pct_net_prof = 100.0 * sum(1 for n in book_net if n > 0)/len(book_net)
    # volatility proxy: median per-book price stddev / mean price (range based)
    import statistics as st
    vol_proxies = []
    for b, tr in books.items():
        prices = [x["price"] for x in tr]
        if len(prices) > 2:
            mp = st.mean(prices)
            sd = st.pstdev(prices)
            vol_proxies.append(sd/mp*10000)  # bps
    med_vol_bps = median(vol_proxies) if vol_proxies else 0
    # price drift per book: (last-first)/first in bps, then median abs and median signed
    drifts = []
    for b, tr in books.items():
        trs = sorted(tr, key=lambda x: x["t"])
        d = (trs[-1]["price"]-trs[0]["price"])/trs[0]["price"]*10000
        drifts.append(d)
    med_signed_drift = median(drifts)
    med_abs_drift = median(abs(d) for d in drifts)
    # rebate bps: median fee/(price*vol) in bps over rows where role allows
    rebate_bps = []
    for r in rows:
        notional = r["price"]*r["vol"]
        if notional > 0:
            rebate_bps.append(r["fee"]/notional*10000)
    med_fee_bps = median(rebate_bps)
    # pct volume on rebate (fee<0) fills
    rebate_vol = sum(r["price"]*r["vol"] for r in rows if r["fee"] < 0)
    tot_vol = sum(r["price"]*r["vol"] for r in rows)
    pct_rebate_vol = 100.0*rebate_vol/tot_vol

    print(f"\n===== {name} ({path}) =====")
    print(f"rows={len(rows)} books={n_books} span={span:.0f}s ({span/60:.1f}min) [{tmin/3600:.2f}h..{tmax/3600:.2f}h]")
    print(f"role: TAKER={n_taker} ({100*n_taker/len(rows):.1f}%) MAKER={n_maker} ({100*n_maker/len(rows):.1f}%)")
    print(f"side: BUY={n_buy} SELL={n_sell} two-sided={min(n_buy,n_sell)/max(n_buy,n_sell):.2f}")
    print(f"fee sum={fee_sum:.3f}  neg(rebate)={n_neg_fee} pos(paid)={n_pos_fee} zero={n_zero_fee}")
    print(f"  -> {100*n_neg_fee/len(rows):.1f}% fills got rebate, {100*n_pos_fee/len(rows):.1f}% paid fee")
    print(f"median fee bps (signed)={med_fee_bps:.3f}  pct notional on rebate fills={pct_rebate_vol:.1f}%")
    print(f"closes(RT)={total_closes} median RTs/book={median(rts_per_book):.0f}")
    print(f"median hold={median(all_holds):.1f}s  median per-book winrate={median(per_book_wr):.3f}")
    print(f"pct books net-profitable(after fees)={pct_net_prof:.1f}%")
    print(f"sum realized PnL={sum(book_pnls):.2f}  sum net(after fee)={sum(book_net):.2f}")
    print(f"VOLATILITY: median per-book price-sd={med_vol_bps:.1f}bps  med|drift|={med_abs_drift:.1f}bps med signed drift={med_signed_drift:.1f}bps")
    return {
        "name": name, "books": n_books, "pct_taker": 100*n_taker/len(rows),
        "med_fee_bps": med_fee_bps, "pct_rebate_vol": pct_rebate_vol,
        "med_vol_bps": med_vol_bps, "med_signed_drift": med_signed_drift,
        "med_hold": median(all_holds), "pct_net_prof": pct_net_prof,
    }

import os
os.chdir("/root/sn-79/dashboard_data")
print("######## PRIOR REGIME REFS ########")
profile("136_trades_top_taker_down_trends.csv", "136 volatile no-rebate downtrend")
profile("126_trades_top_taker.csv", "126 rebate scalp")
profile("109_new_top_maker.csv", "109 maker")
print("\n\n######## CURRENT REGIME (today) ########")
profile("230_trades.csv", "230 FRESHEST today 13:32")
profile("60_new_top.csv", "60 today 02:06")
profile("120_trades.csv", "120 today 05:25")
