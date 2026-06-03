const API = "";
const POLL_MS = 2000;

const MARKER_STYLE = {
  open_long: { position: "aboveBar", shape: "arrowUp", color: "#22c55e", size: 1 },
  open_short: { position: "aboveBar", shape: "arrowDown", color: "#ef4444", size: 1 },
  close_long: { position: "aboveBar", shape: "circle", color: "#22c55e", size: 1 },
  close_short: { position: "aboveBar", shape: "circle", color: "#ef4444", size: 1 },
};

const TABLE_COLUMNS = {
  round_trips: [
    "seq", "closed_at", "book_id", "side", "qty", "entry_avg", "exit_avg",
    "realized_pnl", "hold", "reason",
  ],
  trades: [
    "seq", "action", "time_label", "side", "price", "quantity",
    "pos_before", "pos_after", "fills", "orderId",
  ],
  snapshots: [
    "closed_at", "mid", "signal_trend_bps", "signal_flow", "signal_imb", "action", "pos_qty",
  ],
};

const LINE_OPTS = {
  lineWidth: 2,
  lineType: LightweightCharts.LineType?.WithSteps ?? 1,
  crosshairMarkerVisible: true,
  lastValueVisible: true,
  priceLineVisible: false,
};

let chart;
let midSeries;
let miners = [];
let activeTab = "round_trips";
let pollTimer;
let lastChartKey = null;

const $ = (id) => document.getElementById(id);

function fmt(n, d = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}

function fmtVol(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const v = Number(n);
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
  return v.toFixed(0);
}

async function api(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

function selection() {
  return {
    uid: $("sel-uid").value,
    validator: $("sel-validator").value,
    sim: $("sel-sim").value,
    book: parseInt($("inp-book").value, 10) || 0,
  };
}

function initChart() {
  const el = $("chart");
  chart = LightweightCharts.createChart(el, {
    layout: { background: { color: "#161a22" }, textColor: "#8b95a8" },
    grid: { vertLines: { color: "#2a3140" }, horzLines: { color: "#2a3140" } },
    timeScale: { timeVisible: true, secondsVisible: false },
  });
  midSeries = chart.addLineSeries({ ...LINE_OPTS, color: "#f59e0b", priceLineVisible: true });
  const onResize = () => chart.applyOptions({ width: el.clientWidth });
  window.addEventListener("resize", onResize);
  onResize();
}

function chartKey(uid, validator, sim, book) {
  return `${uid}|${validator}|${sim}|${book}`;
}

function midPoints(payload) {
  return payload?.mid ?? [];
}

function chartMarkers(orders) {
  return (orders || [])
    .filter((o) => o.time != null && o.time >= 0 && MARKER_STYLE[o.action])
    .map((o) => {
      const s = MARKER_STYLE[o.action];
      return { time: o.time, position: s.position, shape: s.shape, color: s.color, size: s.size };
    })
    .sort((a, b) => a.time - b.time);
}

function updateChart(midPayload, orders, fitView) {
  midSeries.setData(midPoints(midPayload));
  midSeries.setMarkers(chartMarkers(orders));
  if (fitView) requestAnimationFrame(() => chart.timeScale().fitContent());
}

function formatCell(row, col) {
  const v = row[col];
  if (v === null || v === undefined) return "";
  if (col === "pos_before" || col === "pos_after") return Number(v).toFixed(3);
  return String(v);
}

function renderTable(tab, data) {
  const cols = TABLE_COLUMNS[tab] || [];
  const rows = data[tab] || [];
  const headers = { time_label: "time" };
  const table = $("data-table");
  table.querySelector("thead").innerHTML =
    `<tr>${cols.map((c) => `<th>${headers[c] || c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${formatCell(r, c)}</td>`).join("")}</tr>`)
    .join("");
}

function updateCards(summary) {
  const snap = summary.latest_snapshot || {};
  $("card-mid").textContent = fmt(snap.mid, 2);
  $("card-spread").textContent = fmt(snap.spread_bps, 2);
  $("card-pos").textContent =
    snap.pos_qty != null ? `${fmt(snap.pos_qty, 3)} @ ${fmt(snap.pos_avg, 2)}` : "—";
  $("card-base").textContent = snap.base_bal != null ? fmt(snap.base_bal, 4) : "—";
  $("card-quote").textContent = snap.quote_bal != null ? fmt(snap.quote_bal, 2) : "—";
  $("card-vol-traded").textContent = fmtVol(snap.traded_volume);
  $("card-vol-cap").textContent = fmtVol(snap.volume_cap);
  const volLeft = snap.volume_remaining;
  $("card-vol-left").textContent = fmtVol(volLeft);
  $("card-vol-left").className =
    "value " +
    (volLeft != null && snap.volume_cap != null && volLeft <= snap.volume_cap * 0.1 ? "negative" : "");
  const upnl = snap.unrealized_pnl;
  $("card-upnl").textContent = fmt(upnl, 2);
  $("card-upnl").className = "value " + (upnl > 0 ? "positive" : upnl < 0 ? "negative" : "");
  const rt = summary.round_trips || {};
  $("card-rt").textContent = rt.n != null ? String(rt.n) : "—";
  $("card-pnl-rt").textContent = fmt(summary.pnl_per_rt, 4);
  $("card-step").textContent = fmt((summary.latest_summary || {}).loop_ms, 1);
}

async function loadMiners() {
  miners = await api("/api/miners");
  const uids = [...new Set(miners.map((m) => m.uid))].sort((a, b) => a - b);
  $("sel-uid").innerHTML = uids.map((u) => `<option value="${u}">${u}</option>`).join("");
  if (!uids.length) {
    $("status").textContent =
      "No telemetry found. Enable TAOS_TELEMETRY_ENABLED=1 and restart miner, or run dashboard/seed_demo.py";
    return;
  }
  onUidChange();
}

function onUidChange() {
  const uid = parseInt($("sel-uid").value, 10);
  const slugs = [...new Set(miners.filter((m) => m.uid === uid).map((m) => m.validator_slug))];
  $("sel-validator").innerHTML = slugs.map((s) => `<option value="${s}">${s}</option>`).join("");
  onValidatorChange();
}

function onValidatorChange() {
  const uid = parseInt($("sel-uid").value, 10);
  const slug = $("sel-validator").value;
  const sims = [
    ...new Set(miners.filter((m) => m.uid === uid && m.validator_slug === slug).map((m) => m.simulation_id)),
  ].sort();
  $("sel-sim").innerHTML = sims.map((s) => `<option value="${s}">${s}</option>`).join("");
  refresh();
}

async function refresh() {
  const { uid, validator, sim, book } = selection();
  if (!uid || !validator || !sim) return;
  const base = `/api/${uid}/${validator}/${sim}`;
  try {
    const [summary, midPayload, roundTrips, { orders }, snapshots] = await Promise.all([
      api(`${base}/summary?book=${book}`),
      api(`${base}/ohlcv?book=${book}&resolution=1&limit=5000`),
      api(`${base}/round_trips?book=${book}&limit=100`),
      api(`${base}/trades?book=${book}&limit=500`),
      api(`${base}/snapshots?book=${book}&limit=80`),
    ]);

    updateCards(summary);
    const key = chartKey(uid, validator, sim, book);
    const fitView = key !== lastChartKey;
    lastChartKey = key;
    updateChart(midPayload, orders, fitView);
    renderTable(activeTab, { round_trips: roundTrips, trades: orders, snapshots });
    $("status").textContent = `Updated ${new Date().toLocaleTimeString()} · uid=${uid} book=${book}`;
  } catch (err) {
    $("status").textContent = `Error: ${err.message}`;
  }
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = btn.dataset.tab;
      refresh();
    });
  });
}

function setupPoll() {
  $("btn-refresh").addEventListener("click", refresh);
  $("sel-uid").addEventListener("change", onUidChange);
  $("sel-validator").addEventListener("change", onValidatorChange);
  $("sel-sim").addEventListener("change", refresh);
  $("inp-book").addEventListener("change", refresh);
  pollTimer = setInterval(refresh, POLL_MS);
}

document.addEventListener("DOMContentLoaded", () => {
  initChart();
  setupTabs();
  setupPoll();
  loadMiners().catch((e) => {
    $("status").textContent = `Failed to load: ${e.message}`;
  });
});
