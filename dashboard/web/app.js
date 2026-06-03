const API = "";
const POLL_MS = 2000;
const DEFAULT_VALIDATOR_ID =
  "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF";
/** UTC epoch anchor so Lightweight Charts shows sim HH:MM:SS, not 1970/local TZ. */
const SIM_CHART_EPOCH = Date.UTC(2000, 0, 1) / 1000;

const ROUTE_RE =
  /^\/validators\/([^/]+)\/agents\/(\d+)\/simulations\/([^/]+)\/books\/(\d+)\/?$/;

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
let applyingRoute = false;

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

function routePath({ validator_id, uid, simulation_id, book_id }) {
  const v = encodeURIComponent(validator_id);
  const s = encodeURIComponent(simulation_id);
  return `/validators/${v}/agents/${uid}/simulations/${s}/books/${book_id}`;
}

function parseRoute(pathname) {
  const m = ROUTE_RE.exec(pathname);
  if (!m) return null;
  return {
    validator_id: decodeURIComponent(m[1]),
    uid: parseInt(m[2], 10),
    simulation_id: decodeURIComponent(m[3]),
    book_id: parseInt(m[4], 10),
  };
}

function catalogHas(sel) {
  return miners.some(
    (m) =>
      m.uid === sel.uid &&
      m.validator_id === sel.validator_id &&
      m.simulation_id === sel.simulation_id,
  );
}

function pickValidator(validators) {
  if (validators.includes(DEFAULT_VALIDATOR_ID)) return DEFAULT_VALIDATOR_ID;
  return validators[0];
}

function defaultSelection() {
  const uids = [...new Set(miners.map((m) => m.uid))].sort((a, b) => a - b);
  const uid = uids[0];
  const validators = [
    ...new Set(miners.filter((m) => m.uid === uid).map((m) => m.validator_id)),
  ].sort();
  const validator_id = pickValidator(validators);
  const sims = [
    ...new Set(
      miners
        .filter((m) => m.uid === uid && m.validator_id === validator_id)
        .map((m) => m.simulation_id),
    ),
  ].sort();
  return {
    uid,
    validator_id,
    simulation_id: sims[0],
    book_id: 0,
  };
}

function selection() {
  return {
    uid: parseInt($("sel-uid").value, 10),
    validator_id: $("sel-validator").value,
    simulation_id: $("sel-sim").value,
    book_id: parseInt($("inp-book").value, 10) || 0,
  };
}

function bookApiPath({ validator_id, uid, simulation_id, book_id }) {
  const v = encodeURIComponent(validator_id);
  const s = encodeURIComponent(simulation_id);
  return `/api/validators/${v}/agents/${uid}/simulations/${s}/books/${book_id}`;
}

function syncUrl(sel, replace) {
  const path = routePath(sel);
  if (location.pathname === path) return;
  const state = { ...sel };
  if (replace) {
    history.replaceState(state, "", path);
  } else {
    history.pushState(state, "", path);
  }
}

function populateUidOptions() {
  const uids = [...new Set(miners.map((m) => m.uid))].sort((a, b) => a - b);
  $("sel-uid").innerHTML = uids.map((u) => `<option value="${u}">${u}</option>`).join("");
}

function validatorsForUid(uid) {
  return [
    ...new Set(miners.filter((m) => m.uid === uid).map((m) => m.validator_id)),
  ].sort();
}

function simsFor(uid, validator_id) {
  return [
    ...new Set(
      miners
        .filter((m) => m.uid === uid && m.validator_id === validator_id)
        .map((m) => m.simulation_id),
    ),
  ].sort();
}

function applySelection(sel, { updateUrl, replaceUrl }) {
  applyingRoute = true;
  populateUidOptions();
  $("sel-uid").value = String(sel.uid);

  const validators = validatorsForUid(sel.uid);
  const validator_id = validators.includes(sel.validator_id)
    ? sel.validator_id
    : pickValidator(validators);
  $("sel-validator").innerHTML = validators
    .map((v) => `<option value="${v}">${v}</option>`)
    .join("");
  $("sel-validator").value = validator_id;

  const sims = simsFor(sel.uid, validator_id);
  const simulation_id = sims.includes(sel.simulation_id) ? sel.simulation_id : sims[0];
  $("sel-sim").innerHTML = sims.map((s) => `<option value="${s}">${s}</option>`).join("");
  $("sel-sim").value = simulation_id;

  const book_id = Number.isFinite(sel.book_id) ? sel.book_id : 0;
  $("inp-book").value = String(book_id);

  applyingRoute = false;

  const resolved = { uid: sel.uid, validator_id, simulation_id, book_id };
  if (updateUrl) syncUrl(resolved, replaceUrl);
  return resolved;
}

function onUidChange() {
  const uid = parseInt($("sel-uid").value, 10);
  const validators = validatorsForUid(uid);
  $("sel-validator").innerHTML = validators
    .map((v) => `<option value="${v}">${v}</option>`)
    .join("");
  $("sel-validator").value = pickValidator(validators);
  onValidatorChange();
}

function onValidatorChange() {
  const uid = parseInt($("sel-uid").value, 10);
  const validator_id = $("sel-validator").value;
  const sims = simsFor(uid, validator_id);
  $("sel-sim").innerHTML = sims.map((s) => `<option value="${s}">${s}</option>`).join("");
  onSelectionChange();
}

function onSelectionChange() {
  if (applyingRoute) return;
  const sel = selection();
  syncUrl(sel, false);
  refresh();
}

function formatSimTimeSec(sec) {
  const s = Math.floor(Number(sec));
  if (!Number.isFinite(s) || s < 0) return "";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function toChartTime(simSec) {
  return SIM_CHART_EPOCH + Math.floor(Number(simSec));
}

function fromChartTime(chartTime) {
  return Math.floor(chartTime) - SIM_CHART_EPOCH;
}

function formatChartAxisTime(chartTime) {
  return formatSimTimeSec(fromChartTime(chartTime));
}

function initChart() {
  const el = $("chart");
  chart = LightweightCharts.createChart(el, {
    layout: { background: { color: "#161a22" }, textColor: "#8b95a8" },
    grid: { vertLines: { color: "#2a3140" }, horzLines: { color: "#2a3140" } },
    timeScale: {
      timeVisible: true,
      secondsVisible: true,
      tickMarkFormatter: formatChartAxisTime,
    },
    localization: {
      timeFormatter: formatChartAxisTime,
    },
  });
  midSeries = chart.addLineSeries({ ...LINE_OPTS, color: "#f59e0b", priceLineVisible: true });
  const onResize = () => chart.applyOptions({ width: el.clientWidth });
  window.addEventListener("resize", onResize);
  onResize();
}

function chartKey(uid, validator_id, simulation_id, book_id) {
  return `${uid}|${validator_id}|${simulation_id}|${book_id}`;
}

function midPoints(payload) {
  return (payload?.mid ?? []).map((p) => ({
    time: toChartTime(p.time),
    value: p.value,
  }));
}

function chartMarkers(orders) {
  return (orders || [])
    .filter((o) => o.time != null && MARKER_STYLE[o.action])
    .map((o) => {
      const s = MARKER_STYLE[o.action];
      return {
        time: toChartTime(o.time),
        position: s.position,
        shape: s.shape,
        color: s.color,
        size: s.size,
      };
    })
    .sort((a, b) => a.time - b.time);
}

function updateChart(midPayload, orders, fitView) {
  midSeries.setData(midPoints(midPayload));
  const markers = chartMarkers(orders);
  midSeries.setMarkers(markers);
  if (fitView) requestAnimationFrame(() => chart.timeScale().fitContent());
}

function formatCell(row, col) {
  const v = row[col];
  if (v === null || v === undefined) return "";
  if (col === "time_label" || col === "closed_at") {
    const sec = row.time_sec ?? row.time;
    if (sec != null) return formatSimTimeSec(sec);
  }
  if (col === "pos_before" || col === "pos_after") return Number(v).toFixed(3);
  return String(v);
}

function renderTable(tab, data) {
  const cols = TABLE_COLUMNS[tab] || [];
  const rows = data[tab] || [];
  const headers = {
    time_label: "time",
    // MeanReversionAgent maps dev_bps into signal_flow (see agent _snap).
    signal_trend_bps: "trend_bps",
    signal_flow: "dev_bps",
    signal_imb: "imb",
    closed_at: "sim_time",
  };
  const table = $("data-table");
  table.querySelector("thead").innerHTML =
    `<tr>${cols.map((c) => `<th>${headers[c] || c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${formatCell(r, c)}</td>`).join("")}</tr>`)
    .join("");
  if (tab === "snapshots") {
    const wrap = document.querySelector(".table-wrap");
    if (wrap) wrap.scrollTop = 0;
  }
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
  miners = await api("/api/catalog");
  if (!miners.length) {
    $("status").textContent =
      "No telemetry found. Enable TAOS_TELEMETRY_ENABLED=1 and restart miner, or run dashboard/seed_demo.py";
    return;
  }

  const fromUrl = parseRoute(location.pathname);
  const initial =
    fromUrl && catalogHas(fromUrl) ? fromUrl : defaultSelection();
  const replaceUrl = !fromUrl || !catalogHas(fromUrl);
  applySelection(initial, { updateUrl: true, replaceUrl });
  refresh();
}

async function refresh() {
  const { uid, validator_id, simulation_id, book_id } = selection();
  if (!uid || !validator_id || !simulation_id) return;
  const base = bookApiPath({ validator_id, uid, simulation_id, book_id });
  try {
    const [summary, midPayload, roundTrips, { orders }, snapshots] = await Promise.all([
      api(`${base}/summary`),
      api(`${base}/mid?resolution=1&limit=5000`),
      api(`${base}/round_trips?limit=100`),
      api(`${base}/trades?limit=500`),
      api(`${base}/snapshots?limit=80`),
    ]);

    updateCards(summary);
    const key = chartKey(uid, validator_id, simulation_id, book_id);
    const fitView = key !== lastChartKey;
    lastChartKey = key;
    updateChart(midPayload, orders, fitView);
    renderTable(activeTab, { round_trips: roundTrips, trades: orders, snapshots });
    const latestSig = snapshots.length ? snapshots[0].closed_at : "";
    $("status").textContent =
      activeTab === "snapshots" && latestSig
        ? `Updated ${new Date().toLocaleTimeString()} · uid=${uid} book=${book_id} · latest signal ${latestSig} (newest row on top)`
        : `Updated ${new Date().toLocaleTimeString()} · uid=${uid} book=${book_id}`;
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
  $("sel-sim").addEventListener("change", onSelectionChange);
  $("inp-book").addEventListener("change", onSelectionChange);
  window.addEventListener("popstate", (ev) => {
    const sel = ev.state || parseRoute(location.pathname) || defaultSelection();
    applySelection(sel, { updateUrl: false, replaceUrl: false });
    lastChartKey = null;
    refresh();
  });
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
