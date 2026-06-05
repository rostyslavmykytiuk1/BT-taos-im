const API = "";
const POLL_MS = 2000;
const DEFAULT_VALIDATOR_ID =
  "5EWwdZB7qCCMaAso5Mzcks4UUcPxKYvpAj32t5Mg1v6HSxoF";
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
    "seq", "action", "reason", "time_label", "side", "price", "quantity",
    "pos_before", "pos_after", "fills", "orderId",
  ],
  snapshots: [
    "closed_at", "mid", "signal_trend_bps", "signal_flow", "signal_imb", "action", "pos_qty",
  ],
};

const TABLE_HEADERS = {
  time_label: "time",
  closed_at: "sim_time",
  signal_trend_bps: "trend_bps",
  signal_flow: "dev_bps",
  signal_imb: "imb",
  reason: "why",
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
let avgSeries;
let kalmanLevelSeries;
let slopeChart;
let slopeSeries;
let slopeZeroSeries;
let miners = [];
let activeAgentClass = null;
let activeTab = "round_trips";
let pollTimer;
let lastChartKey = null;
let cachedTables = null;
let applyingRoute = false;
let refreshInFlight = false;
let refreshSeq = 0;
let pendingRefresh = null;

const $ = (id) => document.getElementById(id);

function setStatusText(msg, { error = false } = {}) {
  const textEl = $("status-text");
  if (textEl) textEl.textContent = msg;
  $("status")?.classList.toggle("is-error", error);
}

function setLoadingUI(active) {
  document.body.classList.toggle("is-loading", active);
  for (const id of ["overlay-chart", "overlay-table"]) {
    $(id)?.setAttribute("aria-hidden", active ? "false" : "true");
  }
  if (active) setStatusText("Loading data from API…");
}

/** Let the browser paint overlays before a long await. */
function paintLoadingFrame() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

function setRefreshButtonState(state) {
  const btn = $("btn-refresh");
  if (!btn) return;
  btn.classList.remove("is-loading", "is-ok");
  if (state === "loading") {
    btn.disabled = true;
    btn.classList.add("is-loading");
    btn.textContent = "…";
    btn.setAttribute("aria-busy", "true");
    return;
  }
  btn.disabled = false;
  btn.removeAttribute("aria-busy");
  if (state === "ok") {
    btn.classList.add("is-ok");
    btn.textContent = "✓";
    window.setTimeout(() => {
      btn.classList.remove("is-ok");
      btn.textContent = "Refresh";
    }, 600);
    return;
  }
  btn.textContent = "Refresh";
}

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

function encPathSegment(value) {
  return encodeURIComponent(value);
}

function routePath(sel) {
  return `/validators/${encPathSegment(sel.validator_id)}/agents/${sel.uid}/simulations/${encPathSegment(sel.simulation_id)}/books/${sel.book_id}`;
}

function bookApiPath(sel) {
  return `/api/validators/${encPathSegment(sel.validator_id)}/agents/${sel.uid}/simulations/${encPathSegment(sel.simulation_id)}/books/${sel.book_id}`;
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
  return validators.includes(DEFAULT_VALIDATOR_ID) ? DEFAULT_VALIDATOR_ID : validators[0];
}

function fillSelect(id, values, selected) {
  const el = $(id);
  el.innerHTML = values.map((v) => `<option value="${v}">${v}</option>`).join("");
  el.value = selected;
}

function validatorsForUid(uid) {
  return [...new Set(miners.filter((m) => m.uid === uid).map((m) => m.validator_id))].sort();
}

function simsFor(uid, validator_id) {
  return [
    ...new Set(
      miners.filter((m) => m.uid === uid && m.validator_id === validator_id).map((m) => m.simulation_id),
    ),
  ].sort();
}

function defaultSelection() {
  const uids = [...new Set(miners.map((m) => m.uid))].sort((a, b) => a - b);
  const uid = uids[0];
  const validators = validatorsForUid(uid);
  const validator_id = pickValidator(validators);
  const sims = simsFor(uid, validator_id);
  return { uid, validator_id, simulation_id: sims[0], book_id: 0 };
}

function selection() {
  return {
    uid: parseInt($("sel-uid").value, 10),
    validator_id: $("sel-validator").value,
    simulation_id: $("sel-sim").value,
    book_id: parseInt($("inp-book").value, 10) || 0,
  };
}

function syncUrl(sel, replace) {
  const path = routePath(sel);
  if (location.pathname === path) return;
  const state = { ...sel };
  if (replace) history.replaceState(state, "", path);
  else history.pushState(state, "", path);
}

function applySelection(sel, { updateUrl, replaceUrl }) {
  applyingRoute = true;
  fillSelect("sel-uid", [...new Set(miners.map((m) => m.uid))].sort((a, b) => a - b), sel.uid);

  const validators = validatorsForUid(sel.uid);
  const validator_id = validators.includes(sel.validator_id) ? sel.validator_id : pickValidator(validators);
  fillSelect("sel-validator", validators, validator_id);

  const sims = simsFor(sel.uid, validator_id);
  const simulation_id = sims.includes(sel.simulation_id) ? sel.simulation_id : sims[0];
  fillSelect("sel-sim", sims, simulation_id);

  $("inp-book").value = String(Number.isFinite(sel.book_id) ? sel.book_id : 0);
  applyingRoute = false;

  const resolved = { uid: sel.uid, validator_id, simulation_id, book_id: parseInt($("inp-book").value, 10) };
  if (updateUrl) syncUrl(resolved, replaceUrl);
  return resolved;
}

function onUidChange() {
  const uid = parseInt($("sel-uid").value, 10);
  fillSelect("sel-validator", validatorsForUid(uid), pickValidator(validatorsForUid(uid)));
  onValidatorChange();
}

function onValidatorChange() {
  const uid = parseInt($("sel-uid").value, 10);
  const validator_id = $("sel-validator").value;
  const sims = simsFor(uid, validator_id);
  fillSelect("sel-sim", sims, sims[0]);
  onSelectionChange();
}

function onSelectionChange() {
  if (applyingRoute) return;
  syncUrl(selection(), false);
  refresh({ withLoading: true });
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

function chartOptions(el) {
  return {
    layout: { background: { color: "#161a22" }, textColor: "#8b95a8" },
    grid: { vertLines: { color: "#2a3140" }, horzLines: { color: "#2a3140" } },
    timeScale: {
      timeVisible: true,
      secondsVisible: true,
      tickMarkFormatter: formatChartAxisTime,
    },
    localization: { timeFormatter: formatChartAxisTime },
    width: el.clientWidth,
  };
}

function initChart() {
  const el = $("chart");
  chart = LightweightCharts.createChart(el, chartOptions(el));
  midSeries = chart.addLineSeries({ ...LINE_OPTS, color: "#f59e0b", priceLineVisible: true });
  avgSeries = chart.addLineSeries({
    lineWidth: 2,
    lineType: LightweightCharts.LineType?.Simple ?? 0,
    lineStyle: LightweightCharts.LineStyle?.Dashed ?? 2,
    color: "#38bdf8",
    crosshairMarkerVisible: false,
    lastValueVisible: true,
    priceLineVisible: false,
  });
  kalmanLevelSeries = chart.addLineSeries({
    lineWidth: 2,
    lineType: LightweightCharts.LineType?.Simple ?? 0,
    lineStyle: LightweightCharts.LineStyle?.Dashed ?? 2,
    color: "#a78bfa",
    crosshairMarkerVisible: false,
    lastValueVisible: true,
    priceLineVisible: false,
    visible: false,
  });

  const slopeEl = $("chart-slope");
  slopeChart = LightweightCharts.createChart(slopeEl, {
    ...chartOptions(slopeEl),
    rightPriceScale: { borderVisible: false },
    timeScale: { visible: false },
  });
  slopeSeries = slopeChart.addLineSeries({
    ...LINE_OPTS,
    lineType: LightweightCharts.LineType?.Simple ?? 0,
    color: "#c084fc",
    priceLineVisible: false,
  });
  slopeZeroSeries = slopeChart.addLineSeries({
    lineWidth: 1,
    lineType: LightweightCharts.LineType?.Simple ?? 0,
    lineStyle: LightweightCharts.LineStyle?.Dashed ?? 2,
    color: "#4b5563",
    crosshairMarkerVisible: false,
    lastValueVisible: false,
    priceLineVisible: false,
  });

  let syncing = false;
  chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (syncing || !range) return;
    syncing = true;
    slopeChart.timeScale().setVisibleLogicalRange(range);
    syncing = false;
  });
  slopeChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (syncing || !range) return;
    syncing = true;
    chart.timeScale().setVisibleLogicalRange(range);
    syncing = false;
  });

  const onResize = () => {
    chart.applyOptions({ width: el.clientWidth });
    slopeChart.applyOptions({ width: slopeEl.clientWidth });
  };
  window.addEventListener("resize", onResize);
  onResize();
}

function chartKey(sel) {
  return `${sel.uid}|${sel.validator_id}|${sel.simulation_id}|${sel.book_id}`;
}

function ordersInChartRange(midPayload, orders) {
  const pts = midPayload?.mid ?? [];
  if (!pts.length) return [];
  let tMin = midPayload?.range?.[0];
  let tMax = midPayload?.range?.[1];
  if (tMin == null || tMax == null) {
    tMin = pts[0].time;
    tMax = pts[pts.length - 1].time;
  }
  return (orders || []).filter(
    (o) => o.time != null && o.time >= tMin && o.time <= tMax && MARKER_STYLE[o.action],
  );
}

function agentClassFor(sel) {
  const row = miners.find(
    (m) =>
      m.uid === sel.uid &&
      m.validator_id === sel.validator_id &&
      m.simulation_id === sel.simulation_id,
  );
  return row?.agent_class ?? null;
}

function setKalmanPanelsVisible(showLevel, showSlope) {
  $("leg-kalman-level")?.classList.toggle("hidden", !showLevel);
  kalmanLevelSeries?.applyOptions({ visible: showLevel });
  $("slope-panel")?.classList.toggle("hidden", !showSlope);
}

function updateChart(midPayload, orders, fitView) {
  const inRange = ordersInChartRange(midPayload, orders);
  midSeries.setData(
    (midPayload?.mid ?? []).map((p) => ({ time: toChartTime(p.time), value: p.value })),
  );
  const avgPts = midPayload?.average ?? [];
  avgSeries.setData(avgPts.map((p) => ({ time: toChartTime(p.time), value: p.value })));

  const levelPts = midPayload?.kalman_level ?? [];
  const slopePts = midPayload?.slope_bps ?? [];
  const isKalman = activeAgentClass === "KalmanMomentumAgent";
  const showLevel = levelPts.length > 0 && (isKalman || levelPts.length >= 3);
  const showSlope = slopePts.length > 0 && (isKalman || slopePts.length >= 3);

  setKalmanPanelsVisible(showLevel, showSlope);
  if (showLevel) {
    kalmanLevelSeries.setData(levelPts.map((p) => ({ time: toChartTime(p.time), value: p.value })));
  } else {
    kalmanLevelSeries.setData([]);
  }
  if (showSlope) {
    const slopeData = slopePts.map((p) => ({ time: toChartTime(p.time), value: p.value }));
    slopeSeries.setData(slopeData);
    if (slopeData.length >= 2) {
      slopeZeroSeries.setData([
        { time: slopeData[0].time, value: 0 },
        { time: slopeData[slopeData.length - 1].time, value: 0 },
      ]);
    } else {
      slopeZeroSeries.setData([]);
    }
  } else {
    slopeSeries.setData([]);
    slopeZeroSeries.setData([]);
  }

  midSeries.setMarkers(
    inRange
      .map((o) => {
        const s = MARKER_STYLE[o.action];
        return { time: toChartTime(o.time), position: s.position, shape: s.shape, color: s.color, size: s.size };
      })
      .sort((a, b) => a.time - b.time),
  );
  if (fitView) {
    requestAnimationFrame(() => {
      chart.timeScale().fitContent();
      if (showSlope) slopeChart.timeScale().fitContent();
    });
  }
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
  const table = $("data-table");
  table.querySelector("thead").innerHTML =
    `<tr>${cols.map((c) => `<th>${TABLE_HEADERS[c] || c}</th>`).join("")}</tr>`;
  let emptyMsg = "No data";
  if (!rows.length && tab === "round_trips") {
    emptyMsg =
      "No completed round trips for this book. Only full flat-to-flat cycles appear here; partial closes are in Market orders.";
  }
  table.querySelector("tbody").innerHTML = rows.length
    ? rows.map((r) => `<tr>${cols.map((c) => `<td>${formatCell(r, c)}</td>`).join("")}</tr>`).join("")
    : `<tr><td colspan="${cols.length}" class="empty-cell">${emptyMsg}</td></tr>`;
  if (tab === "snapshots") document.querySelector(".table-wrap")?.scrollTo(0, 0);
}

function showTableLoading() {
  const cols = TABLE_COLUMNS[activeTab] || ["…"];
  const table = $("data-table");
  table.querySelector("thead").innerHTML =
    `<tr>${cols.map((c) => `<th>${TABLE_HEADERS[c] || c}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML =
    `<tr><td colspan="${cols.length}" class="loading-cell">Loading…</td></tr>`;
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

function statusMessage(uid, book_id, snapshots) {
  const stamp = new Date().toLocaleTimeString();
  if (activeTab === "snapshots" && snapshots?.length) {
    return `Updated ${stamp} · uid=${uid} book=${book_id} · latest signal ${snapshots[0].closed_at} (newest on top)`;
  }
  return `Updated ${stamp} · uid=${uid} book=${book_id}`;
}

async function loadMiners() {
  setStatusText("Loading catalog…");
  try {
    miners = await api("/api/catalog");
  } catch (err) {
    setStatusText(`Failed to load catalog: ${err.message}`, { error: true });
    return;
  }
  if (!miners.length) {
    setStatusText(
      "No telemetry found. Enable TAOS_TELEMETRY_ENABLED=1 and restart miner, or run dashboard/seed_demo.py",
      { error: true },
    );
    return;
  }
  const fromUrl = parseRoute(location.pathname);
  const initial = fromUrl && catalogHas(fromUrl) ? fromUrl : defaultSelection();
  applySelection(initial, { updateUrl: true, replaceUrl: !fromUrl || !catalogHas(fromUrl) });
  await refresh({ withLoading: true });
}

function beginLoadingUI() {
  setLoadingUI(true);
  showTableLoading();
  setRefreshButtonState("loading");
}

async function refresh(opts = {}) {
  const withLoading = Boolean(opts.withLoading);
  const sel = selection();
  if (!sel.uid || !sel.validator_id || !sel.simulation_id) return;

  if (refreshInFlight) {
    if (!withLoading) return;
    refreshSeq += 1;
    pendingRefresh = { withLoading: true };
    beginLoadingUI();
    return;
  }

  const seq = ++refreshSeq;
  refreshInFlight = true;
  if (withLoading) {
    beginLoadingUI();
    await paintLoadingFrame();
  }

  const base = bookApiPath(sel);
  try {
    const [summary, midPayload, roundTrips, { orders }, snapshots] = await Promise.all([
      api(`${base}/summary`),
      api(`${base}/mid?resolution=1&limit=5000`),
      api(`${base}/round_trips?limit=100&chart_limit=5000`),
      api(`${base}/trades?limit=500&chart_limit=5000`),
      api(`${base}/snapshots?limit=80`),
    ]);

    if (seq !== refreshSeq) return;

    activeAgentClass = agentClassFor(sel);
    updateCards(summary);
    const fitView = chartKey(sel) !== lastChartKey;
    lastChartKey = chartKey(sel);
    const chartOrders = ordersInChartRange(midPayload, orders);
    updateChart(midPayload, chartOrders, fitView);

    cachedTables = { round_trips: roundTrips, trades: chartOrders, snapshots };
    renderTable(activeTab, cachedTables);
    setStatusText(statusMessage(sel.uid, sel.book_id, snapshots));
    if (withLoading) setRefreshButtonState("ok");
  } catch (err) {
    if (seq === refreshSeq) {
      setStatusText(`Error: ${err.message}`, { error: true });
      if (withLoading) setRefreshButtonState("idle");
    }
  } finally {
    refreshInFlight = false;
    const current = seq === refreshSeq;
    if (withLoading && current) setLoadingUI(false);
    if (pendingRefresh) {
      const next = pendingRefresh;
      pendingRefresh = null;
      void refresh(next);
    }
  }
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      if (tab === activeTab) return;
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeTab = tab;
      if (cachedTables) {
        renderTable(activeTab, cachedTables);
        const sel = selection();
        setStatusText(statusMessage(sel.uid, sel.book_id, cachedTables.snapshots));
      }
    });
  });
}

function setupPoll() {
  $("btn-refresh").addEventListener("click", () => {
    if (!applyingRoute) syncUrl(selection(), false);
    refresh({ withLoading: true });
  });
  $("sel-uid").addEventListener("change", onUidChange);
  $("sel-validator").addEventListener("change", onValidatorChange);
  $("sel-sim").addEventListener("change", onSelectionChange);
  $("inp-book").addEventListener("change", onSelectionChange);
  window.addEventListener("popstate", (ev) => {
    const sel = ev.state || parseRoute(location.pathname) || defaultSelection();
    applySelection(sel, { updateUrl: false, replaceUrl: false });
    lastChartKey = null;
    cachedTables = null;
    refresh({ withLoading: true });
  });
  pollTimer = setInterval(refresh, POLL_MS);
}

document.addEventListener("DOMContentLoaded", () => {
  initChart();
  setupTabs();
  setupPoll();
  loadMiners().catch((e) => setStatusText(`Failed to load: ${e.message}`, { error: true }));
});
