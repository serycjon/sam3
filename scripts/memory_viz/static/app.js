/* SAM3 streaming memory viz — vanilla JS, no build step. */
"use strict";

const $ = (id) => document.getElementById(id);
const pad6 = (n) => String(n).padStart(6, "0");
const fmt = (x, d = 3) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));

const state = {
  config: {},
  log: [],
  pos: 0,
  playing: false,
  timer: null,
  overlayKey: "final", // final | 0 | 1 | 2 | token0
  mmCache: new Map(), // frame_idx -> multimask meta (or promise)
  chart: null, // geometry cache for hit-testing
};

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ---------------- data loading ---------------- */

async function init() {
  const [meta, logResp] = await Promise.all([
    fetch("/api/meta").then((r) => r.json()),
    fetch("/api/log").then((r) => r.json()),
  ]);
  state.config = meta.config || {};
  state.log = logResp.log || [];
  $("slider").max = Math.max(0, state.log.length - 1);

  buildRoleLegend();
  bindControls();
  drawChart();
  window.addEventListener("resize", drawChart);
  render(0);
}

function multimaskMeta(frame) {
  if (!state.mmCache.has(frame)) {
    state.mmCache.set(
      frame,
      fetch(`/api/multimask/${frame}`).then((r) => r.json()).catch(() => ({ available: false }))
    );
  }
  return state.mmCache.get(frame);
}

/* ---------------- controls ---------------- */

function bindControls() {
  $("btnPrev").onclick = () => seek(state.pos - 1);
  $("btnNext").onclick = () => seek(state.pos + 1);
  $("btnPlay").onclick = togglePlay;
  $("slider").oninput = (e) => seek(Number(e.target.value));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") { seek(state.pos - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { seek(state.pos + 1); e.preventDefault(); }
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
  });
}

function seek(pos) {
  pos = Math.max(0, Math.min(state.log.length - 1, pos));
  if (pos !== state.pos) render(pos);
}

function togglePlay() {
  state.playing = !state.playing;
  $("btnPlay").textContent = state.playing ? "⏸" : "▶";
  clearInterval(state.timer);
  if (state.playing) {
    const fps = Number($("fps").value) || 5;
    state.timer = setInterval(() => {
      if (state.pos >= state.log.length - 1) togglePlay();
      else seek(state.pos + 1);
    }, 1000 / fps);
  }
}

/* ---------------- mask overlays (CSS luminance masks) ---------------- */

function setOverlay(overlayEl, maskUrl) {
  overlayEl.style.display = "none";
  if (!maskUrl) return;
  const probe = new Image();
  probe.onload = () => {
    overlayEl.style.webkitMaskImage = `url("${maskUrl}")`;
    overlayEl.style.maskImage = `url("${maskUrl}")`;
    overlayEl.style.display = "block";
  };
  probe.onerror = () => { overlayEl.style.display = "none"; };
  probe.src = maskUrl;
}

/* ---------------- cells ---------------- */

function makeCell({ frame, roleClass, capRight, tag, trimmed, empty, emptyLabel }) {
  const cell = document.createElement("div");
  if (empty) {
    cell.className = "cell empty";
    cell.textContent = emptyLabel || "empty";
    return cell;
  }
  cell.className = `cell ${roleClass || ""} ${trimmed ? "trimmed" : ""}`;
  const view = document.createElement("div");
  view.className = "frame-view";
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = `/thumbs/${pad6(frame)}.jpg`;
  img.alt = `frame ${frame}`;
  const ov = document.createElement("div");
  ov.className = "mask-overlay";
  view.append(img, ov);
  setOverlay(ov, `/masks/${pad6(frame)}.png`);

  const cap = document.createElement("div");
  cap.className = "cap";
  const left = document.createElement("span");
  left.className = "num";
  left.textContent = `#${frame}`;
  cap.append(left);
  if (capRight !== undefined) {
    const right = document.createElement("span");
    right.className = "num";
    right.textContent = capRight;
    cap.append(right);
  }
  cell.append(view, cap);
  if (tag) {
    const t = document.createElement("div");
    t.className = "tag";
    t.textContent = tag;
    cell.append(t);
  }
  cell.title = `frame ${frame}`;
  cell.style.cursor = "pointer";
  cell.onclick = () => {
    const target = state.log.findIndex((r) => r.frame_idx === frame && r.event === "track");
    if (target >= 0) seek(target);
  };
  return cell;
}

/* ---------------- render ---------------- */

function render(pos) {
  state.pos = pos;
  const rec = state.log[pos];
  if (!rec) return;
  $("slider").value = pos;
  $("posLabel").textContent = `${pos + 1}/${state.log.length}`;
  $("frameNum").textContent = rec.frame_idx;

  const badge = $("eventBadge");
  badge.textContent = rec.event;
  badge.className = `badge ${rec.event === "correct" ? "correct" : ""}`;

  renderMainView(rec);
  renderScores(rec);
  renderEventDetail(rec);
  renderPools(rec);
  renderBank(rec);
  drawChart();
}

function renderMainView(rec) {
  const frame = rec.frame_idx;
  $("mainThumb").src = `/thumbs/${pad6(frame)}.jpg`;
  applyMainOverlay(frame);
  renderOverlayPicker(rec);
}

function applyMainOverlay(frame) {
  const key = state.overlayKey;
  const url = key === "final"
    ? `/masks/${pad6(frame)}.png`
    : `/multimask/${frame}/${key}.png`;
  setOverlay($("mainOverlay"), url);
}

async function renderOverlayPicker(rec) {
  const frame = rec.frame_idx;
  const picker = $("overlayPicker");
  picker.innerHTML = "";
  const mm = await multimaskMeta(frame);
  if (rec !== state.log[state.pos]) return; // stale async result

  const options = [{ key: "final", label: "output mask" }];
  if (mm.available) {
    for (let k = 0; k < (mm.n_candidates || 0); k++) {
      options.push({ key: String(k), label: `cand ${k}`, num: `iou ${fmt(mm.ious[k], 2)}` });
    }
    if (mm.token0_iou !== undefined) {
      options.push({
        key: "token0",
        label: "token0",
        num: `iou ${fmt(mm.token0_iou, 2)} · stab ${fmt(mm.token0_stability, 2)}`,
      });
    }
  }
  if (!options.some((o) => o.key === state.overlayKey)) state.overlayKey = "final";

  for (const opt of options) {
    const chip = document.createElement("button");
    chip.className = `chip ${state.overlayKey === opt.key ? "active" : ""}`;
    chip.append(opt.label);
    if (opt.num) {
      const n = document.createElement("span");
      n.className = "num";
      n.textContent = opt.num;
      chip.append(n);
    }
    chip.onclick = () => {
      state.overlayKey = opt.key;
      applyMainOverlay(frame);
      renderOverlayPicker(rec);
    };
    picker.append(chip);
  }
}

function renderScores(rec) {
  const dl = $("scoreList");
  dl.innerHTML = "";
  const thr = state.config.mf_threshold;
  const s = rec.scores;
  const rows = [];
  if (s) {
    const selectable = thr !== undefined && s.eff_iou_score > thr;
    rows.push([
      "eff_iou_score",
      `${fmt(s.eff_iou_score)} <span class="badge ${selectable ? "ok" : "below"}">` +
        `${selectable ? "selectable" : `≤ thr ${thr}`}</span>`,
    ]);
    rows.push(["object_score_logits", fmt(s.object_score_logits, 2)]);
    rows.push(["iou_score (best cand)", fmt(s.iou_score)]);
  } else {
    rows.push(["scores", "— (no tracked output stored for this frame)"]);
  }
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.innerHTML = v;
    dl.append(dt, dd);
  }
}

function renderEventDetail(rec) {
  const bits = [];
  if (rec.trimmed && rec.trimmed.length) bits.push(`trimmed: ${rec.trimmed.join(", ")}`);
  if (rec.cleared && rec.cleared.length) bits.push(`cleared non-cond: ${rec.cleared.join(", ")}`);
  if (rec.evicted_corrections && rec.evicted_corrections.length)
    bits.push(`evicted corrections: ${rec.evicted_corrections.join(", ")}`);
  $("eventDetail").textContent = bits.join(" · ");
}

function bankScore(rec, frame) {
  const scores = (rec.mem_state && rec.mem_state.non_cond_scores) || {};
  const v = scores[String(frame)] !== undefined ? scores[String(frame)] : scores[frame];
  return v === undefined ? undefined : Number(v);
}

function renderPools(rec) {
  const att = rec.attention;
  const condCells = $("condPool").querySelector(".cells");
  const spatCells = $("spatialPool").querySelector(".cells");
  const ptrCells = $("ptrPool").querySelector(".cells");
  condCells.innerHTML = spatCells.innerHTML = ptrCells.innerHTML = "";

  if (!att) {
    const note = state.log[state.pos].event === "track"
      ? "no attention roster (revisited consolidated frame)"
      : "no memory read this step (mask used as output)";
    condCells.append(makeCell({ empty: true, emptyLabel: note }));
    return;
  }

  const pinned = state.config.keep_first_cond_frame && att.cond_selected.length
    ? Math.min(...att.cond_selected) : null;
  for (const f of att.cond_selected) {
    condCells.append(makeCell({
      frame: f, roleClass: "role-cond",
      tag: f === pinned ? "pinned first" : "correction",
    }));
  }
  for (const f of att.cond_unselected) {
    const cell = makeCell({ frame: f, roleClass: "role-none", tag: "not attended" });
    cell.style.opacity = 0.6;
    condCells.append(cell);
  }

  const numSlots = (state.config.num_maskmem || 7) - 1;
  const bySlot = {};
  const spatialFrames = new Set();
  for (const m of att.spatial_mem) {
    if (!m.is_cond) bySlot[m.t_pos] = m;
    spatialFrames.add(m.frame_idx);
  }
  for (let slot = 1; slot <= numSlots; slot++) {
    const m = bySlot[slot];
    if (!m) {
      spatCells.append(makeCell({ empty: true, emptyLabel: `slot ${slot}` }));
    } else {
      const cell = makeCell({
        frame: m.frame_idx, roleClass: "role-spatial",
        capRight: fmt(bankScore(rec, m.frame_idx), 2),
        tag: `slot ${slot}${slot === numSlots ? " (prev frame)" : ""}`,
      });
      spatCells.append(cell);
    }
  }

  for (const p of att.obj_ptrs) {
    const role = p.is_cond ? "role-cond" : (spatialFrames.has(p.frame_idx) ? "role-spatial" : "role-ptr");
    ptrCells.append(makeCell({
      frame: p.frame_idx, roleClass: role,
      capRight: fmt(bankScore(rec, p.frame_idx), 2),
      tag: p.is_cond ? `cond · dist ${p.pos}` : `rank ${p.pos}${role === "role-ptr" ? " · ptr only" : ""}`,
    }));
  }
}

function renderBank(rec) {
  const cells = $("bankCells");
  cells.innerHTML = "";
  if (!rec.mem_state) return;
  const att = rec.attention || { cond_selected: [], spatial_mem: [], obj_ptrs: [] };
  const spatialSet = new Set(att.spatial_mem.filter((m) => !m.is_cond).map((m) => m.frame_idx));
  const ptrSet = new Set(att.obj_ptrs.filter((p) => !p.is_cond).map((p) => p.frame_idx));
  const condSet = new Set(rec.mem_state.cond);

  const entries = [];
  for (const f of rec.mem_state.cond) entries.push({ frame: f, kind: "bank" });
  for (const k of Object.keys(rec.mem_state.non_cond_scores)) {
    entries.push({ frame: Number(k), kind: "bank" });
  }
  for (const f of rec.trimmed || []) entries.push({ frame: f, kind: "trimmed" });
  entries.sort((a, b) => a.frame - b.frame);

  for (const e of entries) {
    let roleClass = "role-none";
    let tag = "retained, unattended";
    if (e.kind === "trimmed") { roleClass = ""; tag = "trimmed this step"; }
    else if (condSet.has(e.frame)) { roleClass = "role-cond"; tag = "cond"; }
    else if (spatialSet.has(e.frame)) { roleClass = "role-spatial"; tag = "spatial + ptr"; }
    else if (ptrSet.has(e.frame)) { roleClass = "role-ptr"; tag = "ptr only"; }
    cells.append(makeCell({
      frame: e.frame, roleClass, trimmed: e.kind === "trimmed",
      capRight: fmt(bankScore(rec, e.frame), 2),
      tag,
    }));
  }
}

function buildRoleLegend() {
  const legend = $("roleLegend");
  const items = [
    ["--role-cond", "conditioning"],
    ["--role-spatial", "spatial + ptr"],
    ["--role-ptr", "ptr only"],
    ["--border", "retained, unattended"],
    ["--status-trimmed", "trimmed this step"],
  ];
  for (const [v, label] of items) {
    const chip = document.createElement("span");
    chip.className = "chip legend-chip";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = cssVar(v);
    chip.append(sw, label);
    legend.append(chip);
  }
}

/* ---------------- score timeline chart ---------------- */

function chartData() {
  const pts = [];
  const corrections = [];
  for (let i = 0; i < state.log.length; i++) {
    const r = state.log[i];
    if (r.event === "track") {
      pts.push({ pos: i, frame: r.frame_idx, eff: r.scores ? r.scores.eff_iou_score : null });
    } else if (r.event === "correct") {
      corrections.push(r.frame_idx);
    }
  }
  return { pts, corrections };
}

function drawChart() {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.parentElement.clientWidth;
  const cssH = 140;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const { pts, corrections } = chartData();
  if (!pts.length) return;
  const m = { l: 44, r: 10, t: 10, b: 18 };
  const w = cssW - m.l - m.r;
  const h = cssH - m.t - m.b;
  const maxFrame = Math.max(1, pts[pts.length - 1].frame);
  const x = (f) => m.l + (f / maxFrame) * w;
  const y = (v) => m.t + (1 - v) * h;
  state.chart = { pts, x, m, w, h, maxFrame };

  // grid + axis (recessive)
  ctx.strokeStyle = cssVar("--grid");
  ctx.lineWidth = 1;
  ctx.fillStyle = cssVar("--muted");
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "right";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.beginPath();
    ctx.moveTo(m.l, y(v) + 0.5);
    ctx.lineTo(m.l + w, y(v) + 0.5);
    ctx.stroke();
    ctx.fillText(v.toFixed(2), m.l - 6, y(v) + 3);
  }
  ctx.textAlign = "center";
  const step = Math.max(1, Math.pow(10, Math.floor(Math.log10(maxFrame))));
  for (let f = 0; f <= maxFrame; f += step) ctx.fillText(String(f), x(f), cssH - 4);

  // threshold line (status color, dashed, labeled)
  const thr = state.config.mf_threshold;
  if (thr !== undefined) {
    ctx.strokeStyle = cssVar("--status-trimmed");
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(m.l, y(thr));
    ctx.lineTo(m.l + w, y(thr));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--status-trimmed");
    ctx.textAlign = "left";
    ctx.fillText(`mf_threshold ${thr}`, m.l + 4, y(thr) - 4);
  }

  // correction markers (ticks at top)
  ctx.strokeStyle = cssVar("--role-cond");
  ctx.lineWidth = 1.5;
  for (const f of corrections) {
    ctx.beginPath();
    ctx.moveTo(x(f), m.t);
    ctx.lineTo(x(f), m.t + 8);
    ctx.stroke();
  }

  // score line (2px), gaps where eff is null
  ctx.strokeStyle = cssVar("--role-spatial");
  ctx.lineWidth = 2;
  ctx.beginPath();
  let pen = false;
  for (const p of pts) {
    if (p.eff === null || p.eff === undefined) { pen = false; continue; }
    if (pen) ctx.lineTo(x(p.frame), y(p.eff));
    else { ctx.moveTo(x(p.frame), y(p.eff)); pen = true; }
  }
  ctx.stroke();

  // current position marker
  const cur = state.log[state.pos];
  if (cur) {
    ctx.strokeStyle = cssVar("--axis");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(cur.frame_idx) + 0.5, m.t);
    ctx.lineTo(x(cur.frame_idx) + 0.5, m.t + h);
    ctx.stroke();
  }

  canvas.onmousemove = chartHover;
  canvas.onmouseleave = () => { $("chartTip").hidden = true; drawChart(); };
  canvas.onclick = (e) => {
    const p = nearestPoint(e);
    if (p) seek(p.pos);
  };
}

function nearestPoint(e) {
  const c = state.chart;
  if (!c) return null;
  const rect = $("chart").getBoundingClientRect();
  const px = e.clientX - rect.left;
  const frame = ((px - c.m.l) / c.w) * c.maxFrame;
  let best = null;
  for (const p of c.pts) {
    if (!best || Math.abs(p.frame - frame) < Math.abs(best.frame - frame)) best = p;
  }
  return best;
}

function chartHover(e) {
  const p = nearestPoint(e);
  const tip = $("chartTip");
  if (!p) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.textContent = `frame ${p.frame} · eff ${fmt(p.eff)}`;
  tip.style.left = `${e.clientX + 12}px`;
  tip.style.top = `${e.clientY - 28}px`;
}

init();
