"use strict";
const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");

const KO = {
  label: { excavator: "굴착기", forklift: "지게차", "crane hook": "크레인 훅" },
  motion: { approaching: "접근 중", receding: "멀어지는 중", moving: "이동 중", stationary: "정지" },
  state: { idle: "대기", starting: "준비 중", connecting: "연결 중", running: "분석 중", error: "오류" },
};
const COLOR = { ok: "#3fb950", warn: "#f0b429", danger: "#f85149", equip: "#ff9f43", roi: "#e879f9" };

// one id per browser tab: the server keeps each visitor's analysis session separate
const SID = (() => {
  try {
    let v = sessionStorage.getItem("sid");
    if (!v) { v = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now()).replace(/[^A-Za-z0-9_-]/g, ""); sessionStorage.setItem("sid", v); }
    return v;
  } catch (_) { return "tab" + Math.random().toString(36).slice(2, 12) + Date.now(); }
})();
// one id per browser (survives tabs): remembers which demos this visitor has already run
const VID = (() => {
  try {
    let v = localStorage.getItem("vid");
    if (!v) { v = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now()).replace(/[^A-Za-z0-9_-]/g, ""); localStorage.setItem("vid", v); }
    return v;
  } catch (_) { return SID; }
})();
const wsUrl = (path) => (location.protocol === "https:" ? "wss://" : "ws://") + location.host + path + "?sid=" + encodeURIComponent(SID);

const S = {
  public: false, done: new Set(), tab: "demo", demo: null, demos: [], status: { state: "idle" }, meta: null, zones: { danger: [], rest: [] },
  drawing: null, draft: [], lastDecision: null, roi: null, alertUntil: 0, camStream: null, ingest: null, sendTimer: null,
};

// ------------------------------------------------------------------ helpers
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const hhmmss = (ts) => { const d = new Date(ts * 1000), p = (n) => String(n).padStart(2, "0"); return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; };
const helmetKo = (s) => s.startsWith("uncertain") ? "불확실 " + (s.match(/[\d.]+/) || [""])[0]
  : ({ worn: "착용", absent: "미착용", held_in_hand: "손에 파지", not_detected: "미확인" }[s] || s);
const harnessKo = (s) => ({ hooked: "결속", unhooked: "미결속", not_detected: "미확인" }[s] || (s.startsWith("uncertain") ? "불확실" : s));
function reasonKo(r) {
  const m = r.match(/^worker (\d+): (.*)$/); if (!m) return r;
  const who = `작업자 #${m[1]}`, t = m[2], eq = (l) => KO.label[l] || l;
  let x;
  if ((x = t.match(/^(.+) TTC ([\d.]+)s$/))) return `${who}: ${eq(x[1])} 충돌 예상 ${x[2]}초`;
  if ((x = t.match(/^(.+) ([\d.]+)m and closing$/))) return `${who}: ${eq(x[1])} ${x[2]}m 접근 중`;
  if (t === "in danger zone") return `${who}: 위험구역 진입`;
  if ((x = t.match(/^([\d.]+)m up a ladder/))) return `${who}: 사다리 ${x[1]}m 고소 작업, 안전고리 결속 미확인`;
  if ((x = t.match(/^torso tilt (\d+) deg$/))) return `${who}: 상체 기울기 ${x[1]}°`;
  return `${who}: ${t}`;
}
function msg(text, err = false) { const m = $("msg"); m.textContent = text || ""; m.className = "msg" + (err ? " err" : ""); }

async function api(path, body) {
  path += (path.includes("?") ? "&" : "?") + "sid=" + encodeURIComponent(SID) + "&vid=" + encodeURIComponent(VID);
  const r = await fetch(path, { method: body === undefined ? "GET" : "POST", headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  if (!r.ok) { let d = ""; try { d = (await r.json()).detail; } catch (_) {} throw new Error(d || r.statusText); }
  return r.json();
}

// ------------------------------------------------------------------ source panel
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  S.tab = t.dataset.tab;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.id === "pane-" + S.tab));
}));
document.querySelectorAll("input[name=camKind]").forEach((r) => r.addEventListener("change", () => {
  const server = camKind() === "server";
  $("serverCam").hidden = !server;
  $("camHint").textContent = server ? "이 서버(PC)에 연결된 카메라를 직접 엽니다. macOS는 서버를 실행한 터미널/앱에 카메라 권한이 필요합니다."
    : "브라우저가 카메라 권한을 요청합니다. 영상은 이 PC의 서버로만 전송됩니다.";
}));
const camKind = () => document.querySelector("input[name=camKind]:checked").value;

function renderDemos() {
  const box = $("demoList"); box.replaceChildren();
  S.demos.forEach((d) => {
    const b = el("button", "demo" + (S.demo === d.id ? " on" : ""));
    b.type = "button"; b.append(el("b", null, d.title), el("span", null, d.expect));
    if (S.done.has(d.id)) b.append(el("em", "done", "✓ 실행함"));
    b.addEventListener("click", () => { S.demo = d.id; renderDemos(); $("demoWatch").textContent = d.watch; });
    box.append(b);
  });
}

$("probeBtn").addEventListener("click", async () => {
  $("probeMsg").textContent = "검색 중…";
  try {
    const r = await api("/api/cameras");
    const sel = $("camIndex"); sel.replaceChildren();
    r.cameras.forEach((c) => { const o = el("option", null, `카메라 ${c.index}${c.ok ? " (" + c.size + ")" : " – 사용 불가"}`); o.value = c.index; o.disabled = !c.ok; sel.append(o); });
    const ok = r.cameras.find((c) => c.ok);
    if (ok) sel.value = ok.index;
    $("probeMsg").textContent = r.any ? "사용 가능한 카메라를 찾았습니다." : "열린 카메라가 없습니다. 서버를 실행한 터미널에 카메라 권한(시스템 설정 > 개인정보 보호 > 카메라)이 있는지 확인하거나, '이 브라우저의 웹캠'을 사용하세요.";
  } catch (e) { $("probeMsg").textContent = "검색 실패: " + e.message; }
});

// consent (public mode): asked once per browser tab
function askConsent() {
  try { if (sessionStorage.getItem("consent") === "1") return Promise.resolve(true); } catch (_) {}
  return new Promise((resolve) => {
    const box = $("consent"), ok = $("cOk"), agree = $("cAgree");
    agree.checked = false; ok.disabled = true; box.hidden = false; agree.focus();
    const done = (v) => { box.hidden = true; agree.onchange = ok.onclick = $("cCancel").onclick = null; if (v) { try { sessionStorage.setItem("consent", "1"); } catch (_) {} } resolve(v); };
    agree.onchange = () => (ok.disabled = !agree.checked);
    ok.onclick = () => done(true); $("cCancel").onclick = () => done(false);
  });
}
$("privBtn").addEventListener("click", () => { try { sessionStorage.removeItem("consent"); } catch (_) {} askConsent(); });

$("startBtn").addEventListener("click", async () => {
  msg("");
  try {
    if (S.public && !(await askConsent())) return;
    await stopCamera();
    const precision = $("precision").value, loop = $("loop").checked;
    if (S.tab === "demo") {
      if (!S.demo) return msg("데모 시나리오를 선택하세요.", true);
      await api("/api/start", { kind: "demo", demo_id: S.demo, precision, loop });
      S.done.add(S.demo); renderDemos();
    } else if (S.tab === "cctv") {
      const url = $("cctvUrl").value.trim();
      if (!url) return msg("카메라 주소를 입력하세요.", true);
      await api("/api/start", { kind: "url", url, precision });
    } else if (camKind() === "server") {
      await api("/api/start", { kind: "webcam", index: Number($("camIndex").value), precision });
    } else {
      await startBrowserCam(precision);
    }
  } catch (e) { msg(e.message || String(e), true); await stopCamera(); }
});
$("stopBtn").addEventListener("click", async () => { await stopCamera(); await api("/api/stop", {}); });

// Browser webcam: getUserMedia -> JPEG frames over /ws/ingest (~12 fps)
async function startBrowserCam(precision) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("이 브라우저는 웹캠을 지원하지 않거나 보안 컨텍스트(localhost)가 아닙니다.");
  try {
    S.camStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) { throw new Error("웹캠 권한이 거부되었거나 카메라를 찾을 수 없습니다 (" + e.name + ")"); }
  const v = document.createElement("video"); v.muted = true; v.playsInline = true; v.srcObject = S.camStream; await v.play();
  await api("/api/start", { kind: "browser", precision, loop: false });
  const ws = new WebSocket(wsUrl("/ws/ingest"));
  ws.binaryType = "arraybuffer"; S.ingest = ws;
  const off = document.createElement("canvas");
  ws.onopen = () => {
    S.sendTimer = setInterval(() => {
      if (ws.bufferedAmount > 600000 || !v.videoWidth) return;
      const sc = Math.min(1, 960 / v.videoWidth); off.width = Math.round(v.videoWidth * sc); off.height = Math.round(v.videoHeight * sc);
      off.getContext("2d").drawImage(v, 0, 0, off.width, off.height);
      off.toBlob((b) => b && ws.readyState === 1 && ws.send(b), "image/jpeg", 0.72);
    }, 1000 / 12);
  };
  ws.onclose = () => { if (S.sendTimer) { clearInterval(S.sendTimer); S.sendTimer = null; } };
}
async function stopCamera() {
  if (S.sendTimer) { clearInterval(S.sendTimer); S.sendTimer = null; }
  if (S.ingest) { S.ingest.close(); S.ingest = null; }
  if (S.camStream) { S.camStream.getTracks().forEach((t) => t.stop()); S.camStream = null; }
}

// ------------------------------------------------------------------ websocket stream
let ws, pingTimer;
function connect() {
  ws = new WebSocket(wsUrl("/ws/stream"));
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { clearInterval(pingTimer); pingTimer = setInterval(() => ws.readyState === 1 && ws.send("ping"), 15000); };
  ws.onclose = () => { clearInterval(pingTimer); setTimeout(connect, 1500); };
  ws.onmessage = async (e) => {
    if (typeof e.data === "string") return onText(JSON.parse(e.data));
    const n = new DataView(e.data).getUint32(0);
    const meta = JSON.parse(new TextDecoder().decode(e.data.slice(4, 4 + n)));
    const bmp = await createImageBitmap(new Blob([e.data.slice(4 + n)], { type: "image/jpeg" }));
    if (S.lastBitmap) S.lastBitmap.close();
    S.lastBitmap = bmp; S.meta = meta; drawFrame(bmp, meta); // kept so zone drawing can redraw between frames
    updateChip(meta);
  };
}

function onText(m) {
  if (m.type === "status") setStatus(m);
  else if (m.type === "history") { resetBoard(); m.events.forEach(applyEvent); }
  else if (m.type === "event") applyEvent(m);
}

function setStatus(m) {
  S.status = m;
  if (m.state === "starting") resetBoard(); // new run: drop the previous run's actions and sequence
  $("stateText").textContent = KO.state[m.state] || m.state;
  $("statePill").className = "pill " + m.state;
  const running = ["running", "starting", "connecting"].includes(m.state);
  $("startBtn").disabled = running; $("stopBtn").disabled = !running;
  $("empty").style.display = m.state === "running" && S.meta ? "none" : "";
  if (m.state === "idle") { S.meta = null; ctx.clearRect(0, 0, cv.width, cv.height); $("empty").style.display = ""; $("chip").hidden = true; $("think").hidden = true; setViewportLevel(null); }
  if (m.message && m.state !== "running") msg(m.message, m.state === "error");
  else if (m.state === "running") msg("");
}

function updateChip(meta) {
  const c = $("chip"); c.hidden = false;
  c.textContent = `${meta.fps} fps${meta.vision_ms ? " · 비전 " + meta.vision_ms + "ms" : ""} · 작업자 ${meta.workers.length}명${meta.equipment.length ? " · 장비 " + meta.equipment.length : ""}`;
  $("think").hidden = !meta.busy;
  $("empty").style.display = "none";
  const risk = meta.risk.reasons;
  $("reasons").replaceChildren();
  if (!risk.length) $("reasons").append(el("span", "none", "없음"));
  risk.forEach((r) => $("reasons").append(el("span", null, reasonKo(r))));
  if (Date.now() > S.alertUntil) $("alertBar").hidden = true;
}

// ------------------------------------------------------------------ drawing
function pill(text, x, y, color, fs) {
  ctx.font = `600 ${fs}px -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif`;
  const w = ctx.measureText(text).width + fs * 0.9, h = fs * 1.5;
  ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(x, y - h, w, h, 4); ctx.fill();
  ctx.fillStyle = "#06121f"; ctx.textBaseline = "middle"; ctx.fillText(text, x + fs * 0.45, y - h / 2 + 1);
  return h;
}
function poly(points, color, label, fs) {
  if (points.length < 2) return;
  const W = cv.width, H = cv.height;
  ctx.beginPath(); points.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H))); ctx.closePath();
  ctx.fillStyle = color + "26"; ctx.fill(); ctx.setLineDash([8, 6]); ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.stroke(); ctx.setLineDash([]);
  if (label) pill(label, points[0][0] * W, points[0][1] * H, color, fs * 0.85);
}

function drawFrame(img, meta) {
  if (cv.width !== img.width || cv.height !== img.height) { cv.width = img.width; cv.height = img.height; }
  const W = cv.width, H = cv.height, fs = Math.max(12, Math.round(W / 62));
  ctx.drawImage(img, 0, 0);
  const z = { danger: S.zones.danger.length ? S.zones.danger : meta.zones.danger, rest: S.zones.rest.length ? S.zones.rest : meta.zones.rest };
  z.danger.forEach((p) => poly(p, COLOR.danger, "위험구역", fs));
  z.rest.forEach((p) => poly(p, COLOR.ok, "휴게구역", fs));

  meta.equipment.forEach((e) => {
    const [x1, y1, x2, y2] = [e.box[0] * W, e.box[1] * H, e.box[2] * W, e.box[3] * H];
    ctx.lineWidth = 3; ctx.strokeStyle = COLOR.equip; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    pill(`${KO.label[e.label] || e.label} · ${KO.motion[e.motion] || e.motion}`, x1, y1, COLOR.equip, fs);
  });
  meta.workers.forEach((w) => {
    const [x1, y1, x2, y2] = [w.box[0] * W, w.box[1] * H, w.box[2] * W, w.box[3] * H], c = COLOR[w.flag];
    ctx.lineWidth = w.flag === "ok" ? 2 : 4; ctx.strokeStyle = c; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const lines = [`작업자 #${w.id}`, `안전모 ${helmetKo(w.helmet)}`];
    if (w.ladder) lines.push(`사다리 ${w.climb ?? "?"}m · 안전고리 ${harnessKo(w.harness)}`);
    if (w.tilt != null && w.tilt >= 15) lines.push(`상체 기울기 ${Math.round(w.tilt)}°`);
    if (w.eq) lines.push(`${KO.label[w.eq.label] || w.eq.label} ${w.eq.dist}m${w.eq.ttc != null ? " · 충돌 " + w.eq.ttc + "초 전" : ""}`);
    if (w.aware === false) lines.push("장비 인지 못함");
    let y = y2 + 4;
    lines.forEach((t, i) => { y += pill(t, x1, y + fs * 1.5, i === 0 ? c : "#cfd8e3", fs * 0.9) + 2; });
  });

  if (S.roi && Date.now() < S.roi.until) {
    const [x1, y1, x2, y2] = [S.roi.box[0] * W, S.roi.box[1] * H, S.roi.box[2] * W, S.roi.box[3] * H];
    ctx.setLineDash([10, 6]); ctx.lineWidth = 3; ctx.strokeStyle = COLOR.roi; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
    pill(`Jev 확인 영역${S.roi.conf != null ? " " + Math.round(S.roi.conf * 100) + "%" : ""}`, x1, y1, COLOR.roi, fs * 0.9);
  }
  if (S.draft.length) { poly(S.draft, COLOR[S.drawing === "danger" ? "danger" : "ok"], null, fs); S.draft.forEach(([x, y]) => { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(x * W, y * H, 5, 0, 7); ctx.fill(); }); }
}

// ------------------------------------------------------------------ situation board
const levelOf = (d) => (d.p >= 0.8 || d.risk >= 4 ? "danger" : d.p >= 0.5 ? "warn" : "ok");
const LVL = { ok: "정상", warn: "주의 · 관리자 확인 필요", danger: "위험 · 즉시 조치" };

function setViewportLevel(l) { $("viewport").className = "viewport" + (l === "danger" || l === "warn" ? " " + l : ""); }

function resetBoard() {
  $("actList").replaceChildren(); S.lastDecision = null; S.roi = null;
  $("actEmpty").hidden = false; $("actCount").textContent = "";
  $("lvl").className = "lvl idle"; $("lvlLabel").textContent = "대기 중";
  ["pVal", "rVal", "planVal", "modeVal", "latVal"].forEach((i) => ($(i).textContent = "–"));
  document.querySelectorAll("#rBar i").forEach((i) => (i.className = ""));
  $("narr").replaceChildren(el("p", "none", "분석이 시작되면 작업자별 행동 흐름이 표시됩니다.")); setViewportLevel(null);
  $("alertBar").hidden = true;
}

function applyEvent(e) {
  if (e.kind === "notice") msg(e.text);
  else if (e.kind === "decision") applyDecision(e);
  else if (e.kind === "action") addAction(e);
}

function applyDecision(d) {
  S.lastDecision = d;
  const l = levelOf(d);
  $("lvl").className = "lvl " + l; $("lvlLabel").textContent = LVL[l];
  $("pVal").textContent = Math.round(d.p * 100) + "%";
  $("rVal").textContent = d.risk.toFixed(1) + " / 5";
  document.querySelectorAll("#rBar i").forEach((b, i) => (b.className = d.risk > i ? (d.risk >= 4 ? "hi" : "on") : ""));
  $("planVal").textContent = d.plan_ko;
  $("modeVal").textContent = d.mode === "image_fallback" ? "이미지 폴백 (멀티모달)" : "텍스트 시퀀스";
  $("latVal").textContent = `${d.latency_ms}ms · ${d.tokens_in.toLocaleString()}토큰${d.live ? "" : " · 대체응답"}`;
  setViewportLevel(l);
  const n = $("narr"); n.replaceChildren();
  if (!d.narratives.length) n.append(el("p", "none", "작업자가 없습니다."));
  d.narratives.forEach((x) => { const p = el("p"); p.append(el("b", null, `작업자 #${x.id}  `), document.createTextNode(x.text)); n.append(p); });
  if (d.roi_box) S.roi = { box: d.roi_box, conf: d.roi_conf, until: Date.now() + 6000 };
}

function addAction(a) {
  $("actEmpty").hidden = true;
  const list = $("actList"), first = list.firstElementChild;
  // merge repeats of the same action within 5 s (the interlock re-fires on every Jev call while the danger lasts)
  if (first && first.dataset.action === a.action && a.ts - Number(first.dataset.ts) < 5) {
    const n = Number(first.dataset.n) + 1; first.dataset.n = n; first.dataset.ts = a.ts;
    first.querySelector("time").textContent = hhmmss(a.ts); first.querySelector("em").textContent = `×${n}`; return;
  }
  const li = el("li", a.tier); li.dataset.action = a.action; li.dataset.ts = a.ts; li.dataset.n = 1;
  li.append(el("time", null, hhmmss(a.ts)));
  const t = el("div", "t", a.label); t.append(el("em", null, ""));
  li.append(t);
  if (a.snapshot) { const im = el("img"); im.src = a.snapshot; im.alt = "스냅샷"; li.append(im); }
  li.append(el("div", "why", `${a.tier === "immediate" ? "즉시" : "확인"} · P ${Math.round(a.p * 100)}% · 위험 ${Number(a.risk).toFixed(1)} · ${a.mode === "image_fallback" ? "이미지 폴백" : "텍스트"}`));
  list.prepend(li);
  while (list.children.length > 40) list.lastElementChild.remove();
  $("actCount").textContent = `${list.children.length}건`;
  if (Date.now() - a.ts * 1000 < 8000) { // live event, not a history replay
    const bar = $("alertBar"); bar.hidden = false; bar.className = "alertbar" + (a.tier === "immediate" ? "" : " warn");
    bar.textContent = (a.tier === "immediate" ? "즉시 조치 · " : "확인 요청 · ") + a.label; S.alertUntil = Date.now() + 4000;
  }
}

// ------------------------------------------------------------------ zone drawing
document.querySelectorAll(".zbtn").forEach((b) => b.addEventListener("click", () => {
  const z = b.dataset.zone; S.drawing = S.drawing === z ? null : z; S.draft = [];
  document.querySelectorAll(".zbtn").forEach((x) => x.classList.toggle("drawing", x.dataset.zone === S.drawing));
  cv.classList.toggle("drawing", !!S.drawing);
}));
$("zoneClear").addEventListener("click", () => { S.zones = { danger: [], rest: [] }; S.draft = []; finishDrawing(false); pushZones(); });
const norm = (e) => { const r = cv.getBoundingClientRect(); return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))]; };
cv.addEventListener("click", (e) => { if (S.drawing) { S.draft.push(norm(e)); redraw(); } });
cv.addEventListener("dblclick", () => finishDrawing(true));
window.addEventListener("keydown", (e) => { if (!S.drawing) return; if (e.key === "Enter") finishDrawing(true); if (e.key === "Escape") { S.draft = []; finishDrawing(false); } });
function finishDrawing(commit) {
  if (commit && S.drawing && S.draft.length >= 3) { S.zones[S.drawing].push(S.draft.slice()); pushZones(); }
  S.draft = []; S.drawing = null; cv.classList.remove("drawing");
  document.querySelectorAll(".zbtn").forEach((x) => x.classList.remove("drawing")); redraw();
}
function redraw() { if (S.meta && S.lastBitmap) drawFrame(S.lastBitmap, S.meta); }
const pushZones = () => api("/api/zones", S.zones).catch((e) => msg("구역 저장 실패: " + e.message, true));

// remaining demo time (public mode)
setInterval(() => {
  const dl = S.status && S.status.deadline, running = ["running", "starting", "connecting"].includes(S.status.state);
  const pill = $("timerPill");
  if (!dl || !running) { pill.hidden = true; return; }
  const left = Math.max(0, Math.round(dl - Date.now() / 1000));
  pill.hidden = false; $("timerVal").textContent = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
  pill.classList.toggle("warn", left <= 30);
}, 500);

// ------------------------------------------------------------------ boot
(async function boot() {
  try {
    const info = await api("/api/info");
    S.public = !!info.public; S.done = new Set(info.demos_done || []);
    if (S.public) {
      $("demoQuota").hidden = false; // hosted demo: no CCTV URLs or server cameras; the visitor's own browser webcam, the demos, and the source link
      $("tabCctv").hidden = true; $("serverOpt").hidden = true; $("serverCam").hidden = true;
      $("pane-cam").querySelector("input[value=browser]").checked = true;
      $("loop").parentElement.hidden = false;
      $("foot").hidden = false; $("srcLink").href = info.source_url;
      $("cMin").textContent = Math.round(info.limits.session_s / 60);
      document.querySelectorAll(".tab").forEach((t) => (t.style.flex = "1"));
    }
    S.demos = info.demos; S.demo = info.demos[0]?.id || null; renderDemos(); $("demoWatch").textContent = info.demos[0]?.watch || "";
    const p = $("jevPill");
    p.textContent = info.jev.live ? `Jev LIVE · ${info.jev.model}` : "Jev 오프라인 대체 응답기 (실제 모델 아님)";
    p.className = "pill " + (info.jev.live ? "live" : "sim");
    p.title = info.jev.live ? "TypeSafe Jev API에 실제로 호출합니다" : "TYPESAFE_API_KEY가 없어 로컬 휴리스틱 응답기로 동작합니다. 확률은 Jev의 판단이 아닙니다.";
    setStatus(info.status);
  } catch (e) { msg("서버에 연결하지 못했습니다: " + e.message, true); }
  connect();
})();
