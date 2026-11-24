"use strict";
// Maintenance Copilot chat client: login, conversations, SSE streaming, ticket confirmation.

const $ = (id) => document.getElementById(id);
const state = { token: localStorageGet("token"), user: localStorageGet("user"), conv: null, busy: false };

function localStorageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function localStorageSet(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* private mode */ } }

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401 && !path.startsWith("/api/auth")) { logout(); throw new Error("Please log in again"); }
  if (!res.ok) {
    let detail = res.statusText;
    try { const body = await res.json(); detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch { /* not json */ }
    throw new Error(detail);
  }
  return res;
}

// ---------------------------------------------------------------- auth
async function authenticate(kind) {
  $("login-error").textContent = "";
  try {
    const res = await api(`/api/auth/${kind}`, { method: "POST",
      body: JSON.stringify({ username: $("username").value.trim(), password: $("password").value }) });
    const data = await res.json();
    state.token = data.access_token; state.user = data.username;
    localStorageSet("token", state.token); localStorageSet("user", state.user);
    await showApp();
  } catch (e) { $("login-error").textContent = e.message; }
}

function logout() {
  state.token = null; state.user = null; state.conv = null;
  localStorageSet("token", null); localStorageSet("user", null);
  $("app").classList.add("hidden"); $("login").classList.remove("hidden");
}

async function showApp() {
  $("login").classList.add("hidden"); $("app").classList.remove("hidden");
  $("who").textContent = state.user || "";
  await loadConversations();
}

// ------------------------------------------------------- conversations
async function loadConversations() {
  const list = await (await api("/api/conversations")).json();
  const nav = $("conversations"); nav.innerHTML = "";
  for (const c of list) {
    const row = document.createElement("div");
    row.className = "conv" + (c.id === state.conv ? " active" : "");
    const title = document.createElement("button"); title.className = "title"; title.textContent = c.title;
    title.onclick = () => openConversation(c.id);
    const del = document.createElement("button"); del.className = "del"; del.title = "Delete"; del.textContent = "×";
    del.onclick = async () => { await api(`/api/conversations/${c.id}`, { method: "DELETE" }); if (state.conv === c.id) newChat(); await loadConversations(); };
    row.append(title, del); nav.append(row);
  }
}

function newChat() {
  state.conv = null; hideTickets(); closeSource();
  $("messages").innerHTML = document.querySelector("#empty-template")?.innerHTML || emptyHtml;
  bindExamples(); loadConversations();
}

async function openConversation(id) {
  hideTickets(); closeSource();
  const data = await (await api(`/api/conversations/${id}`)).json();
  state.conv = id; $("messages").innerHTML = "";
  for (const m of data.messages) {
    if (m.role === "user") addUser(m.content);
    else renderAssistant(addAssistant(), m.content, m.meta || {});
  }
  if (data.pending) showConfirm(addAssistant(), data.pending);
  await loadConversations(); scrollDown();
}

// ---------------------------------------------------------------- render
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function markdown(text) {
  const lines = escapeHtml(text).split("\n"); let html = ""; let inList = false;
  for (const raw of lines) {
    const line = raw.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[((?:S\d+)(?:\s*,\s*S\d+)*)\]/g, (_, ids) => ids.split(/\s*,\s*/).map((i) => `<button type="button" class="cite" data-sid="${i}" title="Show source ${i}">${i}</button>`).join(""));
    const item = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)/);
    if (item) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${item[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.trim()) html += `<p>${line.replace(/^#+\s*/, "")}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

function addUser(text) {
  const el = document.createElement("div"); el.className = "msg user";
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  clearEmpty(); $("messages").append(el); scrollDown(); return el;
}

function addAssistant() {
  const el = document.createElement("div"); el.className = "msg assistant";
  el.innerHTML = `<div class="bubble"><div class="tools"></div><div class="answer"></div><div class="extra"></div></div>`;
  clearEmpty(); $("messages").append(el); return el;
}

function renderAssistant(el, answer, meta) {
  el.querySelector(".answer").innerHTML = markdown(answer || "");
  const tools = el.querySelector(".tools");
  if (meta.tools && !tools.children.length) for (const t of meta.tools) tools.append(chip(t.tool, t.ok ? "ok" : "err"));
  const extra = el.querySelector(".extra"); extra.innerHTML = "";
  const sources = meta.sources || [];
  el._sources = sources;
  if (sources.length) {
    const cited = new Set(meta.cited || []);
    const items = sources.map((s) => `<li><button type="button" class="src" data-sid="${s.id}"><span class="cite">${s.id}</span> ${escapeHtml(s.label)}${cited.has(s.id) ? "" : " <span class=\"muted\">(not cited)</span>"}</button></li>`).join("");
    extra.insertAdjacentHTML("beforeend", `<details class="sources"><summary>${sources.length} sources, ${cited.size} cited</summary><ul>${items}</ul></details>`);
  }
  if (meta.latency_s != null) {
    const u = meta.usage || {};
    const cost = meta.cost_usd != null ? ` · $${meta.cost_usd.toFixed(5)}` : "";
    extra.insertAdjacentHTML("beforeend", `<div class="meta">${meta.latency_s.toFixed(1)} s · ${u.llm_calls || 0} LLM calls · ${(u.input || 0) + (u.output || 0)} tokens${cost}</div>`);
  }
}

// ------------------------------------------------------------- sources
function slugify(t) { return t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "section"; }

// Readable rendering of tool output: tables instead of raw JSON.
const label = (k) => escapeHtml(String(k).replace(/_/g, " "));
const isScalar = (v) => v === null || typeof v !== "object";

function cell(key, v) {
  if (v === null || v === undefined) return '<span class="muted">-</span>';
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number" && /probability/.test(key)) {
    const p = v * 100;  // keep "99.99%" and "0.01%" distinguishable from 100% / 0%
    return p > 0 && p < 0.05 ? "<0.05%" : `${p.toFixed(p >= 99.95 && p < 100 ? 2 : 1)}%`;
  }
  if (isScalar(v)) return escapeHtml(String(v));
  if (Array.isArray(v) && v.every(isScalar)) return v.length ? v.map((x) => escapeHtml(String(x))).join("<br>") : '<span class="muted">none</span>';
  return smartRender(v);
}

function smartRender(value) {
  if (Array.isArray(value)) {
    if (!value.length) return '<span class="muted">none</span>';
    if (value.every((x) => x && typeof x === "object" && !Array.isArray(x))) {
      const cols = [...new Set(value.flatMap(Object.keys))].slice(0, 12);
      const head = cols.map((c) => `<th>${label(c)}</th>`).join("");
      const rows = value.map((r) => `<tr>${cols.map((c) => `<td>${cell(c, r[c])}</td>`).join("")}</tr>`).join("");
      return `<table class="grid"><tr>${head}</tr>${rows}</table>`;
    }
    return value.map((x) => cell("", x)).join("<br>");
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value).filter(([k]) => k !== "source_id");
    // An object of objects (e.g. risk per horizon) reads best as one table, one row per key.
    if (entries.length && entries.every(([, v]) => v && typeof v === "object" && !Array.isArray(v))) {
      const cols = [...new Set(entries.flatMap(([, v]) => Object.keys(v)))].slice(0, 10);
      const head = `<th></th>${cols.map((c) => `<th>${label(c)}</th>`).join("")}`;
      const rows = entries.map(([k, v]) => `<tr><th>${label(k)}</th>${cols.map((c) => `<td>${cell(c, v[c])}</td>`).join("")}</tr>`).join("");
      return `<table class="grid"><tr>${head}</tr>${rows}</table>`;
    }
    // Simple values in a key/value table; lists and nested objects as full-width sections.
    const simple = entries.filter(([, v]) => isScalar(v) || (Array.isArray(v) && v.every(isScalar)));
    const nested = entries.filter((e) => !simple.includes(e));
    const kv = simple.length ? `<table class="kv">${simple.map(([k, v]) => `<tr><th>${label(k)}</th><td>${cell(k, v)}</td></tr>`).join("")}</table>` : "";
    return kv + nested.map(([k, v]) => `<div class="sub"><p class="sub-title">${label(k)}</p>${smartRender(v)}</div>`).join("");
  }
  return cell("", value);
}

function originHtml(source) {
  const o = source.origin;
  if (!o) return '<p class="muted">Origin details are not stored for this older answer.</p>';
  const kind = o.kind === "model" ? "ML model prediction" : "Database query";
  const window = o.window ? ` · Time window: ${escapeHtml(o.window.from)} to ${escapeHtml(o.window.to)}` : "";
  const buttons = (o.machine_ids || []).slice(0, 5).map((m) => `<button type="button" class="secondary record-btn" data-mid="${m}" data-from="${escapeHtml(o.window?.from || "")}" data-to="${escapeHtml(o.window?.to || "")}">Machine ${m} records</button>`);
  if (o.kind === "model") buttons.push('<a class="button-link" href="/model" target="_blank" rel="noopener">How the model was tested</a>');
  return `<div class="origin"><span class="badge ${o.kind}">${kind}</span><p>${escapeHtml(o.summary)}</p>`
    + `<p class="muted small">Tables: ${escapeHtml((o.tables || []).join(", "))}${window}</p>`
    + `<p class="muted small">Data: <a href="${escapeHtml(o.dataset.url)}" target="_blank" rel="noopener">${escapeHtml(o.dataset.name)}</a></p>`
    + (o.caveat ? `<p class="caveat">${escapeHtml(o.caveat)}</p>` : "")
    + (buttons.length ? `<p class="links">${buttons.join("")}</p>` : "") + "</div>";
}

function technicalHtml(source) {
  const queries = (source.queries || []).map((qr) => `<pre>${escapeHtml(qr.sql)}\n-- parameters: ${escapeHtml(JSON.stringify(qr.params))}</pre>`).join("");
  return `<details class="tech"><summary>Technical details</summary>`
    + `<p class="muted small">Tool call</p><pre>${escapeHtml(source.tool)}(${escapeHtml(JSON.stringify(source.args || {}))})</pre>`
    + (queries ? `<p class="muted small">SQL the tool ran (read-only connection)</p>${queries}` : "")
    + `<p class="muted small">Raw output</p><pre>${escapeHtml(JSON.stringify(source.data || {}, null, 2))}</pre></details>`;
}

async function openSource(source) {
  const panel = $("source-panel"); const body = $("source-body");
  $("source-title").textContent = `${source.id} · ${source.label}`;
  if (source.tool === "search_manuals") {
    let s = source;
    if (!s.text && s.chunk_id) {  // answers saved before passages were stored
      try { s = { ...s, ...(await (await api(`/api/passages/${encodeURIComponent(s.chunk_id)}`)).json()) }; } catch { /* shown below */ }
    }
    const docId = s.doc_id || (s.chunk_id || "").split("#")[0];
    const anchor = s.page ? `page-${s.page}` : slugify(s.section || "");
    const pageNote = s.page ? ` · page ${s.page}` : "";
    const links = [`<a href="/docs/${encodeURIComponent(docId)}#${anchor}" target="_blank" rel="noopener">Open full document${pageNote}</a>`];
    if ((s.source_url || "").startsWith("http")) links.push(`<a href="${escapeHtml(s.source_url)}${s.page ? `#page=${s.page}` : ""}" target="_blank" rel="noopener">Original PDF${pageNote}</a>`);
    body.innerHTML = `<p class="muted">${escapeHtml(s.doc || s.title || "")} &rsaquo; ${escapeHtml(s.section || "")}${pageNote}${s.synthetic ? ' · <span class="badge">synthetic</span>' : ""}</p>`
      + `<blockquote class="passage">${escapeHtml(s.text || "Passage text not available.")}</blockquote>`
      + `<p class="links">${links.join("")}</p>`;
  } else {
    body.innerHTML = originHtml(source)
      + `<h4>What the tool returned</h4>`
      + (source.data ? smartRender(source.data) : '<p class="muted">Tool output was not stored for this older answer.</p>')
      + technicalHtml(source);
  }
  panel.classList.remove("hidden");
  document.querySelector(".main").classList.add("panel-open");
}

// ------------------------------------------------------- machine records
// Normal bands from the (synthetic) fleet handbook, derived from the fleet's own readings.
const SIGNALS = [
  { key: "volt", name: "Voltage", unit: "V", normal: [146, 197] },
  { key: "rotate", name: "Rotation", unit: "rpm", normal: [359, 531] },
  { key: "pressure", name: "Pressure", unit: "psi", normal: [84, 119] },
  { key: "vibration", name: "Vibration", unit: "mm/s", normal: [32, 49] },
];
const toTime = (s) => new Date(String(s).replace(" ", "T")).getTime();
const day = (t) => new Date(t).toISOString().slice(0, 10);

function lineChart(rows, sig, shown, win, events) {
  const W = 660, H = 120, L = 46, R = 8, T = 8, B = 20;
  const t0 = toTime(shown.from), t1 = toTime(shown.to);
  const vals = rows.map((r) => r[sig.key]);
  const lo = Math.min(...vals, sig.normal[0]), hi = Math.max(...vals, sig.normal[1]);
  const x = (t) => L + ((t - t0) / Math.max(t1 - t0, 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / Math.max(hi - lo, 1e-9)) * (H - T - B);
  const pts = rows.map((r) => `${x(toTime(r.ts)).toFixed(1)},${y(r[sig.key]).toFixed(1)}`).join(" ");
  const wx0 = x(Math.max(toTime(win.from), t0)), wx1 = x(Math.min(toTime(win.to), t1));
  const marks = events.map((e) => {
    const ex = x(toTime(e.ts)); const cls = e.kind === "failure" ? "ev-failure" : "ev-error";
    return `<line class="${cls}" x1="${ex}" x2="${ex}" y1="${T}" y2="${H - B}"><title>${escapeHtml(e.label)}</title></line>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${sig.name} readings">`
    + `<rect class="win" x="${wx0}" y="${T}" width="${Math.max(wx1 - wx0, 1)}" height="${H - T - B}"><title>Time window the answer used</title></rect>`
    + `<rect class="normal" x="${L}" y="${y(sig.normal[1])}" width="${W - L - R}" height="${y(sig.normal[0]) - y(sig.normal[1])}"><title>Handbook normal range</title></rect>`
    + marks + `<polyline class="line" points="${pts}"/>`
    + `<text x="4" y="${T + 8}" class="axis">${hi.toFixed(0)}</text><text x="4" y="${H - B}" class="axis">${lo.toFixed(0)}</text>`
    + `<text x="${L}" y="${H - 4}" class="axis">${day(t0)}</text><text x="${W - R}" y="${H - 4}" class="axis" text-anchor="end">${day(t1)}</text>`
    + `<text x="${L + 4}" y="${T + 12}" class="title">${sig.name} (${sig.unit})</text></svg>`;
}

async function openRecord(mid, from, to) {
  const params = new URLSearchParams(); if (from) params.set("start", from.replace(" ", "T")); if (to) params.set("end", to.replace(" ", "T"));
  $("record-title").textContent = `Machine ${mid} - records from the fleet database`;
  $("record-body").innerHTML = '<p class="muted">Loading...</p>';
  $("record-view").classList.remove("hidden");
  try {
    const r = await (await api(`/api/machines/${mid}/record?${params}`)).json();
    const events = [
      ...r.errors.map((e) => ({ ts: e.ts, kind: "error", label: `${e.error} at ${e.ts}` })),
      ...r.maintenance.filter((m) => m.reason === "failure").map((m) => ({ ts: m.ts, kind: "failure", label: `${m.component} failed at ${m.ts}` })),
    ];
    const charts = r.telemetry.length ? SIGNALS.map((s) => lineChart(r.telemetry, s, r.shown, r.window, events)).join("") : '<p class="muted">No sensor readings in this period.</p>';
    $("record-body").innerHTML =
      `<p>Model <strong>${escapeHtml(r.machine.model)}</strong>, ${r.machine.age_years} years in service. Showing ${escapeHtml(r.shown.from)} to ${escapeHtml(r.shown.to)}: ${r.telemetry.length} hourly readings.</p>`
      + `<p class="legend"><span class="key win"></span> time window the answer used <span class="key normal"></span> handbook normal range <span class="key err"></span> error <span class="key fail"></span> failure</p>`
      + charts
      + `<div class="record-tables"><div><h4>Errors (${r.errors.length})</h4>${smartRender(r.errors)}</div>`
      + `<div><h4>Component replacements (${r.maintenance.length})</h4>${smartRender(r.maintenance)}</div>`
      + `<div><h4>Last replacement per component</h4>${smartRender(r.last_replacement)}</div></div>`
      + `<p class="muted small">Source: <a href="${escapeHtml(r.dataset.url)}" target="_blank" rel="noopener">${escapeHtml(r.dataset.name)}</a>. Hover a line for details.</p>`;
  } catch (e) { $("record-body").innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`; }
}

function closeRecord() { $("record-view").classList.add("hidden"); }

function closeSource() {
  $("source-panel").classList.add("hidden");
  document.querySelector(".main").classList.remove("panel-open");
}

function chip(name, cls) { const c = document.createElement("span"); c.className = `chip ${cls}`; c.textContent = name; return c; }

function showConfirm(el, pending) {
  const box = document.createElement("div"); box.className = "confirm";
  const rows = (pending.tickets || []).map((t) => `<dl><dt>Machine</dt><dd>${t.machine_id}</dd><dt>Component</dt><dd>${escapeHtml(t.component || "-")}</dd><dt>Priority</dt><dd>${escapeHtml(t.priority)}</dd><dt>Summary</dt><dd>${escapeHtml(t.summary)}</dd></dl>`).join("");
  box.innerHTML = `<strong>Create this maintenance ticket?</strong>${rows}<div class="row"><button class="yes">Create ticket</button><button class="secondary no">Cancel</button></div>`;
  box.querySelector(".yes").onclick = () => { box.remove(); resume(el, true); };
  box.querySelector(".no").onclick = () => { box.remove(); resume(el, false); };
  el.querySelector(".extra").append(box); scrollDown();
}

function clearEmpty() { const e = $("messages").querySelector(".empty"); if (e) e.remove(); }
function scrollDown() { const m = $("messages"); m.scrollTop = m.scrollHeight; }

// ---------------------------------------------------------------- chat
async function send(text) {
  if (state.busy || !text.trim()) return;
  if (!state.conv) state.conv = (await (await api("/api/conversations", { method: "POST" })).json()).id;
  addUser(text);
  await stream(`/api/chat/${state.conv}`, { message: text }, addAssistant());
  loadConversations();
}

async function resume(el, approved) { await stream(`/api/chat/${state.conv}/confirm`, { approved }, el); }

async function stream(url, body, el) {
  setBusy(true);
  let answer = ""; const tools = el.querySelector(".tools"); const running = {};
  try {
    const res = await api(url, { method: "POST", body: JSON.stringify(body) });
    const reader = res.body.getReader(); const decoder = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const data = block.split("\n").find((l) => l.startsWith("data: "));
        if (!data) continue;
        const ev = JSON.parse(data.slice(6));
        if (ev.type === "token") { answer += ev.text; el.querySelector(".answer").innerHTML = markdown(answer); scrollDown(); }
        else if (ev.type === "tool_start") { const c = chip(ev.tool, "run"); (running[ev.tool] ||= []).push(c); tools.append(c); }
        else if (ev.type === "tool_end") { const c = (running[ev.tool] || []).shift(); if (c) c.className = `chip ${ev.ok ? "ok" : "err"}`; }
        else if (ev.type === "confirm") showConfirm(el, ev);
        else if (ev.type === "done") renderAssistant(el, ev.answer, ev);
        else if (ev.type === "error") el.querySelector(".answer").innerHTML = `<p class="error">${escapeHtml(ev.message)}</p>`;
      }
    }
  } catch (e) { el.querySelector(".answer").innerHTML = `<p class="error">${escapeHtml(e.message)}</p>`; }
  finally { setBusy(false); scrollDown(); }
}

function setBusy(b) { state.busy = b; $("send").disabled = b; }

// ------------------------------------------------------------- tickets
async function showTickets() {
  const list = await (await api("/api/tickets")).json();
  const rows = list.map((t) => `<tr><td>#${t.id}</td><td>${t.machine_id}</td><td>${escapeHtml(t.component || "-")}</td><td>${escapeHtml(t.priority)}</td><td>${escapeHtml(t.summary)}</td><td>${escapeHtml(String(t.created_at).slice(0, 16))}</td></tr>`).join("");
  $("tickets").innerHTML = `<h2>Maintenance tickets</h2>${list.length ? `<table><tr><th>ID</th><th>Machine</th><th>Component</th><th>Priority</th><th>Summary</th><th>Created</th></tr>${rows}</table>` : "<p class=\"muted\">No tickets yet.</p>"}<p><button class="secondary" id="close-tickets">Back to chat</button></p>`;
  $("close-tickets").onclick = hideTickets;
  $("tickets").classList.remove("hidden"); $("messages").classList.add("hidden"); $("chat-form").classList.add("hidden");
}
function hideTickets() { $("tickets").classList.add("hidden"); $("messages").classList.remove("hidden"); $("chat-form").classList.remove("hidden"); }

// ---------------------------------------------------------------- wiring
const emptyHtml = $("messages").innerHTML;
function bindExamples() { for (const b of document.querySelectorAll(".example")) b.onclick = () => send(b.textContent); }

$("login-form").onsubmit = (e) => { e.preventDefault(); authenticate("login"); };
$("register-btn").onclick = () => { if ($("login-form").reportValidity()) authenticate("register"); };
$("logout").onclick = logout;
$("new-chat").onclick = newChat;
$("show-tickets").onclick = showTickets;
$("chat-form").onsubmit = (e) => { e.preventDefault(); const t = $("input").value; $("input").value = ""; send(t); };
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-form").requestSubmit(); } });
$("messages").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-sid]"); if (!btn) return;
  const msg = btn.closest(".msg"); const src = (msg?._sources || []).find((x) => x.id === btn.dataset.sid);
  if (src) openSource(src);
});
$("source-close").onclick = closeSource;
$("source-body").addEventListener("click", (e) => {
  const b = e.target.closest(".record-btn"); if (b) openRecord(b.dataset.mid, b.dataset.from, b.dataset.to);
});
$("record-close").onclick = closeRecord;
$("record-view").addEventListener("click", (e) => { if (e.target.id === "record-view") closeRecord(); });
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("record-view").classList.contains("hidden")) closeRecord(); else closeSource();
});
bindExamples();
if (state.token) showApp().catch(logout); else logout();
