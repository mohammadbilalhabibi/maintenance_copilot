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

function dataTable(obj) {
  const rows = Object.entries(obj || {}).filter(([k]) => k !== "source_id").map(([k, v]) => {
    const val = v !== null && typeof v === "object" ? `<pre>${escapeHtml(JSON.stringify(v, null, 2))}</pre>` : escapeHtml(String(v));
    return `<tr><th>${escapeHtml(k)}</th><td>${val}</td></tr>`;
  }).join("");
  return `<table class="kv">${rows}</table>`;
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
    const args = escapeHtml(JSON.stringify(source.args || {}));
    body.innerHTML = `<p class="muted">What the agent got from <code>${escapeHtml(source.tool)}</code>(${args}):</p>`
      + (source.data ? dataTable(source.data) : '<p class="muted">Tool output was not stored for this older answer.</p>');
  }
  panel.classList.remove("hidden");
  document.querySelector(".main").classList.add("panel-open");
}

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
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSource(); });
bindExamples();
if (state.token) showApp().catch(logout); else logout();
