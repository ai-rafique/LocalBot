// App shell: routing, sidebar (navigation + chat history), status, theme.
import { api } from "./api.js";
import { confirmDialog, fill, h, icon, popover, toastError } from "./ui.js";
import * as chat from "./chat.js";
import * as documents from "./documents.js";
import * as evaluate from "./evaluate.js";
import * as experiments from "./experiments.js";
import * as settings from "./settings.js";

const $ = (id) => document.getElementById(id);
const NAV = [
  { key: "chat", label: "Chat", icon: "chat", href: "#/chat" },
  { key: "documents", label: "Documents", icon: "docs", href: "#/documents" },
  { key: "evaluate", label: "Evaluate", icon: "grade", href: "#/evaluate" },
  { key: "experiments", label: "Experiments", icon: "flask", href: "#/experiments" },
  { key: "settings", label: "Settings", icon: "settings", href: "#/settings" },
];
const VIEWS = { chat, documents, evaluate, experiments, settings };

export const app = {
  status: null,
  conversations: [],
  ungraded: 0,
  refreshHistory, refreshStatus, navigate: (hash) => { location.hash = hash; },
};

let cleanup = null;

function parseRoute() {
  const raw = location.hash.replace(/^#\/?/, "") || "chat";
  const [path, query = ""] = raw.split("?");
  const parts = path.split("/").filter(Boolean);
  return { view: VIEWS[parts[0]] ? parts[0] : "chat", parts: parts.slice(1), query: Object.fromEntries(new URLSearchParams(query)) };
}

async function route() {
  const r = parseRoute();
  if (cleanup) { try { cleanup(); } catch (e) {} cleanup = null; }
  document.querySelectorAll(".popover").forEach((p) => p.remove());
  renderNav(r.view);
  renderHistory();
  document.getElementById("app").classList.remove("menu-open");
  const main = $("main");
  main.replaceChildren();
  try {
    cleanup = (await VIEWS[r.view].render(main, r, app)) || null;
  } catch (e) {
    toastError(e);
  }
}

function renderNav(active) {
  $("nav").replaceChildren(...NAV.map((n) => h("a", { href: n.href, class: n.key === active ? "active" : "" },
    icon(n.icon), n.label,
    n.key === "evaluate" && app.ungraded ? h("span", { class: "count", title: "Answers not graded yet" }, app.ungraded) : null)));
}

function groupLabel(iso) {
  const d = new Date(iso), now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = (day(now) - day(d)) / 86400000;
  if (diff < 1) return "Today";
  if (diff < 2) return "Yesterday";
  if (diff < 7) return "Previous 7 days";
  if (diff < 30) return "Previous 30 days";
  return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function renderHistory() {
  const r = parseRoute();
  const activeId = r.view === "chat" ? r.parts[0] : null;
  const list = $("history");
  if (!app.conversations.length) {
    list.replaceChildren(h("div", { class: "history-empty" }, "No chats yet"));
    return;
  }
  const nodes = [];
  let last = null;
  for (const c of app.conversations) {
    const g = groupLabel(c.updated);
    if (g !== last) { nodes.push(h("div", { class: "history-group" }, g)); last = g; }
    const more = h("button", { class: "icon-btn", "aria-label": "Chat options", onclick: (e) => { e.preventDefault(); chatMenu(more, c); } }, icon("more"));
    nodes.push(h("div", { class: "history-item" + (c.id === activeId ? " active" : "") },
      h("a", { href: `#/chat/${c.id}`, title: c.title }, c.title), more));
  }
  list.replaceChildren(...nodes);
}

function chatMenu(anchor, c) {
  const pop = popover(anchor, h("div", { style: { display: "flex", flexDirection: "column", gap: "2px" } },
    h("button", { class: "btn ghost sm", style: { justifyContent: "flex-start" }, onclick: () => { pop.close(); renameChat(c); } }, icon("edit"), "Rename"),
    h("button", { class: "btn ghost sm danger", style: { justifyContent: "flex-start" }, onclick: () => { pop.close(); deleteChat(c); } }, icon("trash"), "Delete"),
  ), { align: "end" });
  pop.style.width = "170px";
}

export async function renameChat(c) {
  const title = prompt("Rename chat", c.title);
  if (!title || title === c.title) return;
  try { await api.patch(`/api/conversations/${c.id}`, { title }); await refreshHistory(); if (parseRoute().parts[0] === c.id) route(); }
  catch (e) { toastError(e); }
}

export async function deleteChat(c) {
  if (!(await confirmDialog("Delete chat?", `“${c.title}” will be removed from your chats. Its answers stay in Evaluate, with any grades.`, { confirm: "Delete", danger: true }))) return;
  try {
    await api.del(`/api/conversations/${c.id}`);
    await refreshHistory();
    if (parseRoute().parts[0] === c.id) location.hash = "#/chat";
  } catch (e) { toastError(e); }
}

async function refreshHistory() {
  try { app.conversations = await api.get("/api/conversations"); } catch (e) { app.conversations = []; }
  renderHistory();
}

async function refreshStatus() {
  try {
    const [st, ev] = await Promise.all([api.get("/api/status"), api.get("/api/evaluate/stats")]);
    app.status = st;
    app.ungraded = ev.ungraded;
  } catch (e) {
    app.status = null;
  }
  renderFooter();
  renderNav(parseRoute().view);
}

function renderFooter() {
  const st = app.status;
  const rows = [];
  if (!st) {
    rows.push(h("div", { class: "status-line" }, h("span", { class: "dot bad" }), h("span", {}, "LocalBot server unreachable")));
  } else if (!st.ollama) {
    rows.push(h("div", { class: "status-line", title: "Start Ollama, e.g. `ollama serve`" }, h("span", { class: "dot bad" }), h("span", {}, "Ollama isn't running")));
  } else {
    const loaded = st.loaded.find((m) => m.name === st.chat_model || m.name === `${st.chat_model}:latest`);
    const where = loaded ? (loaded.gpu_share >= 0.99 ? "on GPU" : `${Math.round(100 * loaded.gpu_share)}% on GPU`) : "not loaded yet";
    const ok = st.chat_model_installed && st.embed_model_installed;
    rows.push(h("a", { class: "status-line", href: "#/settings/models", title: ok ? "Chat model" : "A configured model isn't installed" },
      h("span", { class: `dot ${ok ? (loaded && loaded.gpu_share < 0.99 ? "warn" : "ok") : "bad"}` }),
      h("span", {}, `${st.chat_model} · ${ok ? where : "not installed"}`)));
  }
  if (st?.running_experiment) {
    const r = st.running_experiment;
    rows.push(h("a", { class: "status-line", href: "#/experiments" }, h("span", { class: "dot busy" }),
      h("span", {}, `Experiment ${r.done}/${r.total}: ${r.name}`)));
  }
  if (st?.job) {
    const j = st.job;
    rows.push(h("a", { class: "status-line", href: "#/documents", title: j.step }, h("span", { class: "dot busy" }),
      h("span", {}, j.total > 1 ? `${j.title} · ${Math.round((100 * j.done) / j.total)}%` : j.title)));
  }
  const dark = document.documentElement.dataset.theme === "dark"
    || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  rows.push(h("div", { class: "footer-row" },
    h("span", { class: "status-line faint" }, h("span", {}, "Private · runs on this computer")),
    h("button", { class: "icon-btn", "aria-label": "Toggle theme", title: dark ? "Light theme" : "Dark theme", onclick: () => {
      const next = dark ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) {}
      renderFooter();
    } }, icon(dark ? "sun" : "moon"))));
  fill($("sidebar-footer"), rows);
}

function init() {
  $("new-chat").replaceChildren(icon("plus"), "New chat");
  $("new-chat").addEventListener("click", () => { location.hash = "#/chat"; if (parseRoute().parts.length === 0) route(); });
  $("sidebar-close").replaceChildren(icon("x"));
  $("sidebar-close").addEventListener("click", () => $("app").classList.remove("menu-open"));
  $("scrim").addEventListener("click", () => $("app").classList.remove("menu-open"));
  window.addEventListener("hashchange", route);
  refreshHistory();
  refreshStatus();
  setInterval(refreshStatus, 8000);
  route();
}

export function openMenu() { $("app").classList.add("menu-open"); }
export function menuButton() {
  return h("button", { class: "icon-btn only-mobile", "aria-label": "Menu", onclick: openMenu }, icon("menu"));
}

init();
