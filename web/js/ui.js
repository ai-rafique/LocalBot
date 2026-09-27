// DOM helpers, icons, toasts, dialogs and formatting shared by all views.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "html") el.innerHTML = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// Views write optional parts as `cond ? node : null`. h() skips those, but
// the DOM's append/replaceChildren would insert them as the text "null".
const present = (nodes) => nodes.flat(Infinity).filter((n) => n !== null && n !== undefined && n !== false);
export function fill(el, ...nodes) { el.replaceChildren(...present(nodes)); return el; }
export function add(el, ...nodes) { el.append(...present(nodes)); return el; }

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const PATHS = {
  chat: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>',
  docs: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>',
  grade: '<path d="M9 11l3 3 8-8"/><path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9"/>',
  flask: '<path d="M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3"/><path d="M7.5 14h9"/>',
  settings: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  send: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
  check: '<path d="M5 12l5 5 9-10"/>',
  up: '<path d="M7 11v9H4v-9zM7 11l4-8a2 2 0 0 1 2 2v4h5.5a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.3 20H7"/>',
  down: '<path d="M7 13V4H4v9zM7 13l4 8a2 2 0 0 0 2-2v-4h5.5a2 2 0 0 0 2-2.3l-1.2-7A2 2 0 0 0 17.3 4H7"/>',
  panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5M4 20h16"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  alert: '<path d="M12 3l9.5 17h-19z"/><path d="M12 10v4M12 17.5v.5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.5"/>',
  play: '<path d="M7 4l13 8-13 8z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  more: '<circle cx="5" cy="12" r="1.2" fill="currentColor"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/><circle cx="19" cy="12" r="1.2" fill="currentColor"/>',
  sliders: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  back: '<path d="M19 12H5M11 6l-6 6 6 6"/>',
  compare: '<path d="M8 3v18M16 3v18M3 8h5M16 16h5"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  sparkle: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M6 18l2.5-2.5M15.5 8.5L18 6"/>',
};
export function icon(name, cls = "") {
  const span = document.createElement("span");
  span.innerHTML = `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true">${PATHS[name] || ""}</svg>`;
  return span.firstChild;
}

export function toast(message, { error = false, ms = 3200 } = {}) {
  const t = h("div", { class: "toast" + (error ? " error" : ""), role: error ? "alert" : "status" }, message);
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), error ? Math.max(ms, 6000) : ms);
}
export const toastError = (e) => toast(e.message || String(e), { error: true });

// Modal dialog. content: Node; buttons: [{label, cls, onClick -> false keeps it open}]
export function dialog({ title, content, buttons = [], wide = false, onClose }) {
  const dlg = h("dialog", { class: wide ? "wide" : "" });
  const close = () => { dlg.close(); };
  dlg.append(
    h("div", { class: "dlg-head" }, h("h3", {}, title),
      h("button", { class: "icon-btn", "aria-label": "Close", onclick: close }, icon("x"))),
    h("div", { class: "dlg-body" }, content),
  );
  if (buttons.length) {
    dlg.append(h("div", { class: "dlg-foot" }, buttons.map((b) => h("button", {
      class: "btn " + (b.cls || ""), onclick: async (e) => {
        const btn = e.currentTarget; btn.disabled = true;
        try { if ((await b.onClick?.()) !== false) close(); } catch (err) { toastError(err); }
        finally { btn.disabled = false; }
      },
    }, b.label))));
  }
  dlg.addEventListener("close", () => { dlg.remove(); onClose?.(); });
  document.body.append(dlg);
  dlg.showModal();
  return dlg;
}

export function confirmDialog(title, text, { confirm = "Confirm", danger = false } = {}) {
  return new Promise((resolve) => {
    let ok = false;
    dialog({
      title, content: h("p", { style: { margin: 0 } }, text), onClose: () => resolve(ok),
      buttons: [{ label: "Cancel" }, { label: confirm, cls: danger ? "danger" : "primary", onClick: () => { ok = true; } }],
    });
  });
}

// Floating panel anchored to an element; closes on outside click / Escape.
export function popover(anchor, content, { align = "start", above = false } = {}) {
  document.querySelectorAll(".popover").forEach((p) => p.remove());
  const pop = h("div", { class: "popover", role: "dialog" }, content);
  document.body.append(pop);
  const r = anchor.getBoundingClientRect();
  const w = pop.offsetWidth, ph = pop.offsetHeight;
  let left = align === "end" ? r.right - w : r.left;
  left = Math.max(12, Math.min(left, window.innerWidth - w - 12));
  let top = above || r.bottom + ph + 8 > window.innerHeight ? r.top - ph - 8 : r.bottom + 8;
  pop.style.left = left + "px"; pop.style.top = Math.max(12, top) + "px";
  const close = () => { pop.remove(); document.removeEventListener("mousedown", out, true); document.removeEventListener("keydown", key, true); };
  const out = (e) => { if (!pop.contains(e.target) && !anchor.contains(e.target)) close(); };
  const key = (e) => { if (e.key === "Escape") close(); };
  setTimeout(() => { document.addEventListener("mousedown", out, true); document.addEventListener("keydown", key, true); });
  pop.close = close;
  return pop;
}

export function switchEl(checked, onChange, label) {
  const input = h("input", { type: "checkbox", checked: !!checked, onchange: (e) => onChange(e.target.checked) });
  return h("label", { class: "switch" }, input, h("span", { class: "track" }), label ? h("span", {}, label) : null);
}

export function segmented(options, value, onChange) {
  const wrap = h("div", { class: "segmented", role: "tablist" });
  const render = (v) => {
    wrap.replaceChildren(...options.map((o) => h("button", {
      class: o.value === v ? "on" : "", role: "tab", "aria-selected": o.value === v ? "true" : "false",
      onclick: () => { render(o.value); onChange(o.value); },
    }, o.label)));
  };
  render(value);
  return wrap;
}

export function debounce(fn, ms = 300) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function timeAgo(iso) {
  if (!iso) return "";
  const d = new Date(iso.length <= 19 ? iso : iso);
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)} d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric" });
}
export const pct = (x, digits = 0) => (x === null || x === undefined ? "—" : `${(100 * x).toFixed(digits)}%`);
export const num = (x, digits = 2) => (x === null || x === undefined ? "—" : Number(x).toFixed(digits));
export function bytes(n) {
  if (n === null || n === undefined) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch (e) { toast("Couldn't copy", { error: true }); }
}
export function download(filename, text, type = "application/json") {
  const a = h("a", { href: URL.createObjectURL(new Blob([text], { type })), download: filename });
  document.body.append(a); a.click(); a.remove();
}
export function tip(text) { return h("span", { class: "info-tip", "data-tip": text, "data-tip-wide": "" }, "?"); }
export function gradeBadge(grade) {
  if (!grade) return h("span", { class: "badge" }, "Not graded");
  return h("span", { class: `badge ${grade}` }, grade[0].toUpperCase() + grade.slice(1));
}
