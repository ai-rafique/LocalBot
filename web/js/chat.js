// Chat: conversation thread, streaming answers, sources, grading, inspector.
import { api, stream } from "./api.js";
import { renderMarkdown } from "./markdown.js";
import { app, menuButton, renameChat } from "./main.js";
import { control, effectText, invalidateSettings, loadModels, loadSettings } from "./tuning.js";
import { add, copyText, fill, h, icon, num, popover, segmented, switchEl, thumb, toast, toastError } from "./ui.js";

const STAGES = { rewrite: "Understanding your question", search: "Searching your documents", rerank: "Reranking passages", write: "Writing the answer",
                 verify: "Checking each statement against its sources" };
let tagVocab = null;

export async function render(main, route) {
  const cid = route.parts[0] || null;
  const state = {
    conv: null, turns: [], selected: null, controller: null,
    inspector: localStorage.getItem("inspector") === "1", tab: "sources",
  };
  if (cid) {
    try {
      const conv = await api.get(`/api/conversations/${cid}`);
      state.conv = conv; state.turns = conv.turns;
      state.selected = conv.turns.length ? conv.turns[conv.turns.length - 1].id : null;
    } catch (e) {
      toastError(e); location.hash = "#/chat"; return;
    }
  }
  if (!tagVocab) api.get("/api/experiments/meta").then((m) => { tagVocab = m.tags; }).catch(() => {});

  const thread = h("div", { class: "thread" });
  const inner = h("div", { class: "thread-inner" });
  thread.append(inner);
  const inspector = h("aside", { class: "inspector" });
  const titleEl = h("div", { class: "title", title: "Rename", onclick: () => state.conv && editTitle() });
  const inspectBtn = h("button", { class: "icon-btn", "aria-label": "Details", title: "Sources and details", onclick: () => toggleInspector() }, icon("panel"));
  const topbar = h("div", { class: "topbar" }, menuButton(), titleEl, h("div", { class: "spacer" }), inspectBtn);

  const textarea = h("textarea", { rows: 1, placeholder: "Ask about your documents…", "aria-label": "Message" });
  const sendBtn = h("button", { class: "send-btn", "aria-label": "Send", disabled: true }, icon("send"));
  const tuneBtn = h("button", { class: "tune-btn", onclick: () => quickTune(tuneBtn) }, icon("sliders"), h("span", {}, "…"));
  // Search in: the whole library or one document (e.g. one documentation archive).
  const readScope = () => { try { return localStorage.getItem("scope") || ""; } catch { return ""; } };
  const scopeSelect = h("select", { class: "scope-select", "aria-label": "Search in", title: "Which documents answers are searched in",
    onchange: () => { try { localStorage.setItem("scope", scopeSelect.value); } catch { /* private window */ } } },
    h("option", { value: "" }, "All documents"));
  api.get("/api/documents").then((lib) => {
    const names = lib.documents.filter((d) => d.chunks).map((d) => d.name);
    scopeSelect.append(...names.map((n) => h("option", { value: n }, n)));
    scopeSelect.value = names.includes(readScope()) ? readScope() : "";
    scopeSelect.hidden = names.length < 2;
  }).catch(() => { scopeSelect.hidden = true; });
  const composer = h("div", { class: "composer-wrap" },
    h("div", { class: "composer" }, textarea, sendBtn),
    h("div", { class: "composer-foot" },
      h("span", {}, "Answers come only from your documents — check the cited sources."),
      h("label", { class: "hint", style: { display: "flex", alignItems: "center", gap: "6px" } }, "Search in", scopeSelect), tuneBtn));

  const chatMain = h("div", { class: "chat-main" }, topbar, thread, composer);
  const root = h("div", { class: "chat" }, chatMain, inspector);
  main.append(root);

  // ---- rendering
  function renderTitle() {
    titleEl.textContent = state.conv ? state.conv.title : "New chat";
    titleEl.style.cursor = state.conv ? "text" : "default";
  }
  function editTitle() {
    const input = h("input", { type: "text", class: "title-edit", value: state.conv.title });
    titleEl.replaceWith(input);
    input.focus(); input.select();
    const done = async (save) => {
      input.replaceWith(titleEl);
      const title = input.value.trim();
      if (save && title && title !== state.conv.title) {
        try { await api.patch(`/api/conversations/${state.conv.id}`, { title }); state.conv.title = title; app.refreshHistory(); }
        catch (e) { toastError(e); }
      }
      renderTitle();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); });
    input.addEventListener("blur", () => done(true));
  }

  function renderThread() {
    if (!state.turns.length) {
      inner.replaceChildren(welcome());
      return;
    }
    inner.replaceChildren(...state.turns.flatMap((t) => turnNodes(t)));
  }

  function welcome() {
    const st = app.status;
    if (st && st.documents === 0) {
      return h("div", { class: "welcome" },
        h("div", { class: "empty-state" }, h("div", { class: "icon-circle" }, icon("docs")),
          h("h3", {}, "Add documents to get started"),
          h("p", {}, "LocalBot answers questions from your own PDF, Word, Markdown and text files. They never leave this computer."),
          h("a", { class: "btn primary", href: "#/documents" }, icon("upload"), "Add documents")));
    }
    const ideas = [
      ["Summarize", "Summarize the main points of my documents"],
      ["Exact values", "What exact values, ids or limits are specified?"],
      ["Procedures", "What are the steps to get started?"],
      ["Details", "What does the documentation say about error handling?"],
    ];
    return h("div", { class: "welcome" },
      h("span", { class: "brand-mark", style: { display: "inline-block", width: "44px", height: "44px", borderRadius: "12px", backgroundSize: "26px" } }),
      h("h2", {}, "What would you like to know?"),
      h("p", {}, st ? `Searching ${st.documents} document${st.documents === 1 ? "" : "s"} (${st.chunks} passages), privately on this computer.` : ""),
      h("div", { class: "suggestions" }, ideas.map(([t, q]) =>
        h("button", { class: "suggestion", onclick: () => { textarea.value = q; autosize(); send(); } }, h("b", {}, t), q))));
  }

  function usedHits(t) { return (t.hits || []).filter((x) => x.used); }

  function turnNodes(t) {
    const user = h("div", { class: "msg user" }, h("div", { class: "bubble" }, t.question));
    return [user, botNode(t)];
  }

  function botNode(t) {
    const used = usedHits(t);
    const body = h("div", { class: "body" });
    const md = h("div", { class: "md", html: renderMarkdown(t.answer, used.length, claimMarks(t)) });
    md.addEventListener("click", (e) => {
      const c = e.target.closest(".cite");
      if (c) { state.selected = t.id; openInspector("sources", c.dataset.cite); }
    });
    body.append(md);
    const meta = h("div", { class: "msg-meta" });
    used.forEach((x, i) => meta.append(h("button", { class: "src-chip", title: `${x.source}${x.path ? " › " + x.path : ""}${x.section ? " — " + x.section : ""}`,
      onclick: () => { state.selected = t.id; openInspector("sources", String(i + 1)); } },
      h("span", { class: "n" }, i + 1), h("span", { class: "t" }, shortName(x.path ? x.path.split("/").pop() : x.source) + pages(x)))));
    (t.memory || []).forEach((m, i) => meta.append(h("button", { class: "src-chip", title: "Your verified answer to: " + m.question,
      onclick: () => { state.selected = t.id; openInspector("sources", `V${i + 1}`); } },
      h("span", { class: "n" }, `V${i + 1}`), h("span", { class: "t" }, "Verified answer"))));
    for (const w of t.checks?.warnings || []) {
      meta.append(w.kind === "claims"
        ? h("button", { class: "warning-chip", style: { border: 0, cursor: "pointer" }, title: "Show which statements",
            onclick: () => { state.selected = t.id; openInspector("checks"); } }, icon("alert"), w.text)
        : h("span", { class: "warning-chip", title: w.text }, icon("alert"), w.text));
    }
    if (meta.childNodes.length) body.append(meta);
    body.append(actions(t));
    return h("div", { class: "msg bot" + (t.id === state.selected && state.inspector ? " selected" : ""), "data-id": t.id },
      h("div", { class: "avatar" }), body);
  }

  function actions(t) {
    const m = t.metrics || {};
    const facts = [m.total_s != null ? `${num(m.total_s, 1)} s` : null,
      m.confidence != null ? `confidence ${num(m.confidence)}` : null].filter(Boolean).join(" · ");
    const up = h("button", { class: "icon-btn" + (t.grade === "good" ? " on" : ""), title: "Good answer", "aria-label": "Good answer",
      onclick: () => quickGrade(t, "good") }, icon("up"));
    const down = h("button", { class: "icon-btn" + (t.grade === "bad" ? " on" : ""), title: "Bad answer — say why", "aria-label": "Bad answer",
      onclick: (e) => gradePopover(e.currentTarget, t, "bad") }, icon("down"));
    const more = h("button", { class: "icon-btn" + (t.grade === "partial" ? " on" : ""), title: "Grade with details", "aria-label": "Grade with details",
      onclick: (e) => gradePopover(e.currentTarget, t, t.grade || "partial") }, icon("grade"));
    return h("div", { class: "msg-actions" },
      h("button", { class: "icon-btn", title: "Copy", "aria-label": "Copy answer", onclick: () => copyText(t.answer) }, icon("copy")),
      up, down, more,
      h("button", { class: "icon-btn", title: "Sources and details", "aria-label": "Inspect", onclick: () => { state.selected = t.id; openInspector(state.tab); } }, icon("panel")),
      h("span", { class: "sep" }),
      t.grade ? h("span", { class: "graded-note" }, h("span", { class: `grade-dot ${t.grade}` }), `Graded ${t.grade}`) : null,
      h("span", { class: "facts" }, facts));
  }

  async function quickGrade(t, grade) {
    const next = t.grade === grade ? null : grade;  // clicking again clears it
    await saveGrade(t, { grade: next, tags: next ? t.tags || [] : [], reason: t.reason || "", correction: t.correction || "" });
  }

  async function saveGrade(t, g) {
    try {
      const r = await api.put(`/api/interactions/${t.id}/grade`, g);
      Object.assign(t, g);
      toast(g.grade ? (r.learned ? "Saved — this answer will be reused for similar questions" : "Grade saved") : "Grade cleared");
      if (r.note) toast(r.note, { error: true });
      rerenderTurn(t);
      app.refreshStatus();
    } catch (e) { toastError(e); }
  }

  function gradePopover(anchor, t, preset) {
    let grade = preset;
    const tags = new Set(t.tags || []);
    const reason = h("textarea", { rows: 3, placeholder: "Why? (what's wrong or missing)" }, t.reason || "");
    const correction = h("textarea", { rows: 3, placeholder: "The correct answer (optional) — reused for similar questions" }, t.correction || "");
    const tagRow = h("div", { class: "tag-row" }, (tagVocab || []).map((tg) => {
      const b = h("button", { class: "chip" + (tags.has(tg.key) ? " on" : ""), title: tg.hint, onclick: () => {
        tags.has(tg.key) ? tags.delete(tg.key) : tags.add(tg.key); b.classList.toggle("on");
      } }, tg.label);
      return b;
    }));
    const pop = popover(anchor, h("div", {},
      h("h4", {}, "Grade this answer"),
      segmented([{ value: "good", label: "Good" }, { value: "partial", label: "Partial" }, { value: "bad", label: "Bad" }], grade, (v) => { grade = v; }),
      h("div", { class: "hint", style: { margin: "12px 0 6px" } }, "What went wrong?"),
      tagRow,
      h("div", { style: { marginTop: "10px" } }, reason),
      h("details", { style: { marginTop: "8px" } }, h("summary", { class: "hint", style: { cursor: "pointer" } }, "Add the correct answer"), h("div", { style: { marginTop: "6px" } }, correction)),
      h("div", { class: "row", style: { justifyContent: "flex-end", marginBottom: 0 } },
        h("a", { href: `#/evaluate?id=${t.id}`, class: "hint", style: { marginRight: "auto" } }, "Open in Evaluate"),
        h("button", { class: "btn sm", onclick: () => pop.close() }, "Cancel"),
        h("button", { class: "btn sm primary", onclick: async () => {
          pop.close();
          await saveGrade(t, { grade, tags: [...tags], reason: reason.value, correction: correction.value });
        } }, "Save"))), { align: "start", above: true });
    reason.focus();
  }

  function rerenderTurn(t) {
    const old = inner.querySelector(`.msg.bot[data-id="${t.id}"]`);
    if (old) old.replaceWith(botNode(t));
    if (state.inspector && state.selected === t.id) renderInspector();
  }

  // ---- inspector
  function toggleInspector(force) {
    state.inspector = force ?? !state.inspector;
    localStorage.setItem("inspector", state.inspector ? "1" : "0");
    renderInspector();
    inner.querySelectorAll(".msg.bot").forEach((n) => n.classList.toggle("selected", state.inspector && n.dataset.id === state.selected));
  }
  function openInspector(tab, flash) {
    state.tab = tab || "sources";
    toggleInspector(true);
    if (flash) {
      const card = inspector.querySelector(`[data-n="${flash}"]`);
      if (card) { card.scrollIntoView({ block: "center", behavior: "smooth" }); card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); card.querySelector("details")?.setAttribute("open", ""); }
    }
  }

  function renderInspector() {
    inspector.classList.toggle("hidden", !state.inspector);
    inspectBtn.classList.toggle("on", state.inspector);
    if (!state.inspector) return;
    const t = state.turns.find((x) => x.id === state.selected) || state.live;
    const tabs = h("div", { class: "tabs" }, [["sources", "Sources"], ["checks", "Checks"], ["trace", "Trace"], ["prompt", "Prompt"]].map(([k, l]) =>
      h("a", { href: "#", class: state.tab === k ? "active" : "", onclick: (e) => { e.preventDefault(); state.tab = k; renderInspector(); } }, l)));
    const body = h("div", { class: "inspector-body" });
    fill(inspector,
      h("div", { class: "inspector-head" }, "Answer details",
        h("button", { class: "icon-btn", "aria-label": "Close details", onclick: () => toggleInspector(false) }, icon("x"))),
      tabs, body);
    if (!t) { body.append(h("p", { class: "muted" }, "Ask a question to see which passages were used, how long each step took, and the exact prompt.")); return; }
    if (state.tab === "sources") add(body, sourcesPanel(t));
    else if (state.tab === "checks") add(body, checksPanel(t));
    else if (state.tab === "trace") add(body, tracePanel(t));
    else promptPanel(t, body);
  }

  renderTitle(); renderThread(); renderInspector(); updateTune();
  requestAnimationFrame(() => { thread.scrollTop = thread.scrollHeight; });

  // ---- composer
  function autosize() {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 220) + "px";
    sendBtn.disabled = !state.controller && !textarea.value.trim();
  }
  textarea.addEventListener("input", autosize);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!state.controller) send(); }
  });
  sendBtn.addEventListener("click", () => (state.controller ? state.controller.abort() : send()));
  textarea.focus();

  async function send() {
    const message = textarea.value.trim();
    if (!message || state.controller) return;
    textarea.value = ""; autosize();
    if (!state.turns.length) inner.replaceChildren();
    const live = { id: "live", question: message, answer: "", hits: [], memory: [], metrics: {}, checks: {} };
    state.live = live;
    const userNode = h("div", { class: "msg user" }, h("div", { class: "bubble" }, message));
    const stageEl = h("div", { class: "stage" }, h("span", { class: "spinner" }), h("span", {}, STAGES.search));
    const mdEl = h("div", { class: "md caret" });
    const bot = h("div", { class: "msg bot" }, h("div", { class: "avatar" }), h("div", { class: "body" }, stageEl, mdEl));
    inner.append(userNode, bot);
    const nearBottom = () => thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
    thread.scrollTop = thread.scrollHeight;

    state.controller = new AbortController();
    sendBtn.classList.add("stop"); sendBtn.replaceChildren(icon("stop")); sendBtn.disabled = false; sendBtn.setAttribute("aria-label", "Stop");
    let pending = false;
    const paint = () => {
      if (pending) return; pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const stick = nearBottom();
        mdEl.innerHTML = renderMarkdown(live.answer);
        if (stick) thread.scrollTop = thread.scrollHeight;
      });
    };
    try {
      await stream("/api/chat", { message, conversation_id: state.conv?.id || null, sources: scopeSelect.value ? [scopeSelect.value] : null }, (ev) => {
        if (ev.type === "conversation" && !state.conv) {
          state.conv = ev.conversation;
          history.replaceState(null, "", `#/chat/${ev.conversation.id}`);
          renderTitle(); app.refreshHistory();
        } else if (ev.type === "stage") {
          stageEl.lastChild.textContent = ev.stage === "search" && ev.query ? `Searching for “${ev.query}”`
            : ev.stage === "search" && ev.rerank ? "Searching and reranking passages" : STAGES[ev.stage] || ev.stage;
          stageEl.classList.remove("hidden");
          if (ev.stage === "verify") { mdEl.classList.remove("caret"); bot.querySelector(".body").append(stageEl); }
        } else if (ev.type === "sources") {
          live.hits = ev.hits; live.memory = ev.memory;
          if (state.inspector && !state.selected) renderInspector();
        } else if (ev.type === "token") {
          stageEl.classList.add("hidden");
          live.answer += ev.text; paint();
        } else if (ev.type === "done") {
          state.turns.push(ev.turn);
          state.selected = ev.turn.id;
          bot.replaceWith(botNode(ev.turn));
          if (state.inspector) renderInspector();
          app.refreshStatus();
        } else if (ev.type === "error") {
          throw new Error(ev.message);
        }
      }, state.controller.signal);
    } catch (e) {
      stageEl.remove();
      mdEl.classList.remove("caret");
      if (e.name === "AbortError") {
        bot.querySelector(".body").append(h("div", { class: "hint", style: { marginTop: "6px" } }, "Stopped. This partial answer wasn't saved."));
      } else {
        bot.querySelector(".body").append(h("div", { class: "warning-chip", style: { marginTop: "6px" } }, icon("alert"), e.message),
          h("div", { style: { marginTop: "8px" } }, h("button", { class: "btn sm", onclick: () => { userNode.remove(); bot.remove(); textarea.value = message; autosize(); send(); } }, icon("refresh"), "Try again")));
      }
    } finally {
      state.controller = null; state.live = null;
      sendBtn.classList.remove("stop"); sendBtn.replaceChildren(icon("send")); sendBtn.setAttribute("aria-label", "Send");
      autosize(); textarea.focus();
    }
  }

  // ---- quick tuning
  async function updateTune() {
    try {
      const s = await loadSettings();
      tuneBtn.lastChild.textContent = `${s.values.llm_model} · ${presetName(s)}`;
    } catch (e) { tuneBtn.lastChild.textContent = "Settings"; }
  }

  async function quickTune(anchor) {
    let s;
    try { s = await loadSettings(true); } catch (e) { toastError(e); return; }
    const [models] = await Promise.all([loadModels()]);
    const v = { ...s.values };
    const ins = s.insights;
    const spec = Object.fromEntries(s.schema.map((x) => [x.key, x]));
    const save = async (changes, msg) => {
      try {
        await api.put("/api/settings", { changes });
        Object.assign(v, changes); invalidateSettings(); await updateTune(); toast(msg || "Saved");
      } catch (e) { toastError(e); }
    };
    const eff = (key) => h("div", { class: "hint", style: { marginTop: "4px" } }, effectText(key, v[key], v, ins, models.installed));
    const content = h("div", {});
    const draw = () => {
      fill(content,
        h("h4", {}, "Answer settings"),
        segmented(Object.entries(s.presets).map(([k, p]) => ({ value: k, label: p.label })), presetKey(s.presets, v), async (k) => {
          await save(s.presets[k].values, `${s.presets[k].label} preset applied`); draw();
        }),
        h("div", { class: "hint", style: { marginTop: "6px" } }, s.presets[presetKey(s.presets, v)]?.summary || "Custom settings"),
        h("div", { class: "row" }, h("b", {}, "Chat model"), control(spec.llm_model, v.llm_model, (x) => { save({ llm_model: x }, `Model: ${x}`).then(draw); }, { models: models.installed })),
        eff("llm_model"),
        h("div", { class: "row" }, h("b", {}, "Rerank passages"), switchEl(v.rerank, (x) => { save({ rerank: x }).then(draw); })),
        eff("rerank"),
        h("div", { style: { marginTop: "12px" } }, h("b", {}, "Passages per answer")),
        control(spec.top_k, v.top_k, (x) => { v.top_k = x; effTop.textContent = effectText("top_k", x, v, ins); saveTopK(x); }),
        effTop = h("div", { class: "hint", style: { marginTop: "4px" } }, effectText("top_k", v.top_k, v, ins)),
        h("div", { class: "row", style: { marginBottom: 0 } }, h("a", { href: "#/settings" }, "All settings →")),
      );
    };
    let effTop;
    const saveTopK = debounceSave((x) => save({ top_k: x }, `Passages per answer: ${x}`));
    draw();
    popover(anchor, content, { align: "end", above: true });
  }
}

function debounceSave(fn) { let t; return (x) => { clearTimeout(t); t = setTimeout(() => fn(x), 500); }; }

function presetKey(presets, v) {
  return Object.keys(presets).find((k) => Object.entries(presets[k].values).every(([key, val]) => v[key] === val)) || null;
}
function presetName(s) {
  const k = presetKey(s.presets, s.values);
  return k ? s.presets[k].label : "Custom";
}

// Unsupported statements, for highlighting in the answer text.
export function claimMarks(t) {
  return (t.checks?.claims || []).filter((c) => !c.supported)
    .map((c) => ({ text: c.text, title: `Not supported by the cited sources (support ${c.score.toFixed(2)})` }));
}

// Every checked statement with its support score.
export function checksPanel(t) {
  const claims = t.checks?.claims;
  const out = [];
  if (!claims) out.push(h("p", { class: "muted" }, "Statements weren't checked for this answer (the check was off, or the answer came before it existed)."));
  else if (!claims.length) out.push(h("p", { class: "muted" }, "No statements to check (a refusal, or only short sentences)."));
  else {
    const bad = claims.filter((c) => !c.supported).length;
    out.push(h("p", { class: "hint", style: { marginTop: 0 } },
      `${claims.length - bad} of ${claims.length} statements supported by their sources. Each is checked against the passages it cites (all passages, if it cites none); scores are the model's confidence that the passage supports it.`));
    out.push(h("div", {}, claims.map((c) => h("div", { class: "claim-row " + (c.supported ? "ok" : "bad") },
      h("span", { class: "mark" }, c.supported ? "✓" : "⚠"),
      h("span", {}, c.text, c.cites.length ? h("span", { class: "faint" }, ` · checked against [${c.cites.join("], [")}]`) : h("span", { class: "faint" }, " · checked against all sources")),
      h("span", { class: "score" }, c.score.toFixed(2))))));
  }
  const w = (t.checks?.warnings || []).filter((x) => x.kind !== "claims");
  if (w.length) out.push(h("div", { class: "section-title", style: { marginTop: "16px" } }, "Other checks"),
    ...w.map((x) => h("div", { class: "warning-chip", style: { marginBottom: "6px" } }, icon("alert"), x.text)));
  return out;
}

function shortName(name) { return name.length > 28 ? name.slice(0, 25) + "…" : name; }
function pages(x) {
  const a = x.page_start, b = x.page_end;
  return a ? ` · p. ${a === b ? a : `${a}–${b}`}` : "";
}

export function sourcesPanel(t) {
  const out = [];
  (t.memory || []).forEach((m, i) => out.push(h("div", { class: "src-card", "data-n": `V${i + 1}` },
    h("div", { class: "src-head" }, h("span", { class: "badge accent" }, `V${i + 1}`), h("span", { class: "name" }, "Your verified answer"),
      h("span", { class: "faint" }, `match ${num(m.score)}`)),
    h("div", { class: "src-where" }, m.question),
    h("div", { class: "src-text" }, m.answer))));
  let n = 0;
  const hits = t.hits || [];
  if (!hits.length && !out.length) out.push(h("p", { class: "muted" }, "No passages were retrieved."));
  for (const x of hits) {
    const used = !!x.used;
    const label = used ? String(++n) : null;
    const kind = "rerank" in x ? "relevance" : "similarity";
    const found = [x.dense_rank ? "meaning" : null, x.keyword_rank ? "keywords" : null].filter(Boolean).join(" + ");
    const where = [x.page_start ? `p. ${x.page_start === x.page_end ? x.page_start : `${x.page_start}–${x.page_end}`}` : null, x.section].filter(Boolean).join(" · ");
    out.push(h("div", { class: "src-card" + (used ? "" : " off"), "data-n": label || "" },
      h("div", { class: "src-head" },
        used ? h("span", { class: "badge accent" }, label) : h("span", { class: "badge", title: x.cut ? "Relevant, but the comparison already had its share of passages for this side" : "Below the relevance cutoff, so not sent to the model" }, "not sent"),
        h("span", { class: "name", title: x.source }, x.source)),
      x.path ? h("div", { class: "src-where src-path", title: x.path }, x.path) : null,
      where ? h("div", { class: "src-where", title: where }, where) : null,
      (x.images || []).length ? h("div", { class: "src-thumbs" }, x.images.map(thumb),
        h("span", { class: "hint" }, "Text read from this picture. ",
          h("a", { href: `#/documents?doc=${encodeURIComponent(x.source)}&picture=${x.images[0]}` }, "Check or correct this reading"))) : null,
      h("div", { class: "score-bar" }, h("div", { style: { width: `${Math.max(2, Math.min(100, (x.score ?? 0) * 100))}%` } })),
      h("div", { class: "src-scores" },
        h("span", {}, `${kind} ${num(x.score)}`),
        found ? h("span", {}, `found by ${found}`) : null,
        x.found_for ? h("span", {}, `for ${x.found_for.join(", ")}`) : null,
        x.dense != null && kind === "relevance" ? h("span", {}, `similarity ${num(x.dense)}`) : null),
      h("details", {}, h("summary", { class: "hint", style: { cursor: "pointer", marginTop: "6px" } }, "Show passage"),
        h("div", { class: "src-text" }, x.text))));
  }
  return out;
}

export function tracePanel(t) {
  const m = t.metrics || {};
  if (!m.total_s && m.total_s !== 0) return [h("p", { class: "muted" }, "Details appear when the answer is complete.")];
  const search = (m.embed_ms + m.search_ms) / 1000, rerank = (m.rerank_ms || 0) / 1000;
  const verify = (m.verify_ms || 0) / 1000;
  const gen = Math.max(0, m.total_s - search - rerank - verify);
  const total = Math.max(m.total_s, 0.001);
  const seg = (v, color) => h("div", { style: { width: `${(100 * v) / total}%`, background: color } });
  const st = t.settings || {};
  const rows = [
    ["Searched for", m.search_query ? `“${m.search_query}”` : "the question as asked"],
    m.compared ? ["Compared, each searched too", m.compared.join(" · ")] : null,
    ["Passages sent", `${m.passages_used}${m.passages_trimmed ? ` (${m.passages_trimmed} trimmed to fit)` : ""}`],
    ["Relevance cutoff", num(m.min_score)], ["Best passage score", num(m.top_score)],
    ["Gap to 2nd", num(m.score_gap)], ["Verified answers used", m.memory_used],
    ["Earlier turns sent", (m.history_msgs ? m.history_msgs / 2 : 0) + (m.history_dropped ? ` (${m.history_dropped} on other subjects left out)` : "")],
    ["Prompt tokens", m.prompt_tokens], ["Answer tokens", `${m.answer_tokens}${m.hit_length_cap ? " (hit the limit)" : ""}`],
    ["Speed", m.tok_per_s ? `${m.tok_per_s} tokens/s` : "—"], ["First token after", `${num(m.ttft_s, 1)} s`],
    ["Confidence", m.confidence != null ? `${num(m.confidence)} (least sure token ${num(m.min_token_p)})` : "—"],
    ["Model load", m.load_s > 1 ? `${num(m.load_s, 1)} s (cold start)` : "already loaded"],
  ];
  return [
    h("div", { class: "section-title" }, `Took ${num(m.total_s, 1)} s`),
    h("div", { class: "timeline" }, seg(search, "#60a5fa"), seg(rerank, "#a78bfa"), seg(gen, "var(--accent)"), seg(verify, "#f59e0b")),
    h("div", { class: "legend" },
      h("span", {}, h("i", { style: { background: "#60a5fa" } }), `Search ${num(search, 2)} s`),
      h("span", {}, h("i", { style: { background: "#a78bfa" } }), m.rerank ? `Rerank ${num(rerank, 1)} s` : "No reranking"),
      h("span", {}, h("i", { style: { background: "var(--accent)" } }), `Answer ${num(gen, 1)} s`),
      m.claims_checked ? h("span", {}, h("i", { style: { background: "#f59e0b" } }), `Check ${m.claims_checked} statements ${num(verify, 1)} s`) : null),
    h("div", { class: "section-title", style: { marginTop: "18px" } }, "Details"),
    h("dl", { class: "kv" }, rows.filter(Boolean).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v ?? "—")])),
    st.llm_model ? h("div", { class: "hint", style: { marginTop: "14px" } },
      `${st.llm_model} · ${st.top_k} passages · ${st.rerank ? "reranked" : "not reranked"} · ${st.hybrid ? "keyword + meaning" : "meaning only"} · temperature ${st.temperature}`) : null,
  ];
}

async function promptPanel(t, body) {
  if (!t.id || t.id === "live") { body.append(h("p", { class: "muted" }, "Available when the answer is complete.")); return; }
  body.append(h("div", { class: "stage" }, h("span", { class: "spinner" }), "Loading"));
  try {
    const row = await api.get(`/api/interactions/${t.id}`);
    fill(body,
      h("p", { class: "hint", style: { marginTop: 0 } }, "Exactly what the model received for this answer."),
      ...row.messages.flatMap((m) => [h("div", { class: "role-tag" }, m.role), h("pre", { class: "prompt" }, m.content)]),
      h("button", { class: "btn sm", onclick: () => copyText(row.messages.map((m) => `### ${m.role}\n${m.content}`).join("\n\n")) }, icon("copy"), "Copy prompt"));
  } catch (e) { body.replaceChildren(h("p", { class: "muted" }, e.message)); }
}
