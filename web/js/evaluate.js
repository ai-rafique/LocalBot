// Evaluate: a grading console for every logged answer (chat and experiments).
import { api } from "./api.js";
import { sourcesPanel, tracePanel } from "./chat.js";
import { renderMarkdown } from "./markdown.js";
import { app, menuButton } from "./main.js";
import { dialog, fill, gradeBadge, h, icon, num, pct, segmented, switchEl, timeAgo, toast, toastError } from "./ui.js";

const GRADE_KEYS = { 1: "good", 2: "partial", 3: "bad" };

export async function render(main, route) {
  const q = route.query;
  const f = { status: q.status || (q.id || q.run ? "all" : "ungraded"), source: q.source || (q.run ? "experiment" : "all"),
              search: "", run: q.run || "", flagged: false };
  let items = [], total = 0, selected = q.id || null, detail = null, stats = null, runs = [], tags = [];
  let grade = null, tagSet = new Set(), dirty = false;

  const page = h("div", { class: "page" });
  const inner = h("div", { class: "page-inner", style: { maxWidth: "1400px", paddingBottom: "20px" } });
  page.append(inner);
  main.append(page);

  const statsEl = h("div", { class: "stat-strip" });
  const listEl = h("div", { class: "eval-list", role: "listbox", "aria-label": "Answers" });
  const detailEl = h("div", { class: "eval-detail" });
  const searchInput = h("input", { type: "search", placeholder: "Search questions and answers", value: f.search });
  let searchT;
  searchInput.addEventListener("input", () => { clearTimeout(searchT); searchT = setTimeout(() => { f.search = searchInput.value; loadList(); }, 250); });
  const runSelect = h("select", { onchange: (e) => { f.run = e.target.value; if (f.run) f.source = "all"; drawToolbar(); loadList(); } });
  const toolbar = h("div", { class: "toolbar" });

  function drawToolbar() {
    runSelect.replaceChildren(h("option", { value: "" }, "All runs"), ...runs.map((r) => h("option", { value: r.id, selected: r.id === f.run }, r.name)));
    fill(toolbar,
      segmented([{ value: "ungraded", label: "To grade" }, { value: "all", label: "All" }, { value: "good", label: "Good" },
        { value: "partial", label: "Partial" }, { value: "bad", label: "Bad" }], f.status, (v) => { f.status = v; loadList(); }),
      h("select", { onchange: (e) => { f.source = e.target.value; loadList(); } },
        [["all", "Chat and experiments"], ["chat", "Chat only"], ["experiment", "Experiments only"]].map(([v, l]) => h("option", { value: v, selected: f.source === v }, l))),
      runs.length ? runSelect : null,
      switchEl(f.flagged, (v) => { f.flagged = v; loadList(); }, "Flagged only"),
      searchInput);
  }

  inner.append(
    h("div", { class: "page-head" },
      h("div", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, menuButton(),
        h("div", {}, h("h1", {}, "Evaluate"), h("p", {}, "Grade answers and say why. Grades feed experiment results, calibrate warnings, and good chat answers are reused for similar questions."))),
      h("div", { class: "actions" }, h("button", { class: "btn", onclick: exportDialog }, icon("download"), "Export"))),
    statsEl, toolbar, h("div", { class: "eval-layout" }, listEl, detailEl));

  async function loadStats() {
    stats = await api.get("/api/evaluate/stats");
    tags = stats.tag_vocabulary;
    const g = stats.grades, n = Math.max(1, stats.graded);
    fill(statsEl,
      h("div", { class: "item" }, h("b", {}, `${stats.graded} / ${stats.total}`), h("span", {}, "graded")),
      h("div", { class: "item" },
        h("div", { class: "gradebar", title: `good ${g.good} · partial ${g.partial} · bad ${g.bad}` },
          h("div", { class: "g", style: { width: `${(100 * g.good) / n}%` } }), h("div", { class: "p", style: { width: `${(100 * g.partial) / n}%` } }),
          h("div", { class: "b", style: { width: `${(100 * g.bad) / n}%` } })),
        h("span", { style: { marginTop: "4px" } }, `${g.good} good · ${g.partial} partial · ${g.bad} bad`)),
      h("div", { class: "item" }, h("b", {}, stats.learned), h("span", {}, "learned answers")),
      h("div", { class: "item", style: { marginLeft: "auto" } }, h("span", {}, "Shortcuts: "),
        h("span", {}, h("span", { class: "kbd" }, "1"), " ", h("span", { class: "kbd" }, "2"), " ", h("span", { class: "kbd" }, "3"), " grade · ",
          h("span", { class: "kbd" }, "J"), " ", h("span", { class: "kbd" }, "K"), " move · ", h("span", { class: "kbd" }, "Ctrl"), "+", h("span", { class: "kbd" }, "Enter"), " save")));
  }

  async function loadList(keepSelection = true) {
    const params = new URLSearchParams({ status: f.status, source: f.source, search: f.search, run_id: f.run, flagged: f.flagged, limit: 300 });
    try {
      const r = await api.get(`/api/interactions?${params}`);
      items = r.items; total = r.total;
    } catch (e) { toastError(e); return; }
    // Keep the current answer selected if it's still listed (or was linked to
    // directly); otherwise start at the top of the list.
    const linked = keepSelection && selected && selected === q.id;
    if (!keepSelection || (!linked && !items.some((x) => x.id === selected))) selected = items[0]?.id || null;
    drawList();
    if (selected) loadDetail(selected); else drawDetail();
  }

  function drawList() {
    if (!items.length) {
      listEl.replaceChildren(h("div", { class: "empty-state" }, h("div", { class: "icon-circle" }, icon("check")),
        h("h3", {}, f.status === "ungraded" ? "All caught up" : "Nothing here"),
        h("p", {}, f.status === "ungraded" ? "Every answer matching these filters is graded." : "No answers match these filters.")));
      return;
    }
    listEl.replaceChildren(
      h("div", { class: "hint", style: { padding: "8px 14px", borderBottom: "1px solid var(--border)" } }, `${total} answer${total === 1 ? "" : "s"}`),
      ...items.map((x) => h("div", { class: "eval-item" + (x.id === selected ? " active" : ""), role: "option", "aria-selected": x.id === selected ? "true" : "false",
        "data-id": x.id, onclick: () => select(x.id) },
        h("span", { class: `grade-dot ${x.grade || ""}`, title: x.grade || "not graded" }),
        h("div", { style: { minWidth: 0, flex: 1 } },
          h("div", { class: "q" }, x.question),
          h("div", { class: "meta" },
            h("span", {}, x.origin === "chat" ? "Chat" : x.qid ? `Experiment · ${x.qid}` : "Experiment"),
            h("span", {}, timeAgo(x.ts)),
            x.warnings ? h("span", { style: { color: "var(--warn)" }, title: "Has warnings" }, `⚠ ${x.warnings}`) : null,
            x.confidence != null ? h("span", {}, `conf ${num(x.confidence)}`) : null)))));
  }

  async function select(id) {
    if (dirty && !confirm("Discard your unsaved grade for this answer?")) return;
    selected = id;
    listEl.querySelectorAll(".eval-item").forEach((n) => n.classList.toggle("active", n.dataset.id === id));
    await loadDetail(id);
    listEl.querySelector(".eval-item.active")?.scrollIntoView({ block: "nearest" });
  }

  async function loadDetail(id) {
    try { detail = await api.get(`/api/interactions/${id}`); }
    catch (e) { detail = null; toastError(e); }
    grade = detail?.grade || null; tagSet = new Set(detail?.tags || []); dirty = false;
    drawDetail();
  }

  function drawDetail() {
    if (!detail) {
      detailEl.replaceChildren(h("div", { class: "empty-state" }, h("p", {}, "Select an answer to grade it.")));
      return;
    }
    const d = detail, m = d.metrics || {};
    const used = (d.hits || []).filter((x) => x.used);
    const source = d.origin === "chat"
      ? h("span", {}, "Chat", d.conversation_id ? [" · ", h("a", { href: `#/chat/${d.conversation_id}` }, d.conversation_title || "open conversation")] : null)
      : h("span", {}, "Experiment", d.run_id ? [" · ", h("a", { href: `#/experiments/run/${d.run_id}` }, d.run_name || "run")] : null, d.qid ? ` · ${d.qid}` : "");
    const answer = h("div", { class: "md", html: renderMarkdown(d.answer, used.length) });
    const sourcesBody = h("div", { class: "fold-body" });
    answer.addEventListener("click", (e) => {
      const c = e.target.closest(".cite");
      if (!c) return;
      sourcesFold.open = true;
      const card = sourcesBody.querySelector(`[data-n="${c.dataset.cite}"]`);
      if (card) { card.scrollIntoView({ block: "center", behavior: "smooth" }); card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); card.querySelector("details")?.setAttribute("open", ""); }
    });
    sourcesBody.append(...sourcesPanel(d));
    const sourcesFold = h("details", { class: "fold" }, h("summary", {}, `Sources (${used.length} sent of ${(d.hits || []).length} retrieved)`), sourcesBody);
    const warnings = d.checks?.warnings || [];
    const expected = d.expected ? h("div", { class: "expected" },
      h("b", {}, d.expected.answerable === false ? "Expected: a refusal (the documents don't cover this)" : "Expected facts"),
      d.expected.answerable !== false ? h("ul", { style: { margin: "6px 0 0", paddingLeft: "20px" } },
        d.expected.facts.map((alts) => h("li", {}, alts.map((a, i) => [i ? h("span", { class: "faint" }, " or ") : null, h("code", {}, a)])))) : null,
      d.expected.note ? h("div", { class: "hint", style: { marginTop: "4px" } }, d.expected.note) : null,
      h("div", { class: "hint", style: { marginTop: "4px" } }, `Variant: ${d.expected.variant}`)) : null;

    const scroll = h("div", { class: "eval-scroll" },
      h("div", { class: "hint", style: { display: "flex", gap: "10px", flexWrap: "wrap" } }, source, h("span", {}, new Date(d.ts).toLocaleString())),
      h("div", { class: "eval-q" }, d.question),
      h("div", { class: "eval-answer" }, answer),
      warnings.length ? h("div", { class: "msg-meta" }, warnings.map((w) => h("span", { class: "warning-chip" }, icon("alert"), w.text))) : null,
      h("div", { class: "hint", style: { marginTop: "8px" } },
        [m.total_s != null ? `${num(m.total_s, 1)} s` : null, m.confidence != null ? `confidence ${num(m.confidence)}` : null,
         `${used.length} passages sent`, d.settings?.llm_model || d.settings?.llm].filter(Boolean).join(" · ")),
      expected, sourcesFold,
      h("details", { class: "fold" }, h("summary", {}, "Trace"), h("div", { class: "fold-body" }, ...tracePanel(d))),
      h("details", { class: "fold" }, h("summary", {}, "Exact prompt"), h("div", { class: "fold-body" },
        ...(d.messages || []).flatMap((msg) => [h("div", { class: "role-tag" }, msg.role), h("pre", { class: "prompt" }, msg.content)]))));
    fill(detailEl, scroll, grader());
  }

  const reasonEl = h("textarea", { rows: 2, placeholder: "Why? What's wrong, missing, or especially good — this drives tuning." });
  const correctionEl = h("textarea", { rows: 2, placeholder: "Correct answer (optional). For chat answers it's reused for similar questions." });
  reasonEl.addEventListener("input", () => { dirty = true; });
  correctionEl.addEventListener("input", () => { dirty = true; });

  function grader() {
    reasonEl.value = detail.reason || "";
    correctionEl.value = detail.correction || "";
    const buttons = h("div", { class: "grade-buttons" }, ["good", "partial", "bad"].map((g, i) =>
      h("button", { class: `grade-btn ${g}` + (grade === g ? " on" : ""), onclick: () => setGrade(g) },
        h("span", { class: `grade-dot ${g}` }), g[0].toUpperCase() + g.slice(1), h("span", { class: "kbd" }, i + 1))));
    const tagRow = h("div", { class: "tag-row" }, tags.map((t) => h("button", { class: "chip" + (tagSet.has(t.key) ? " on" : ""), title: t.hint,
      onclick: (e) => { tagSet.has(t.key) ? tagSet.delete(t.key) : tagSet.add(t.key); e.currentTarget.classList.toggle("on"); dirty = true; } }, t.label)));
    const hasCorrection = !!detail.correction;
    return h("div", { class: "grader" },
      buttons, tagRow, reasonEl,
      h("details", { open: hasCorrection }, h("summary", { class: "hint", style: { cursor: "pointer" } }, "Correct answer"), h("div", { style: { marginTop: "6px" } }, correctionEl)),
      h("div", { class: "grader-foot" },
        h("span", { class: "grow" }, detail.graded_at ? `Graded ${timeAgo(detail.graded_at)}` : "Not graded yet"),
        detail.grade ? h("button", { class: "btn ghost sm", onclick: () => save(null, true) }, "Clear grade") : null,
        h("button", { class: "btn", onclick: () => move(1, true) }, "Skip"),
        h("button", { class: "btn primary", onclick: () => save(grade) }, "Save and next", h("span", { class: "kbd", style: { background: "transparent", color: "inherit", borderColor: "currentColor" } }, "Ctrl ↵"))));
  }

  function setGrade(g) {
    grade = g; dirty = true;
    detailEl.querySelectorAll(".grade-btn").forEach((b) => b.classList.toggle("on", b.classList.contains(g)));
  }

  async function save(g, clear = false) {
    if (!g && !clear) { toast("Pick a grade first: good, partial or bad", { error: true }); return; }
    try {
      const r = await api.put(`/api/interactions/${detail.id}/grade`, { grade: g, tags: g ? [...tagSet] : [], reason: reasonEl.value, correction: correctionEl.value });
      toast(g ? (r.learned ? "Saved · will be reused for similar questions" : "Saved") : "Grade cleared");
      if (r.note) toast(r.note, { error: true });
      dirty = false;
      const idx = items.findIndex((x) => x.id === detail.id);
      if (idx >= 0) {
        items[idx].grade = g; items[idx].tags = [...tagSet];
        if (f.status === "ungraded" && g) { items.splice(idx, 1); total--; }
      }
      const next = items[f.status === "ungraded" && g ? idx : idx + 1] || items[idx] || null;
      loadStats(); app.refreshStatus();
      drawList();
      if (next && g) await select(next.id); else if (!items.length) { selected = null; detail = null; drawDetail(); } else await loadDetail(detail.id);
    } catch (e) { toastError(e); }
  }

  function move(step, skip = false) {
    if (!items.length) return;
    const idx = Math.max(0, items.findIndex((x) => x.id === selected));
    const next = items[Math.min(items.length - 1, Math.max(0, idx + step))];
    if (next && next.id !== selected) { if (skip) dirty = false; select(next.id); }
  }

  function exportDialog() {
    const opts = { only_graded: true, include_text: true, anonymize: true };
    const row = (key, label, hint) => h("div", {}, switchEl(opts[key], (v) => { opts[key] = v; }, label), h("div", { class: "hint", style: { marginLeft: "46px" } }, hint));
    dialog({
      title: "Export grades",
      content: h("div", { style: { display: "flex", flexDirection: "column", gap: "14px" } },
        h("p", { class: "muted", style: { margin: 0 } }, "A JSON file with grades, reasons, settings and retrieval scores, for analysis or tuning elsewhere. Passage text and prompts are never included."),
        row("only_graded", "Graded answers only", "Leave out answers you haven't graded."),
        row("include_text", "Include questions and answers", "Answers can quote your documents. Turn off to share only grades and numbers."),
        row("anonymize", "Hide document names", "Replace file names with doc1, doc2, …")),
      buttons: [{ label: "Cancel" }, { label: "Download", cls: "primary", onClick: () => {
        const a = h("a", { href: `/api/export?${new URLSearchParams(opts)}` });
        document.body.append(a); a.click(); a.remove();
      } }],
    });
  }

  const onKey = (e) => {
    if (e.target.closest("dialog")) return;
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); if (detail) save(grade); return; }
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    if (GRADE_KEYS[e.key] && detail) { e.preventDefault(); setGrade(GRADE_KEYS[e.key]); }
    else if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); move(-1); }
  };
  document.addEventListener("keydown", onKey);

  try {
    runs = (await api.get("/api/runs")).filter((r) => r.config && !r.config.retrieval_only);
    await loadStats();
  } catch (e) { toastError(e); }
  drawToolbar();
  await loadList(true);
  return () => document.removeEventListener("keydown", onKey);
}
