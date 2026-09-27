// Experiments: question sets, runs under changed settings, results, comparison.
import { api } from "./api.js";
import { app, menuButton } from "./main.js";
import { control, effectText, formatValue, loadModels, loadSettings } from "./tuning.js";
import { add, confirmDialog, dialog, fill, gradeBadge, h, icon, num, pct, segmented, timeAgo, tip, toast, toastError } from "./ui.js";

// What each metric means, and whether higher is better (for coloring deltas).
const METRICS = [
  { key: "good_rate", label: "Graded good", fmt: pct, up: true, help: "Share of graded answers you marked good. The most trustworthy measure — grade in Evaluate." },
  { key: "auto_correct", label: "Auto-correct", fmt: pct, up: true, help: "Answerable questions whose answer contains every expected fact and isn't a refusal. Can't see wrong extra claims." },
  { key: "fact_coverage", label: "Fact coverage", fmt: pct, up: true, help: "Average share of expected facts found in the answers." },
  { key: "context_recall", label: "Context recall", fmt: pct, up: true, help: "Answerable questions where the passages sent to the model contained every expected fact. Measures retrieval." },
  { key: "refusal_accuracy", label: "Refused when it should", fmt: pct, up: true, help: "Unanswerable questions answered with \"I don't know\"." },
  { key: "offtopic_blocked", label: "Blocked before the model", fmt: pct, up: true, help: "Unanswerable questions where no passage passed the cutoff, so the model wasn't even asked." },
  { key: "false_refusals", label: "Wrongly refused", fmt: pct, up: false, help: "Answerable questions answered with \"I don't know\"." },
  { key: "generation_misses", label: "Misread passages", fmt: (x) => x ?? "—", up: false, help: "The expected facts were in the passages sent, but not in the answer: the model misread or ignored them." },
  { key: "flagged", label: "Flagged terms", fmt: (x) => x ?? "—", up: false, help: "Answers containing numbers or identifiers that appear in none of their sources." },
  { key: "latency_s", label: "Seconds per question", fmt: (x) => num(x, 1), up: false, help: "Average time per question." },
];

export async function render(main, route) {
  const [sub, arg] = route.parts;
  const page = h("div", { class: "page" });
  const inner = h("div", { class: "page-inner" });
  page.append(inner);
  main.append(page);
  const ctx = { inner, route };
  if (sub === "run" && arg) return runDetail(ctx, arg);
  if (sub === "compare" && arg) return compareView(ctx, arg.split(","));
  return listView(ctx, sub === "sets" ? "sets" : "runs");
}

function header(title, subtitle, actions = []) {
  return h("div", { class: "page-head" },
    h("div", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, menuButton(),
      h("div", {}, h("h1", {}, title), subtitle ? h("p", {}, subtitle) : null)),
    h("div", { class: "actions" }, actions));
}

function changesChips(run, schema) {
  const ov = run.config?.overrides || {};
  const keys = Object.keys(ov);
  const byKey = Object.fromEntries((schema || []).map((s) => [s.key, s]));
  const chips = keys.map((k) => h("span", { class: "chip", title: byKey[k]?.label || k }, `${byKey[k]?.label || k}: ${byKey[k] ? formatValue(byKey[k], ov[k]) : ov[k]}`));
  return h("div", { class: "run-changes" }, chips.length ? chips : h("span", { class: "faint" }, "current settings"),
    run.config?.retrieval_only ? h("span", { class: "badge accent" }, "passages only") : null);
}

function statusBadge(r) {
  const map = { done: ["good", "Done"], running: ["accent", "Running"], queued: ["", "Queued"], failed: ["bad", "Failed"],
                cancelled: ["", "Cancelled"], interrupted: ["partial", "Interrupted"] };
  const [cls, label] = map[r.status] || ["", r.status];
  return h("span", { class: `badge ${cls}` }, r.status === "running" ? h("span", { class: "spinner", style: { width: "10px", height: "10px" } }) : null, label);
}

// ---------------------------------------------------------------- list view
async function listView({ inner, route }, tab) {
  let runs = [], sets = [], settings = null, selected = new Set(), timer = null;
  const body = h("div", {});
  const tabs = h("div", { class: "tabs" },
    h("a", { href: "#/experiments", class: tab === "runs" ? "active" : "" }, "Runs"),
    h("a", { href: "#/experiments/sets", class: tab === "sets" ? "active" : "" }, "Question sets"));
  add(inner, header("Experiments", "Measure how a change in settings affects answers, on a fixed set of questions.",
    [h("button", { class: "btn primary", onclick: () => newRunDialog({ sets, settings, onStarted: reload }) }, icon("play"), "New run")]), tabs, body);

  async function reload() {
    try { [runs, sets, settings] = await Promise.all([api.get("/api/runs"), api.get("/api/question-sets"), loadSettings()]); }
    catch (e) { toastError(e); return; }
    tab === "runs" ? drawRuns() : drawSets();
    clearTimeout(timer);
    if (runs.some((r) => r.status === "running" || r.status === "queued")) timer = setTimeout(reload, 2000);
  }

  function drawRuns() {
    if (!runs.length) {
      body.replaceChildren(h("div", { class: "card card-pad" }, h("div", { class: "empty-state" },
        h("div", { class: "icon-circle" }, icon("flask")), h("h3", {}, "No runs yet"),
        h("p", {}, sets.length ? "Start a run to measure your current settings, then change one thing and compare." : "First add a question set: questions with the facts a correct answer must contain."),
        sets.length ? h("button", { class: "btn primary", onclick: () => newRunDialog({ sets, settings, onStarted: reload }) }, icon("play"), "New run")
          : h("a", { class: "btn primary", href: "#/experiments/sets" }, "Add a question set"))));
      return;
    }
    const compareBtn = h("button", { class: "btn", disabled: selected.size < 2, onclick: () => { location.hash = `#/experiments/compare/${[...selected].join(",")}`; } }, icon("compare"), "Compare selected");
    const table = h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "Run"), h("th", {}, "Changes"), h("th", {}, "Status"),
        h("th", { class: "num" }, "Graded good"), h("th", { class: "num" }, "Auto-correct"), h("th", { class: "num" }, "Context recall"),
        h("th", { class: "num" }, "s / question"), h("th", {}, "Started"))),
      h("tbody", {}, runs.map((r) => {
        const s = r.summary || {};
        const box = h("input", { type: "checkbox", checked: selected.has(r.id), "aria-label": `Select ${r.name}`, onclick: (e) => {
          e.stopPropagation(); e.target.checked ? selected.add(r.id) : selected.delete(r.id); compareBtn.disabled = selected.size < 2; } });
        const progress = r.status === "running" || r.status === "queued"
          ? h("div", { style: { minWidth: "110px" } }, statusBadge(r), h("div", { class: "progress", style: { marginTop: "6px" } }, h("div", { style: { width: `${(100 * r.done) / Math.max(1, r.total)}%` } })), h("div", { class: "hint" }, `${r.done} / ${r.total}`))
          : statusBadge(r);
        return h("tr", { class: "clickable", onclick: () => { location.hash = `#/experiments/run/${r.id}`; } },
          h("td", {}, box),
          h("td", {}, h("div", { style: { fontWeight: 600 } }, r.name), h("div", { class: "hint" }, `${r.set_name || "deleted set"} · ${r.total} questions`)),
          h("td", {}, changesChips(r, settings?.schema)), h("td", {}, progress),
          h("td", { class: "num" }, s.graded ? h("span", {}, pct(s.good_rate), h("div", { class: "hint" }, `${s.graded} graded`)) : h("span", { class: "faint" }, "—")),
          h("td", { class: "num" }, pct(s.auto_correct)), h("td", { class: "num" }, pct(s.context_recall)),
          h("td", { class: "num" }, num(s.latency_s, 1)), h("td", { class: "hint" }, timeAgo(r.created)));
      })));
    body.replaceChildren(h("div", { class: "toolbar" }, compareBtn, h("span", { class: "hint" }, "Tick two or more runs to compare them.")),
      h("div", { class: "card", style: { overflowX: "auto" } }, table));
  }

  function drawSets() {
    const fileInput = h("input", { type: "file", accept: ".jsonl,.json,.txt", class: "hidden", onchange: async (e) => {
      const file = e.target.files[0]; e.target.value = "";
      if (!file) return;
      try {
        const r = await api.post("/api/question-sets", { name: file.name.replace(/\.(jsonl|json|txt)$/i, ""), text: await file.text() });
        toast(`Added ${r.count} questions`); reload();
      } catch (err) { toastError(err); }
    } });
    const list = sets.length ? h("div", { class: "card" }, sets.map((s) => h("div", { class: "set-card" },
      h("div", { class: "grow" }, h("div", { style: { fontWeight: 600 } }, s.name),
        h("div", { class: "hint" }, `${s.count} questions · added ${timeAgo(s.created)}`),
        h("div", { class: "run-changes", style: { marginTop: "6px" } }, Object.entries(s.variants).map(([v, n]) => h("span", { class: "chip" }, `${v} ${n}`)))),
      h("button", { class: "btn sm", onclick: () => viewSet(s.id) }, "View"),
      h("a", { class: "btn sm", href: `/api/question-sets/${s.id}/download` }, icon("download")),
      h("button", { class: "btn sm primary", onclick: () => newRunDialog({ sets, settings, setId: s.id, onStarted: () => { location.hash = "#/experiments"; } }) }, icon("play"), "Run"),
      h("button", { class: "icon-btn", "aria-label": `Delete ${s.name}`, onclick: async () => {
        if (!(await confirmDialog(`Delete “${s.name}”?`, "Runs that used it keep their results.", { confirm: "Delete", danger: true }))) return;
        await api.del(`/api/question-sets/${s.id}`).catch(toastError); reload();
      } }, icon("trash")))))
      : h("div", { class: "card card-pad" }, h("div", { class: "empty-state" }, h("div", { class: "icon-circle" }, icon("grade")),
        h("h3", {}, "No question sets"), h("p", {}, "Upload a .jsonl file, or let the model draft questions from your documents to get started.")));
    body.replaceChildren(
      h("div", { class: "toolbar" }, fileInput,
        h("button", { class: "btn", onclick: () => fileInput.click() }, icon("upload"), "Upload .jsonl"),
        h("button", { class: "btn", onclick: draftDialog }, icon("sparkle"), "Draft from documents")),
      list, formatHelp());
  }

  function draftDialog() {
    const n = h("input", { type: "number", min: 1, max: 30, value: 10 });
    const name = h("input", { type: "text", placeholder: "Drafted questions" });
    dialog({ title: "Draft questions from your documents", content: h("div", { style: { display: "flex", flexDirection: "column", gap: "12px" } },
      h("p", { class: "muted", style: { margin: 0 } }, "The model picks random passages and writes one question each, with the phrase that answers it as the expected fact. Drafted questions reuse the documents' wording, so they're easier than real ones — review them and add harder ones."),
      h("label", { class: "field" }, "Number of questions", n), h("label", { class: "field" }, "Name", name)),
      buttons: [{ label: "Cancel" }, { label: "Draft", cls: "primary", onClick: async () => {
        toast("Drafting… this takes a few seconds per question");
        const r = await api.post("/api/question-sets/draft", { n: +n.value || 10, name: name.value });
        toast(`Drafted ${r.count} questions`); reload();
      } }] });
  }

  async function viewSet(id) {
    try {
      const s = await api.get(`/api/question-sets/${id}`);
      dialog({ title: s.name, wide: true, content: h("table", { class: "table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Id"), h("th", {}, "Variant"), h("th", {}, "Question"), h("th", {}, "Expected"))),
        h("tbody", {}, s.items.map((q) => h("tr", {}, h("td", { class: "mono" }, q.id), h("td", {}, q.variant), h("td", {}, q.question),
          h("td", {}, q.answerable === false ? h("span", { class: "badge" }, "refusal") : q.facts.map((f) => h("div", { class: "mono" }, f.join(" | ")))))))) });
    } catch (e) { toastError(e); }
  }

  if (route.query.new) {
    await reload();
    newRunDialog({ sets, settings, preset: route.query, onStarted: reload });
  } else await reload();
  return () => clearTimeout(timer);
}

function formatHelp() {
  const example = [
    '{"id": "q1", "variant": "lookup", "question": "What baud rate does the service port use?", "facts": [["115200"]]}',
    '{"id": "q2", "variant": "exact", "question": "Which bytes reset the device?", "facts": [["aa 01 00 ff"]]}',
    '{"id": "q3", "variant": "unanswerable", "question": "Who made the enclosure?", "answerable": false, "facts": []}',
  ].join("\n");
  return h("details", { class: "fold", style: { marginTop: "16px" } }, h("summary", {}, "Question set format"),
    h("div", { class: "fold-body" },
      h("p", { class: "muted" }, "One JSON object per line. ", h("code", {}, "facts"), " lists what a correct answer must contain: each inner list is one fact, and any spelling in it counts (case, spaces and separators are ignored). Mark questions your documents can't answer with ", h("code", {}, "\"answerable\": false"), "."),
      h("pre", { class: "prompt" }, example),
      h("p", { class: "muted" }, "Variants group questions by what they test, so results show where a change helps or hurts:"),
      h("div", { class: "run-changes" }, ["lookup", "exact", "paraphrase", "multihop", "crossdoc", "numeric", "procedure", "false_premise", "unanswerable", "terse"].map((v) => h("span", { class: "chip" }, v)))));
}

// ---------------------------------------------------------------- new run
export async function newRunDialog({ sets, settings, setId, preset = {}, onStarted }) {
  if (!sets?.length) { toast("Add a question set first", { error: true }); location.hash = "#/experiments/sets"; return; }
  settings = settings || await loadSettings();
  const models = (await loadModels()).installed || [];
  const meta = await api.get("/api/experiments/meta");
  const schema = settings.schema.filter((s) => s.key !== "rerank_model");
  const byKey = Object.fromEntries(schema.map((s) => [s.key, s]));
  const v = settings.values;
  const state = { set: setId || sets[0].id, mode: "answers", overrides: [], sweep: preset.sweep ? { key: preset.sweep, values: "" } : null,
                  variants: null, limit: "", name: "" };
  if (preset.sweep && byKey[preset.sweep]) state.sweep.values = suggestValues(byKey[preset.sweep], v[preset.sweep]);

  const content = h("div", { style: { display: "flex", flexDirection: "column", gap: "16px" } });
  const estimateEl = h("div", { class: "hint" });

  function settingSelect(value, onChange, exclude = []) {
    return h("select", { onchange: (e) => onChange(e.target.value) },
      h("option", { value: "" }, "Choose a setting…"),
      settings.groups.map((g) => h("optgroup", { label: g.label }, schema.filter((s) => s.group === g.key && (!exclude.includes(s.key) || s.key === value))
        .map((s) => h("option", { value: s.key, selected: s.key === value }, s.label)))));
  }

  function draw() {
    const set = sets.find((s) => s.id === state.set);
    const effVals = { ...v, ...Object.fromEntries(state.overrides.filter((o) => o.key).map((o) => [o.key, o.value])) };
    const rows = state.overrides.map((o, i) => {
      const spec = byKey[o.key];
      const effect = spec ? effectText(o.key, o.value, effVals, settings.insights, models) : "";
      return h("div", { class: "override-row" },
        settingSelect(o.key, (k) => { o.key = k; o.value = k ? v[k] : null; draw(); }, state.overrides.map((x) => x.key)),
        spec ? h("div", {}, control(spec, o.value, (x) => { o.value = x; updateEffects(); }, { models }), h("div", { class: "hint" }, `now: ${formatValue(spec, v[o.key])}`)) : h("div", {}),
        h("button", { class: "icon-btn", "aria-label": "Remove change", onclick: () => { state.overrides.splice(i, 1); draw(); } }, icon("x")),
        effect ? h("div", { class: "effect" }, effect) : null);
    });
    const sweepSpec = state.sweep && byKey[state.sweep.key];
    const variants = set ? Object.keys(set.variants) : [];
    content.replaceChildren(
      h("label", { class: "field" }, "Question set",
        h("select", { onchange: (e) => { state.set = e.target.value; state.variants = null; draw(); } },
          sets.map((s) => h("option", { value: s.id, selected: s.id === state.set }, `${s.name} (${s.count})`)))),
      h("div", {}, h("div", { style: { fontWeight: 550, marginBottom: "6px" } }, "What to run"),
        segmented([{ value: "answers", label: "Full answers" }, { value: "passages", label: "Passages only" }], state.mode, (x) => { state.mode = x; updateEffects(); }),
        h("div", { class: "hint", style: { marginTop: "6px" } }, state.mode === "answers"
          ? "Answers every question and scores it. Answers can be graded in Evaluate."
          : "Only checks which passages would be sent — much faster. For tuning search, cutoff and chunking.")),
      h("div", {}, h("div", { style: { fontWeight: 550 } }, "Change settings for this run"),
        h("div", { class: "hint" }, "Anything not changed here uses your current settings. Changing chunking builds a temporary index; your library isn't touched."),
        rows,
        h("button", { class: "btn sm", style: { marginTop: "8px" }, onclick: () => { state.overrides.push({ key: "", value: null }); draw(); } }, icon("plus"), "Change a setting")),
      h("div", {}, h("label", { class: "switch" }, h("input", { type: "checkbox", checked: !!state.sweep, onchange: (e) => { state.sweep = e.target.checked ? { key: "", values: "" } : null; draw(); } }),
        h("span", { class: "track" }), h("span", {}, "Compare several values of one setting")),
        state.sweep ? h("div", { style: { display: "grid", gridTemplateColumns: "220px 1fr", gap: "10px", marginTop: "10px" } },
          settingSelect(state.sweep.key, (k) => { state.sweep.key = k; state.sweep.values = k ? suggestValues(byKey[k], v[k]) : ""; draw(); }),
          h("div", {}, h("input", { type: "text", value: state.sweep.values, placeholder: "e.g. 2, 3, 4", oninput: (e) => { state.sweep.values = e.target.value; updateEffects(); } }),
            h("div", { class: "hint" }, sweepSpec ? `One run per value, comma-separated. Now: ${formatValue(sweepSpec, v[state.sweep.key])}.` : "Pick a setting."))) : null),
      variants.length > 1 ? h("div", {}, h("div", { style: { fontWeight: 550, marginBottom: "6px" } }, "Question variants"),
        h("div", { class: "tag-row" }, variants.map((k) => h("button", { class: "chip" + (!state.variants || state.variants.includes(k) ? " on" : ""),
          title: meta.variants.find((x) => x.key === k)?.hint, onclick: () => {
            const cur = state.variants || [...variants];
            state.variants = cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k];
            if (state.variants.length === variants.length) state.variants = null;
            draw();
          } }, `${k} ${set.variants[k]}`)))) : null,
      h("div", { style: { display: "grid", gridTemplateColumns: "1fr 160px", gap: "12px" } },
        h("label", { class: "field" }, "Name", h("input", { type: "text", value: state.name, placeholder: autoName(), oninput: (e) => { state.name = e.target.value; } })),
        h("label", { class: "field" }, "Question limit", h("input", { type: "number", min: 1, value: state.limit, placeholder: "all", oninput: (e) => { state.limit = e.target.value; updateEffects(); } }))),
      estimateEl);
    updateEffects();
  }

  function updateEffects() {
    const set = sets.find((s) => s.id === state.set);
    if (!set) return;
    let n = state.variants ? Object.entries(set.variants).filter(([k]) => state.variants.includes(k)).reduce((a, [, c]) => a + c, 0) : set.count;
    if (+state.limit > 0) n = Math.min(n, +state.limit);
    const eff = { ...v, ...Object.fromEntries(state.overrides.filter((o) => o.key).map((o) => [o.key, o.value])) };
    const per = settings.insights.rerank_ms_per_candidate || 200;
    const retrieval = 0.15 + (eff.rerank ? (eff.rerank_candidates * per) / 1000 : 0);
    const perQ = state.mode === "answers" ? retrieval + 4 : retrieval;
    const runsN = state.sweep ? Math.max(1, parseValues(state.sweep).length) : 1;
    const secs = n * perQ * runsN;
    const reindex = ["chunk_size", "chunk_overlap", "embed_model"].some((k) => k in eff && state.overrides.some((o) => o.key === k));
    estimateEl.textContent = `${runsN > 1 ? `${runsN} runs × ` : ""}${n} questions · roughly ${secs < 90 ? `${Math.round(secs)} seconds` : `${Math.round(secs / 60)} minutes`}${reindex ? " plus building a temporary index" : ""}. Runs go one at a time in the background; you can keep chatting.`;
    content.querySelectorAll(".override-row").forEach((row, i) => {
      const o = state.overrides[i];
      const effEl = row.querySelector(".effect");
      if (o?.key && effEl) effEl.textContent = effectText(o.key, o.value, eff, settings.insights, models);
    });
  }

  function autoName() {
    const parts = state.overrides.filter((o) => o.key).map((o) => `${o.key}=${formatValue(byKey[o.key], o.value)}`);
    if (state.sweep?.key) parts.push(`${state.sweep.key} sweep`);
    return parts.join(", ") || "current settings";
  }

  function parseValues(sw) {
    const spec = byKey[sw.key];
    if (!spec) return [];
    return sw.values.split(",").map((x) => x.trim()).filter(Boolean).map((x) => {
      if (spec.type === "int") return parseInt(x, 10);
      if (spec.type === "float") return parseFloat(x);
      if (spec.type === "bool") return ["on", "true", "yes", "1"].includes(x.toLowerCase());
      if (spec.type === "choice") return spec.choices.find((c) => String(c) === x) ?? x;
      return x;
    });
  }

  draw();
  dialog({ title: "New experiment run", wide: true, content,
    buttons: [{ label: "Cancel" }, { label: "Start", cls: "primary", onClick: async () => {
      const overrides = Object.fromEntries(state.overrides.filter((o) => o.key).map((o) => [o.key, o.value]));
      const body = { name: state.name || autoName(), set_id: state.set, overrides, retrieval_only: state.mode === "passages",
                     variants: state.variants, limit: +state.limit > 0 ? +state.limit : null };
      if (state.sweep) {
        const values = parseValues(state.sweep);
        if (!state.sweep.key || values.length < 2) throw new Error("Pick a setting and at least two values to compare");
        body.sweep = { key: state.sweep.key, values };
      }
      const r = await api.post("/api/runs", body);
      toast(r.ids.length > 1 ? `${r.ids.length} runs queued` : "Run queued");
      app.refreshStatus();
      onStarted?.();
    } }] });
}

function suggestValues(spec, current) {
  if (!spec) return "";
  if (spec.type === "bool") return "on, off";
  if (spec.type === "choice") return spec.choices.join(", ");
  if (spec.type === "int" || spec.type === "float") {
    const step = spec.step * (spec.type === "int" ? Math.max(1, Math.round((spec.max - spec.min) / spec.step / 8)) : 2);
    const vals = [current - step, current, current + step].filter((x) => x >= spec.min && x <= spec.max);
    return vals.map((x) => (spec.type === "float" ? +x.toFixed(2) : x)).join(", ");
  }
  return String(current ?? "");
}

// ---------------------------------------------------------------- run detail
async function runDetail({ inner }, id) {
  let timer = null;
  const settings = await loadSettings().catch(() => null);
  async function load() {
    let r;
    try { r = await api.get(`/api/runs/${id}`); } catch (e) { toastError(e); location.hash = "#/experiments"; return; }
    const s = r.summary;
    const live = r.status === "running" || r.status === "queued";
    const actions = [
      !r.config.retrieval_only ? h("a", { class: "btn primary", href: `#/evaluate?run=${r.id}&status=ungraded` }, icon("grade"), "Grade answers") : null,
      live ? h("button", { class: "btn", onclick: async () => { await api.post(`/api/runs/${r.id}/cancel`).catch(toastError); load(); } }, "Cancel") : null,
      h("button", { class: "btn", onclick: () => pickCompare(r) }, icon("compare"), "Compare…"),
      h("button", { class: "icon-btn", "aria-label": "Delete run", onclick: async () => {
        if (!(await confirmDialog(`Delete run “${r.name}”?`, "Its answers stay in Evaluate (unlinked from the run).", { confirm: "Delete", danger: true }))) return;
        await api.del(`/api/runs/${r.id}`).catch(toastError); location.hash = "#/experiments";
      } }, icon("trash")),
    ];
    const cards = METRICS.filter((m) => s[m.key] !== undefined && s[m.key] !== null).map((m) => h("div", { class: "stat" },
      h("div", { class: "label" }, m.label, tip(m.help)), h("div", { class: "value" }, m.fmt(s[m.key])),
      m.key === "good_rate" ? h("div", { class: "sub" }, `${s.graded} of ${s.questions} graded`) : null));
    if (!s.graded && !r.config.retrieval_only && r.status === "done") cards.unshift(h("div", { class: "stat", style: { gridColumn: "span 2" } },
      h("div", { class: "label" }, "Graded good"), h("div", { class: "value", style: { fontSize: "15px", fontWeight: 500 } }, "Not graded yet"),
      h("a", { href: `#/evaluate?run=${r.id}&status=ungraded`, class: "sub" }, "Grade the answers →")));
    const variants = Object.entries(s.by_variant || {});
    fill(inner,
      h("a", { href: "#/experiments", class: "hint", style: { display: "inline-flex", gap: "6px", alignItems: "center", marginBottom: "8px" } }, icon("back"), "All runs"),
      header(r.name, `${r.set_name || "deleted question set"} · ${r.total} questions · ${r.config.retrieval_only ? "passages only" : "full answers"} · started ${timeAgo(r.created)}`, actions),
      h("div", { style: { display: "flex", gap: "10px", alignItems: "center", marginBottom: "16px", flexWrap: "wrap" } }, statusBadge(r), changesChips(r, settings?.schema),
        r.error ? h("span", { class: "warning-chip" }, r.error) : null),
      live ? h("div", { class: "card card-pad", style: { marginBottom: "16px" } }, h("div", { class: "muted" }, `${r.done} of ${r.total} questions`),
        h("div", { class: "progress", style: { marginTop: "8px" } }, h("div", { style: { width: `${(100 * r.done) / Math.max(1, r.total)}%` } }))) : null,
      h("div", { class: "stat-grid", style: { marginBottom: "20px" } }, cards),
      variants.length ? h("div", { class: "card", style: { marginBottom: "20px", overflowX: "auto" } }, h("table", { class: "table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Variant"), h("th", { class: "num" }, "Questions"), h("th", { class: "num" }, "Context recall"),
          r.config.retrieval_only ? null : h("th", { class: "num" }, "Auto-correct"), h("th", {}, "Grades"))),
        h("tbody", {}, variants.map(([k, x]) => h("tr", {}, h("td", {}, k), h("td", { class: "num" }, x.n), h("td", { class: "num" }, pct(x.context_recall)),
          r.config.retrieval_only ? null : h("td", { class: "num" }, pct(x.auto_correct)),
          h("td", {}, x.good + x.partial + x.bad ? h("span", { class: "run-changes" }, h("span", { class: "badge good" }, x.good), h("span", { class: "badge partial" }, x.partial), h("span", { class: "badge bad" }, x.bad)) : h("span", { class: "faint" }, "—"))))))) : null,
      h("h2", { class: "section-title", style: { margin: "4px 0 10px" } }, "Questions"),
      h("div", { class: "card", style: { overflowX: "auto" } }, h("table", { class: "table" },
        h("thead", {}, h("tr", {}, h("th", {}, "Id"), h("th", {}, "Variant"), h("th", {}, "Question"),
          r.config.retrieval_only ? null : h("th", {}, "Auto"), h("th", { class: "num" }, "Context"),
          r.config.retrieval_only ? null : h("th", {}, "Grade"), h("th", { class: "num" }, "s"))),
        h("tbody", {}, r.results.map((x) => h("tr", { class: x.interaction_id ? "clickable" : "", onclick: () => { if (x.interaction_id) location.hash = `#/evaluate?id=${x.interaction_id}&run=${r.id}`; } },
          h("td", { class: "mono" }, x.qid), h("td", {}, x.variant), h("td", { style: { maxWidth: "420px" } }, x.question),
          r.config.retrieval_only ? null : h("td", {}, x.auto_label ? h("span", { class: `label-chip ${x.auto_label}` }, x.auto_label) : "—"),
          h("td", { class: "num" }, x.answerable ? pct(x.context_coverage) : h("span", { class: "faint" }, "n/a")),
          r.config.retrieval_only ? null : h("td", {}, x.grade ? gradeBadge(x.grade) : h("span", { class: "faint" }, "—")),
          h("td", { class: "num" }, num(x.total_s, 1))))))));
    clearTimeout(timer);
    if (live) timer = setTimeout(load, 2000);
  }
  async function pickCompare(r) {
    const runs = (await api.get("/api/runs")).filter((x) => x.id !== r.id && x.status === "done");
    if (!runs.length) { toast("No other finished runs to compare with"); return; }
    const sel = h("select", {}, runs.map((x) => h("option", { value: x.id }, `${x.name} (${timeAgo(x.created)})`)));
    dialog({ title: "Compare with", content: h("label", { class: "field" }, "Run", sel),
      buttons: [{ label: "Cancel" }, { label: "Compare", cls: "primary", onClick: () => { location.hash = `#/experiments/compare/${sel.value},${r.id}`; } }] });
  }
  await load();
  return () => clearTimeout(timer);
}

// ---------------------------------------------------------------- compare
async function compareView({ inner }, ids) {
  let data;
  try { data = await api.get(`/api/runs/compare?ids=${ids.join(",")}`); } catch (e) { toastError(e); return; }
  const settings = await loadSettings().catch(() => null);
  const runs = data.runs;
  const letters = runs.map((_, i) => String.fromCharCode(65 + i));
  const base = runs[0].summary;
  const cell = (m, r) => {
    const val = r.summary[m.key];
    const b = base[m.key];
    let delta = null;
    if (r !== runs[0] && val != null && b != null && val !== b) {
      const d = val - b;
      const better = m.up ? d > 0 : d < 0;
      const txt = m.fmt === pct ? `${d > 0 ? "+" : ""}${Math.round(100 * d)} pts` : `${d > 0 ? "+" : ""}${Number.isInteger(d) ? d : d.toFixed(1)}`;
      delta = h("span", { class: `delta ${better ? "up" : "down"}` }, txt);
    }
    return h("td", { class: "num" }, m.fmt(val), delta);
  };
  const variants = [...new Set(runs.flatMap((r) => Object.keys(r.summary.by_variant || {})))];
  add(inner,
    h("a", { href: "#/experiments", class: "hint", style: { display: "inline-flex", gap: "6px", alignItems: "center", marginBottom: "8px" } }, icon("back"), "All runs"),
    header("Compare runs", "Differences are shown against run A. Green is better, red is worse."),
    h("div", { class: "card", style: { overflowX: "auto", marginBottom: "20px" } }, h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Metric"), runs.map((r, i) => h("th", { class: "num" },
        h("a", { href: `#/experiments/run/${r.id}` }, `${letters[i]} · ${r.name}`))))),
      h("tbody", {},
        h("tr", {}, h("td", { class: "muted" }, "Changes"), runs.map((r) => h("td", {}, changesChips(r, settings?.schema)))),
        METRICS.filter((m) => runs.some((r) => r.summary[m.key] != null)).map((m) => h("tr", {},
          h("td", {}, h("span", { class: "metric-name" }, m.label, tip(m.help))), runs.map((r) => cell(m, r)))),
        h("tr", {}, h("td", { class: "muted" }, "Graded"), runs.map((r) => h("td", { class: "num" }, `${r.summary.graded} / ${r.summary.questions}`)))))),
    variants.length ? h("div", { class: "card", style: { overflowX: "auto", marginBottom: "20px" } }, h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Variant"), runs.map((_, i) => h("th", { class: "num" }, `${letters[i]} context`)),
        runs.map((r, i) => r.config.retrieval_only ? null : h("th", { class: "num" }, `${letters[i]} auto-correct`)))),
      h("tbody", {}, variants.map((v) => h("tr", {}, h("td", {}, v),
        runs.map((r) => h("td", { class: "num" }, pct(r.summary.by_variant?.[v]?.context_recall))),
        runs.map((r) => r.config.retrieval_only ? null : h("td", { class: "num" }, pct(r.summary.by_variant?.[v]?.auto_correct)))))))) : null,
    h("h2", { class: "section-title", style: { margin: "4px 0 10px" } }, `Questions that changed (${letters[0]} → ${letters[letters.length - 1]})`),
    data.changes.length ? h("div", { class: "card", style: { overflowX: "auto" } }, h("table", { class: "table" },
      h("thead", {}, h("tr", {}, h("th", {}, "Id"), h("th", {}, "Variant"), h("th", {}, "Question"), h("th", {}, "Before"), h("th", {}, "After"))),
      h("tbody", {}, data.changes.map((c) => h("tr", {}, h("td", { class: "mono" }, c.qid), h("td", {}, c.variant), h("td", {}, c.question),
        h("td", {}, h("span", { class: `label-chip ${labelClass(c.before)}` }, c.before)), h("td", {}, h("span", { class: `label-chip ${labelClass(c.after)}` }, c.after)))))))
      : h("p", { class: "muted" }, "No question changed outcome."));
}

function labelClass(x) { return { good: "correct", bad: "wrong" }[x] || x; }
