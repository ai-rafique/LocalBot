// Settings: presets, models, and every tunable parameter with a live
// explanation of what the chosen value will do.
import { api } from "./api.js";
import { app, menuButton } from "./main.js";
import { control, effectText, invalidateSettings, loadModels, loadSettings } from "./tuning.js";
import { confirmDialog, debounce, fill, h, icon, switchEl, toast, toastError } from "./ui.js";

export async function render(main, route) {
  const page = h("div", { class: "page" });
  const inner = h("div", { class: "page-inner" });
  page.append(inner);
  main.append(page);
  inner.append(h("div", { class: "stage" }, h("span", { class: "spinner" }), "Loading"));

  let s, models, draft, pollTimer = null;
  let advanced = localStorage.getItem("advanced") === "1";
  const effects = {};   // key -> element showing the effect line
  const rows = {};      // key -> setting row element
  const savebar = h("div", { class: "savebar hidden" });

  async function load() {
    invalidateSettings();
    [s, models] = await Promise.all([loadSettings(true), loadModels(true)]);
    s.insights.indexed_embed = s.index.embed_model || s.values.embed_model;
    draft = { ...s.values };
    draw();
    const target = route.parts[0];
    if (target) requestAnimationFrame(() => document.getElementById(`group-${target}`)?.scrollIntoView());
  }

  const changedKeys = () => Object.keys(draft).filter((k) => draft[k] !== s.values[k]);

  function draw() {
    const presetKey = Object.keys(s.presets).find((k) => Object.entries(s.presets[k].values).every(([key, v]) => draft[key] === v));
    const presets = h("div", { class: "presets" }, Object.entries(s.presets).map(([k, p]) => h("button", { class: "preset" + (k === presetKey ? " on" : ""),
      onclick: () => { Object.assign(draft, p.values); draw(); } },
      h("b", {}, p.label, k === presetKey ? h("span", { class: "badge accent" }, "Current") : null), h("span", {}, p.summary))));
    const nav = h("nav", { class: "settings-nav" }, s.groups.map((g) => h("a", { href: `#/settings/${g.key}`, onclick: (e) => {
      e.preventDefault(); document.getElementById(`group-${g.key}`)?.scrollIntoView({ behavior: "smooth" }); } }, g.label)));
    const groups = s.groups.map((g) => {
      const items = s.schema.filter((x) => x.group === g.key && (advanced || !x.advanced));
      const card = h("div", { class: "card" }, items.map(settingRow));
      const extra = g.key === "models" ? modelsCard() : g.key === "chunking"
        ? h("p", { class: "hint", style: { margin: "8px 2px 0" } }, "See exactly how your documents would be split in ", h("a", { href: "#/documents" }, "Documents"), ".") : null;
      return h("section", { class: "settings-group", id: `group-${g.key}` }, h("h2", {}, g.label), card, extra);
    });
    fill(inner,
      h("div", { class: "page-head" },
        h("div", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, menuButton(),
          h("div", {}, h("h1", {}, "Settings"), h("p", {}, "Every change shows what it will do. Nothing is applied until you save."))),
        h("div", { class: "actions" }, switchEl(advanced, (v) => { advanced = v; localStorage.setItem("advanced", v ? "1" : "0"); draw(); }, "Show advanced"))),
      presets,
      h("div", { class: "settings-layout" }, nav, h("div", {}, groups, savebar)));
    updateAll();
  }

  function settingRow(spec) {
    const key = spec.key;
    const effect = h("div", { class: "effect" }, icon("info"), h("span", {}));
    effects[key] = effect;
    const ctl = control(spec, draft[key], (v) => { draft[key] = v; updateAll(key); }, { models: models.installed });
    const testable = ["int", "float", "bool", "choice", "model"].includes(spec.type);
    const links = h("div", { class: "links" });
    const row = h("div", { class: "setting" + (spec.type === "text" ? " wide" : "") },
      h("div", {},
        h("div", { class: "label" }, spec.label,
          spec.reindex ? h("span", { class: "badge warn", title: "Documents must be re-indexed after changing this" }, "re-index") : null,
          spec.advanced ? h("span", { class: "badge" }, "advanced") : null),
        h("div", { class: "help" }, spec.help, spec.effect ? " " + spec.effect : ""),
        links),
      h("div", { class: "control" }, ctl),
      effect);
    row.links = links; row.spec = spec; row.testable = testable;
    rows[key] = row;
    return row;
  }

  // Refresh effect lines, "changed" markers, links and the save bar.
  function updateAll() {
    for (const [key, row] of Object.entries(rows)) {
      if (!row.isConnected) continue;
      const spec = row.spec;
      const text = effectText(key, draft[key], draft, s.insights, models.installed);
      effects[key].classList.toggle("hidden", !text);
      effects[key].lastChild.textContent = text;
      row.classList.toggle("changed", draft[key] !== s.values[key]);
      fill(row.links,
        draft[key] !== spec.default ? h("a", { href: "#", onclick: (e) => { e.preventDefault(); draft[key] = spec.default; draw(); } }, "Reset to default") : null,
        row.testable ? h("a", { href: `#/experiments?new=1&sweep=${key}` }, "Compare values in an experiment") : null);
    }
    if (["chunk_size", "chunk_overlap"].some((k) => draft[k] !== s.values[k])) previewSoon();
    else if (s.insights.preview) { s.insights.preview = null; }
    const changed = changedKeys();
    savebar.classList.toggle("hidden", !changed.length);
    fill(savebar,
      h("span", { class: "grow" }, `${changed.length} unsaved change${changed.length === 1 ? "" : "s"}`,
        changed.some((k) => s.schema.find((x) => x.key === k)?.reindex) ? " · requires re-indexing" : ""),
      h("button", { class: "btn ghost", onclick: () => { draft = { ...s.values }; draw(); } }, "Discard"),
      h("button", { class: "btn primary", onclick: save }, "Save"));
  }

  const previewSoon = debounce(async () => {
    if (draft.chunk_overlap >= draft.chunk_size) return;
    try {
      s.insights.preview = await api.post("/api/chunking/preview", { chunk_size: draft.chunk_size, chunk_overlap: draft.chunk_overlap });
      for (const k of ["chunk_size", "chunk_overlap"]) if (effects[k]) effects[k].lastChild.textContent = effectText(k, draft[k], draft, s.insights);
    } catch (e) { /* preview is best-effort */ }
  }, 400);

  async function save() {
    const changes = Object.fromEntries(changedKeys().map((k) => [k, draft[k]]));
    try {
      const r = await api.put("/api/settings", { changes });
      toast("Settings saved");
      await load();
      app.refreshStatus();
      if (r.reindex.length && (await confirmDialog("Re-index documents now?",
        "Chunking or the embedding model changed. Until you re-index, answers keep using the old passages.", { confirm: "Re-index now" }))) {
        toast("Re-indexing…");
        const x = await api.post("/api/documents/reindex");
        toast(`Re-indexed · ${x.index.chunks} passages`);
        await load();
      }
    } catch (e) { toastError(e); }
  }

  // ---- models card
  function modelsCard() {
    const gpu = models.gpu;
    const loaded = Object.fromEntries((models.loaded || []).map((m) => [m.name, m]));
    const nameInput = h("input", { type: "text", placeholder: "e.g. qwen3.5:4b or llama3.2:3b" });
    const install = async () => {
      const name = nameInput.value.trim();
      if (!name) return;
      try { await api.post("/api/models/pull", { name }); toast(`Installing ${name}…`); nameInput.value = ""; pollPulls(); }
      catch (e) { toastError(e); }
    };
    nameInput.addEventListener("keydown", (e) => { if (e.key === "Enter") install(); });
    const pulls = Object.entries(models.pulls || {}).filter(([, p]) => p.state !== "done" || Date.now() - (p.seen || Date.now()) < 5000);
    const list = models.ollama === false
      ? h("div", { class: "card-pad muted" }, models.error || "Ollama isn't running.")
      : models.installed.map((m) => {
        const l = loaded[m.name];
        return h("div", { class: "model-row" },
          h("div", { class: "grow" }, h("div", { class: "name" }, m.name),
            h("div", { class: "meta" }, [m.role === "embedding" ? "embedding" : "chat", m.parameters, m.quantization, `${m.size_gb} GB`].filter(Boolean).join(" · "))),
          m.fits_gpu === false ? h("span", { class: "badge warn", title: "Larger than your GPU memory allows: part runs on the CPU, slower" }, "may not fit GPU")
            : m.fits_gpu ? h("span", { class: "badge good" }, "fits GPU") : null,
          l ? h("span", { class: "badge accent", title: "Currently loaded" }, l.gpu_share >= 0.99 ? "loaded · GPU" : `loaded · ${Math.round(100 * l.gpu_share)}% GPU`) : null,
          m.name === s.values.llm_model || m.name === `${s.values.llm_model}:latest` ? h("span", { class: "badge" }, "chat") : null,
          m.name === s.values.embed_model || m.name === `${s.values.embed_model}:latest` ? h("span", { class: "badge" }, "embeddings") : null);
      });
    return h("div", { class: "card", style: { marginTop: "12px" } },
      h("div", { class: "card-pad", style: { paddingBottom: "8px" } }, h("h2", {}, "Installed models"),
        h("p", { class: "hint", style: { margin: 0 } }, gpu ? `${gpu.name}: ${(gpu.total_mb / 1024).toFixed(1)} GB of GPU memory. Models that fit entirely run fastest.` : "No NVIDIA GPU detected; models run on the CPU.")),
      list,
      h("div", { class: "card-pad", style: { borderTop: "1px solid var(--border)" } },
        h("h2", {}, "Add a model"),
        h("p", { class: "hint", style: { marginTop: 0 } }, "Enter a model name from ", h("a", { href: "https://ollama.com/library", target: "_blank", rel: "noopener" }, "ollama.com/library"),
          ". It's downloaded once by Ollama and runs locally afterwards. For chat, pick an instruction-tuned model smaller than your GPU memory; after installing, choose it as the chat model above and run an experiment to compare it with the current one. Changing the embedding model requires re-indexing."),
        h("div", { style: { display: "flex", gap: "8px" } }, nameInput, h("button", { class: "btn primary", onclick: install }, icon("download"), "Install")),
        h("div", { id: "pulls" }, pulls.map(([name, p]) => pullRow(name, p)))));
  }

  function pullRow(name, p) {
    const frac = p.total ? p.completed / p.total : 0;
    return h("div", { style: { marginTop: "10px" } },
      h("div", { style: { display: "flex", justifyContent: "space-between", fontSize: "13px" } }, h("b", { class: "mono" }, name),
        h("span", { class: p.state === "error" ? "" : "hint", style: p.state === "error" ? { color: "var(--bad)" } : {} },
          p.state === "done" ? "Installed" : p.state === "error" ? p.status : `${p.status}${p.total ? ` · ${Math.round(100 * frac)}%` : ""}`)),
      p.state === "running" ? h("div", { class: "progress", style: { marginTop: "6px" } }, h("div", { style: { width: `${100 * frac}%` } })) : null);
  }

  async function pollPulls() {
    clearTimeout(pollTimer);
    try {
      const m = await api.get("/api/models");
      const box = document.getElementById("pulls");
      if (box) box.replaceChildren(...Object.entries(m.pulls).map(([n, p]) => pullRow(n, p)));
      if (Object.values(m.pulls).some((p) => p.state === "running")) pollTimer = setTimeout(pollPulls, 1000);
      else { models = await loadModels(true); draw(); }
    } catch (e) { pollTimer = setTimeout(pollPulls, 3000); }
  }

  try { await load(); } catch (e) { toastError(e); }
  if (Object.values(models?.pulls || {}).some((p) => p.state === "running")) pollPulls();
  const warn = (e) => { if (draft && changedKeys().length) { e.preventDefault(); e.returnValue = ""; } };
  window.addEventListener("beforeunload", warn);
  return () => { clearTimeout(pollTimer); window.removeEventListener("beforeunload", warn); };
}
