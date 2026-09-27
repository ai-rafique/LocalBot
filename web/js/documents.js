// Documents: the library, uploads, and chunking with a live preview.
import { api } from "./api.js";
import { app, menuButton } from "./main.js";
import { control, effectText, invalidateSettings, loadSettings } from "./tuning.js";
import { bytes, confirmDialog, debounce, dialog, fill, h, icon, thumb, toast, toastError } from "./ui.js";

export async function render(main) {
  const page = h("div", { class: "page" });
  const inner = h("div", { class: "page-inner" });
  page.append(inner);
  main.append(page);

  let lib, settings, busy = null;
  const fileInput = h("input", { type: "file", multiple: true, accept: ".pdf,.docx,.pptx,.xlsx,.txt,.md,.png,.jpg,.jpeg,.webp", class: "hidden",
    onchange: (e) => { upload([...e.target.files]); e.target.value = ""; } });
  const draft = {};

  async function load() {
    [lib, settings] = await Promise.all([api.get("/api/documents"), loadSettings(true)]);
    draft.chunk_size = settings.values.chunk_size;
    draft.chunk_overlap = settings.values.chunk_overlap;
    draw();
    refreshPreview();
  }

  function draw() {
    const st = lib.index;
    const docs = lib.documents;
    const total = docs.reduce((a, d) => a + d.chunks, 0);
    const head = h("div", { class: "page-head" },
      h("div", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, menuButton(),
        h("div", {}, h("h1", {}, "Documents"),
          h("p", {}, docs.length ? `${docs.length} document${docs.length === 1 ? "" : "s"} · ${total} passages${st.built ? ` · indexed ${st.built}` : ""}`
            : "Add the files LocalBot should answer from."))),
      h("div", { class: "actions" },
        docs.length ? h("button", { class: "btn", onclick: () => reindex(), disabled: !!busy }, icon("refresh"), "Re-index") : null,
        h("button", { class: "btn primary", onclick: () => fileInput.click(), disabled: !!busy }, icon("upload"), "Add documents"),
        fileInput));
    const banners = [];
    if (busy) banners.push(h("div", { class: "banner info" }, h("span", { class: "spinner" }), h("div", { class: "grow" }, busy)));
    else if (st.stale?.length && docs.length) {
      banners.push(h("div", { class: "banner warn" }, icon("alert"),
        h("div", { class: "grow" }, h("b", {}, "Settings changed since the index was built. "),
          `${st.stale.map(label).join(", ")} differ${st.stale.length === 1 ? "s" : ""} — answers still use the old index until you re-index.`),
        h("button", { class: "btn sm primary", onclick: () => reindex() }, "Re-index now")));
    }
    const list = docs.length ? h("div", { class: "card" }, docs.map(docRow), dropzone(true))
      : h("div", { class: "card card-pad" }, h("div", { class: "empty-state", style: { padding: "24px 0 16px" } },
          h("div", { class: "icon-circle" }, icon("docs")), h("h3", {}, "No documents yet"),
          h("p", {}, "PDF, Word, PowerPoint and Excel files, Markdown and text, and pictures. Text in pictures is read too. Files are copied into LocalBot's data folder and never leave this computer.")),
        dropzone(false));
    fill(inner, head, banners, h("div", { class: "two-col" }, list, chunkPanel()));
  }

  function label(k) { return { chunk_size: "Chunk size", chunk_overlap: "Overlap", embed_model: "Embedding model" }[k] || k; }

  function docRow(d) {
    const meta = [`${d.chunks} passages`, d.sections ? `${d.sections} sections` : null, d.pages ? `${d.pages} pages` : null,
      d.pictures ? `${d.pictures} picture${d.pictures === 1 ? "" : "s"} read` : null,
      bytes(d.size_bytes), d.added ? `added ${d.added}` : null].filter(Boolean).join(" · ");
    return h("div", { class: "doc-row" },
      h("div", { class: `doc-icon ${["png", "jpg", "jpeg", "webp"].includes(d.type) ? "img" : d.type}` }, d.type.toUpperCase()),
      h("div", { class: "doc-info" }, h("div", { class: "name", title: d.name }, d.name), h("div", { class: "meta" }, meta),
        !d.stored ? h("div", { class: "hint" }, "Original file missing: can't be re-indexed. Upload it again.") : null),
      h("button", { class: "btn sm", onclick: () => showChunks(d.name) }, "View passages"),
      h("button", { class: "icon-btn", title: "Remove", "aria-label": `Remove ${d.name}`, onclick: () => remove(d) }, icon("trash")));
  }

  function dropzone(compact) {
    const z = h("div", { class: "dropzone", style: compact ? { margin: "12px", padding: "16px" } : {}, onclick: () => fileInput.click(),
      role: "button", tabindex: 0 },
      icon("upload"), h("div", {}, h("b", {}, "Drop files here"), " or click to browse"),
      h("div", { class: "hint" }, "PDF · Word · PowerPoint · Excel · Markdown · text · pictures (PNG, JPG, WEBP) — re-adding a file with the same name replaces it"));
    return z;
  }

  // ---- chunking panel
  const previewBox = h("div", {});
  const sampleSelect = h("select", { onchange: () => refreshPreview() });
  function chunkPanel() {
    const spec = Object.fromEntries(settings.schema.map((s) => [s.key, s]));
    const changed = draft.chunk_size !== lib.index.chunk_size || draft.chunk_overlap !== lib.index.chunk_overlap;
    sampleSelect.replaceChildren(h("option", { value: "" }, "All documents"), ...lib.documents.filter((d) => d.stored).map((d) =>
      h("option", { value: d.name }, d.name)));
    return h("div", { class: "card card-pad", style: { position: "sticky", top: "0" } },
      h("h2", {}, "How documents are split"),
      h("p", { class: "muted", style: { marginTop: 0 } }, "Documents are split into passages along their own paragraphs and headings. The model reads the best few passages for each question."),
      h("div", { style: { marginTop: "14px" } }, h("b", {}, "Chunk size"),
        h("div", { class: "hint" }, spec.chunk_size.effect),
        control(spec.chunk_size, draft.chunk_size, (v) => { draft.chunk_size = v; refreshPreviewSoon(); })),
      h("div", { style: { marginTop: "14px" } }, h("b", {}, "Overlap"),
        h("div", { class: "hint" }, spec.chunk_overlap.effect),
        control(spec.chunk_overlap, draft.chunk_overlap, (v) => { draft.chunk_overlap = v; refreshPreviewSoon(); })),
      h("div", { style: { marginTop: "16px", borderTop: "1px solid var(--border)", paddingTop: "14px" } },
        h("div", { style: { display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" } }, h("b", {}, "Preview"), sampleSelect),
        previewBox),
      h("div", { class: "actions", style: { marginTop: "14px" } },
        h("button", { class: "btn primary", disabled: !lib.documents.length || !!busy, onclick: () => applyChunking() },
          changed || lib.index.stale?.length ? "Apply and re-index" : "Re-index"),
        h("button", { class: "btn ghost", onclick: () => { const d = spec; draft.chunk_size = d.chunk_size.default; draft.chunk_overlap = d.chunk_overlap.default; draw(); refreshPreview(); } }, "Defaults")));
  }

  let previewSeq = 0;
  async function refreshPreview() {
    if (!lib.documents.some((d) => d.stored)) { previewBox.replaceChildren(h("p", { class: "hint" }, "Add a document to preview how it will be split.")); return; }
    if (draft.chunk_overlap >= draft.chunk_size) { previewBox.replaceChildren(h("p", { class: "warning-chip" }, "Overlap must be smaller than the chunk size")); return; }
    const seq = ++previewSeq;
    previewBox.style.opacity = ".6";
    try {
      const p = await api.post("/api/chunking/preview", { chunk_size: draft.chunk_size, chunk_overlap: draft.chunk_overlap, name: sampleSelect.value || null });
      if (seq !== previewSeq) return;
      const now = sampleSelect.value ? lib.documents.find((d) => d.name === sampleSelect.value)?.chunks : lib.documents.reduce((a, d) => a + d.chunks, 0);
      const max = Math.max(1, ...p.histogram);
      const ins = settings.insights;
      const reads = effectText("top_k", settings.values.top_k, settings.values, { ...ins, avg_chunk_chars: p.avg_chars });
      fill(previewBox,
        h("div", { class: "compare-line" }, h("span", { class: "big" }, p.chunks), h("span", { class: "muted" }, `passages${now !== undefined && now !== p.chunks ? ` (now ${now})` : ""}`)),
        h("div", { class: "muted" }, `average ${p.avg_chars} characters · shortest ${p.min_chars} · longest ${p.max_chars}`),
        h("div", { class: "histogram", title: "Passage lengths, from short (left) to the chunk size (right)" }, p.histogram.map((n) => h("div", { style: { height: `${(100 * n) / max}%` } }))),
        h("div", { class: "hint" }, "Passage lengths — mostly full-size passages means paragraphs pack well."),
        h("div", { class: "hint", style: { marginTop: "8px" } }, reads),
        p.samples.length ? h("button", { class: "btn sm", style: { marginTop: "10px" }, onclick: () => showSamples(p.samples, sampleSelect.value) }, `Show the ${p.samples.length} first passages`) : null);
    } catch (e) {
      previewBox.replaceChildren(h("p", { class: "hint" }, e.message));
    } finally { previewBox.style.opacity = ""; }
  }
  const refreshPreviewSoon = debounce(refreshPreview, 350);

  function chunkCards(items) {
    return items.map((c) => h("div", { class: "chunk-card" },
      h("div", { class: "head" }, h("span", { class: "badge" }, `#${c.chunk}`), c.pages ? h("span", {}, `p. ${c.pages}`) : null,
        c.section ? h("span", {}, c.section) : null, h("span", { style: { marginLeft: "auto" } }, `${c.chars} chars`)),
      (c.images || []).length ? h("div", { class: "src-thumbs" }, c.images.map(thumb)) : null,
      h("div", { class: "text" }, c.text)));
  }

  function showSamples(samples, name) {
    dialog({ title: `How “${name}” would be split`, wide: true, content: h("div", {}, chunkCards(samples)) });
  }

  async function showChunks(name) {
    const list = h("div", {}, h("div", { class: "stage" }, h("span", { class: "spinner" }), "Loading"));
    const search = h("input", { type: "search", placeholder: "Filter passages…" });
    dialog({ title: name, wide: true, content: h("div", { style: { display: "flex", flexDirection: "column", gap: "12px" } }, search, list) });
    try {
      const chunks = await api.get(`/api/documents/${encodeURIComponent(name)}/chunks`);
      const drawList = () => {
        const q = search.value.toLowerCase();
        const items = chunks.filter((c) => !q || c.text.toLowerCase().includes(q) || c.section.toLowerCase().includes(q));
        list.replaceChildren(h("div", { class: "hint", style: { marginBottom: "8px" } }, `${items.length} of ${chunks.length} passages`), ...chunkCards(items));
      };
      search.addEventListener("input", debounce(drawList, 150));
      drawList();
    } catch (e) { list.replaceChildren(h("p", { class: "muted" }, e.message)); }
  }

  // ---- actions
  async function upload(files) {
    files = files.filter((f) => /\.(pdf|docx|pptx|xlsx|txt|md|png|jpe?g|webp)$/i.test(f.name));
    if (!files.length) { toast("Use PDF, Word, PowerPoint, Excel, Markdown, text or picture files", { error: true }); return; }
    busy = `Adding ${files.length} document${files.length === 1 ? "" : "s"}: reading text and pictures, splitting into passages and embedding… (pictures take a second or two each)`;
    draw();
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    try {
      const r = await api.upload("/api/documents", form);
      r.added.forEach((a) => toast(`Added ${a.document}: ${a.chunks} passages`
        + (a.pictures ? `, ${a.pictures} picture${a.pictures === 1 ? "" : "s"} read` : "")
        + (a.pictures_skipped ? ` (${a.pictures_skipped} skipped: the picture reader model can't read images; see Settings → Models)` : "")
        + (a.pictures_unreadable ? ` (${a.pictures_unreadable} couldn't be read)` : "")));
      r.errors.forEach((e) => toast(e, { error: true }));
    } catch (e) { toastError(e); }
    busy = null;
    await load();
    app.refreshStatus();
  }

  async function remove(d) {
    if (!(await confirmDialog(`Remove ${d.name}?`, "Its passages are removed from the index and the stored copy is deleted. Past answers that cited it stay in Evaluate.", { confirm: "Remove", danger: true }))) return;
    try { await api.del(`/api/documents/${encodeURIComponent(d.name)}`); toast(`Removed ${d.name}`); await load(); app.refreshStatus(); }
    catch (e) { toastError(e); }
  }

  async function reindex() {
    const n = lib.documents.filter((d) => d.stored).length;
    if (!(await confirmDialog("Re-index all documents?", `${n} document${n === 1 ? "" : "s"} will be split and embedded again with the current settings (chunk size ${settings.values.chunk_size}, overlap ${settings.values.chunk_overlap}, ${settings.values.embed_model}). Usually takes seconds.`, { confirm: "Re-index" }))) return;
    busy = "Re-indexing: splitting and embedding every document again…";
    draw();
    try {
      const r = await api.post("/api/documents/reindex");
      toast(`Re-indexed ${r.reindexed.length} document${r.reindexed.length === 1 ? "" : "s"} · ${r.index.chunks} passages`);
      r.errors.forEach((e) => toast(e, { error: true }));
    } catch (e) { toastError(e); }
    busy = null;
    invalidateSettings();
    await load();
    app.refreshStatus();
  }

  async function applyChunking() {
    const changes = {};
    if (draft.chunk_size !== settings.values.chunk_size) changes.chunk_size = draft.chunk_size;
    if (draft.chunk_overlap !== settings.values.chunk_overlap) changes.chunk_overlap = draft.chunk_overlap;
    if (Object.keys(changes).length) {
      try { await api.put("/api/settings", { changes }); invalidateSettings(); settings = await loadSettings(true); }
      catch (e) { toastError(e); return; }
    }
    await reindex();
  }

  // Drag and drop anywhere on the page.
  let overlay = null, depth = 0;
  const onEnter = (e) => { if (!e.dataTransfer?.types?.includes("Files")) return; e.preventDefault(); if (depth++ === 0) { overlay = h("div", { class: "drop-overlay" }, "Drop to add documents"); document.body.append(overlay); } };
  const onLeave = () => { if (--depth <= 0) { depth = 0; overlay?.remove(); overlay = null; } };
  const onOver = (e) => { if (e.dataTransfer?.types?.includes("Files")) e.preventDefault(); };
  const onDrop = (e) => { e.preventDefault(); depth = 0; overlay?.remove(); overlay = null; if (e.dataTransfer.files.length) upload([...e.dataTransfer.files]); };
  window.addEventListener("dragenter", onEnter); window.addEventListener("dragleave", onLeave);
  window.addEventListener("dragover", onOver); window.addEventListener("drop", onDrop);

  inner.append(h("div", { class: "stage" }, h("span", { class: "spinner" }), "Loading"));
  try { await load(); } catch (e) { toastError(e); }

  return () => {
    window.removeEventListener("dragenter", onEnter); window.removeEventListener("dragleave", onLeave);
    window.removeEventListener("dragover", onOver); window.removeEventListener("drop", onDrop);
    overlay?.remove();
  };
}
