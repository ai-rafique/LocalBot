// Controls for settings, and a live explanation of what each value will do,
// computed from measurements (chunk sizes, reranker speed, recent scores).
import { api } from "./api.js";
import { h, switchEl } from "./ui.js";

let cache = null;
export async function loadSettings(force = false) {
  if (!cache || force) cache = await api.get("/api/settings");
  return cache;
}
export function invalidateSettings() { cache = null; }

let modelsCache = null;
export async function loadModels(force = false) {
  if (!modelsCache || force) modelsCache = await api.get("/api/models").catch(() => ({ installed: [], loaded: [] }));
  return modelsCache;
}

const share = (list, cutoff) => (list.length ? list.filter((x) => x >= cutoff).length / list.length : null);
const fmtS = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

// A one-line consequence of `value` for setting `key`, given other values `v`.
export function effectText(key, value, v, ins, models = []) {
  const avg = ins.avg_chunk_chars || v.chunk_size * 0.7;
  const per = ins.rerank_ms_per_candidate || 200;
  const tps = ins.tok_per_s;
  switch (key) {
    case "llm_model": case "rerank_model": {
      if (key === "rerank_model" && !value) return "Uses the chat model — no extra memory.";
      const m = models.find((x) => x.name === value || x.name === `${value}:latest`);
      if (!m) return `Not installed. Install it under Models below, or run: ollama pull ${value}`;
      const gpu = ins.gpu ? ` Your GPU has ${(ins.gpu.total_mb / 1024).toFixed(1)} GB.` : "";
      const fit = m.fits_gpu === false ? " May not fit fully on the GPU, so it will run slower." : m.fits_gpu ? " Fits on the GPU." : "";
      return `${m.parameters || ""} ${m.quantization || ""} · ${m.size_gb} GB.${fit}${gpu}`.trim();
    }
    case "embed_model":
      return value !== ins.indexed_embed ? "Different from the index: re-index documents after saving." : "Matches the current index.";
    case "top_k": {
      const chars = value * avg;
      return `The model reads about ${Math.round(chars).toLocaleString()} characters (~${Math.round(chars / 3.5).toLocaleString()} tokens) of passages per question.`;
    }
    case "hybrid":
      return value ? "Exact terms (ids, commands, numbers) are found even when meaning-search misses them."
                   : "Meaning-search only: identifiers and exact values are found less reliably.";
    case "rerank":
      return value ? `Adds ≈ ${fmtS(v.rerank_candidates * per)} per question (${v.rerank_candidates} candidates × ${fmtS(per)}). Cutoff uses relevance scores, which separate off-topic questions well.`
                   : "Faster. The cutoff falls back to similarity, which lets more off-topic questions through.";
    case "rerank_candidates":
      return `Reranking takes ≈ ${fmtS(value * per)} per question.`;
    case "min_score_reranked": {
      const s = share(ins.rerank_scores || [], value);
      return s === null ? "Higher = more \"don't know\"; lower = more answers from weak passages."
        : `Of ${ins.rerank_scores.length} recently retrieved passages, ${Math.round(100 * s)}% would be sent to the model.`;
    }
    case "min_score_similarity": {
      const s = share(ins.dense_scores || [], value);
      return s === null ? "" : `Of recent passages, ${Math.round(100 * s)}% would pass (only used with reranking off).`;
    }
    case "temperature":
      return value === 0 ? "Most repeatable: the model always picks its most likely word. Some answers still change wording between runs, because GPU arithmetic isn't bit-exact."
        : "Wording varies between runs, so a grade may not hold next time.";
    case "num_predict":
      return `Answers stop after ≈ ${Math.round(value * 0.75)} words` + (tps ? `, taking up to ≈ ${(value / tps).toFixed(0)} s at ${tps} tokens/s.` : ".");
    case "num_ctx": {
      const room = (value - v.num_predict) * 3;
      const fits = Math.floor((room - 800) / (avg + 60));
      const warn = fits < v.top_k ? ` Too small for ${v.top_k} passages: weaker ones will be dropped.` : "";
      return `Room for about ${Math.max(0, fits)} passages plus instructions and history.${warn}`;
    }
    case "history_turns":
      return value === 0 ? "Each question is answered on its own." : `The last ${value} question/answer pair${value > 1 ? "s are" : " is"} sent with each new question.`;
    case "low_confidence": {
      const c = ins.confidence_by_grade || {};
      const parts = ["good", "partial", "bad"].filter((g) => c[g]?.mean != null).map((g) => `${g} ${c[g].mean.toFixed(2)}`);
      return parts.length ? `Your graded answers average: ${parts.join(" · ")}. Answers below ${value.toFixed(2)} get a warning.` : "Grade some answers to calibrate this.";
    }
    case "chunk_size": case "chunk_overlap":
      return ins.preview ? `${ins.preview.chunks} chunks, average ${ins.preview.avg_chars} characters (currently ${ins.total_chunks}).` : "Re-index after saving to apply.";
    case "use_memory":
      return value ? "Close repeats of questions you graded get your verified answer as context." : "Graded answers are never reused.";
    case "force_gpu":
      return value ? "Faster when the model fits. Falls back to CPU+GPU automatically if memory runs out." : "Ollama decides; on small GPUs it often moves part of the model to the CPU.";
    default:
      return "";
  }
}

// An input for one setting. onChange(value) fires on every edit.
export function control(spec, value, onChange, { models = [] } = {}) {
  const t = spec.type;
  if (t === "bool") return switchEl(value, onChange, value ? "On" : "Off");
  if (t === "text") {
    return h("textarea", { rows: 5, oninput: (e) => onChange(e.target.value) }, value);
  }
  if (t === "choice") {
    return h("select", { onchange: (e) => onChange(spec.choices.find((c) => String(c) === e.target.value)) },
      spec.choices.map((c) => h("option", { value: c, selected: c === value }, choiceLabel(spec, c))));
  }
  if (t === "model") {
    const role = spec.role || "chat";
    const opts = models.filter((m) => m.role === role).map((m) => m.name);
    const current = value && !opts.includes(value) && !opts.includes(`${value}:latest`) ? [value] : [];
    const sel = h("select", { onchange: (e) => onChange(e.target.value) },
      spec.key === "rerank_model" ? h("option", { value: "", selected: !value }, "Same as chat model") : null,
      [...current, ...opts].map((n) => h("option", { value: n, selected: n === value || n === `${value}:latest` },
        n + (current.includes(n) ? " (not installed)" : ""))));
    return sel;
  }
  // int / float: slider + number box kept in sync.
  const range = h("input", { type: "range", min: spec.min, max: spec.max, step: spec.step, value });
  const box = h("input", { type: "number", min: spec.min, max: spec.max, step: spec.step, value });
  const fill = () => range.style.setProperty("--fill", `${(100 * (range.value - spec.min)) / (spec.max - spec.min)}%`);
  const parse = (x) => (t === "int" ? parseInt(x, 10) : parseFloat(x));
  range.addEventListener("input", () => { box.value = range.value; fill(); onChange(parse(range.value)); });
  box.addEventListener("change", () => {
    let x = parse(box.value);
    if (Number.isNaN(x)) x = value;
    x = Math.min(spec.max, Math.max(spec.min, x));
    box.value = x; range.value = x; fill(); onChange(x);
  });
  fill();
  return h("div", { class: "control-row" }, range, box, spec.unit ? h("span", { class: "hint" }, spec.unit) : null);
}

function choiceLabel(spec, c) {
  if (spec.key === "keep_alive") return { "5m": "5 minutes", "30m": "30 minutes", "2h": "2 hours", "-1": "Always" }[c] || c;
  if (spec.key === "num_ctx") return `${c.toLocaleString()} tokens`;
  return String(c);
}

export function formatValue(spec, v) {
  if (spec.type === "bool") return v ? "on" : "off";
  if (spec.type === "choice") return choiceLabel(spec, v);
  if (spec.type === "text") return `${String(v).slice(0, 24)}…`;
  if (spec.type === "model") return v || "same as chat";
  return String(v);
}
