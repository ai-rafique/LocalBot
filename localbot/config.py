"""Paths, fixed constants, and the tunable settings.

Tunable settings are described by SCHEMA (type, range, help text, what
changing them does) so the UI can render and explain them. Values the user
changes are saved to data/settings.json; everything else stays at its
default. Code reads settings at call time (`S.top_k`), so changes apply to
the next question without a restart.

Experiments override settings per thread (`with S.override({...})`), so a
background run can't change what the chat is using.
"""
import json
import os
import threading
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# All user data lives under one folder: documents, index, logs, settings.
DATA_DIR = os.environ.get("LOCALBOT_DATA") or os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "documents")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
DB_PATH = os.path.join(DATA_DIR, "localbot.db")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
INDEX_STATE_PATH = os.path.join(DATA_DIR, "index_state.json")
WEB_DIR = os.path.join(BASE_DIR, "web")

HOST, PORT = "127.0.0.1", int(os.environ.get("LOCALBOT_PORT", 7860))

SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".md"]
EMBED_BATCH = 32        # chunks per Ollama embed call
RRF_K = 60              # standard reciprocal-rank-fusion constant
CHARS_PER_TOKEN = 3.0   # conservative estimate used to keep prompts inside the context window

# Returned without calling the model when no passage clears the cutoff: a
# fixed refusal is more reliable than asking a small model to refuse.
NO_CONTEXT_ANSWER = "I don't know. I couldn't find a relevant passage in your documents."

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered passages from the user's "
    "documents, and any verified answer, given with each question. If they "
    "don't contain the answer, say you don't know. Never guess. Be concise. "
    "Cite what you used by its number, like [1]."
)


def embed_prefixes(model):
    """Task prefixes some embedding models were trained with. nomic-embed-text
    retrieves noticeably worse without them; other models don't use them."""
    if "nomic" in model:
        return "search_document: ", "search_query: "
    return "", ""


def _s(key, group, label, type_, default, help_, effect="", **extra):
    return {"key": key, "group": group, "label": label, "type": type_, "default": default,
            "help": help_, "effect": effect, **extra}


# `reindex`: changing it requires rebuilding the index. `advanced`: hidden
# unless the user asks for advanced settings.
SCHEMA = [
    # --- Models
    _s("llm_model", "models", "Chat model", "model", "qwen3.5:2b",
       "Writes the answers (and reranks passages unless a separate reranker is set).",
       "Bigger models read passages more accurately but are slower and need more VRAM.",
       role="chat"),
    _s("embed_model", "models", "Embedding model", "model", "nomic-embed-text",
       "Turns passages and questions into vectors for meaning-based search.",
       "Changing it requires re-indexing every document.", role="embedding", reindex=True),
    _s("rerank_model", "models", "Reranking model", "model", "",
       "Scores which passages answer the question. Empty = use the chat model, which is already in memory.",
       "A separate model needs its own VRAM; on a 4 GB GPU, keep it empty.", role="chat", advanced=True),
    # --- Retrieval
    _s("top_k", "retrieval", "Passages per answer", "int", 4,
       "How many passages are sent to the model with each question.",
       "More gives more context but more neighbouring text to confuse; fewer is more focused.",
       min=1, max=10, step=1),
    _s("hybrid", "retrieval", "Keyword + meaning search", "bool", True,
       "Combine keyword search (exact identifiers, numbers, names) with meaning-based search.",
       "Helps technical documents a lot; costs a few milliseconds."),
    _s("rerank", "retrieval", "Rerank passages", "bool", True,
       "Ask the model how well each candidate passage answers the question, and keep the best.",
       "Much better passage choice and a reliable relevance cutoff; adds about 2 seconds."),
    _s("rerank_candidates", "retrieval", "Candidates to rerank", "int", 10,
       "How many search results the reranker reads before choosing the best.",
       "More finds buried passages but each adds ~0.2 s.", min=2, max=20, step=1),
    _s("min_score_reranked", "retrieval", "Relevance cutoff", "float", 0.5,
       "Passages the reranker scores below this aren't sent. With none left, the bot says it doesn't know.",
       "Higher: more \"don't know\", fewer answers from weak passages. Lower: the reverse.",
       min=0.0, max=0.95, step=0.05),
    _s("min_score_similarity", "retrieval", "Similarity cutoff (no reranking)", "float", 0.45,
       "Cutoff on vector similarity, used only when reranking is off.",
       "Similarity separates on- and off-topic questions poorly; prefer reranking.",
       min=0.0, max=0.95, step=0.05, advanced=True),
    _s("candidates", "retrieval", "Search candidates", "int", 20,
       "Results taken from each search method before merging.", "Rarely worth changing.",
       min=5, max=50, step=5, advanced=True),
    # --- Answering
    _s("system_prompt", "answering", "Instructions", "text", DEFAULT_SYSTEM_PROMPT,
       "Standing instructions sent with every question.",
       "Keep them short: small models follow a few clear rules better than many."),
    _s("temperature", "answering", "Temperature", "float", 0.0,
       "Randomness of the wording.",
       "0 = as repeatable as the GPU allows: the model always picks its most likely word, but GPU "
       "arithmetic isn't bit-exact, so some answers still change wording between runs.",
       min=0.0, max=1.5, step=0.05),
    _s("num_predict", "answering", "Maximum answer length", "int", 384,
       "Upper limit on answer tokens; also stops a runaway loop.", "",
       min=64, max=2048, step=32, unit="tokens"),
    _s("num_ctx", "answering", "Context window", "choice", 4096,
       "How much text the model can read at once (instructions + passages + history + answer).",
       "Larger fits more passages and history but uses more VRAM.", choices=[2048, 4096, 8192, 16384],
       unit="tokens"),
    _s("history_turns", "answering", "Conversation memory", "int", 4,
       "How many earlier question/answer pairs are sent along with a new question.",
       "Search still uses only the new question.", min=0, max=10, step=1, unit="exchanges"),
    _s("verify_claims", "answering", "Check each statement", "bool", True,
       "After answering, the model checks every sentence against the passage it cites and marks "
       "sentences the passages don't support.",
       "Catches wrong details inside otherwise good answers; adds about 0.2 s per sentence."),
    # Calibrated on 100 hand-graded answers (qwen3.5:2b): at 0.7 the check
    # flagged 37% of partial/bad answers and 13 of 27 containing a false
    # statement, with false alarms on 11% of good ones; 0.5 caught only 20%.
    _s("claim_min_score", "answering", "Statement support threshold", "float", 0.7,
       "Sentences the check scores below this are marked as not supported by their sources.",
       "Higher marks more sentences: more caution, more false alarms.",
       min=0.05, max=0.95, step=0.05, advanced=True),
    _s("low_confidence", "answering", "Low-confidence warning", "float", 0.62,
       "Answers whose average token probability is below this get a warning.",
       "Model-specific: recalibrate from your grades when you change the chat model.",
       min=0.0, max=1.0, step=0.01),
    _s("repeat_penalty", "answering", "Repeat penalty", "float", 1.1,
       "Discourages repeating the same phrase.", "Guards against loops at temperature 0.",
       min=1.0, max=1.5, step=0.05, advanced=True),
    _s("presence_penalty", "answering", "Presence penalty", "float", 0.0,
       "Discourages reusing any word already written.",
       "Keep at 0 for technical answers that must repeat ids and names.",
       min=0.0, max=2.0, step=0.1, advanced=True),
    _s("seed", "answering", "Seed", "int", 42, "Random seed, used when temperature > 0.", "",
       min=0, max=2**31 - 1, step=1, advanced=True),
    _s("force_gpu", "answering", "Keep the whole model on the GPU", "bool", True,
       "Load every layer onto the GPU instead of letting Ollama estimate.",
       "Ollama often puts part of a model on the CPU when it would fit; forcing it is ~1.5× faster. "
       "Falls back automatically if VRAM runs out.", advanced=True),
    _s("keep_alive", "answering", "Keep models loaded", "choice", "30m",
       "How long Ollama keeps models in memory after the last request.",
       "Longer avoids a slow first answer after a pause.", choices=["5m", "30m", "2h", "-1"], advanced=True),
    # --- Chunking
    _s("chunk_size", "chunking", "Chunk size", "int", 800,
       "Target size of a passage. Paragraphs are packed up to this; a new section starts a new chunk.",
       "Smaller: more precise matches, less context each. Larger: more context, blurrier matches.",
       min=200, max=2000, step=50, unit="characters", reindex=True),
    _s("chunk_overlap", "chunking", "Overlap", "int", 120,
       "Text carried from one chunk into the next within a section.",
       "Keeps sentences that straddle a boundary findable.", min=0, max=400, step=10,
       unit="characters", reindex=True),
    # --- Learning
    _s("use_memory", "learning", "Reuse graded answers", "bool", True,
       "Answers you grade good (or correct) are shown to the model when a nearly identical question comes up.",
       "Only affects questions very close to one you graded."),
    _s("memory_k", "learning", "Graded answers per question", "int", 2,
       "At most this many graded answers are added to a prompt.", "",
       min=0, max=5, step=1, advanced=True),
    _s("memory_min_score", "learning", "Similarity to reuse", "float", 0.85,
       "How close a new question must be to a graded one.", "0.85 ≈ a paraphrase of the same question.",
       min=0.5, max=1.0, step=0.01, advanced=True),
]
SCHEMA_BY_KEY = {s["key"]: s for s in SCHEMA}
GROUPS = [
    ("models", "Models"), ("retrieval", "Retrieval"), ("answering", "Answering"),
    ("chunking", "Chunking"), ("learning", "Learning"),
]

# One-click starting points. Accurate = the defaults.
PRESETS = {
    "accurate": {"label": "Accurate", "summary": "Reranking on, 4 passages. Best answers, ~2 s slower.",
                 "values": {"rerank": True, "rerank_candidates": 10, "top_k": 4, "hybrid": True}},
    "balanced": {"label": "Balanced", "summary": "Reranks fewer candidates, 3 passages.",
                 "values": {"rerank": True, "rerank_candidates": 6, "top_k": 3, "hybrid": True}},
    "fast": {"label": "Fast", "summary": "No reranking. Quickest, but off-topic questions are caught less reliably.",
             "values": {"rerank": False, "top_k": 3, "hybrid": True}},
}


def _check_combination(values):
    if values["chunk_overlap"] >= values["chunk_size"]:
        raise ValueError("Overlap must be smaller than the chunk size")


def validate(key, value):
    """Coerce and check one setting value; raises ValueError with a readable message."""
    spec = SCHEMA_BY_KEY.get(key)
    if spec is None:
        raise ValueError(f"Unknown setting: {key}")
    t = spec["type"]
    try:
        if t == "int":
            value = int(value)
        elif t == "float":
            value = float(value)
        elif t == "bool":
            value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        elif t == "choice":
            choices = spec["choices"]
            value = type(choices[0])(value)
            if value not in choices:
                raise ValueError
        else:
            value = str(value)
    except (TypeError, ValueError):
        raise ValueError(f"{spec['label']}: invalid value {value!r}")
    if t in ("int", "float") and not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{spec['label']} must be between {spec['min']} and {spec['max']}")
    if t == "model" and key != "rerank_model" and not value.strip():
        raise ValueError(f"{spec['label']} can't be empty")
    return value


_GLOBAL = {}


class Settings:
    """Effective settings: thread-local overrides, then saved values, then defaults."""

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        self._local = threading.local()
        _GLOBAL.update({s["key"]: s["default"] for s in SCHEMA})
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    saved = json.load(f)
                staged = dict(_GLOBAL)
                staged.update({k: validate(k, v) for k, v in saved.items() if k in SCHEMA_BY_KEY})
                _check_combination(staged)
                _GLOBAL.update(staged)
            except (ValueError, OSError):
                pass  # a damaged settings file falls back to defaults

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        for layer in reversed(getattr(self._local, "stack", [])):
            if key in layer:
                return layer[key]
        if key in _GLOBAL:
            return _GLOBAL[key]
        raise AttributeError(key)

    def values(self):
        """Saved values (ignores any thread-local override)."""
        return dict(_GLOBAL)

    def effective(self):
        """Values as seen by the current thread."""
        return {k: getattr(self, k) for k in _GLOBAL}

    def update(self, changes):
        """Validate and save; returns the keys whose change needs a re-index."""
        with self._lock:
            clean = {k: validate(k, v) for k, v in changes.items()}
            _check_combination({**_GLOBAL, **clean})
            _GLOBAL.update(clean)
            self._save()
            return [k for k in clean if SCHEMA_BY_KEY[k].get("reindex")]

    def reset(self, keys=None):
        with self._lock:
            for s in SCHEMA:
                if keys is None or s["key"] in keys:
                    _GLOBAL[s["key"]] = s["default"]
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        changed = {s["key"]: _GLOBAL[s["key"]] for s in SCHEMA if _GLOBAL[s["key"]] != s["default"]}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(changed, f, indent=2)

    @contextmanager
    def override(self, changes):
        """Apply settings for the current thread only (experiments)."""
        layer = {k: validate(k, v) for k, v in (changes or {}).items()}
        _check_combination({**self.effective(), **layer})
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        stack.append(layer)
        try:
            yield
        finally:
            stack.pop()

    # Derived values
    def cutoff(self, rerank=None):
        rerank = self.rerank if rerank is None else rerank
        return self.min_score_reranked if rerank else self.min_score_similarity

    def rerank_model_name(self):
        return self.rerank_model or self.llm_model


S = Settings(SETTINGS_PATH)
