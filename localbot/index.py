"""The document library and its search indexes.

Originals are kept in DOCS_DIR so everything can be re-indexed when
chunking or the embedding model changes. Chunks and their vectors live in
a Chroma collection; a BM25 keyword index is built from it in memory.
"""
import json
import math
import os
import re
import shutil
import statistics
import threading
import time
import uuid
from collections import Counter

import chromadb
import ollama
from chromadb.config import Settings as ChromaSettings

from . import documents, vision
from .config import CHROMA_DIR, DOCS_DIR, EMBED_BATCH, INDEX_STATE_PATH, SUPPORTED_TYPES, S, embed_prefixes

# anonymized_telemetry=False stops Chroma from sending usage data to PostHog.
_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False))
_ephemeral = None
_local = threading.local()


def _open(name, client=None):
    # Cosine space so `1 - distance` is a real similarity score. Chroma's
    # default is squared L2, which doesn't map onto a 0..1 relevance scale.
    return (client or _client).get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})


_collection = _open("docs")
# Questions the user graded, embedded, pointing at their verified answers.
# Separate from the documents so clearing the library doesn't erase them.
memory = _open("feedback_memory")


def collection():
    """The active document collection: a thread's temporary index (see
    `use_temporary`) or the library."""
    return getattr(_local, "collection", None) or _collection


# --- Embedding ---------------------------------------------------------------
def embed(texts, kind="document"):
    """Embed strings via Ollama, in batches. kind: "document" or "query"."""
    doc_prefix, query_prefix = embed_prefixes(S.embed_model)
    prefix = query_prefix if kind == "query" else doc_prefix
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [prefix + t for t in texts[i:i + EMBED_BATCH]]
        vectors.extend(ollama.embed(model=S.embed_model, input=batch, keep_alive=S.keep_alive)["embeddings"])
    return vectors


# --- Keyword index (BM25) ----------------------------------------------------
STOPWORDS = set(
    "a an and are as at be by can do does did for from has have how i in is it its of on or that "
    "the their there these this to was were what when where which who why will with you your".split()
)
# Identifiers stay whole ("CAN_ID_MASK", "Foo::bar", "0x1A3", "src/main.cpp")
# and are also indexed by their parts, so "baud rate" finds "baudRate".
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_.:/-]+[A-Za-z0-9]+)*")
SUBTOKEN_RE = re.compile(r"[_.:/-]+|(?<=[a-z])(?=[A-Z])")


def keyword_tokens(text):
    out = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if low not in STOPWORDS:
            out.append(low)
        parts = [p.lower() for p in SUBTOKEN_RE.split(tok) if p]
        if len(parts) > 1:
            out += [p for p in parts if p not in STOPWORDS]
    return out


# Built lazily per collection and dropped whenever that collection changes.
_keyword_cache = {}


def invalidate_keyword_index(col=None):
    _keyword_cache.pop((col or collection()).name, None)


def _keyword_index(col):
    idx = _keyword_cache.get(col.name)
    if idx is None:
        data = col.get(include=["documents", "metadatas"])
        docs = []
        for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            tf = Counter(keyword_tokens(documents.chunk_header(meta) + "\n" + text))
            docs.append((cid, tf, sum(tf.values()) or 1))
        df = Counter(t for _, tf, _ in docs for t in tf)
        avgdl = statistics.mean(n for _, _, n in docs) if docs else 1
        idx = _keyword_cache[col.name] = {"docs": docs, "df": df, "avgdl": avgdl}
    return idx


def keyword_search(query, n, k1=1.5, b=0.75):
    """Top-n (score, chunk id) by BM25 over the active collection."""
    idx = _keyword_index(collection())
    terms = set(keyword_tokens(query))
    total = len(idx["docs"])
    idf = {t: math.log(1 + (total - idx["df"][t] + 0.5) / (idx["df"][t] + 0.5)) for t in terms if idx["df"][t]}
    scored = []
    for cid, tf, length in idx["docs"]:
        s = sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * length / idx["avgdl"]))
                for t in idf if tf[t])
        if s > 0:
            scored.append((s, cid))
    return sorted(scored, reverse=True)[:n]


# --- Library -----------------------------------------------------------------
def _read_pictures(paras, transcribe=True):
    """Replace picture placeholders with their transcripts, marked as such
    so the model (and you) know the text was read from a picture.
    transcribe=False (previews) uses remembered transcripts only.
    Returns (paragraphs, pictures read, pictures skipped, pictures unreadable)."""
    out, read, skipped, failed = [], 0, 0, 0
    readable = S.read_images and (not transcribe or vision.can_read())
    for p in paras:
        if "image" not in p:
            out.append(p)
            continue
        image_id = vision.store(p["image"]) if S.read_images else None
        if not image_id:
            continue
        if not readable:
            skipped += 1
            continue
        if transcribe:
            try:
                text = vision.transcribe(image_id)
            except Exception as e:
                # One unreadable picture must not fail the whole document.
                if "connect" in str(e).lower():
                    raise
                failed += 1
                continue
        else:
            text = vision.cached(image_id) or "(picture: read when added)"
        if len(text.strip()) < 15:
            failed += not text.strip()
            continue  # nothing legible or describable
        where = f", p. {p['page']}" if p.get("page") else ""
        out.append({"text": f"[Picture{where}]\n{text}", "page": p["page"], "level": 0, "image_id": image_id})
        read += 1
    return out, read, skipped, failed


def _chunk_file(path, chunk_size, overlap, transcribe=True):
    paras, read, skipped, failed = _read_pictures(documents.load_paragraphs(path), transcribe)
    chunks = documents.chunk_paragraphs(paras, int(chunk_size), int(overlap))
    return chunks, {"pictures": read, "pictures_skipped": skipped, "pictures_unreadable": failed}


def _add_chunks(col, fname, chunks, chunk_size, overlap):
    metas = [{"source": fname, "chunk": i, "chars": len(c["text"]), "section": c["section"],
              "page_start": c["page_start"], "page_end": c["page_end"],
              "images": ",".join(c.get("images", [])),
              "chunk_size": chunk_size, "overlap": overlap} for i, c in enumerate(chunks)]
    vectors = embed([documents.chunk_header(m) + "\n" + c["text"] for m, c in zip(metas, chunks)])
    col.delete(where={"source": fname})  # re-adding a file replaces it
    col.add(ids=[str(uuid.uuid4()) for _ in chunks], embeddings=vectors,
            documents=[c["text"] for c in chunks], metadatas=metas)


def ingest(path, name=None):
    """Chunk, embed and store one file (copying it into the library).
    Returns a report dict; raises ValueError for unusable files."""
    name = os.path.basename(name or path)
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"{name}: unsupported type (use {', '.join(SUPPORTED_TYPES)})")
    t0 = time.perf_counter()
    chunks, info = _chunk_file(path, S.chunk_size, S.chunk_overlap)
    if not chunks:
        raise ValueError(f"{name}: no extractable text or legible pictures")
    t1 = time.perf_counter()
    _add_chunks(_collection, name, chunks, S.chunk_size, S.chunk_overlap)
    invalidate_keyword_index(_collection)
    os.makedirs(DOCS_DIR, exist_ok=True)
    stored = os.path.join(DOCS_DIR, name)
    if os.path.abspath(path) != os.path.abspath(stored):
        shutil.copy2(path, stored)
    _save_state()
    return {"document": name, "chunks": len(chunks), "embed_s": round(time.perf_counter() - t1, 2),
            "read_s": round(t1 - t0, 2), **info}


def reindex(progress=None):
    """Rebuild every document with the current chunking and embedding settings."""
    files = stored_documents()
    reports, errors = [], []
    for i, name in enumerate(files):
        if progress:
            progress(i, len(files), name)
        try:
            reports.append(ingest(os.path.join(DOCS_DIR, name)))
        except Exception as e:  # keep going; report per file
            errors.append(f"{name}: {e}")
    # Chunks of documents whose original is gone can't be rebuilt; drop them
    # so the index doesn't mix chunkings.
    for name in set(library_counts()) - set(files):
        _collection.delete(where={"source": name})
    invalidate_keyword_index(_collection)
    _forget_unused_pictures()
    _save_state()
    return reports, errors


def _forget_unused_pictures():
    ids = set()
    for m in _collection.get(include=["metadatas"])["metadatas"] or []:
        ids.update(i for i in (m.get("images") or "").split(",") if i)
    vision.remove_unused(ids)


def remove(name):
    _collection.delete(where={"source": name})
    invalidate_keyword_index(_collection)
    stored = os.path.join(DOCS_DIR, name)
    if os.path.isfile(stored):
        os.remove(stored)
    _forget_unused_pictures()


def stored_documents():
    if not os.path.isdir(DOCS_DIR):
        return []
    return sorted(f for f in os.listdir(DOCS_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_TYPES)


def library_counts():
    """{document: {"chunks", "chars", "sections", "pages"}} from the index."""
    data = _collection.get(include=["metadatas"])
    out = {}
    for m in data["metadatas"] or []:
        d = out.setdefault(m["source"], {"chunks": 0, "chars": 0, "sections": set(), "pages": 0, "pictures": set()})
        d["pictures"].update(i for i in (m.get("images") or "").split(",") if i)
        d["chunks"] += 1
        d["chars"] += m.get("chars", 0)
        if m.get("section"):
            d["sections"].add(m["section"])
        d["pages"] = max(d["pages"], m.get("page_end") or 0)
    for d in out.values():
        d["sections"] = len(d["sections"])
        d["pictures"] = len(d["pictures"])
    return out


def library():
    counts = library_counts()
    rows = []
    for name in sorted(set(counts) | set(stored_documents())):
        path = os.path.join(DOCS_DIR, name)
        c = counts.get(name, {"chunks": 0, "chars": 0, "sections": 0, "pages": 0, "pictures": 0})
        rows.append({
            "name": name, "type": os.path.splitext(name)[1].lstrip(".").lower(),
            "size_bytes": os.path.getsize(path) if os.path.isfile(path) else None,
            "added": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))) if os.path.isfile(path) else None,
            "stored": os.path.isfile(path), **c,
        })
    return rows


def chunks_of(name, limit=500):
    data = _collection.get(where={"source": name}, include=["documents", "metadatas"])
    items = sorted(zip(data["documents"], data["metadatas"]), key=lambda x: x[1].get("chunk", 0))
    return [{"chunk": m.get("chunk"), "section": m.get("section", ""), "pages": documents.page_range(m),
             "chars": len(t), "text": t, "images": [i for i in (m.get("images") or "").split(",") if i]}
            for t, m in items[:limit]]


def preview(chunk_size, overlap, name=None, samples=40):
    """What chunking with these settings would produce, without embedding.
    For one document (with sample chunks) or the whole library."""
    names = [name] if name else stored_documents()
    sizes, per_doc, sample = [], {}, []
    for n in names:
        # Pictures aren't read for a preview; remembered transcripts are used.
        chunks, _ = _chunk_file(os.path.join(DOCS_DIR, n), chunk_size, overlap, transcribe=False)
        per_doc[n] = len(chunks)
        sizes += [len(c["text"]) for c in chunks]
        if name:
            sample = [{"chunk": i, "section": c["section"], "pages": documents.page_range(c),
                       "chars": len(c["text"]), "text": c["text"], "images": c.get("images", [])}
                      for i, c in enumerate(chunks[:samples])]
    bins = [0] * 10
    for s in sizes:
        bins[min(9, int(10 * s / (chunk_size + 1)))] += 1
    return {
        "chunks": len(sizes), "per_document": per_doc,
        "avg_chars": round(statistics.mean(sizes)) if sizes else 0,
        "min_chars": min(sizes) if sizes else 0, "max_chars": max(sizes) if sizes else 0,
        "histogram": bins, "samples": sample,
    }


# --- Index state --------------------------------------------------------------
def _save_state():
    state = {"embed_model": S.embed_model, "chunk_size": S.chunk_size, "chunk_overlap": S.chunk_overlap,
             "built": time.strftime("%Y-%m-%d %H:%M")}
    with open(INDEX_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def state():
    """How the index was built, and whether current settings differ."""
    try:
        with open(INDEX_STATE_PATH, encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    stale = [k for k in ("embed_model", "chunk_size", "chunk_overlap") if k in st and st[k] != getattr(S, k)]
    return {**st, "stale": stale, "chunks": _collection.count()}


# --- Temporary indexes (experiments) -------------------------------------------
class use_temporary:
    """Within this block, the current thread searches an in-memory index
    built from the stored originals with other chunk settings. The library
    itself is never modified."""

    def __init__(self, chunk_size, overlap):
        self.chunk_size, self.overlap = int(chunk_size), int(overlap)

    def __enter__(self):
        global _ephemeral
        if _ephemeral is None:
            _ephemeral = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
        self.col = _open(f"tmp-{uuid.uuid4().hex[:10]}", _ephemeral)
        for name in stored_documents():
            chunks, _ = _chunk_file(os.path.join(DOCS_DIR, name), self.chunk_size, self.overlap)
            if chunks:
                _add_chunks(self.col, name, chunks, self.chunk_size, self.overlap)
        _local.collection = self.col
        return self.col

    def __exit__(self, *exc):
        _local.collection = None
        _keyword_cache.pop(self.col.name, None)
        try:
            _ephemeral.delete_collection(self.col.name)
        except Exception:
            pass
