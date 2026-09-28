"""The document library and its search indexes.

Originals are kept in DOCS_DIR so everything can be re-indexed when
chunking or the embedding model changes. Chunks and their vectors live in
a Chroma collection; a BM25 keyword index is built from it in memory.
"""
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import threading
import time
import uuid
import zipfile
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
# An inverted index (term -> passages containing it), so a query only
# touches passages with its words: tens of thousands of passages stay fast.
_keyword_cache = {}
# Library counts, rebuilt only after the library changes: the UI asks often,
# and counting every passage's metadata gets slow for large archives.
_library_cache = {}


def invalidate_keyword_index(col=None):
    _keyword_cache.pop((col or collection()).name, None)
    _library_cache.clear()


def _keyword_index(col):
    idx = _keyword_cache.get(col.name)
    if idx is None:
        data = col.get(include=["documents", "metadatas"])
        postings, lengths, sources = {}, {}, {}
        for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            tf = Counter(keyword_tokens(documents.chunk_header(meta) + "\n" + text))
            lengths[cid] = sum(tf.values()) or 1
            sources[cid] = meta.get("source")
            for term, f in tf.items():
                postings.setdefault(term, []).append((cid, f))
        avgdl = statistics.mean(lengths.values()) if lengths else 1
        idx = _keyword_cache[col.name] = {"postings": postings, "lengths": lengths, "sources": sources, "avgdl": avgdl}
    return idx


# Words that say what kind of question it is, or talk about the chat itself,
# not what it's about.
QUESTION_WORDS = set(
    "about above according again also answer answers any based between both compare compared comparing "
    "comparison contrast describe detail details difference differences different differ document "
    "documents does doing each earlier explain example examples finding give going into its just know "
    "like list mean means mentioned more most need now okay other passage passages please previous said "
    "same say should show similar tell terms than them then they think those use used using versus want "
    "way ways what's work works would".split()
)


def distinctive_terms(text, max_share=0.03):
    """The words in text that pick out a subject: in the library, but in at
    most max_share of its passages ("dijkstra", "bfs"), unlike words common
    to the whole library ("algorithm", "graph"), question words ("compare")
    and words the library never uses (typos, chat: "going", "think")."""
    idx = _keyword_index(collection())
    total = len(idx["lengths"]) or 1
    return {t for t in set(keyword_tokens(text))
            if len(t) > 2 and not t.isdigit() and t not in QUESTION_WORDS
            and 0 < len(idx["postings"].get(t, ())) <= max_share * total}


def keyword_search(query, n, sources=None, k1=1.5, b=0.75):
    """Top-n (score, chunk id) by BM25 over the active collection,
    optionally only within some documents."""
    idx = _keyword_index(collection())
    total = len(idx["lengths"])
    scores = {}
    for term in set(keyword_tokens(query)):
        plist = idx["postings"].get(term)
        if not plist:
            continue
        idf = math.log(1 + (total - len(plist) + 0.5) / (len(plist) + 0.5))
        for cid, f in plist:
            if sources and idx["sources"][cid] not in sources:
                continue
            dl = idx["lengths"][cid]
            scores[cid] = scores.get(cid, 0.0) + idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / idx["avgdl"]))
    return sorted(((s, cid) for cid, s in scores.items()), reverse=True)[:n]


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


# Parsed paragraphs, remembered per file (by modification time) so that
# chunk-size previews and re-indexing don't parse large PDFs and archives again.
_para_cache = {}
_PARA_CACHE_FILES = 24


def _remember(key, sig, value):
    if len(_para_cache) >= _PARA_CACHE_FILES:
        _para_cache.pop(next(iter(_para_cache)))
    _para_cache[key] = (sig, value)
    return value


def _signature(path):
    st = os.stat(path)
    return (st.st_mtime, st.st_size, S.read_images)


def _paragraphs(path):
    key, sig = os.path.abspath(path), _signature(path)
    hit = _para_cache.get(key)
    return hit[1] if hit and hit[0] == sig else _remember(key, sig, documents.load_paragraphs(path))


def _zip_readable():
    types = {".pdf", ".docx", ".pptx", ".xlsx"} | set(documents.TEXT_TYPES + documents.HTML_TYPES + documents.CODE_TYPES)
    return types | (set(documents.IMAGE_TYPES) if S.read_images else set())


def _zip_units(path, progress=None, cached_only=False):
    """The readable files in an archive as (path inside, paragraphs), plus
    counts of what was skipped and why. None if cached_only and not parsed yet."""
    key, sig = os.path.abspath(path) + "#zip", _signature(path)
    hit = _para_cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    if cached_only:
        return None
    units, skipped, seen = [], Counter(), set()
    with zipfile.ZipFile(path) as zf:
        plan = documents.zip_plan(zf.infolist(), _zip_readable())
        skipped.update(r for _, r in plan if r)
        todo = [info for info, r in plan if r is None]
        for n, info in enumerate(todo, 1):
            if progress:
                progress(n, len(todo), f"Reading {info.filename}")
            data = zf.read(info)
            digest = hashlib.sha1(data).hexdigest()
            if digest in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(digest)
            ext = os.path.splitext(info.filename)[1].lower()
            if ext in documents.CODE_TYPES and documents.looks_minified(documents._decode(data)):
                skipped["minified or generated code"] += 1
                continue
            try:
                paras = documents.paragraphs_from_bytes(info.filename, data)
            except Exception:
                skipped["could not be read"] += 1
                continue
            if not any(p.get("text") or "image" in p for p in paras):
                skipped["no readable text"] += 1
                continue
            units.append((info.filename, paras))
    return _remember(key, sig, (units, dict(skipped)))


def _chunk_file(path, chunk_size, overlap, transcribe=True, progress=None, cached_only=False):
    """Chunks for one library file (every readable file, for an archive)."""
    info = {"pictures": 0, "pictures_skipped": 0, "pictures_unreadable": 0}
    if path.lower().endswith(".zip"):
        found = _zip_units(path, progress, cached_only)
        if found is None:
            return None, info
        units, skipped = found
        info.update(files=len(units), skipped=skipped)
    else:
        units = [("", _paragraphs(path))]
    chunks = []
    for member, paras in units:
        paras, read, pskip, failed = _read_pictures(paras, transcribe)
        info["pictures"] += read
        info["pictures_skipped"] += pskip
        info["pictures_unreadable"] += failed
        for c in documents.chunk_paragraphs(paras, int(chunk_size), int(overlap)):
            c["path"] = member
            chunks.append(c)
    return chunks, info


EMBED_GROUP = 256  # passages embedded and stored per step, for progress and memory


def _add_chunks(col, fname, chunks, chunk_size, overlap, progress=None):
    """Store a document's chunks, replacing its old ones only once all new
    ones are in: if embedding fails halfway, the old passages remain."""
    old = col.get(where={"source": fname}, include=[])["ids"]
    new_ids = []
    try:
        for start in range(0, len(chunks), EMBED_GROUP):
            group = chunks[start:start + EMBED_GROUP]
            metas = [{"source": fname, "path": c.get("path", ""), "chunk": start + i, "chars": len(c["text"]),
                      "section": c["section"], "page_start": c["page_start"], "page_end": c["page_end"],
                      "images": ",".join(c.get("images", [])), "chunk_size": chunk_size, "overlap": overlap}
                     for i, c in enumerate(group)]
            ids = [str(uuid.uuid4()) for _ in group]
            col.add(ids=ids, embeddings=embed([documents.chunk_header(m) + "\n" + c["text"] for m, c in zip(metas, group)]),
                    documents=[c["text"] for c in group], metadatas=metas)
            new_ids += ids
            if progress:
                progress(min(start + EMBED_GROUP, len(chunks)), len(chunks), "Embedding passages")
    except BaseException:
        for i in range(0, len(new_ids), 5000):
            col.delete(ids=new_ids[i:i + 5000])
        raise
    for i in range(0, len(old), 5000):
        col.delete(ids=old[i:i + 5000])


def ingest(path, name=None, progress=None, move=False):
    """Read, chunk, embed and store one file or archive, keeping a copy in the
    library. progress(done, total, step) reports along the way. Returns a
    report dict; raises ValueError for unusable files."""
    name = os.path.basename(name or path)
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_TYPES:
        raise ValueError(f"{name}: unsupported type (use {', '.join(SUPPORTED_TYPES)})")
    t0 = time.perf_counter()
    chunks, info = _chunk_file(path, S.chunk_size, S.chunk_overlap, progress=progress)
    if not chunks:
        if ext == ".zip":
            why = ", ".join(f"{n} {r}" for r, n in sorted(info.get("skipped", {}).items(), key=lambda x: -x[1]))
            raise ValueError(f"{name}: no readable documents in the archive" + (f" (skipped: {why})" if why else ""))
        raise ValueError(f"{name}: no extractable text" + (" or legible pictures" if S.read_images else ""))
    t1 = time.perf_counter()
    _add_chunks(_collection, name, chunks, S.chunk_size, S.chunk_overlap, progress)
    invalidate_keyword_index(_collection)
    os.makedirs(DOCS_DIR, exist_ok=True)
    stored = os.path.join(DOCS_DIR, name)
    if os.path.abspath(path) != os.path.abspath(stored):
        (shutil.move if move else shutil.copy2)(path, stored)
        # Keep what was just parsed under the stored path (both keep the
        # modification time), so previews of a large archive work right away.
        for suffix in ("", "#zip"):
            hit = _para_cache.pop(os.path.abspath(path) + suffix, None)
            if hit:
                _remember(os.path.abspath(stored) + suffix, *hit)
    _save_state()
    return {"document": name, "chunks": len(chunks), "embed_s": round(time.perf_counter() - t1, 2),
            "read_s": round(t1 - t0, 2), **info}


def warm_up():
    """Build the keyword index now (after adding documents) rather than on
    the next question, which would otherwise wait for it."""
    _keyword_index(_collection)


def reindex(progress=None):
    """Rebuild every document with the current chunking and embedding settings."""
    files = stored_documents()
    reports, errors = [], []
    for i, name in enumerate(files):
        step = (lambda i, name: (lambda d, t, s: progress(d, t, f"{name} ({i + 1}/{len(files)}): {s}")))(i, name) if progress else None
        if progress:
            progress(0, 1, f"{name} ({i + 1}/{len(files)})")
        try:
            reports.append(ingest(os.path.join(DOCS_DIR, name), progress=step))
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


def pictures_of(name):
    """The pictures read from a document, in order, with their current text."""
    data = _collection.get(where={"source": name}, include=["metadatas"])
    first_page = {}
    for m in sorted(data["metadatas"] or [], key=lambda m: m.get("chunk", 0)):
        for i in (m.get("images") or "").split(","):
            if i and i not in first_page:
                first_page[i] = m.get("page_start") or 0
    return [{"id": i, "page": p, "text": vision.cached(i) or "", "model_text": vision.model_reading(i) or "",
             "corrected": vision.is_corrected(i)} for i, p in first_page.items()]


def documents_with_picture(image_id):
    data = _collection.get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"] or [] if image_id in (m.get("images") or "").split(",")})


def refresh(names):
    """Rebuild these documents' passages (after a picture's text changed).
    Other pictures come from the cache, so this takes seconds."""
    return [ingest(os.path.join(DOCS_DIR, n)) for n in names if os.path.isfile(os.path.join(DOCS_DIR, n))]


def _forget_unused_pictures():
    if not os.path.isdir(vision.IMAGES_DIR):
        return
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
    _para_cache.clear()
    _forget_unused_pictures()


def stored_documents():
    if not os.path.isdir(DOCS_DIR):
        return []
    return sorted(f for f in os.listdir(DOCS_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_TYPES)


def library_counts():
    """{document: {"chunks", "chars", "sections", "pages", "pictures", "files"}}."""
    if "counts" in _library_cache:
        return _library_cache["counts"]
    data = _collection.get(include=["metadatas"])
    out = {}
    for m in data["metadatas"] or []:
        d = out.setdefault(m["source"], {"chunks": 0, "chars": 0, "sections": set(), "pages": 0,
                                         "pictures": set(), "files": set()})
        d["pictures"].update(i for i in (m.get("images") or "").split(",") if i)
        d["chunks"] += 1
        d["chars"] += m.get("chars", 0)
        if m.get("section"):
            d["sections"].add(m["section"])
        if m.get("path"):
            d["files"].add(m["path"])
        d["pages"] = max(d["pages"], m.get("page_end") or 0)
    for d in out.values():
        for k in ("sections", "pictures", "files"):
            d[k] = len(d[k])
    _library_cache["counts"] = out
    return out


def library():
    counts = library_counts()
    rows = []
    for name in sorted(set(counts) | set(stored_documents())):
        path = os.path.join(DOCS_DIR, name)
        c = counts.get(name, {"chunks": 0, "chars": 0, "sections": 0, "pages": 0, "pictures": 0, "files": 0})
        rows.append({
            "name": name, "type": os.path.splitext(name)[1].lstrip(".").lower(),
            "size_bytes": os.path.getsize(path) if os.path.isfile(path) else None,
            "added": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))) if os.path.isfile(path) else None,
            "stored": os.path.isfile(path), **c,
        })
    return rows


def chunks_of(name, limit=500):
    data = _collection.get(where={"source": name}, include=["documents", "metadatas"], limit=limit)
    items = sorted(zip(data["documents"], data["metadatas"]), key=lambda x: x[1].get("chunk", 0))
    return [{"chunk": m.get("chunk"), "section": m.get("section", ""), "path": m.get("path", ""),
             "pages": documents.page_range(m), "chars": len(t), "text": t,
             "images": [i for i in (m.get("images") or "").split(",") if i]} for t, m in items]


def preview(chunk_size, overlap, name=None, samples=40):
    """What chunking with these settings would produce, without embedding.
    For one document (with sample chunks) or the whole library. Archives are
    only included once parsed (after adding or re-indexing them), since
    parsing thousands of pages is too slow for a live preview."""
    names = [name] if name else stored_documents()
    sizes, per_doc, sample, not_previewed = [], {}, [], []
    for n in names:
        # Pictures aren't read for a preview; remembered transcripts are used.
        chunks, _ = _chunk_file(os.path.join(DOCS_DIR, n), chunk_size, overlap, transcribe=False, cached_only=True)
        if chunks is None:
            not_previewed.append(n)
            continue
        per_doc[n] = len(chunks)
        sizes += [len(c["text"]) for c in chunks]
        if name:
            sample = [{"chunk": i, "section": c["section"], "path": c.get("path", ""), "pages": documents.page_range(c),
                       "chars": len(c["text"]), "text": c["text"], "images": c.get("images", [])}
                      for i, c in enumerate(chunks[:samples])]
    bins = [0] * 10
    for s in sizes:
        bins[min(9, int(10 * s / (chunk_size + 1)))] += 1
    return {
        "chunks": len(sizes), "per_document": per_doc, "not_previewed": not_previewed,
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
