"""Finding the passages that answer a question.

Vector search (meaning) + BM25 keyword search (exact identifiers, numbers)
merged by reciprocal rank fusion, then optionally reranked by the chat
model, which reads each candidate and scores P("yes, this answers it").
"""
import math
import time

import ollama

from . import documents, index
from .config import RRF_K, S

RERANK_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)


def gen_options(force_gpu=None):
    force = S.force_gpu if force_gpu is None else force_gpu
    opts = {"temperature": S.temperature, "seed": S.seed, "repeat_penalty": S.repeat_penalty,
            "presence_penalty": S.presence_penalty, "num_ctx": S.num_ctx, "num_predict": S.num_predict}
    if force:
        # Ollama's memory estimate is conservative: on a 4 GB GPU it put 37%
        # of a 2.7 GB model on the CPU (34 tok/s); all layers on GPU: 54 tok/s.
        opts["num_gpu"] = 99
    return opts


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _hit(cid, doc, meta):
    return {
        "id": cid, "text": doc, "source": meta["source"], "chunk": meta.get("chunk", 0),
        "section": meta.get("section", ""), "page_start": meta.get("page_start", 0),
        "page_end": meta.get("page_end", 0), "chunk_size": meta.get("chunk_size"),
        "overlap": meta.get("overlap"), "used": False,
    }


def rerank_score(query, hit):
    """P(yes) that the passage answers the question, from the model's first
    token: the Qwen3-Reranker recipe run on the chat model already loaded.
    Same options as answering, so Ollama doesn't reload the model."""
    passage = f"{documents.chunk_header(hit)}\n{hit['text']}"
    r = ollama.chat(
        model=S.rerank_model_name(), think=False, logprobs=True, top_logprobs=10,
        keep_alive=S.keep_alive, options={**gen_options(), "num_predict": 1},
        messages=[
            {"role": "system", "content": RERANK_SYSTEM},
            # The question comes before the passage so Ollama can reuse the
            # cached prompt prefix across the candidates of one question.
            {"role": "user", "content": (
                "<Instruct>: Given a question about a technical document, judge whether the "
                f"passage contains information that answers it\n<Query>: {query}\n<Document>: {passage}"
            )},
        ],
    )
    yes = no = 0.0
    for alt in (r.logprobs[0].top_logprobs or []) if r.logprobs else []:
        word = alt.token.strip().lower()
        if word == "yes":
            yes += math.exp(alt.logprob)
        elif word == "no":
            no += math.exp(alt.logprob)
    return yes / (yes + no) if yes + no else 0.0


def retrieve(query, k=None, hybrid=None, rerank=None):
    """Return (hits, timings, query_vector). Timings are in milliseconds.

    hits are the k best passages, best first, each with its scores: dense
    (cosine), keyword (BM25), rerank (P(yes)) and `score` — the one the
    relevance cutoff applies to (rerank if used, else cosine). Callers apply
    the cutoff, so the scores of rejected passages still get logged.
    """
    k = S.top_k if k is None else int(k)
    hybrid = S.hybrid if hybrid is None else hybrid
    rerank = S.rerank if rerank is None else rerank
    col = index.collection()
    count = col.count()
    timings = {"embed_ms": 0.0, "search_ms": 0.0, "rerank_ms": 0.0}
    if count == 0:
        return [], timings, None
    t0 = time.perf_counter()
    q_vec = index.embed([query], kind="query")[0]
    t1 = time.perf_counter()
    n = min(S.candidates, count)
    res = col.query(query_embeddings=[q_vec], n_results=n)
    cand = {}
    for rank, (cid, doc, meta, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                       res["metadatas"][0], res["distances"][0]), 1):
        cand[cid] = {**_hit(cid, doc, meta), "dense": round(1 - dist, 4), "dense_rank": rank}
    if hybrid:
        kw = index.keyword_search(query, n)
        new = [cid for _, cid in kw if cid not in cand]
        if new:
            got = col.get(ids=new, include=["documents", "metadatas", "embeddings"])
            for cid, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
                cand[cid] = {**_hit(cid, doc, meta), "dense": round(_cosine(q_vec, emb), 4)}
        for rank, (s, cid) in enumerate(kw, 1):
            cand[cid].update(keyword=round(s, 3), keyword_rank=rank)
    for h in cand.values():
        h["fused"] = sum(1 / (RRF_K + h[r]) for r in ("dense_rank", "keyword_rank") if h.get(r))
    ranked = sorted(cand.values(), key=lambda h: h["fused"], reverse=True)
    t2 = time.perf_counter()
    if rerank:
        ranked = ranked[:S.rerank_candidates]
        for h in ranked:
            h["rerank"] = round(rerank_score(query, h), 4)
        ranked = sorted(ranked, key=lambda h: h["rerank"], reverse=True)  # ties keep fused order
    t3 = time.perf_counter()
    hits = ranked[:k]
    for h in hits:
        h["score"] = h["rerank"] if rerank else h["dense"]
    timings.update(embed_ms=(t1 - t0) * 1000, search_ms=(t2 - t1) * 1000, rerank_ms=(t3 - t2) * 1000)
    return hits, timings, q_vec


def recall(q_vec):
    """Graded answers to past questions nearly identical to this one."""
    n = index.memory.count()
    if q_vec is None or not n or S.memory_k <= 0:
        return []
    r = index.memory.query(query_embeddings=[q_vec], n_results=min(S.memory_k, n))
    found = []
    for mid, answer, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
        if 1 - dist >= S.memory_min_score:
            found.append({"id": mid, "question": meta["question"], "answer": answer, "score": round(1 - dist, 4)})
    return found
