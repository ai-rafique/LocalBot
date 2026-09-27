"""Answering: retrieve → build prompt → stream the answer → check → log.

`answer()` is the single path used by the chat and by experiments, so both
measure exactly the same behaviour.
"""
import math
import re
import statistics
import time

import ollama

from . import documents, retrieval, storage
from .config import CHARS_PER_TOKEN, NO_CONTEXT_ANSWER, S

REFUSAL_RE = re.compile(
    r"(don't|do not|doesn't|does not|cannot|can't) (know|contain|have|mention|find|answer|state|specify)"
    r"|not (found|mentioned|provided|included|stated|specified|defined) in|no information|isn't (stated|defined|specified)",
    re.I,
)

# Terms a technical answer must not invent: hex values, multi-digit numbers,
# versions/ranges, and identifiers (snake_case, camelCase, Foo::bar, ABC12).
TECH_TERM_RE = re.compile(
    r"0x[0-9A-Fa-f]+|\d+(?:[.,:/-]\d+)+|\d{2,}"
    r"|[A-Za-z_]\w*(?:::\w+)+|[A-Za-z]\w*_\w+|[a-z]+[A-Z]\w*|[A-Z]{2,}\w*\d\w*|[A-Za-z]+-\d\w*"
)


def build_prompt(query, hits, memory_items=()):
    parts = []
    if memory_items:
        refs = "\n\n".join(f"[V{i}] Q: {m['question']}\nA: {m['answer']}" for i, m in enumerate(memory_items, 1))
        parts.append("Verified answer to a similar earlier question (checked by the user):\n" + refs)
    if hits:
        # Numbered to match the sources shown in the UI, so [2] is checkable.
        blocks = [f"[{i}] {documents.passage_label(h)}\n{h['text']}" for i, h in enumerate(hits, 1)]
        parts.append("Passages from documents:\n\n" + "\n\n".join(blocks))
    else:
        parts.append("Passages from documents:\n(no relevant passages found)")
    parts.append(
        f"Question: {query}\n\n"
        "Answer using only the text above. Copy names, numbers and identifiers exactly "
        "as written there. Cite what you used, like [1]. If the answer is not there, "
        "say you don't know."
    )
    return "\n\n---\n\n".join(parts)


def fit_context(question, history, hits, memory_items, system_prompt):
    """Trim until the prompt fits the context window: oldest history first,
    then the weakest passages. On overflow Ollama would silently drop the
    *start* of the prompt — the instructions."""
    budget = (S.num_ctx - S.num_predict) * CHARS_PER_TOKEN
    history, hits = list(history)[-2 * S.history_turns:] if S.history_turns else [], list(hits)
    while True:
        prompt = build_prompt(question, hits, memory_items)
        size = len(system_prompt) + len(prompt) + sum(len(m["content"]) for m in history)
        if size <= budget:
            break
        if history:
            history = history[2:]
        elif len(hits) > 1:
            hits = hits[:-1]
        else:
            break
    return history, hits, prompt


def clean_answer(text):
    # Qwen models can emit <think>…</think> reasoning; never show it as the answer.
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


def friendly_error(e):
    msg = str(e)
    if isinstance(e, ConnectionError) or "connect" in msg.lower():
        return "Can't reach Ollama. Make sure it's running (`ollama serve`)."
    if "not found" in msg.lower():
        return f"{msg}. Install it from Settings → Models, or run `ollama pull <model>`."
    return msg


def chat_stream(messages, logprobs=False, model=None):
    kwargs = dict(model=model or S.llm_model, messages=messages, stream=True, keep_alive=S.keep_alive)
    # Thinking mode makes a small model several times slower for little gain
    # on short grounded answers. Log-probabilities measure how sure the model
    # was of each token. Older Ollama versions reject either flag.
    extras = {"think": False}
    if logprobs:
        extras["logprobs"] = True
    try:
        yield from ollama.chat(**extras, **kwargs, options=retrieval.gen_options())
    except ollama.ResponseError as e:
        msg = str(e).lower()
        if "memory" in msg or "cuda" in msg:
            # Forcing every layer onto the GPU failed (another app holds VRAM):
            # let Ollama split the model with the CPU — slower, but works.
            yield from ollama.chat(**extras, **kwargs, options=retrieval.gen_options(force_gpu=False))
        elif any(flag in msg for flag in extras):
            yield from ollama.chat(**kwargs, options=retrieval.gen_options())
        else:
            raise


def confidence(token_lps):
    """confidence: geometric-mean token probability (1.0 = never hesitated).
    min_token_p: the least certain token. unsure_tokens: tokens under 50%."""
    if not token_lps:
        return {"confidence": None, "min_token_p": None, "unsure_tokens": None}, []
    lps = [lp for _, lp in token_lps]
    weakest = sorted(token_lps, key=lambda t: t[1])[:8]
    return {
        "confidence": round(math.exp(statistics.mean(lps)), 3),
        "min_token_p": round(math.exp(min(lps)), 3),
        "unsure_tokens": sum(lp < math.log(0.5) for lp in lps),
    }, [{"token": t, "p": round(math.exp(lp), 3)} for t, lp in weakest]


def faithfulness(answer, hits, memory_items, question):
    """Cheap checks that an answer stays inside its sources.

    unsupported: technical terms in the answer found in none of the passages
    sent, verified answers or the question — the usual shape of an invented
    register, value or version. bad_citations: [n] beyond the passages sent.
    """
    body = re.sub(r"\[V?\d+\]", " ", answer)
    # Compared with spacing and separators removed: PDF extraction turns
    # "12/34" into "12 /34", and a file name may be written with or without
    # underscores.
    squash = lambda t: re.sub(r"[\s,_\-/.:]+", "", t.lower())
    haystack = squash(" ".join([question] + [f"{h['source']} {documents.chunk_header(h)} {h['text']}" for h in hits]
                               + [m["answer"] for m in memory_items]))
    unsupported = [t for t in dict.fromkeys(TECH_TERM_RE.findall(body)) if squash(t) not in haystack]
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    return {
        "unsupported": unsupported,
        "bad_citations": sorted({n for n in cited if not 1 <= n <= len(hits)}),
        "cited": bool(cited) or bool(re.search(r"\[V\d+\]", answer)),
    }


def warnings(answer, hits_used, checks, metrics):
    """Human-readable cautions shown under an answer."""
    out = []
    if checks.get("unsupported"):
        out.append({"kind": "unsupported", "text": "Not found in the sources: " + ", ".join(checks["unsupported"][:6])})
    if checks.get("bad_citations"):
        out.append({"kind": "citation", "text": "Cites " + ", ".join(f"[{n}]" for n in checks["bad_citations"])
                    + ", which isn't one of the sources"})
    if hits_used and not checks.get("cited") and not REFUSAL_RE.search(answer):
        out.append({"kind": "uncited", "text": "No source cited"})
    c = metrics.get("confidence")
    if c is not None and c < S.low_confidence:
        out.append({"kind": "confidence", "text": f"Low model confidence ({c:.2f})"})
    if metrics.get("hit_length_cap"):
        out.append({"kind": "truncated", "text": "Stopped at the maximum answer length"})
    return out


def history_for(conversation_id):
    msgs = []
    for turn in storage.conversation_turns(conversation_id):
        # The bare question and answer, not the passage-stuffed prompt, so
        # earlier turns don't crowd the small model's context window.
        msgs += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
    return msgs


def answer(question, conversation_id=None, origin="chat", run_id=None, qid=None,
           use_memory=None, retrieval_only=False):
    """Yield events while answering; the last is {"type": "done", "interaction": id}.

    Events: stage (search / rerank / write), sources (passages and verified
    answers used), token (answer text), done.
    """
    use_memory = S.use_memory if use_memory is None else use_memory
    rerank = S.rerank
    min_score = S.cutoff()
    t_start = time.perf_counter()
    yield {"type": "stage", "stage": "search", "rerank": rerank}
    all_hits, timings, q_vec = retrieval.retrieve(question)
    memory_items = retrieval.recall(q_vec) if use_memory else []
    passing = [h for h in all_hits if h["score"] >= min_score]
    history = history_for(conversation_id) if conversation_id else []
    system_prompt = S.system_prompt
    history_sent, hits, prompt = fit_context(question, history, passing, memory_items, system_prompt)
    for h in hits:
        h["used"] = True
    yield {"type": "sources", "hits": all_hits, "memory": memory_items}

    messages = [{"role": "system", "content": system_prompt}, *history_sent, {"role": "user", "content": prompt}]
    text, first_token_at, final, token_lps = "", None, None, []
    if retrieval_only:
        text = ""
    elif hits or memory_items:
        yield {"type": "stage", "stage": "write"}
        for part in chat_stream(messages, logprobs=True):
            token = part.message.content or ""
            if token and first_token_at is None:
                first_token_at = time.perf_counter()
            text += token
            token_lps += [(lp.token, lp.logprob) for lp in (getattr(part, "logprobs", None) or [])]
            if part.done:
                final = part
            if token:
                yield {"type": "token", "text": token}
    else:
        text = NO_CONTEXT_ANSWER
        yield {"type": "token", "text": text}

    t_end = time.perf_counter()
    text = clean_answer(text)
    eval_count = (final.eval_count or 0) if final else 0
    eval_s = (final.eval_duration or 0) / 1e9 if final else 0
    conf, weak_tokens = confidence(token_lps)
    checks = (faithfulness(text, hits, memory_items, question) if (hits or memory_items) and text
              else {"unsupported": [], "bad_citations": [], "cited": False})
    scores = [h["score"] for h in all_hits]
    metrics = {
        "k": S.top_k, "min_score": float(min_score), "hybrid": S.hybrid, "rerank": rerank,
        "top_score": round(scores[0], 3) if scores else None,
        # A clear winner vs. several near-equal passages is itself an
        # uncertainty signal for retrieval.
        "score_gap": round(scores[0] - scores[1], 3) if len(scores) > 1 else None,
        "passages_used": len(hits), "passages_trimmed": len(passing) - len(hits),
        "memory_used": len(memory_items), "history_msgs": len(history_sent),
        **conf,
        "unsupported_terms": len(checks["unsupported"]), "bad_citations": len(checks["bad_citations"]),
        "cited": checks["cited"],
        "embed_ms": round(timings["embed_ms"], 1), "search_ms": round(timings["search_ms"], 1),
        "rerank_ms": round(timings["rerank_ms"], 1),
        "ttft_s": round((first_token_at or t_end) - t_start, 2), "total_s": round(t_end - t_start, 2),
        "prompt_tokens": (final.prompt_eval_count or 0) if final else 0, "answer_tokens": eval_count,
        "hit_length_cap": eval_count >= S.num_predict,
        "tok_per_s": round(eval_count / eval_s, 1) if eval_s else 0.0,
        "load_s": round((final.load_duration or 0) / 1e9, 2) if final else 0.0,
    }
    checks["warnings"] = warnings(text, hits, checks, metrics)
    iid = storage.log_interaction(
        origin=origin, conversation_id=conversation_id, run_id=run_id, qid=qid,
        question=question, answer=text, messages=messages, hits=all_hits, memory=memory_items,
        settings=S.effective(), metrics=metrics, weak_tokens=weak_tokens, checks=checks,
    )
    if conversation_id:
        storage.touch_conversation(conversation_id)
    yield {"type": "done", "interaction": iid}
