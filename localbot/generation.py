"""Answering: retrieve → build prompt → stream the answer → check → log.

`answer()` is the single path used by the chat and by experiments, so both
measure exactly the same behaviour.
"""
import math
import re
import statistics
import time

import ollama

from . import documents, index, retrieval, storage
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


CLAIM_SYSTEM = (
    'Judge whether the Document supports the Statement. Note that the answer can only be "yes" or "no".'
)
MAX_CLAIMS = 8  # bounds the added time on long answers


def split_claims(answer):
    """The sentences of an answer worth checking, each with the passage
    numbers it cites. Skips headings, lead-ins ("Answer:") and refusals."""
    out = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", line).strip()
        if not line or line.startswith(("#", "|", "```")):
            continue
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`*\[(])", line):
            sent = sent.strip()
            claim = re.sub(r"\s*\[V?\d+\]", "", sent).strip(" *")
            if len(claim) < 25 or claim.endswith(":") or REFUSAL_RE.search(claim):
                continue
            out.append({"text": sent, "claim": claim, "cites": sorted({int(n) for n in re.findall(r"\[(\d+)\]", sent)})})
    return out[:MAX_CLAIMS]


def check_claims(answer, hits, memory_items=()):
    """Score how well each statement is supported by the passages it cites
    (all passages sent, when it cites none). A cheap second read that catches
    wrong details the term check can't: swapped rows, invented behaviour,
    a false premise repeated as fact."""
    claims = split_claims(answer)
    extra = [{"source": "verified answer", "section": "", "text": m["answer"]} for m in memory_items]
    for c in claims:
        cited = [hits[n - 1] for n in c["cites"] if 1 <= n <= len(hits)]
        passages = (cited or list(hits)) + extra
        doc = "\n\n".join(f"{documents.chunk_header(h)}\n{h['text']}" for h in passages)
        # Document first, statement last: consecutive statements that cite
        # the same passages reuse Ollama's cached prompt prefix.
        c["score"] = round(retrieval.yes_probability(CLAIM_SYSTEM, f"<Document>: {doc}\n<Statement>: {c['claim']}"), 3)
        c["supported"] = c["score"] >= S.claim_min_score
    return claims


REWRITE_SYSTEM = (
    "You turn the user's latest message into one standalone search query for finding passages "
    "in their technical documents. Use the conversation to resolve references such as 'it', "
    "'that one' or 'the second'. Fix obvious typos and complete fragments. Keep names, "
    "identifiers, numbers and commands exactly as written. If the message already names what it "
    "asks about, keep those names. Output only the query, on one line.\n\n"
    "Example: after a conversation about the Bellman-Ford algorithm, \"how does it compare with "
    "Dijkstra?\" becomes \"How does the Bellman-Ford algorithm compare with Dijkstra's algorithm?\""
)


# Words that point back into the conversation.
REFERS_BACK_RE = re.compile(r"\b(it|its|they|them|their|this|that|these|those|former|latter|above|same|"
                            r"first one|second one|other one|previous)\b", re.I)


def search_query(question, history=()):
    """What to search for: a follow-up rewritten into a standalone query
    using the conversation, or the message itself. Only used for finding
    passages. Standalone questions are never rewritten: measured on 32
    questions, rewriting them lost 1-2 passages and gained none (the 2B
    paraphrases away terms the search needed)."""
    if not S.rewrite_queries or not history:
        return question
    # A message that names its own subject and doesn't point back is
    # standalone even mid-conversation; rewriting one lost its subject
    # ("binary tree vs hash map?" -> "...their traversal strategies?").
    if index.distinctive_terms(question) and not REFERS_BACK_RE.search(question):
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in list(history)[-4:])
    user = (f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Latest message: {question}"
    try:
        r = ollama.chat(model=S.llm_model, think=False, keep_alive=S.keep_alive,
                        options={**retrieval.gen_options(), "num_predict": 80},
                        messages=[{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}])
    except ollama.ResponseError:
        return question
    lines = [l.strip().strip('"').strip() for l in clean_answer(r.message.content or "").splitlines() if l.strip()]
    q = re.sub(r"^(search query|query)\s*:\s*", "", lines[0], flags=re.I) if lines else ""
    # A rewrite much longer than the message has wandered off, and one that
    # drops the message's own subject has replaced it (seen: "how does a
    # binary tree differ from a hash map?" -> "how do they differ?").
    if not q or len(q) > 3 * len(question) + 150:
        return question
    return question if index.distinctive_terms(question) - set(index.keyword_tokens(q)) else q


COMPARE_RE = re.compile(r"\b(compar\w*|differ\w*|contrast\w*|versus|vs\.?|similarit\w*|better than|"
                        r"worse than|faster than|slower than|pros and cons|trade-?offs?)(?!\w)", re.I)
SPLIT_SYSTEM = (
    "List the things the question compares, one per line, each as a short name exactly as written. "
    "Only the things themselves, not what they are compared on. Output only the names. If it "
    "doesn't compare two or more named things, output NONE.\n\n"
    "Example: \"How does Bellman-Ford compare with Dijkstra's algorithm for shortest paths?\" "
    "gives:\nBellman-Ford\nDijkstra's algorithm"
)


def comparison_parts(query):
    """The things a comparison question compares (["Bellman-Ford", "Dijkstra's
    algorithm"]), or [] for any other question. Only questions with a
    comparison word reach the model, so others cost nothing."""
    if not S.split_comparisons or not COMPARE_RE.search(query):
        return []
    try:
        r = ollama.chat(model=S.llm_model, think=False, keep_alive=S.keep_alive,
                        options={**retrieval.gen_options(), "num_predict": 60},
                        messages=[{"role": "system", "content": SPLIT_SYSTEM}, {"role": "user", "content": query}])
    except ollama.ResponseError:
        return []
    parts = []
    for line in clean_answer(r.message.content or "").splitlines():
        name = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').rstrip(".").strip()
        if name and name.upper() != "NONE" and len(name) <= 80 and name.lower() not in (p.lower() for p in parts):
            parts.append(name)
    return parts[:4] if len(parts) >= 2 else []


def find_passages(question, query, sources=None):
    """Search for the answer's passages; the same path for chat and experiments.
    Returns (hits, timings, query_vector, compared parts, split ms)."""
    t0 = time.perf_counter()
    parts = comparison_parts(query)
    split_ms = (time.perf_counter() - t0) * 1000
    if parts:
        hits, timings, q_vec = retrieval.retrieve_comparison(query, parts, keyword_text=question, sources=sources)
    else:
        hits, timings, q_vec = retrieval.retrieve(query, keyword_text=question, sources=sources)
    return hits, timings, q_vec, parts, split_ms


def relevant_turns(turns, question, query):
    """Earlier exchanges worth sending with this question. A small model
    mixes whatever the history contains into the answer: asked to compare
    Bellman-Ford with Dijkstra after a BFS/DFS discussion, it compared
    Dijkstra with BFS. So older exchanges are sent only when they share a
    distinctive word (a name, not "algorithm") with the question or its
    search query; the previous exchange is also kept when the new message
    has no subject of its own ("why?", "and the second one?")."""
    if not S.focus_history or not turns:
        return list(turns)
    words = index.distinctive_terms(f"{question} {query}")
    keep = [t for t in turns if words & index.distinctive_terms(
        f"{t['question']} {(t.get('metrics') or {}).get('search_query') or ''}")]
    if turns[-1] not in keep and not index.distinctive_terms(question):
        keep.append(turns[-1])
    return keep


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
    bad = [x for x in checks.get("claims", []) if not x["supported"]]
    if bad:
        out.append({"kind": "claims", "text": f"{len(bad)} statement{'s' if len(bad) > 1 else ''} not supported by the cited sources"})
    c = metrics.get("confidence")
    if c is not None and c < S.low_confidence:
        out.append({"kind": "confidence", "text": f"Low model confidence ({c:.2f})"})
    if metrics.get("hit_length_cap"):
        out.append({"kind": "truncated", "text": "Stopped at the maximum answer length"})
    return out


def history_for(conversation_id=None, turns=None):
    msgs = []
    for turn in storage.conversation_turns(conversation_id) if turns is None else turns:
        # The bare question and answer, not the passage-stuffed prompt, so
        # earlier turns don't crowd the small model's context window.
        msgs += [{"role": "user", "content": turn["question"]}, {"role": "assistant", "content": turn["answer"]}]
    return msgs


def answer(question, conversation_id=None, origin="chat", run_id=None, qid=None,
           use_memory=None, retrieval_only=False, sources=None):
    """Yield events while answering; the last is {"type": "done", "interaction": id}.

    Events: stage (search / rerank / write), sources (passages and verified
    answers used), token (answer text), done. sources limits the search
    to some documents.
    """
    use_memory = S.use_memory if use_memory is None else use_memory
    rerank = S.rerank
    min_score = S.cutoff()
    t_start = time.perf_counter()
    all_turns = storage.conversation_turns(conversation_id) if conversation_id else []
    if S.rewrite_queries and all_turns:
        yield {"type": "stage", "stage": "rewrite"}
    query = search_query(question, history_for(turns=all_turns[-2:]))
    rewrite_ms = (time.perf_counter() - t_start) * 1000
    turns = all_turns[-S.history_turns:] if S.history_turns else []
    kept = relevant_turns(turns, question, query)
    history = history_for(turns=kept)
    yield {"type": "stage", "stage": "search", "rerank": rerank, "query": query if query != question else None}
    all_hits, timings, q_vec, parts, split_ms = find_passages(question, query, sources)
    memory_items = retrieval.recall(q_vec) if use_memory else []
    passing = retrieval.sendable(all_hits)
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

    text = clean_answer(text)
    claims, verify_ms = [], 0.0
    if S.verify_claims and hits and text and not REFUSAL_RE.search(text) and not retrieval_only:
        yield {"type": "stage", "stage": "verify"}
        t_v = time.perf_counter()
        claims = check_claims(text, hits, memory_items)
        verify_ms = (time.perf_counter() - t_v) * 1000
    t_end = time.perf_counter()
    eval_count = (final.eval_count or 0) if final else 0
    eval_s = (final.eval_duration or 0) / 1e9 if final else 0
    conf, weak_tokens = confidence(token_lps)
    checks = (faithfulness(text, hits, memory_items, question) if (hits or memory_items) and text
              else {"unsupported": [], "bad_citations": [], "cited": False})
    checks["claims"] = claims
    scores = [h["score"] for h in all_hits]
    metrics = {
        "k": S.top_k, "min_score": float(min_score), "hybrid": S.hybrid, "rerank": rerank,
        "top_score": round(scores[0], 3) if scores else None,
        # A clear winner vs. several near-equal passages is itself an
        # uncertainty signal for retrieval.
        "score_gap": round(scores[0] - scores[1], 3) if len(scores) > 1 else None,
        "passages_used": len(hits), "passages_trimmed": len(passing) - len(hits),
        "memory_used": len(memory_items), "history_msgs": len(history_sent),
        "history_dropped": len(turns) - len(kept), "compared": parts or None, "split_ms": round(split_ms, 1),
        **conf,
        "unsupported_terms": len(checks["unsupported"]), "bad_citations": len(checks["bad_citations"]),
        "cited": checks["cited"],
        "embed_ms": round(timings["embed_ms"], 1), "search_ms": round(timings["search_ms"], 1),
        "rerank_ms": round(timings["rerank_ms"], 1), "verify_ms": round(verify_ms, 1),
        "search_query": query if query != question else None, "scope": sorted(sources) if sources else None, "rewrite_ms": round(rewrite_ms, 1),
        "claims_checked": len(claims), "claims_unsupported": sum(1 for c in claims if not c["supported"]),
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
