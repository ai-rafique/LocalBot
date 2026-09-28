"""Experiments: run a question set under changed settings and measure.

A question set is a list of questions with the facts a correct answer must
contain. A run answers every question through the same pipeline as the
chat, with setting overrides applied to the worker thread only, then
scores the answers automatically. Graded answers (Evaluate) feed back into
the run's summary.

Question format:
  {"id": "q1", "variant": "lookup", "question": "...", "answerable": true,
   "facts": [["115200"], ["8n1", "8-n-1"]], "note": ""}
facts: each inner list is one required fact; any spelling in it counts.
"""
import json
import queue
import random
import re
import statistics
import threading
import traceback
from collections import Counter
from contextlib import nullcontext

import ollama

from . import generation, index, retrieval, storage
from .config import S, SCHEMA_BY_KEY
from .generation import REFUSAL_RE

VARIANTS = [
    {"key": "lookup", "label": "Lookup", "hint": "One fact, asked in the document's words"},
    {"key": "exact", "label": "Exact values", "hint": "Byte sequences, ids, register values copied exactly"},
    {"key": "paraphrase", "label": "Paraphrase", "hint": "Same kind of fact, different words"},
    {"key": "multihop", "label": "Multi-step", "hint": "Combines two or more places in the documents"},
    {"key": "crossdoc", "label": "Cross-document", "hint": "Combines two documents"},
    {"key": "numeric", "label": "Numeric", "hint": "A small calculation from documented values"},
    {"key": "procedure", "label": "Procedure", "hint": "\"How do I …\""},
    {"key": "false_premise", "label": "False premise", "hint": "Assumes something wrong; should be corrected"},
    {"key": "unanswerable", "label": "Unanswerable", "hint": "On topic, but not in the documents"},
    {"key": "terse", "label": "Terse / typos", "hint": "Fragments and misspellings"},
]
VARIANT_KEYS = [v["key"] for v in VARIANTS]
INDEX_KEYS = ("chunk_size", "chunk_overlap", "embed_model")


# --- Question sets -----------------------------------------------------------
def validate_item(q):
    problems = []
    if not str(q.get("id", "")).strip():
        problems.append("missing id")
    if not str(q.get("question", "")).strip():
        problems.append("missing question")
    facts = q.get("facts", [])
    if not isinstance(facts, list) or any(not isinstance(f, list) or not f for f in facts):
        problems.append("facts must be a list of non-empty lists")
    if q.get("answerable", True) and not facts:
        problems.append("answerable questions need at least one fact")
    if q.get("variant", "lookup") not in VARIANT_KEYS:
        problems.append(f"unknown variant {q.get('variant')!r}")
    return problems


def parse_jsonl(text):
    items, errors, ids = [], [], set()
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            q = json.loads(line)
        except ValueError as e:
            errors.append(f"line {n}: not valid JSON ({e.msg})")
            continue
        q = {"variant": "lookup", "answerable": True, "facts": [], "note": "", **q}
        problems = validate_item(q)
        if q.get("id") in ids:
            problems.append(f"duplicate id {q['id']}")
        if problems:
            errors.append(f"line {n}: " + "; ".join(problems))
        else:
            ids.add(q["id"])
            items.append(q)
    return items, errors


def create_set(name, items):
    sid = storage.new_id()
    storage.query("INSERT INTO question_sets VALUES (?, ?, ?, ?)", (sid, name, storage.now(), json.dumps(items)))
    return sid


def list_sets():
    out = []
    for r in storage.query("SELECT * FROM question_sets ORDER BY created DESC"):
        items = json.loads(r["items"])
        out.append({"id": r["id"], "name": r["name"], "created": r["created"], "count": len(items),
                    "variants": dict(Counter(q["variant"] for q in items))})
    return out


def get_set(sid):
    rows = storage.query("SELECT * FROM question_sets WHERE id = ?", (sid,))
    return storage.decode(rows[0]) if rows else None


def delete_set(sid):
    storage.query("DELETE FROM question_sets WHERE id = ?", (sid,))


def draft_set(n=10):
    """Draft questions from random passages. Each fact is a short phrase the
    model quoted from its passage (verified to be there). Drafted questions
    reuse the documents' wording, so review them and add harder ones."""
    data = index.collection().get(include=["documents", "metadatas"])
    pool = [d for d in data["documents"] if len(d) > 200]
    if not pool:
        raise ValueError("Add documents first")
    items = []
    for i, doc in enumerate(random.sample(pool, min(int(n), len(pool))), 1):
        prompt = (f"Passage:\n{doc}\n\nWrite ONE specific question that this passage answers, and the "
                  "shortest exact phrase from the passage that answers it. Use this format:\n"
                  "Question: <the question>\nAnswer: <phrase copied exactly from the passage>")
        text = generation.clean_answer("".join(p.message.content or "" for p in generation.chat_stream(
            [{"role": "user", "content": prompt}])))
        q = re.search(r"question\s*:\s*(.+)", text, re.I)
        a = re.search(r"answer\s*:\s*(.+)", text, re.I)
        phrase = a.group(1).strip(' "*.') if a else ""
        if q and phrase and norm(phrase) in norm(doc):
            items.append({"id": f"draft-{i:02d}", "variant": "lookup", "question": q.group(1).strip(' "*'),
                          "answerable": True, "facts": [[phrase]], "note": "drafted from a passage; review"})
    return items


# --- Scoring (pure) ------------------------------------------------------------
def norm(text):
    return " ".join(str(text).lower().split())


def _squash(text):
    return re.sub(r"[\s,_\-/.:]+", "", str(text).lower())


def contains(text, alt):
    """Short plain tokens ("cc", "5") must match as whole words, so they
    aren't found inside longer words; anything else is compared with
    spacing and separators removed ("02 A1" = "02a1")."""
    alt = str(alt).lower().strip()
    if re.fullmatch(r"[a-z0-9]{1,3}", alt):
        return re.search(rf"(?<![a-z0-9]){re.escape(alt)}(?![a-z0-9])", str(text).lower()) is not None
    return _squash(alt) in _squash(text)


def fact_coverage(text, facts):
    if not facts:
        return None
    return sum(any(contains(text, alt) for alt in fact) for fact in facts) / len(facts)


def auto_label(q, answer, coverage):
    """correct / partial / wrong / missed for answerable questions;
    correct / answered for unanswerable ones."""
    refused = bool(REFUSAL_RE.search(answer or ""))
    if not q.get("answerable", True):
        return "correct" if refused else "answered"
    if coverage == 1 and not refused:
        return "correct"
    if refused and not coverage:
        return "missed"
    return "partial" if coverage else "wrong"


# --- Runs --------------------------------------------------------------------
_queue = queue.Queue()
_cancel = set()


def create_run(name, set_id, overrides=None, retrieval_only=False, variants=None, limit=None, use_memory=False):
    qs = get_set(set_id)
    if not qs:
        raise ValueError("Question set not found")
    overrides = {k: v for k, v in (overrides or {}).items()}
    with S.override(overrides):  # validates the values now, not when the run starts
        pass
    items = [q for q in qs["items"] if not variants or q["variant"] in variants][:limit or None]
    if not items:
        raise ValueError("No questions match the selected variants")
    rid = storage.new_id()
    config = {"overrides": overrides, "retrieval_only": bool(retrieval_only), "variants": variants or None,
              "limit": limit, "use_memory": bool(use_memory), "question_ids": [q["id"] for q in items]}
    storage.query("INSERT INTO runs (id, name, question_set_id, config, status, created, total) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (rid, name or "run", set_id, json.dumps(config), "queued", storage.now(), len(items)))
    _queue.put(rid)
    return rid


def cancel_run(rid):
    _cancel.add(rid)
    storage.query("UPDATE runs SET status = 'cancelled', finished = ? WHERE id = ? AND status = 'queued'", (storage.now(), rid))


def delete_run(rid):
    cancel_run(rid)
    storage.query("DELETE FROM run_results WHERE run_id = ?", (rid,))
    storage.query("UPDATE interactions SET run_id = NULL WHERE run_id = ?", (rid,))
    storage.query("DELETE FROM runs WHERE id = ?", (rid,))


def _unload_other_llms():
    """Keep only this run's chat model in VRAM: two chat models plus the
    embedder may not fit a 4 GB GPU."""
    try:
        keep = {S.llm_model, S.embed_model, f"{S.embed_model}:latest", S.rerank_model_name()}
        for m in ollama.ps().models:
            if m.model not in keep:
                ollama.generate(model=m.model, prompt="", keep_alive=0)
    except Exception:
        pass


def _execute(rid):
    run = storage.decode(storage.query("SELECT * FROM runs WHERE id = ?", (rid,))[0])
    if run["status"] != "queued" or rid in _cancel:
        return
    cfg = run["config"]
    items = {q["id"]: q for q in get_set(run["question_set_id"])["items"]}
    todo = [items[i] for i in cfg["question_ids"] if i in items]
    storage.query("UPDATE runs SET status = 'running', started = ? WHERE id = ?", (storage.now(), rid))
    try:
        with S.override(cfg["overrides"]):
            state = index.state()
            rebuild = any(k in cfg["overrides"] and cfg["overrides"][k] != state.get(k) for k in INDEX_KEYS)
            with (index.use_temporary(S.chunk_size, S.chunk_overlap) if rebuild else nullcontext()):
                _unload_other_llms()
                for n, q in enumerate(todo, 1):
                    if rid in _cancel:
                        storage.query("UPDATE runs SET status = 'cancelled', finished = ? WHERE id = ?", (storage.now(), rid))
                        return
                    iid, result = _run_question(rid, q, cfg)
                    storage.query("INSERT OR REPLACE INTO run_results VALUES (?, ?, ?, ?)",
                                  (rid, q["id"], iid, json.dumps(result)))
                    storage.query("UPDATE runs SET done = ? WHERE id = ?", (n, rid))
        storage.query("UPDATE runs SET status = 'done', finished = ? WHERE id = ?", (storage.now(), rid))
        carry_over_grades(rid)
    except Exception as e:
        traceback.print_exc()
        storage.query("UPDATE runs SET status = 'failed', error = ?, finished = ? WHERE id = ?",
                      (generation.friendly_error(e), storage.now(), rid))


def _run_question(rid, q, cfg):
    base = {"variant": q["variant"], "answerable": q.get("answerable", True)}
    if cfg["retrieval_only"]:
        # Passages only: nothing is logged for grading, there's no answer.
        query = generation.search_query(q["question"])
        hits, t, _, _, split_ms = generation.find_passages(q["question"], query)
        sent = retrieval.sendable(hits)
        return None, {**base, "context_coverage": fact_coverage(" ".join(h["text"] for h in sent), q["facts"]),
                      "passages_sent": len(sent), "top_score": hits[0]["score"] if hits else None,
                      "total_s": round((sum(t.values()) + split_ms) / 1000, 3)}
    iid = None
    for ev in generation.answer(q["question"], origin="experiment", run_id=rid, qid=q["id"],
                                use_memory=cfg.get("use_memory", False)):
        if ev["type"] == "done":
            iid = ev["interaction"]
    row = storage.get_interaction(iid)
    m = row["metrics"]
    sent_text = " ".join(h["text"] for h in row["hits"] if h.get("used"))
    cov = fact_coverage(row["answer"], q["facts"])
    return iid, {**base, "coverage": cov, "context_coverage": fact_coverage(sent_text, q["facts"]),
                 "refused": bool(REFUSAL_RE.search(row["answer"])), "auto_label": auto_label(q, row["answer"], cov),
                 "passages_sent": m["passages_used"], "top_score": m["top_score"], "confidence": m["confidence"],
                 "unsupported_terms": m["unsupported_terms"], "claims_unsupported": m.get("claims_unsupported"),
                 "answer_tokens": m["answer_tokens"],
                 "total_s": m["total_s"]}


def carry_over_grades(rid):
    """Give each ungraded answer of a run the grade of an identical answer
    to the same question that the user graded by hand. Only word-for-word
    identical answers qualify: a one-word change can make an answer wrong.
    Never copies from carried grades, so nothing is copied twice removed.
    Corrections aren't copied (they teach the chat). Returns the count."""
    rows = storage.query("SELECT i.id, i.question, i.answer FROM run_results rr JOIN interactions i "
                         "ON i.id = rr.interaction_id WHERE rr.run_id = ? AND i.grade IS NULL", (rid,))
    n = 0
    for r in rows:
        src = storage.query("SELECT id, grade, tags, reason FROM interactions WHERE question = ? AND answer = ? "
                            "AND grade IS NOT NULL AND grade_source IS NULL AND id != ? ORDER BY graded_at DESC LIMIT 1",
                            (r["question"], r["answer"], r["id"]))
        if src:
            s = src[0]
            storage.query("UPDATE interactions SET grade = ?, tags = ?, reason = ?, graded_at = ?, grade_source = ? "
                          "WHERE id = ?", (s["grade"], s["tags"], s["reason"], storage.now(), f"carried:{s['id']}", r["id"]))
            n += 1
    return n


def _worker():
    while True:
        rid = _queue.get()
        try:
            _execute(rid)
        finally:
            _cancel.discard(rid)


def start_worker():
    # Runs left "running" by a previous process can't be resumed.
    storage.query("UPDATE runs SET status = 'interrupted' WHERE status = 'running'")
    # Grades given since a run finished can pre-grade its identical answers.
    for r in storage.query("SELECT id FROM runs WHERE status IN ('done', 'interrupted', 'cancelled')"):
        carry_over_grades(r["id"])
    for r in storage.query("SELECT id FROM runs WHERE status = 'queued' ORDER BY created"):
        _queue.put(r["id"])
    threading.Thread(target=_worker, daemon=True, name="experiments").start()


# --- Summaries -----------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(statistics.mean(xs)), 3) if xs else None


def list_runs():
    runs = storage.query("SELECT r.*, s.name AS set_name FROM runs r LEFT JOIN question_sets s "
                         "ON s.id = r.question_set_id ORDER BY r.created DESC")
    return [{**storage.decode(r), "summary": summarize(r["id"], brief=True)} for r in runs]


def get_run(rid):
    rows = storage.query("SELECT r.*, s.name AS set_name FROM runs r LEFT JOIN question_sets s "
                         "ON s.id = r.question_set_id WHERE r.id = ?", (rid,))
    if not rows:
        return None
    run = storage.decode(rows[0])
    items = {q["id"]: q for q in (get_set(run["question_set_id"]) or {"items": []})["items"]}
    order = {qid: n for n, qid in enumerate(run["config"].get("question_ids", []))}
    results = []
    for r in sorted(_results(rid), key=lambda r: order.get(r["qid"], len(order))):
        q = items.get(r["qid"], {})
        results.append({**r, "question": q.get("question", ""), "facts": q.get("facts", [])})
    return {**run, "summary": summarize(rid), "results": results}


def _results(rid):
    rows = storage.query("SELECT rr.qid, rr.interaction_id, rr.result, i.grade, i.tags, i.grade_source FROM run_results rr "
                         "LEFT JOIN interactions i ON i.id = rr.interaction_id WHERE rr.run_id = ?", (rid,))
    return [{"qid": r["qid"], "interaction_id": r["interaction_id"], "grade": r["grade"],
             "carried": bool(r["grade_source"]), "tags": json.loads(r["tags"] or "[]"),
             **json.loads(r["result"])} for r in rows]


def summarize(rid, brief=False):
    res = _results(rid)
    run = storage.decode(storage.query("SELECT config FROM runs WHERE id = ?", (rid,))[0])
    retrieval_only = run["config"].get("retrieval_only")
    ans = [r for r in res if r["answerable"]]
    unans = [r for r in res if not r["answerable"]]
    graded = [r for r in res if r["grade"]]
    s = {
        "questions": len(res), "retrieval_only": retrieval_only,
        # Did the passages sent contain every required fact? (retrieval quality)
        "context_recall": _mean([1.0 if r.get("context_coverage") == 1 else 0.0 for r in ans]),
        # Unanswerable questions refused before reaching the model.
        "offtopic_blocked": _mean([1.0 if not r.get("passages_sent") else 0.0 for r in unans]),
        "latency_s": _mean([r.get("total_s") for r in res]),
        "graded": len(graded),
        "carried": sum(1 for r in graded if r.get("carried")),
        "good_rate": _mean([1.0 if r["grade"] == "good" else 0.0 for r in graded]),
    }
    if not retrieval_only:
        s.update({
            "auto_correct": _mean([1.0 if r.get("auto_label") == "correct" else 0.0 for r in ans]),
            "fact_coverage": _mean([r.get("coverage") for r in ans]),
            "refusal_accuracy": _mean([1.0 if r.get("refused") else 0.0 for r in unans]),
            "false_refusals": _mean([1.0 if r.get("refused") else 0.0 for r in ans]),
            # The facts were in the passages, but not in the answer: the model misread.
            "generation_misses": sum(1 for r in ans if r.get("context_coverage") == 1 and (r.get("coverage") or 0) < 1),
            "flagged": sum(1 for r in res if r.get("unsupported_terms")),
            # None for runs made before the claim check existed.
            "claims_flagged": (sum(1 for r in res if r.get("claims_unsupported"))
                               if any(r.get("claims_unsupported") is not None for r in res) else None),
            "confidence": _mean([r.get("confidence") for r in res]),
        })
    if brief:
        return s
    s["grades"] = dict(Counter(r["grade"] for r in graded))
    s["tags"] = dict(Counter(t for r in graded for t in r["tags"]))
    s["by_variant"] = {}
    present = {r["variant"] for r in res}
    for v in [k for k in VARIANT_KEYS if k in present] + sorted(present - set(VARIANT_KEYS)):
        rs = [r for r in res if r["variant"] == v]
        g = Counter(r["grade"] for r in rs if r["grade"])
        s["by_variant"][v] = {
            "n": len(rs),
            "context_recall": _mean([1.0 if r.get("context_coverage") == 1 else 0.0 for r in rs if r["answerable"]]),
            "auto_correct": None if retrieval_only else _mean([1.0 if r.get("auto_label") == "correct" else 0.0 for r in rs]),
            "good": g.get("good", 0), "partial": g.get("partial", 0), "bad": g.get("bad", 0),
        }
    return s


def compare(run_ids):
    runs = [get_run(r) for r in run_ids]
    runs = [r for r in runs if r]
    changes = []
    if len(runs) >= 2:
        first = {r["qid"]: r for r in runs[0]["results"]}
        for r in runs[-1]["results"]:
            a = first.get(r["qid"])
            if a and (a.get("auto_label") != r.get("auto_label") or (a["grade"] and r["grade"] and a["grade"] != r["grade"])):
                changes.append({"qid": r["qid"], "variant": r["variant"], "question": r["question"],
                                "before": a.get("grade") or a.get("auto_label"),
                                "after": r.get("grade") or r.get("auto_label")})
    return {"runs": [{k: r[k] for k in ("id", "name", "set_name", "config", "status", "created", "summary")}
                     for r in runs], "changes": changes}


def overridable():
    """Settings an experiment may change (everything tunable)."""
    return list(SCHEMA_BY_KEY)
