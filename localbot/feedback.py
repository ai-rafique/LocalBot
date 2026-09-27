"""Grading answers, learning from grades, statistics, and export."""
import json
import os
import re
import statistics
from collections import Counter

from . import index, storage
from .generation import REFUSAL_RE, friendly_error

GRADES = ["good", "partial", "bad"]
# What went wrong, as tags. Keys are stable; labels are for the UI.
TAGS = [
    {"key": "retrieval", "label": "Wrong passages", "hint": "The right passage wasn't among the sources"},
    {"key": "wrong", "label": "False statement", "hint": "Says something the sources don't support"},
    {"key": "incomplete", "label": "Incomplete", "hint": "Right, but leaves out part of the answer"},
    {"key": "missed", "label": "Missed answer", "hint": "Said it doesn't know, but the sources have it"},
    {"key": "premise", "label": "Accepted false premise", "hint": "Went along with a wrong assumption in the question"},
    {"key": "should_refuse", "label": "Should have refused", "hint": "Answered although the documents don't cover it"},
    {"key": "citation", "label": "Citation problem", "hint": "Missing, wrong or made-up citation"},
    {"key": "verbose", "label": "Rambling", "hint": "Too long, repeats itself, or echoes instructions"},
]
TAG_KEYS = {t["key"] for t in TAGS}


def sync_memory(row):
    """Make a graded answer available to future questions — or withdraw it.

    The user's correction wins; otherwise a "good" answer is kept. Refusals
    aren't kept: after documents are added, an old "I don't know" should not
    answer for them.
    """
    reference = (row.get("correction") or "").strip()
    if not reference and row.get("grade") == "good" and not REFUSAL_RE.search(row.get("answer") or ""):
        reference = row["answer"]
    if not reference:
        index.memory.delete(ids=[row["id"]])
        return False
    # Its [n] citations pointed at that answer's passages, not future ones.
    reference = re.sub(r"\s*\[\d+\]", "", reference).strip()
    index.memory.upsert(ids=[row["id"]], embeddings=index.embed([row["question"]], kind="query"),
                        documents=[reference], metadatas=[{"question": row["question"], "grade": row.get("grade") or ""}])
    return True


def save_grade(iid, grade, tags=(), reason="", correction="", learn=True):
    row = storage.get_interaction(iid)
    if not row:
        raise ValueError("Answer not found")
    if grade not in GRADES and grade is not None:
        raise ValueError("Grade must be good, partial or bad")
    tags = [t for t in (tags or []) if t in TAG_KEYS]
    # Grading by hand replaces a carried-over grade: it's now the user's own.
    storage.query("UPDATE interactions SET grade = ?, tags = ?, reason = ?, correction = ?, graded_at = ?, "
                  "grade_source = NULL WHERE id = ?",
                  (grade, json.dumps(tags), (reason or "").strip(), (correction or "").strip(),
                   storage.now() if grade else None, iid))
    row.update(grade=grade, correction=(correction or "").strip())
    learned, note = False, ""
    # Experiment answers are graded to measure the pipeline; learning from
    # them would leak answers into later runs, so only chat answers learn.
    if learn and row.get("origin") == "chat":
        try:
            learned = sync_memory(row)
        except Exception as e:
            note = f"Saved, but not added to learned answers: {friendly_error(e)}"
    return {"learned": learned, "note": note}


def _preview(text, n=160):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def list_interactions(status="all", source="all", search="", run_id=None, flagged=False,
                      conversation_id=None, limit=200, offset=0):
    where, params = [], []
    if status == "ungraded":
        where.append("grade IS NULL")
    elif status in GRADES:
        where.append("grade = ?")
        params.append(status)
    elif status == "graded":
        where.append("grade IS NOT NULL")
    elif status == "carried":
        where.append("grade_source LIKE 'carried:%'")
    if source == "chat":
        where.append("origin = 'chat'")
    elif source == "experiment":
        where.append("origin = 'experiment'")
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    if conversation_id:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if search:
        where.append("(question LIKE ? OR answer LIKE ?)")
        params += [f"%{search}%"] * 2
    if flagged:
        where.append("checks LIKE '%\"kind\"%'")
    sql = ("SELECT id, ts, origin, run_id, qid, conversation_id, question, answer, grade, tags, metrics, checks, grade_source "
           "FROM interactions" + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY ts DESC, rowid DESC LIMIT ? OFFSET ?")
    rows = storage.query(sql, params + [limit, offset])
    total = storage.query("SELECT COUNT(*) AS n FROM interactions" + (" WHERE " + " AND ".join(where) if where else ""),
                          params)[0]["n"]
    out = []
    for r in rows:
        m = json.loads(r["metrics"] or "{}")
        checks = json.loads(r["checks"] or "{}")
        out.append({
            "id": r["id"], "ts": r["ts"], "origin": r["origin"], "run_id": r["run_id"], "qid": r["qid"],
            "conversation_id": r["conversation_id"], "question": _preview(r["question"], 200),
            "answer": _preview(r["answer"]), "grade": r["grade"], "tags": json.loads(r["tags"] or "[]"),
            "carried": bool(r["grade_source"]),
            "confidence": m.get("confidence"), "warnings": len(checks.get("warnings", [])),
            "total_s": m.get("total_s"),
        })
    return {"items": out, "total": total}


def stats():
    rows = storage.query("SELECT origin, grade, tags, metrics, grade_source FROM interactions")
    grades = Counter(r["grade"] for r in rows if r["grade"])
    tags = Counter(t for r in rows for t in json.loads(r["tags"] or "[]"))
    conf = {}
    for g in GRADES:
        vals = [json.loads(r["metrics"] or "{}").get("confidence") for r in rows if r["grade"] == g]
        vals = [v for v in vals if v is not None]
        conf[g] = {"mean": round(statistics.mean(vals), 3) if vals else None, "n": len(vals)}
    return {
        "total": len(rows), "graded": sum(grades.values()),
        "ungraded": len(rows) - sum(grades.values()),
        "grades": {g: grades.get(g, 0) for g in GRADES},
        "carried": sum(1 for r in rows if r["grade"] and r["grade_source"]),
        "tags": dict(tags), "confidence_by_grade": conf,
        "chat": sum(1 for r in rows if r["origin"] == "chat"),
        "learned": index.memory.count(),
    }


def export(only_graded=True, include_text=True, anonymize=True):
    """Grades plus tuning signals, for sharing. Passage text and prompts are
    never included, so the file doesn't carry the documents."""
    rows = [storage.decode(r) for r in storage.query("SELECT * FROM interactions ORDER BY ts")]
    if only_graded:
        rows = [r for r in rows if r["grade"]]
    aliases = {}

    def name(source):
        if not anonymize:
            return source
        if source not in aliases:
            aliases[source] = f"doc{len(aliases) + 1}{os.path.splitext(source)[1]}"
        return aliases[source]

    kb = [{"doc": name(d["name"]), "chunks": d["chunks"], "chars": d["chars"]} for d in index.library()]
    prompts, records = [], []
    for r in rows:
        settings = dict(r["settings"])
        prompt = settings.pop("system_prompt", None)
        if prompt not in prompts:
            prompts.append(prompt)
        rec = {
            "id": r["id"], "time": r["ts"], "origin": r["origin"], "run_id": r["run_id"], "qid": r["qid"],
            "grade": r["grade"], "tags": r["tags"], "reason": r["reason"],
            "settings": settings, "system_prompt_version": prompts.index(prompt), "metrics": r["metrics"],
            "retrieval": [
                {"doc": name(h["source"]), "chunk": h["chunk"], "score": round(h["score"], 3), "used": h.get("used"),
                 "chars": len(h["text"]),
                 **{k: h[k] for k in ("dense", "dense_rank", "keyword", "keyword_rank", "rerank",
                                      "page_start", "page_end", "chunk_size", "overlap") if k in h},
                 # Headings are document text: only shared along with the answers.
                 **({"section": h.get("section", "")} if include_text else {})}
                for h in r["hits"] if "score" in h
            ],
            "memory_used": [{"from_id": m["id"], "score": m["score"]} for m in r["memory"]],
        }
        if include_text:
            rec.update(question=r["question"], answer=r["answer"], correction=r["correction"],
                       weakest_tokens=r["weak_tokens"], unsupported_terms=r["checks"].get("unsupported", []))
        records.append(rec)
    return {
        "exported_at": storage.now(),
        "contains": ("questions, answers, grades, retrieval scores" if include_text
                     else "grades, retrieval scores only") + "; no passage text or prompts",
        "system_prompts": prompts, "knowledge_base": kb, "interactions": records,
    }
