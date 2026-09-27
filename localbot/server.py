"""HTTP API and the web UI. Bound to 127.0.0.1 only: nothing is reachable
from other machines, and there is no public tunnel."""
import json
import os
import shutil
import statistics
import tempfile
import threading
import webbrowser

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import experiments, feedback, generation, index, models, storage, vision
from .config import GROUPS, HOST, PORT, PRESETS, SCHEMA, S, SUPPORTED_TYPES, WEB_DIR

app = FastAPI(title="LocalBot", docs_url=None, redoc_url=None, openapi_url=None)


def _bad(e):
    raise HTTPException(status_code=400, detail=str(e))


# --- Status & settings ---------------------------------------------------------
@app.get("/api/status")
def status():
    lib = index.library()
    running = storage.query("SELECT id, name, done, total FROM runs WHERE status = 'running' LIMIT 1")
    return {**models.status(), "documents": len(lib), "chunks": sum(d["chunks"] for d in lib),
            "index": index.state(), "running_experiment": running[0] if running else None}


def insights():
    """Measured numbers that let the UI say what a setting will do."""
    rows = storage.query("SELECT metrics, hits, settings FROM interactions WHERE origin IN ('chat', 'experiment') "
                         "ORDER BY ts DESC LIMIT 300")
    rerank_scores, dense_scores, per_candidate, tps, claim_scores = [], [], [], [], []
    for r in storage.query("SELECT checks FROM interactions ORDER BY ts DESC LIMIT 300"):
        claim_scores += [c["score"] for c in json.loads(r["checks"] or "{}").get("claims", [])]
    for r in rows:
        m, hits, st = json.loads(r["metrics"] or "{}"), json.loads(r["hits"] or "[]"), json.loads(r["settings"] or "{}")
        rerank_scores += [h["rerank"] for h in hits if "rerank" in h]
        dense_scores += [h["dense"] for h in hits if "dense" in h]
        if m.get("rerank_ms") and st.get("rerank_candidates"):
            per_candidate.append(m["rerank_ms"] / st["rerank_candidates"])
        if m.get("tok_per_s"):
            tps.append(m["tok_per_s"])
    counts = index.library_counts()
    chunks = sum(d["chunks"] for d in counts.values())
    try:
        sizes = {m["name"]: m["size_gb"] for m in models.installed()}
    except Exception:
        sizes = {}
    return {
        "total_chunks": chunks,
        "avg_chunk_chars": round(sum(d["chars"] for d in counts.values()) / chunks) if chunks else None,
        "rerank_ms_per_candidate": round(statistics.median(per_candidate)) if per_candidate else 200,
        "tok_per_s": round(statistics.median(tps), 1) if tps else None,
        "rerank_scores": [round(x, 3) for x in rerank_scores[:800]],
        "dense_scores": [round(x, 3) for x in dense_scores[:800]],
        "claim_scores": claim_scores[:800],
        "confidence_by_grade": feedback.stats()["confidence_by_grade"],
        "gpu": models.gpu(), "model_sizes": sizes,
    }


@app.get("/api/settings")
def get_settings():
    return {"schema": SCHEMA, "groups": [{"key": k, "label": v} for k, v in GROUPS], "presets": PRESETS,
            "values": S.values(), "index": index.state(), "insights": insights()}


class SettingsChange(BaseModel):
    changes: dict


@app.put("/api/settings")
def put_settings(body: SettingsChange):
    try:
        reindex = S.update(body.changes)
    except ValueError as e:
        _bad(e)
    return {"values": S.values(), "reindex": reindex, "index": index.state()}


class KeysBody(BaseModel):
    keys: list | None = None


@app.post("/api/settings/reset")
def reset_settings(body: KeysBody):
    S.reset(body.keys)
    return {"values": S.values(), "index": index.state()}


@app.get("/api/models")
def get_models():
    try:
        return {"installed": models.installed(), "loaded": models.loaded(), "gpu": models.gpu(),
                "pulls": models.pull_status(), "ollama": True}
    except Exception as e:
        return {"installed": [], "loaded": [], "gpu": models.gpu(), "pulls": models.pull_status(),
                "ollama": False, "error": generation.friendly_error(e)}


class PullBody(BaseModel):
    name: str


@app.post("/api/models/pull")
def pull_model(body: PullBody):
    if not body.name.strip():
        _bad("Enter a model name, e.g. qwen3.5:2b")
    models.pull(body.name)
    return {"ok": True}


# --- Conversations & chat ----------------------------------------------------------
def _turn(row):
    return {k: row[k] for k in ("id", "ts", "question", "answer", "hits", "memory", "metrics", "checks",
                                "settings", "grade", "tags", "reason", "correction")}


@app.get("/api/conversations")
def conversations():
    return storage.list_conversations()


class TitleBody(BaseModel):
    title: str


@app.patch("/api/conversations/{cid}")
def rename(cid: str, body: TitleBody):
    storage.rename_conversation(cid, body.title)
    return storage.get_conversation(cid)


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    storage.delete_conversation(cid)
    return {"ok": True}


@app.get("/api/conversations/{cid}")
def conversation(cid: str):
    conv = storage.get_conversation(cid)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return {**conv, "turns": [_turn(t) for t in storage.conversation_turns(cid)]}


class ChatBody(BaseModel):
    message: str
    conversation_id: str | None = None


@app.post("/api/chat")
def chat(body: ChatBody):
    """Streams newline-delimited JSON events while the answer is produced."""
    message = body.message.strip()
    if not message:
        _bad("Empty message")
    conv = storage.get_conversation(body.conversation_id) if body.conversation_id else None
    if not conv:
        title = " ".join(message.split())
        conv = storage.create_conversation(title[:60] + ("…" if len(title) > 60 else ""))

    def events():
        yield json.dumps({"type": "conversation", "conversation": conv}) + "\n"
        if index.collection().count() == 0:
            yield json.dumps({"type": "error", "message": "Your library is empty. Add documents first."}) + "\n"
            return
        try:
            for ev in generation.answer(message, conversation_id=conv["id"]):
                if ev["type"] == "done":
                    ev = {"type": "done", "turn": _turn(storage.get_interaction(ev["interaction"]))}
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": generation.friendly_error(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# --- Evaluate --------------------------------------------------------------------------
@app.get("/api/interactions")
def interactions(status: str = "all", source: str = "all", search: str = "", run_id: str = "",
                 flagged: bool = False, limit: int = 200, offset: int = 0):
    return feedback.list_interactions(status, source, search.strip(), run_id or None, flagged, limit=limit, offset=offset)


@app.get("/api/interactions/{iid}")
def interaction(iid: str):
    row = storage.get_interaction(iid)
    if not row:
        raise HTTPException(404, "Answer not found")
    run = storage.query("SELECT name, question_set_id FROM runs WHERE id = ?", (row["run_id"],)) if row["run_id"] else []
    conv = storage.get_conversation(row["conversation_id"]) if row["conversation_id"] else None
    expected = None
    if run and row["qid"]:
        qs = experiments.get_set(run[0]["question_set_id"])
        item = next((q for q in (qs or {}).get("items", []) if q["id"] == row["qid"]), None)
        if item:
            expected = {k: item.get(k) for k in ("variant", "answerable", "facts", "note")}
    carried_from = None
    if (row.get("grade_source") or "").startswith("carried:"):
        src = storage.query("SELECT i.id, i.ts, i.origin, r.name AS run_name FROM interactions i "
                            "LEFT JOIN runs r ON r.id = i.run_id WHERE i.id = ?", (row["grade_source"][8:],))
        carried_from = src[0] if src else {"id": row["grade_source"][8:]}
    return {**row, "run_name": run[0]["name"] if run else None, "expected": expected,
            "carried_from": carried_from, "conversation_title": conv["title"] if conv else None}


class GradeBody(BaseModel):
    grade: str | None
    tags: list = []
    reason: str = ""
    correction: str = ""


@app.put("/api/interactions/{iid}/grade")
def grade(iid: str, body: GradeBody):
    try:
        return feedback.save_grade(iid, body.grade, body.tags, body.reason, body.correction)
    except ValueError as e:
        _bad(e)


@app.get("/api/evaluate/stats")
def evaluate_stats():
    return {**feedback.stats(), "tag_vocabulary": feedback.TAGS}


@app.get("/api/export")
def export(only_graded: bool = True, include_text: bool = True, anonymize: bool = True):
    data = feedback.export(only_graded, include_text, anonymize)
    return Response(json.dumps(data, indent=2, ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="localbot-feedback-{storage.now()[:10]}.json"'})


# --- Documents ------------------------------------------------------------------------
@app.get("/api/documents")
def list_documents():
    return {"documents": index.library(), "index": index.state(), "supported": SUPPORTED_TYPES}


@app.post("/api/documents")
def upload(files: list[UploadFile] = File(...)):
    reports, errors = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = os.path.basename(f.filename or "document")
            path = os.path.join(tmp, name)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            try:
                reports.append(index.ingest(path, name))
            except Exception as e:
                errors.append(generation.friendly_error(e) if "ollama" in type(e).__module__ else str(e))
    return {"added": reports, "errors": errors, "index": index.state()}


@app.get("/api/images/{image_id}")
def image(image_id: str):
    path = vision.path_of(image_id)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Picture not found")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.delete("/api/documents/{name}")
def remove_document(name: str):
    index.remove(name)
    return {"ok": True}


@app.get("/api/documents/{name}/chunks")
def document_chunks(name: str):
    return index.chunks_of(name)


@app.post("/api/documents/reindex")
def reindex():
    try:
        reports, errors = index.reindex()
    except Exception as e:
        _bad(generation.friendly_error(e))
    return {"reindexed": reports, "errors": errors, "index": index.state()}


class PreviewBody(BaseModel):
    chunk_size: int
    chunk_overlap: int
    name: str | None = None


@app.post("/api/chunking/preview")
def chunking_preview(body: PreviewBody):
    if body.chunk_overlap >= body.chunk_size:
        _bad("Overlap must be smaller than the chunk size")
    return index.preview(body.chunk_size, body.chunk_overlap, body.name)


# --- Experiments ------------------------------------------------------------------------
@app.get("/api/experiments/meta")
def experiments_meta():
    return {"variants": experiments.VARIANTS, "tags": feedback.TAGS}


@app.get("/api/question-sets")
def question_sets():
    return experiments.list_sets()


class SetBody(BaseModel):
    name: str
    text: str


@app.post("/api/question-sets")
def create_question_set(body: SetBody):
    items, errors = experiments.parse_jsonl(body.text)
    if errors:
        _bad("; ".join(errors[:8]) + (f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""))
    if not items:
        _bad("No questions found")
    return {"id": experiments.create_set(body.name.strip() or "Question set", items), "count": len(items)}


class DraftBody(BaseModel):
    n: int = 10
    name: str = ""


@app.post("/api/question-sets/draft")
def draft_question_set(body: DraftBody):
    try:
        items = experiments.draft_set(max(1, min(body.n, 30)))
    except Exception as e:
        _bad(generation.friendly_error(e))
    if not items:
        _bad("The model couldn't draft usable questions; try again")
    return {"id": experiments.create_set(body.name or f"Drafted {storage.now()[:10]}", items), "count": len(items)}


@app.get("/api/question-sets/{sid}")
def question_set(sid: str):
    qs = experiments.get_set(sid)
    if not qs:
        raise HTTPException(404, "Question set not found")
    return qs


@app.get("/api/question-sets/{sid}/download")
def download_question_set(sid: str):
    qs = experiments.get_set(sid)
    if not qs:
        raise HTTPException(404, "Question set not found")
    text = "\n".join(json.dumps(q, ensure_ascii=False) for q in qs["items"]) + "\n"
    return Response(text, media_type="application/jsonl",
                    headers={"Content-Disposition": f'attachment; filename="{qs["name"]}.jsonl"'})


@app.delete("/api/question-sets/{sid}")
def delete_question_set(sid: str):
    experiments.delete_set(sid)
    return {"ok": True}


class RunBody(BaseModel):
    name: str = ""
    set_id: str
    overrides: dict = {}
    retrieval_only: bool = False
    variants: list | None = None
    limit: int | None = None
    sweep: dict | None = None  # {"key": "top_k", "values": [2, 3, 4]}


@app.post("/api/runs")
def create_run(body: RunBody):
    try:
        if body.sweep and body.sweep.get("values"):
            key = body.sweep["key"]
            ids = [experiments.create_run(f"{body.name or 'sweep'} · {key}={v}", body.set_id,
                                          {**body.overrides, key: v}, body.retrieval_only, body.variants, body.limit)
                   for v in body.sweep["values"]]
        else:
            ids = [experiments.create_run(body.name, body.set_id, body.overrides, body.retrieval_only,
                                          body.variants, body.limit)]
    except ValueError as e:
        _bad(e)
    return {"ids": ids}


@app.get("/api/runs")
def runs():
    return experiments.list_runs()


@app.get("/api/runs/compare")
def compare_runs(ids: str):
    return experiments.compare([i for i in ids.split(",") if i])


@app.get("/api/runs/{rid}")
def run(rid: str):
    r = experiments.get_run(rid)
    if not r:
        raise HTTPException(404, "Run not found")
    return r


@app.post("/api/runs/{rid}/cancel")
def cancel(rid: str):
    experiments.cancel_run(rid)
    return {"ok": True}


@app.delete("/api/runs/{rid}")
def delete_run(rid: str):
    experiments.delete_run(rid)
    return {"ok": True}


# --- Web UI -------------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(WEB_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


@app.exception_handler(Exception)
def unhandled(request, exc):
    return JSONResponse(status_code=500, content={"detail": generation.friendly_error(exc)})


def main():
    import uvicorn
    storage.init()
    experiments.start_worker()
    url = f"http://{HOST}:{PORT}"
    print(f"LocalBot running at {url}  (Ctrl+C to stop)")
    if not os.environ.get("LOCALBOT_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # host=127.0.0.1: reachable from this machine only.
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
