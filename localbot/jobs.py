"""Background jobs for slow library work (adding documents, re-indexing).

A large archive takes minutes to read and embed; running it in the request
would time out the browser and give no progress. Jobs run one at a time on
a single worker thread, so two of them never write the index at once. They
live in memory only: after a restart there is nothing to resume, because a
document's passages are replaced only once all its new ones are stored."""
import itertools
import queue
import threading
import time

_jobs = {}
_queue = queue.Queue()
_ids = itertools.count(1)
_lock = threading.Lock()
_worker = None
KEEP = 20  # finished jobs remembered for the UI


def submit(kind, title, fn):
    """Queue fn(progress) and return the job id. fn returns the result dict;
    progress(done, total, step) updates what the UI shows."""
    global _worker
    with _lock:
        jid = str(next(_ids))
        _jobs[jid] = {"id": jid, "kind": kind, "title": title, "status": "queued", "done": 0, "total": 0,
                      "step": "Waiting for the previous job", "result": None, "error": None,
                      "started": None, "finished": None}
        finished = [j for j in _jobs.values() if j["finished"]]
        for old in sorted(finished, key=lambda j: j["finished"])[:-KEEP]:
            _jobs.pop(old["id"], None)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_work, daemon=True, name="localbot-jobs")
            _worker.start()
    _queue.put((jid, fn))
    return jid


def _work():
    while True:
        jid, fn = _queue.get()
        job = _jobs[jid]

        def progress(done, total, step=None):
            job.update(done=done, total=total, **({"step": step} if step else {}))

        job.update(status="running", started=time.time(), step="Starting")
        try:
            job["result"] = fn(progress)
            job["status"] = "done"
        except Exception as e:
            job.update(status="failed", error=str(e) or type(e).__name__)
        job["finished"] = time.time()


def get(jid):
    return _jobs.get(jid)


def all_jobs():
    return sorted(_jobs.values(), key=lambda j: int(j["id"]), reverse=True)


def active():
    """The running job, else the first queued one, else None."""
    live = [j for j in _jobs.values() if j["status"] in ("running", "queued")]
    return min(live, key=lambda j: (j["status"] != "running", int(j["id"]))) if live else None
