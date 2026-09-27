"""Ollama models and the GPU: what's installed, what's loaded, what fits."""
import subprocess
import threading

import ollama

from .config import S

_capabilities = {}      # model -> list of capabilities (cached; `ollama show` is slow-ish)
_pulls = {}             # model -> progress dict for installs started from the UI
_pull_lock = threading.Lock()


def gpu():
    """Total/used VRAM in MB from nvidia-smi, or None without an NVIDIA GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        name, total, used = [x.strip() for x in out[0].split(",")]
        return {"name": name, "total_mb": int(total), "used_mb": int(used)}
    except Exception:
        return None


def _caps(name):
    if name not in _capabilities:
        try:
            _capabilities[name] = list(ollama.show(name).capabilities or [])
        except Exception:
            _capabilities[name] = []
    return _capabilities[name]


def installed():
    """Installed models with size, details and role (chat / embedding)."""
    g = gpu()
    out = []
    for m in ollama.list().models:
        caps = _caps(m.model)
        d = m.details
        size_gb = (m.size or 0) / 1e9
        role = "embedding" if "embedding" in caps else "chat"
        # Weights plus roughly 0.5 GB for the context cache and runtime.
        need_gb = size_gb + (0.5 if role == "chat" else 0.1)
        out.append({
            "name": m.model, "size_gb": round(size_gb, 2), "role": role, "capabilities": caps,
            "family": getattr(d, "family", None), "parameters": getattr(d, "parameter_size", None),
            "quantization": getattr(d, "quantization_level", None),
            "fits_gpu": None if not g else need_gb * 1024 <= g["total_mb"] * 0.95,
        })
    return sorted(out, key=lambda x: (x["role"], x["size_gb"]))


def loaded():
    return [{"name": m.model, "size_gb": round((m.size or 0) / 1e9, 2),
             "gpu_share": round((m.size_vram or 0) / m.size, 2) if m.size else None,
             "context": getattr(m, "context_length", None)} for m in ollama.ps().models]


def status():
    try:
        names = {m["name"] for m in installed()}
        up = True
    except Exception:
        names, up = set(), False

    def present(name):
        return bool(name) and (name in names or f"{name}:latest" in names)

    return {
        "ollama": up,
        "chat_model": S.llm_model, "chat_model_installed": present(S.llm_model),
        "embed_model": S.embed_model, "embed_model_installed": present(S.embed_model),
        "loaded": loaded() if up else [], "gpu": gpu(),
    }


def pull(name):
    """Install a model in the background. This downloads from the Ollama
    registry — the one kind of network request the app makes, and only
    when the user asks for it. Progress: see pull_status()."""
    name = name.strip()
    with _pull_lock:
        if _pulls.get(name, {}).get("state") == "running":
            return
        _pulls[name] = {"state": "running", "status": "starting", "completed": 0, "total": 0}

    def work():
        try:
            for p in ollama.pull(name, stream=True):
                _pulls[name].update(status=p.status or "", completed=p.completed or _pulls[name]["completed"],
                                    total=p.total or _pulls[name]["total"])
            _pulls[name]["state"] = "done"
            _capabilities.pop(name, None)
        except Exception as e:
            _pulls[name].update(state="error", status=str(e))

    threading.Thread(target=work, daemon=True).start()


def pull_status():
    return dict(_pulls)
