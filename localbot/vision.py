"""Reading pictures: the picture-reader model (S.vision_model) transcribes the text in a picture and
briefly describes what isn't text, so images in documents become
searchable. Transcripts are cached by picture content, so re-indexing
doesn't read the same picture twice. Everything stays in DATA_DIR."""
import hashlib
import io
import os
import re

import ollama
from PIL import Image

from . import models, retrieval
from .config import IMAGES_DIR, S

PROMPT = (
    "Transcribe all text in this image exactly as written, keeping line breaks and table rows "
    "(separate table cells with ' | '). Then, if the image shows a diagram, chart, screenshot or "
    "photo, add one line starting with 'Description:' saying what it shows. Output nothing else."
)
MAX_SIDE = 1600   # larger pictures are scaled down; small print stays legible at this size
MIN_SIDE = 48     # smaller ones are icons, bullets or spacers
ID_RE = re.compile(r"[0-9a-f]{16}")


def store(data):
    """Normalise a picture to PNG, save it for thumbnails, return its id —
    or None if it's too small to be content or can't be decoded."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    if min(img.size) < MIN_SIDE:
        return None
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")  # transparent areas read as white, not black
        bg.paste(img, mask=img.split()[-1])
        img = bg
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    png = buf.getvalue()
    image_id = hashlib.sha1(png).hexdigest()[:16]
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, f"{image_id}.png")
    if not os.path.isfile(path):
        with open(path, "wb") as f:
            f.write(png)
    return image_id


def path_of(image_id):
    return os.path.join(IMAGES_DIR, f"{image_id}.png") if ID_RE.fullmatch(image_id or "") else None


def _cache_path(image_id):
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", S.vision_model)
    return os.path.join(IMAGES_DIR, f"{image_id}.{model}.txt")


def cached(image_id):
    try:
        with open(_cache_path(image_id), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def can_read():
    """Whether the picture reader accepts images (e.g. qwen3.5 does)."""
    return "vision" in models._caps(S.vision_model)


def _ask(png, extra):
    r = ollama.chat(model=S.vision_model, think=False, keep_alive=S.keep_alive,
                    options={**retrieval.gen_options(), "num_predict": 1024, **extra},
                    messages=[{"role": "user", "content": PROMPT, "images": [png]}])
    return re.sub(r"<think>.*?(</think>|$)", "", r.message.content or "", flags=re.S).strip()


def transcribe(image_id):
    """The picture's transcript ("" if it couldn't be read). Results,
    including failures, are cached so re-indexing doesn't retry them."""
    text = cached(image_id)
    if text is not None:
        return text
    with open(path_of(image_id), "rb") as f:
        png = f.read()
    try:
        text = _ask(png, {})
    except ollama.ResponseError as e:
        if "repeat" not in str(e).lower():
            raise
        # Grids and tables can make a small model loop until Ollama aborts;
        # one retry with a stronger repeat penalty and a shorter limit.
        try:
            text = _ask(png, {"repeat_penalty": 1.3, "num_predict": 512})
        except ollama.ResponseError:
            text = ""
    with open(_cache_path(image_id), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def remove_unused(keep_ids):
    """Delete stored pictures and transcripts no passage refers to any more
    (after a document is removed, its pictures shouldn't linger)."""
    if not os.path.isdir(IMAGES_DIR):
        return
    for name in os.listdir(IMAGES_DIR):
        if name[:16] not in keep_ids:
            try:
                os.remove(os.path.join(IMAGES_DIR, name))
            except OSError:
                pass
