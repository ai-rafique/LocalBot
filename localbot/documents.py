"""Reading documents into paragraphs, and packing paragraphs into chunks.

Pure functions: no model or database access.
"""
import os
import re

import docx
from pypdf import PdfReader

# Documents are read as paragraphs that keep their page and whether they are
# a heading, so chunks can follow the document's own structure and cite pages.

# "3.2 Frame format", "A.1 Setup", "## Setup". Numbered lines must start with
# a capital and not end like a sentence, so "1. id must be 0." stays a list item.
NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\.?\s+[A-Z(\"'].{0,85}$")
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
BULLET_RE = re.compile(r"^(?:[-•*▪●◦–]|\d+[.)]|[a-z][.)])\s+")
TABLE_ROW_RE = re.compile(r"\S\s{3,}\S|^\s*\|")  # column gaps, or a markdown table row


def _heading_level(line):
    """Heading depth for a numbered heading line, or 0 if it isn't one."""
    line = line.strip()
    if len(line) > 90 or line.endswith((".", ",", ";")) or not NUMBERED_HEADING_RE.match(line):
        return 0
    head = line.split()[0].rstrip(".")
    # "3.2" is depth 2, but a lettered appendix "A.1" is top-level.
    return head.count(".") + (0 if head[0].isalpha() else 1)


def _para(text, page=0, level=0):
    return {"text": text.strip(), "page": page, "level": level}


def _lines_to_paragraphs(lines, page=0, typical_width=80):
    """Rebuild paragraphs from extracted lines (PDF text has no blank lines).

    A paragraph ends at a heading, a list item, a table, or a short line
    ending in sentence punctuation. Table rows stay on their own lines.
    """
    paras, buf, table = [], [], []

    def flush():
        if buf:
            paras.append(_para(" ".join(buf), page))
            buf.clear()
        if table:
            paras.append(_para("\n".join(table), page))
            table.clear()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        level = _heading_level(line)
        if level:
            flush()
            paras.append({**_para(line, page, level), "numbered": True})
            continue
        if TABLE_ROW_RE.search(line):
            if buf:
                flush()
            table.append(line.strip())
            continue
        if table or BULLET_RE.match(line.strip()):
            flush()
        line = line.strip()
        if buf and buf[-1].endswith("-") and line[:1].islower():
            buf[-1] = buf[-1] + line  # re-join a word or identifier broken at the line end
        else:
            buf.append(line)
        if line.endswith((".", "!", "?", ":")) and len(line) < 0.8 * typical_width:
            flush()
    flush()
    return paras


def _markdown_paragraphs(text):
    paras, buf, fence = [], [], None

    def flush():
        if buf:
            paras.extend(_lines_to_paragraphs(buf))
            buf.clear()

    for line in text.splitlines():
        if line.strip().startswith(("```", "~~~")):
            if fence is None:
                flush()
                fence = [line]
            else:
                fence.append(line)
                paras.append(_para("\n".join(fence)))  # a code block is never split mid-way
                fence = None
            continue
        if fence is not None:
            fence.append(line)
            continue
        m = MD_HEADING_RE.match(line.strip())
        if m:
            flush()
            paras.append(_para(m.group(2), level=len(m.group(1))))
            continue
        if not line.strip():
            flush()
        else:
            buf.append(line)
    if fence:
        paras.append(_para("\n".join(fence)))
    flush()
    return paras


def _image(data, page=0):
    """A picture waiting to be read. index.ingest turns it into text by
    transcription; everything here stays free of model calls."""
    return {"text": "", "page": page, "level": 0, "image": data}


def _docx_paragraphs(path):
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(path)
    paras = []
    # Walk the body in order so tables and pictures stay where they appear.
    for el in d.element.body.iterchildren():
        if el.tag.endswith("}p"):
            p = Paragraph(el, d)
            style = (p.style.name if p.style is not None else "") or ""
            level = int(style.split()[-1]) if style.startswith("Heading") and style.split()[-1].isdigit() else 0
            if style == "Title":
                level = 1
            if p.text.strip():
                paras.append(_para(p.text, level=level))
            for blip in el.iter(qn("a:blip")):
                part = d.part.related_parts.get(blip.get(qn("r:embed")))
                if part is not None and getattr(part, "blob", None):
                    paras.append(_image(part.blob))
        elif el.tag.endswith("}tbl"):
            rows = [" | ".join(c.text.strip() for c in row.cells) for row in Table(el, d).rows]
            if rows:
                paras.append(_para("\n".join(rows)))
    return paras


MIN_IMAGE_PT = 40  # smaller PDF pictures are icons and bullets, not content


def _pdf_paragraphs(path):
    """Text, tables and pictures in reading order, page by page.

    Tables are read cell by cell (rows kept whole, values exact), so they
    aren't flattened into running text. Pictures are rendered and read
    later by transcription. Pictures repeated on more than two pages (logos,
    stamps, watermarks) and ones in the top/bottom margins are skipped."""
    import io
    from collections import Counter
    import pdfplumber

    def key(im):  # the same embedded picture object, wherever it's placed
        return (im.get("name"), tuple(im.get("srcsize") or ()), round(im["x1"] - im["x0"]), round(im["bottom"] - im["top"]))

    regions, seen = [], set()
    with pdfplumber.open(path) as pdf:
        pages_with = Counter()
        for page in pdf.pages:
            pages_with.update({key(im) for im in page.images})
        for n, page in enumerate(pdf.pages, 1):
            w, hgt = float(page.width), float(page.height)
            blocks = []
            for t in page.find_tables():
                rows = [" | ".join(" ".join((c or "").split()) for c in row) for row in t.extract()]
                rows = [r for r in rows if r.strip(" |")]
                if rows:
                    blocks.append((t.bbox[1], t.bbox[3], "table", "\n".join(rows)))
            for im in page.images:
                x0, top, x1, bottom = (max(0.0, im["x0"]), max(0.0, im["top"]), min(w, im["x1"]), min(hgt, im["bottom"]))
                k = key(im)
                if (x1 - x0 < MIN_IMAGE_PT or bottom - top < MIN_IMAGE_PT or bottom < 0.08 * hgt or top > 0.92 * hgt
                        or pages_with[k] > 2 or k in seen):
                    continue
                seen.add(k)
                try:
                    buf = io.BytesIO()
                    page.crop((x0, top, x1, bottom)).to_image(resolution=150).original.save(buf, "PNG")
                except Exception:
                    continue
                blocks.append((top, top, "image", buf.getvalue()))
            y = 0.0
            for top, bottom, kind, payload in sorted(blocks, key=lambda b: b[0]):
                if top > y + 1:
                    regions.append((n, "text", page.crop((0, y, w, top)).extract_text() or ""))
                regions.append((n, kind, payload))
                y = max(y, bottom)
            if hgt > y + 1:
                regions.append((n, "text", page.crop((0, y, w, hgt)).extract_text() or ""))
    widths = sorted(len(l) for _, k, t in regions if k == "text" for l in t.splitlines() if l.strip())
    typical = widths[int(0.9 * (len(widths) - 1))] if widths else 80
    paras = []
    for n, kind, payload in regions:
        if kind == "text":
            paras += _lines_to_paragraphs(payload.splitlines(), n, typical)
        elif kind == "table":
            paras.append({**_para(payload, n), "table": True})
        else:
            paras.append(_image(payload, n))
    return paras


def _pdf_paragraphs_basic(path):
    """Plain text extraction, used if the table-aware reader fails."""
    pages = [(i, (p.extract_text() or "").splitlines()) for i, p in enumerate(PdfReader(path).pages, 1)]
    widths = sorted(len(l) for _, lines in pages for l in lines if l.strip())
    typical = widths[int(0.9 * (len(widths) - 1))] if widths else 80
    return [p for i, lines in pages for p in _lines_to_paragraphs(lines, i, typical)]


def _pptx_paragraphs(path):
    """One section per slide: title, text, tables (rows whole), pictures and
    speaker notes. The slide number is used like a page number."""
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    paras = []
    for n, slide in enumerate(Presentation(path).slides, 1):
        title_shape = slide.shapes.title
        title = title_shape.text_frame.text.strip() if title_shape is not None and title_shape.has_text_frame else ""
        paras.append(_para(f"Slide {n}" + (f": {title}" if title else ""), n, 1))

        def walk(shapes):
            for sh in shapes:
                if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                    walk(sh.shapes)
                    continue
                if title_shape is not None and sh.shape_id == title_shape.shape_id:
                    continue
                if sh.has_text_frame:
                    for p in sh.text_frame.paragraphs:
                        text = "".join(r.text for r in p.runs).strip()
                        if text:
                            paras.append(_para(text, n))
                if getattr(sh, "has_table", False):
                    rows = [" | ".join(" ".join(c.text.split()) for c in row.cells) for row in sh.table.rows]
                    paras.append({**_para("\n".join(rows), n), "table": True})
                if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        paras.append(_image(sh.image.blob, n))
                    except Exception:
                        pass  # linked (not embedded) picture

        walk(slide.shapes)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                paras.append(_para("Speaker notes: " + notes, n))
    return paras


MAX_SHEET_ROWS = 5000


def _cell(v):
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return "" if v is None else " ".join(str(v).split())


def _xlsx_paragraphs(path):
    """One section per sheet; every row written with its column names
    ("Port: COM4 · Baud: 115200"), so a row still makes sense on its own
    after the sheet is split into passages. Calculated values, not formulas."""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    paras = []
    try:
        for ws in wb.worksheets:
            header, rows = None, []
            for row in ws.iter_rows(values_only=True):
                cells = [_cell(v) for v in row]
                if not any(cells):
                    continue
                if header is None:
                    header = [c or f"Column {j + 1}" for j, c in enumerate(cells)]
                    continue
                parts = [f"{header[j] if j < len(header) else f'Column {j + 1}'}: {c}" for j, c in enumerate(cells) if c]
                rows.append(_para(" · ".join(parts)))
                if len(rows) >= MAX_SHEET_ROWS:
                    break
            if header is None:
                continue
            paras.append(_para(f"Sheet: {ws.title}", level=1))
            paras += rows or [_para(" · ".join(header))]
    finally:
        wb.close()
    return paras


def _check_heading_numbers(paras):
    """Keep a numbered line as a heading only if it continues the document's
    numbering. Table rows ("2 RESET 0x20 …") and numbered lists inside a
    section ("2. The parser reads past the end") also start with numbers;
    they would otherwise reset the section path."""
    top = 0
    for p in paras:
        if not p.get("numbered"):
            continue
        first = p["text"].split()[0]
        parts = first.rstrip(".").split(".")
        if not parts[0].isdigit():
            continue  # lettered appendix headings: A.1, B.0
        n = int(parts[0])
        if len(parts) == 1:
            ok = first.endswith(".") and n == top + 1  # "3. Title" must follow section 2
            top = n if ok else top
        else:
            ok = n == top  # "3.2 Title" must sit inside section 3
        if not ok:
            p["level"] = 0
    return paras


IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")


def load_paragraphs(file_path: str):
    """List of {"text", "page", "level"}: level > 0 marks a heading.
    Pictures come back as {"image": bytes} placeholders (see _image)."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            paras = _pdf_paragraphs(file_path)
        except Exception:
            paras = _pdf_paragraphs_basic(file_path)
        return _check_heading_numbers(paras)

    if ext == ".docx":
        return _docx_paragraphs(file_path)

    if ext == ".pptx":
        return _pptx_paragraphs(file_path)

    if ext == ".xlsx":
        return _xlsx_paragraphs(file_path)

    if ext in IMAGE_TYPES:
        with open(file_path, "rb") as f:
            return [_image(f.read())]

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return _check_heading_numbers(_markdown_paragraphs(f.read()))

    raise ValueError(f"Unsupported file type: {ext}")


def load_text(file_path: str) -> str:
    return "\n".join(p["text"] for p in load_paragraphs(file_path))


# --- Chunking ----------------------------------------------------------------
def chunk_text(text: str, chunk_size, overlap):
    """Split into ~chunk_size-character chunks, overlapping by ~overlap.

    Cuts are snapped to word boundaries so no chunk starts or ends mid-word
    (half-words embed poorly), and the last chunk is never a pure repeat of
    the tail of the previous one. Used for paragraphs too long for one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    text = " ".join(text.split())  # normalize whitespace
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return [c for c in chunks if c]


def _split_long(text, chunk_size, overlap):
    """A paragraph bigger than a chunk: tables and code are split between
    lines (rows stay whole), prose between words."""
    if "\n" not in text:
        return chunk_text(text, chunk_size, overlap)
    pieces, cur = [], ""
    for line in text.split("\n"):
        if len(line) > chunk_size:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces += chunk_text(line, chunk_size, overlap)
        elif cur and len(cur) + len(line) + 1 > chunk_size:
            pieces.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    return pieces + ([cur] if cur else [])


def chunk_paragraphs(paras, chunk_size, overlap):
    """Pack whole paragraphs into chunks of up to ~chunk_size characters.

    A new section starts a new chunk (unless the current one is still tiny),
    so a chunk rarely mixes topics. Each chunk records its section path
    ("3 Protocol > 3.2 Frame format"), pages, and the ids of pictures its
    text was transcribed from. Overlap is carried only within a section.
    Returns dicts: text, section, page_start, page_end, images.
    """
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    chunks, cur, carry, stack = [], [], 0, []
    sections = []

    def section_path():
        return " > ".join(t for _, t in stack)

    def flush(keep_overlap):
        nonlocal cur, carry, sections
        text = "\n".join(t for t, *_ in cur).strip()
        pages = [p for _, p, _ in cur if p]
        if content() > 0:
            chunks.append({
                "text": text,
                "section": " | ".join(dict.fromkeys(s for s in sections if s)),
                "page_start": min(pages) if pages else 0,
                "page_end": max(pages) if pages else 0,
                "images": list(dict.fromkeys(i for *_, i in cur if i)),
            })
        cur, carry, sections = [], 0, [section_path()]
        if keep_overlap and overlap and text:
            tail = text[-overlap:]
            space = tail.find(" ")
            tail = tail[space + 1:] if 0 <= space < len(tail) - 1 else tail
            cur, carry = [(tail, pages[-1] if pages else 0, None)], len(tail)

    def size():
        return sum(len(t) + 1 for t, *_ in cur)

    def content():  # characters that aren't overlap from the previous chunk
        return sum(len(t) for t, *_ in cur) - carry

    for p in paras:
        text, page, level = p["text"], p["page"], p["level"]
        if not text:
            continue
        if level:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text[:80]))
            if content() >= chunk_size // 4:
                flush(keep_overlap=False)
            elif carry:
                cur, carry = cur[1:], 0  # overlap text belongs to the previous section
            sections.append(section_path())
            cur.append((text, page, None))
            continue
        if not sections:
            sections.append(section_path())
        pieces = _split_long(text, chunk_size, overlap) if len(text) > chunk_size else [text]
        for piece in pieces:
            if cur and size() + len(piece) > chunk_size:
                if content() > 0:
                    flush(keep_overlap=True)
                if cur and size() + len(piece) > chunk_size:
                    cur, carry = [], 0  # the overlap doesn't fit beside this piece
            cur.append((piece, page, p.get("image_id")))
    if cur:
        flush(keep_overlap=False)
    return chunks


def chunk_header(meta):
    """Title and section line put in front of a chunk for embedding and
    keyword search, so "the baud rate" can find a chunk under "UART setup"."""
    title = os.path.splitext(meta.get("source", ""))[0].replace("_", " ")
    return f"{title} — {meta['section']}" if meta.get("section") else title


def page_range(meta):
    """"4" or "4–5" for a chunk's pages ("" if unknown)."""
    a, b = meta.get("page_start") or 0, meta.get("page_end") or 0
    return "" if not a else (f"{a}" if a == b else f"{a}–{b}")


def passage_label(h):
    """"file.pdf, p. 4 — 3.2 Frame format": where a passage came from."""
    pages = page_range(h)
    return h["source"] + (f", p. {pages}" if pages else "") + (f" — {h['section']}" if h.get("section") else "")
