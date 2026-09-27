# LocalBot

Private question answering over your own documents. Add PDF, Word,
Markdown or text files, ask questions in a chat, and get answers grounded
in those documents with citations you can check. Everything runs on your
computer: the models (through Ollama), the search index and all data.

Beyond chatting, LocalBot is a workbench for making the answers better:
grade answers and say why, tune every setting with a live explanation of
its effect, and run experiments that measure whether a change helped.

Built and tuned on a laptop with a 4 GB GPU; works with any model Ollama
can run.

**Three things make it more than a chat box:**

- **Model loading made visible** — pick, install and switch models from
  the UI, see whether each fits your GPU and how much of it is actually on
  the GPU, and keep the whole model in VRAM on small cards.
- **A proper front end** — chat with streamed, cited answers; a details
  panel with every passage, score, timing and the exact prompt; a grading
  console; settings that tell you what each value will do.
- **An experiment area** — fixed question sets, runs with changed
  settings, automatic scoring plus your grades, and side-by-side
  comparison, so tuning is measured instead of guessed.

## Requirements

LocalBot does not include a model. You need:

| | What | Why |
|---|---|---|
| **Ollama** | [ollama.com](https://ollama.com) (Windows, macOS, Linux) | Runs the models locally |
| **A chat model** | e.g. `qwen3.5:2b` | Reads the passages and writes answers (also reranks passages) |
| **An embedding model** | `nomic-embed-text` (recommended) | Turns passages and questions into vectors for search |
| **Python** | 3.10 or newer | Runs LocalBot |

A GPU is optional but makes answers much faster. Any NVIDIA GPU with
about 3 GB free runs the default setup entirely on the GPU; without one,
Ollama uses the CPU.

## Tested setup

Developed and measured on:

| | |
|---|---|
| Hardware/Software | Description |
| OS | Windows 11 (build 26200) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM (driver 616.56) |
| Ollama | 0.34 |
| Python | 3.12 |
| Chat model | `qwen3.5:2b` — 2.3 B parameters, Q8_0, 2.7 GB |
| Embedding model | `nomic-embed-text` — 274 MB |
| Documents | two private technical specifications (~80 passages) |

### How well it works

Measured with the built-in experiment tools on that setup, with answers
graded by hand:

| | Result |
|---|---|
| Speed | ~50 tokens/s, fully on the GPU (both models together use ~3.1 GB); 4–8 s per answer including reranking |
| Finding the right passage | ranked first for 26 of 30 test questions (16 of 30 before structure-aware chunking, keyword search and reranking) |
| Answer quality, 100 graded questions | 54 good · 24 partial · 22 bad |
| Direct lookups | 80% good; small calculations from documented values: 5 of 5 |
| Questions the documents don't cover | 5 of 8 correctly refused; one made-up answer |
| Weak spots | multi-step questions (2 of 10 good), questions with a false premise (0 of 5), exact byte sequences from dense tables |

What that means in practice: a 2 B model on a 4 GB laptop GPU is a solid
**finder and first reader** — it reliably finds where something is
documented and answers direct questions with checkable citations. It is
not yet reliable for multi-step reasoning or for copying long exact values;
the cited sources and the warnings under each answer are there so you can
check. In 34 of the 43 imperfect answers the right text *was* in front of
the model, so a stronger model is the biggest remaining lever — the
experiment area is built to measure exactly that.

For comparison, `qwen3.5:0.8b` runs about 1.6× faster (~80 tokens/s) on
the same GPU but reads passages less accurately.

## Quick start

1. **Install Ollama** from <https://ollama.com> and pull a chat model and
   an embedding model:

   ```bash
   ollama pull qwen3.5:2b
   ollama pull nomic-embed-text
   ```

   On a small GPU, choose a chat model that fits in its memory (Settings
   shows this for every installed model).

2. **Install LocalBot** (Python 3.10+):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run it**:

   ```bash
   python app.py
   ```

   Your browser opens <http://127.0.0.1:7860>. Go to **Documents**, add
   files, then ask away in **Chat**.

## The five screens

### Chat
A familiar chat: conversations in the sidebar, answers streamed as they're
written. Every answer cites its sources as numbered chips; click one (or a
`[2]` in the text) to open the passage. Warnings appear under an answer
when something looks off: a number or identifier that isn't in any
source, a citation to a source that doesn't exist, no citation at all, or
low model confidence.

- 👍 / 👎 grade an answer on the spot; the grade icon opens a detailed
  grade (partial, what went wrong, why, the correct answer).
- The **details panel** (right) shows every retrieved passage with its
  score and whether it was sent, a timing breakdown, and the exact prompt
  the model saw.
- The settings chip under the message box switches presets (Accurate /
  Balanced / Fast), the chat model, reranking and passages per answer.

### Documents
Your library. Drop files anywhere on the page to add them. Each document
shows its passages, sections and pages; **View passages** shows exactly
what the model can read.

The **How documents are split** panel previews new chunk settings live —
passage count, sizes, sample passages — before you commit. **Apply and
re-index** rebuilds the index; originals are kept, so this never needs
the files again.

### Evaluate
A grading console for every answer, from chat and experiments. Filter to
what still needs grading, read the question, answer, sources and prompt,
then grade:

- **Good**, **Partial** or **Bad** (keys `1` `2` `3`)
- **What went wrong** — wrong passages, false statement, incomplete,
  missed answer, accepted a false premise, should have refused, citation
  problem, rambling
- **Why** — free text; this is what tuning decisions rest on
- **Correct answer** (optional)

`Ctrl+Enter` saves and moves to the next one; `J`/`K` move between
answers. Good or corrected chat answers are **reused**: when a nearly
identical question comes up, your verified answer is given to the model
alongside the passages. **Export** writes grades and tuning signals to a
JSON file; passage text and prompts are never included.

### Experiments
Measure a change instead of guessing. A **question set** is a list of
questions with the facts a correct answer must contain. A **run** answers
all of them with some settings changed — or compares several values of
one setting — and reports:

| Metric | Meaning |
|---|---|
| Graded good | Share of graded answers you marked good — the most trustworthy number |
| Auto-correct | Answers containing every expected fact (can't see wrong extra claims) |
| Context recall | The passages sent contained the expected facts — measures search |
| Misread passages | Facts were in the passages but not in the answer — measures the model |
| Refused when it should / Wrongly refused | Behaviour on questions the documents can't answer, and on ones they can |
| Flagged terms | Answers with numbers or identifiers found in no source |

Runs go one at a time in the background. **Passages only** runs skip
answer writing and take seconds — ideal for tuning search, the relevance
cutoff and chunking. Changing chunk settings in a run builds a temporary
index; your library is untouched. Grade a run's answers from its page,
then **Compare** runs side by side (green = better).

Question sets are JSON Lines files:

```json
{"id": "q1", "variant": "lookup", "question": "What baud rate does the service port use?", "facts": [["115200"]]}
{"id": "q2", "variant": "exact", "question": "Which bytes reset the device?", "facts": [["aa 01 00 ff"]]}
{"id": "q3", "variant": "unanswerable", "question": "Who made the enclosure?", "answerable": false, "facts": []}
```

Each inner list in `facts` is one required fact; any spelling in it
counts (case, spaces and separators are ignored). Variants — `lookup`,
`exact`, `paraphrase`, `multihop`, `crossdoc`, `numeric`, `procedure`,
`false_premise`, `unanswerable`, `terse` — show where a change helps or
hurts. No set yet? **Draft from documents** writes questions from random
passages (easy ones; add harder ones yourself).

Experiments also run from the command line:

```bash
python -m localbot.cli sets
python -m localbot.cli add-set my-questions.jsonl --name "My set"
python -m localbot.cli run "My set" --set top_k=3
python -m localbot.cli run "My set" --sweep min_score_reranked=0.3,0.5,0.7 --passages-only
python -m localbot.cli report <run>
```

### Settings
Every tunable parameter, grouped (Models, Retrieval, Answering, Chunking,
Learning), each with a plain explanation and a **live line showing what
the chosen value will do** — computed from your own data: how many
characters the model will read, how long reranking will take, what share
of recent passages would pass a cutoff, whether a model fits your GPU.
Nothing applies until you save; settings that need re-indexing say so.
**Compare values in an experiment** under any setting sets up a sweep.

## Adding a model

Settings → Models → **Add a model**: type a name from
<https://ollama.com/library> (for example `qwen3.5:4b` or `llama3.2:3b`)
and press Install; or run `ollama pull <name>` yourself. Then choose it as
the chat model.

Choosing: instruction-tuned models that fit entirely in your GPU memory
are fastest (Settings shows fit for each installed model). To decide
between two, run the same question set with each (`llm_model` in an
experiment) and compare. After switching models, recalibrate the
low-confidence warning: Settings shows how your graded answers score.

Changing the **embedding** model changes how passages are indexed; you'll
be asked to re-index.

## Tuning in practice

1. Add documents and a question set (draft one, then add real questions).
2. Run it with current settings and grade the answers: that's your
   baseline.
3. Change one thing — Settings shows what to expect — and run again. Use
   passages-only runs for search and chunking changes.
4. Compare. Keep a change when it helps without hurting the
   `unanswerable` and `false_premise` variants.

Temperature 0 makes answers as repeatable as possible, but not perfectly:
GPU arithmetic isn't bit-exact, so small differences in scores and wording
appear between identical runs. Two runs with identical settings on a
100-question set differed by 1–2 points overall and up to one question per
variant; treat differences that small as noise. Run your baseline twice
to see the noise level for your own setup.

## Your data and privacy

Everything LocalBot stores is in the `data/` folder:

| Path | Contents |
|---|---|
| `data/documents/` | Copies of the files you added |
| `data/chroma/` | The search index |
| `data/localbot.db` | Conversations, every answer with its prompt and sources, grades, question sets, runs |
| `data/settings.json` | Settings you changed from the defaults |

- The server listens on `127.0.0.1` only: not reachable from other
  machines, no public link.
- The web UI loads nothing from the internet (system fonts, no CDNs).
- Chroma's anonymous telemetry is switched off.
- The only outside connection is downloading a model when you click
  Install (or run `ollama pull`) — handled by Ollama.

Back up or move LocalBot by copying `data/`. Set `LOCALBOT_DATA` to keep
it elsewhere; `LOCALBOT_PORT` changes the port.

## For developers

```
app.py                  entry point (python app.py)
localbot/
  config.py             paths, settings schema (types, ranges, help, effects), presets
  documents.py          reading files into paragraphs; structure-aware chunking
  index.py              library, embeddings, BM25 keyword index, temporary indexes
  retrieval.py          hybrid search, reciprocal rank fusion, LLM reranking
  generation.py         prompt, streaming answer, confidence, faithfulness checks
  feedback.py           grading, reuse of graded answers, stats, export
  experiments.py        question sets, background runs, scoring, comparisons
  models.py             Ollama models, GPU, installs
  storage.py            SQLite schema and helpers
  server.py             HTTP API + static UI (FastAPI)
  cli.py                experiments from the command line
web/                    the UI: plain HTML, CSS and JavaScript modules, no build step
tests/                  unit tests for the pure logic
```

```bash
python -m unittest discover -s tests
```

New settings go in `SCHEMA` in `config.py` (the UI renders them
automatically; add an effect line in `web/js/tuning.js`). Code reads
settings at call time (`S.top_k`), and experiments override them per
thread, so a background run never changes what the chat uses.

## Troubleshooting

- **"Ollama isn't running"** — start it (`ollama serve`, or the Ollama app).
- **Model "not installed"** — install it in Settings → Models, or `ollama pull <name>`.
- **Slow answers, sidebar shows "x% on GPU"** — the model doesn't fully
  fit in GPU memory. Pick a smaller model, reduce the context window, or
  close other GPU-heavy apps.
- **"Settings changed since the index was built"** — re-index in Documents.
- **Scanned PDFs add no passages** — they contain images, not text; run
  OCR on them first.
