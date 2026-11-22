<div align="center">

# ◈ LocalRAG

**Chat with very large scanned books — Arabic, English, or both — without ever leaving your machine.**

A fully offline desktop RAG app built around Ollama, FastAPI and a small vanilla-JS UI.
No cloud APIs. No telemetry. Your PDFs, OCR text, embeddings and conversations stay in `data/`.

</div>

---

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | PDF import, OCR, checkpoints, review UI, benchmark | ✅ implemented |
| 2 | Chunking, parent–child chunks, FTS5 BM25 + dense bge-m3 indexing | ✅ implemented |
| 3 | Hybrid retrieval, RRF, cross-encoder reranking, debug panel | ✅ implemented + verified |
| 4 | Grounded generation, memory, conversations | ✅ implemented + verified |
| 5 | pywebview shell, PyInstaller, `run.bat` | ✅ implemented + verified |

## Why this exists

Most RAG demos assume clean, digital, Latin-text PDFs. This one is built for the hard case:
500-page books that are mostly **scanned images**, written in **Arabic**, sometimes mixed with
English. That means OCR is a first-class pipeline stage, every page is checkpointed so a crash
never restarts a 500-page job, and the Arabic text is normalized consistently for display and
for search.

## Features (Phase 1)

- **Drag-and-drop PDF import** with automatic page count and script detection (Arabic / English / mixed).
- **Embedded text first** (PyMuPDF); pages with too little text are rasterized at 200–300 DPI.
- **Four pluggable OCR backends**, all local:
  - `paddleocr-vl` — `seriouswebby/paddleocr-vl-1.6` (default: fastest and most faithful on scanned Arabic)
  - `glm-ocr`
  - `deepseek-ocr`
  - Tesseract with `ara+eng` (fallback)
- **OCR benchmark panel** — run every engine side by side on up to 10 pages you choose, with
  per-page timing and text previews; the chosen engine is remembered per book.
- **Durable job queue** — progress, ETA, pause / resume / cancel, retry of failed or low-confidence
  pages, and **one SQLite commit per page**. Killing the process mid-job resumes from the last
  committed page.
- **Arabic text cleanup** — removes tatweel and control characters, repairs hyphenation and broken
  line wraps, NFKC-normalizes presentation forms, keeps the original and a display copy, and
  builds a separate BM25-normalized copy (strip tashkeel, fold `أإآ→ا`, `ى→ي`, `ة→ه`).
- **Bilingual UI** — Arabic by default with an English toggle, dark/light themes, automatic
  RTL/LTR per text block, page image + OCR text preview, and clear errors when Ollama or a model
  is missing.
- **VRAM discipline** — one large model at a time, explicit `keep_alive=0` unloads after each job
  and between benchmark engines to fit an 8 GB GPU.

## Features (Phase 2)

- **Sentence-safe chunking** — ~500-token child chunks with ~15% overlap, Arabic and English
  punctuation (`، ؛ ؟ . !`) respected, small title/blank pages skipped.
- **Parent–child windows** — retrieval searches small children; the larger parent window is linked
  to every child and is what the LLM will read in Phase 4.
- **BM25 / FTS5** over a separate Arabic-normalized copy of each chunk (tashkeel removed,
  `أإآ→ا`, `ى→ي`, `ة→ه`, tatweel removed, digits folded). The same normalization is applied to
  queries, so `الْكِتَابُ` matches `الكتاب`.
- **Dense embeddings** via Ollama `bge-m3` (1024 dimensions), batched and cached by content hash
  in SQLite. Re-indexing is resumable: identical text is never embedded twice.
- **One database file** — `chunks`, `chunks_fts`, `chunk_embeddings`, `embedding_cache` and
  `book_index_status` live alongside books/pages/jobs. Vectors are float32 BLOBs searched with
  NumPy cosine similarity (sqlite-vec has no Python 3.14 Windows wheel; LanceDB would add a large
  native dependency).
- **Indexing API** — enqueue status, chunk browser, BM25 debug search and global stats under `/api`.
- **VRAM discipline** — `LOCALRAG_EMBED_CPU=1` (or `indexing.embed_num_gpu: 0`) forces bge-m3 onto
  the CPU while an OCR job still holds the GPU; the model is explicitly unloaded afterwards.

## Features (Phase 3)

- **Hybrid retrieval** — optional LLM query rewrite, `bge-m3` dense top-50 + FTS5 BM25 top-50,
  Reciprocal Rank Fusion to 40 candidates, then `BAAI/bge-reranker-v2-m3` reranking down to 6–8
  passages. Dense and lexical scores are never compared directly; RRF uses ranks only.
- **Debug payload for every hit** — dense rank, BM25 rank, fused rank, reranker score, book/page,
  the searched child text and its larger parent window. The UI shows all of it in the **Search**
  tab (Arabic/English labels, automatic RTL/LTR per passage).
- **Filters** — all books, the current book, or a page range.
- **Offline cross-encoder** — the reranker is loaded lazily from the Hugging Face cache with
  `local_files_only: true`, and unloaded on CUDA or on app shutdown. CPU is the default because the
  local `torch` build has no CUDA support and OCR may still own the GPU.
- **Eval script** — 20 Q&A pairs over the sample books compare dense-only, BM25-only, hybrid and
  hybrid+rerank using recall@5, recall@8 and MRR.

## Features (Phase 4)

- **Grounded chat with streaming** — `POST /api/chat` streams SSE `delta` events into the Chat tab.
  The prompt contains only retrieved parent windows and confirmed memories; unsupported questions
  get an exact refusal in the question's language. Unresolved citations fail closed, and empty
  model responses are surfaced as errors instead of successful blank answers.
- **Citations** — every claim carries a clickable `[Book Title, p.N]` marker. Clicking it opens the
  page image and OCR text beside the chat with the cited passage highlighted. Combined brackets
  such as `[Book, p.1; Book, p.2]` are normalized into two separate clickable citations.
- **Answer language** — detected from the question (Arabic/English) and enforced in the system
  prompt. A single bounded repair pass fixes models that answer in the source language or omit
  citations. Citations are ignored during language detection. Ollama thinking is disabled by
  default so hidden reasoning does not consume the answer token budget.
- **Memory** — only user-confirmed facts/preferences are stored in the `memories` table, embedded
  with `bge-m3` and retrieved by relevance. The Memory tab supports add/edit/delete; nothing is
  written automatically by the assistant.
- **Conversations** — multiple saved chats, each scoped to chosen books, with a sliding window of
  recent turns and a rolling summary every N messages.
- **Modes** — normal Q&A, map-reduce "Summarize page range", and "Compare across books".
- **Model picker + settings** — the Settings tab lists installed Ollama models and persists model,
  temperature, context budget and history length to `config.local.yaml`. Changing the model
  rebuilds the chat client on the next answer.

## Features (Phase 5)

- **One-command desktop launch** — `run.bat` starts the local Uvicorn server in a background
  thread and opens the same UI in a pywebview (WebView2) window. Closing the window stops the
  server, the ingest/index workers and the single-instance lock.
- **Auto free port** — the desktop shell asks Windows for an unused loopback port by default
  (`desktop.port: 0`) and writes the chosen URL to `data/desktop.url` for diagnostics. Fixed
  port 8765 remains available with `python -m localrag serve`.
- **Single instance** — an OS-held file lock rejects a second copy with exit code 2 instead of
  letting two workers fight over the same SQLite file.
- **Startup preflight** — checks Ollama and the required Ollama models before opening the main
  window. If anything is missing, a bilingual guidance page shows the exact `ollama pull` or
  `ollama serve` command and still links to the app for Settings.
- **Optional system tray** — set `desktop.system_tray: true` to add a show/quit tray icon
  (`pystray` is installed on Windows).
- **Packaging** — `LocalRAG.spec` builds a `dist/LocalRAG/` onedir bundle. Bundled resources are
  read from `_MEIPASS`; user data is written beside `LocalRAG.exe` so it survives upgrades.

The reranker model must be downloaded **once** while online:

```bat
python -m pip install hf_xet
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')"
```

`hf_xet` is only needed for the one-time download and makes the 2.3 GB weight file much faster.
After that, retrieval makes no network calls.

## Requirements

- Windows 64-bit, Python 3.11+
- [Ollama](https://ollama.com/) running locally with these models pulled:

  ```bat
  ollama pull bge-m3
  ollama pull qwen3.5:latest
  ollama pull gemma4:latest
  ollama pull llama3.2:latest
  ollama pull seriouswebby/paddleocr-vl-1.6
  ollama pull glm-ocr
  ollama pull deepseek-ocr
  ```

- The Hugging Face cache containing `BAAI/bge-reranker-v2-m3` (one-time download command above)
- Tesseract OCR with Arabic + English language data (**optional** fallback backend)

## Quick start

```bat
git clone <your-repo-url> LocalRAG
cd LocalRAG
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
REM The default torch wheel may include CUDA; this machine has a CPU build:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
run.bat
```

`run.bat` reuses `.venv` when present (otherwise it uses `python`), installs missing
dependencies once, checks Ollama, picks a free port and opens the desktop window. To run only
the browser server on the fixed port:

```bat
python -m localrag serve
```

Open <http://127.0.0.1:8765/> and drop PDFs onto the folio panel.

Command line equivalents:

```bat
python -m localrag ingest "D:\Books\كتاب عربي.pdf" --engine paddleocr-vl
python -m localrag status
python scripts/resume_ingest.py <book-id> --timeout 7200
python scripts/retry_pages.py <book-id> --statuses failed flagged
```

After a book has been OCR'd, build its search index:

```bat
python -m localrag index <book-id>
python -m localrag search "الكتاب والمكتبة" --limit 5
python -m localrag search "libraries and books" --limit 5
python -m localrag retrieve "الكتاب والمكتبة" --book-ids <book-id> --top-k 8
python -m localrag retrieve "libraries and books" --no-rerank
```

Then open the **Chat** tab, or call the API directly:

```bat
curl -N -X POST http://127.0.0.1:8765/api/chat -H "Content-Type: application/json" ^
  -d "{\"query\":\"ما هو الكتاب؟\",\"language\":\"ar\"}"
```

`resume_ingest.py` reattaches to an existing OCR job after a crash without creating a duplicate
book; it skips every page already committed to SQLite. `index` is resumable and idempotent too.
Add `--force` to rebuild chunks from scratch.
If OCR is still running, keep embeddings off the GPU:

```bat
set LOCALRAG_EMBED_CPU=1
python -m localrag index <book-id>
```

## Configuration

Defaults live in [`config.yaml`](config.yaml). Settings changed in the UI are written to
`config.local.yaml`, which is merged over the defaults and is git-ignored.

| Key | Meaning | Default |
| --- | --- | --- |
| `ingestion.dpi` | Rasterization DPI for scanned pages | `250` |
| `ingestion.min_embedded_chars` | Below this, a page is treated as scanned | `80` |
| `ingestion.low_confidence_threshold` | Pages below this are flagged for review | `45` |
| `ocr.default_engine` | Default OCR backend | `paddleocr-vl` |
| `ocr.tesseract.cmd` | Full path to `tesseract.exe` | `C:/Program Files/Tesseract-OCR/tesseract.exe` |
| `ollama.host` | Ollama endpoint | `http://127.0.0.1:11434` |
| `indexing.target_tokens` | Target child-chunk size | `500` |
| `indexing.overlap_ratio` | Overlap between consecutive children | `0.15` |
| `indexing.parent_multiplier` | Parent window size vs. child target | `3` |
| `indexing.embed_batch_size` | Texts per `/api/embed` call | `16` |
| `indexing.embed_num_gpu` | `null` = auto, `0` = force CPU embeddings | `null` |
| `indexing.vector_backend` | Vector storage implementation | `numpy` |
| `retrieval.dense_top_k` / `retrieval.bm25_top_k` | Candidates per retriever | `50` / `50` |
| `retrieval.rrf_k` / `retrieval.rrf_weights` | RRF smoothing and list weights | `60` / `[1,1]` |
| `retrieval.fused_top_k` / `retrieval.final_top_k` | Rerank input / final passages | `40` / `8` |
| `retrieval.query_rewrite` | LLM rewrite using chat history | `false` |
| `retrieval.reranker.enabled` / `.backend` | Reranker on/off and implementation | `true` / `cross-encoder` |
| `retrieval.reranker.model` / `.device` / `.fp16` | Cross-encoder model and device | `BAAI/bge-reranker-v2-m3` / `cpu` / `false` |
| `retrieval.reranker.local_files_only` | Never contact HF at runtime | `true` |
| `generation.model` | Default chat model (`null` falls back to `ollama.models.llm`) | `null` |
| `generation.temperature` / `.num_predict` | Sampling and max answer length | `0.2` / `1024` |
| `generation.context_tokens` | Ollama `num_ctx` and prompt-budget warning threshold base | `8192` |
| `generation.history_turns` / `.summary_every_messages` | Short-term window / rolling-summary cadence | `6` / `8` |
| `generation.memory_top_k` | Confirmed memories injected per answer | `5` |
| `generation.citations` / `.unload_llm_after_answer` | Citation policy / explicit model unload | `true` / `false` |
| `desktop.port` | Desktop HTTP port (`0` = auto-select a free one) | `0` |
| `desktop.window_width` / `.window_height` | Initial pywebview window size | `1440` / `900` |
| `desktop.system_tray` | Optional show/quit tray icon | `false` |

All runtime state lives under `data/`:

```text
data/localrag.db                 books, pages, jobs, chunks, FTS5 + vector index
data/books/<book-id>/source/      original PDF copy
data/books/<book-id>/pages/       cached page rasters
data/logs/localrag.log            UTF-8 logs
```

## User guide

### Windows quick start

1. Start Ollama and keep it running:

   ```bat
   ollama serve
   ```

2. From the project folder, start the desktop app:

   ```bat
   run.bat
   ```

   To open the browser interface instead, run:

   ```bat
   python -m localrag serve
   ```

   Then open `http://127.0.0.1:8765/`.

### Import and OCR a PDF

1. In the **Library** panel, choose an OCR engine. PaddleOCR VL is the default.
2. Drag a PDF onto the drop zone, or use **Choose PDF files**.
3. The book is added to the library immediately and OCR begins in the background.
4. The card shows the page count, completed pages and current job status. Use
   **Pause**, **Resume**, **Cancel** or **Retry** as needed.
5. Large jobs may take several hours. LocalRAG checkpoints every page in SQLite,
   so restarting the app or killing a process does not require OCR from page one.

### Inspect pages

1. Select a book in the library.
2. Open the **Pages** section.
3. Choose a page number to compare the rasterized page image with its original
   and cleaned OCR text.
4. Failed or low-confidence pages appear with review status. Retry only those
   pages after changing the OCR engine.

### Build the search index

After OCR finishes, open the **Index** section and start indexing. LocalRAG:

- creates small search chunks and larger parent passages;
- builds the Arabic-aware BM25 index;
- embeds text with `bge-m3`;
- resumes incomplete indexing without duplicating cached work.

Indexing is required before retrieval or chat.

### Search

Open **Search**, enter a question or phrase, and optionally choose one book or
a page range. The debug panel shows dense rank, BM25 rank, RRF rank and reranker
score for every returned passage.

### Chat with the books

1. Open **Chat** and create or select a saved conversation.
2. Choose all books or scope the conversation to selected books.
3. Ask in Arabic or English. The answer is written in the question language.
4. Every grounded claim includes a clickable citation such as
   `[Book Title, p.12]`.
5. Clicking a citation opens the cited page image and OCR text beside the chat.

If the sources do not contain the answer, LocalRAG says that it does not know
instead of inventing an answer.

### Memory

Open **Memory** to add, edit or delete user-confirmed facts and preferences.
The assistant does not add memory entries automatically; it only retrieves
facts that you confirmed.

### Settings

Open **Settings** to change:

- the chat and OCR models;
- temperature and answer length;
- context budget and history length;
- reranking device;
- Arabic/English UI and dark/light theme.

Changes are saved to `config.local.yaml`. Defaults remain in `config.yaml`.

### Troubleshooting

- **Ollama is down:** run `ollama serve`, then reopen LocalRAG.
- **A model is missing:** run `ollama pull <model-name>` from the exact command
  shown by the startup guidance page.
- **OCR is unavailable:** keep PaddleOCR VL, GLM OCR or DeepSeek OCR pulled in
  Ollama. Tesseract is an optional fallback and must be installed separately.
- **The PDF is not visible after upload:** confirm the Upload toast appeared,
  open [the log](data/logs/localrag.log), and verify that the job is in
  `queued`, `running` or `done` state.
- **The app cannot keep up:** pause OCR, let it unload, and avoid loading two
  large models at once. The 8 GB GPU configuration prefers CPU reranking.
- **No answers are returned:** finish OCR and indexing, verify that the selected
  books contain text, and confirm that `bge-m3` is installed.

## Packaging (Windows)

```bat
python -m pip install pyinstaller pywebview pystray
pyinstaller --noconfirm --clean LocalRAG.spec
dist\LocalRAG\LocalRAG.exe
```

The build is `onedir` so the large CPU torch/sentence-transformers wheels are not unpacked on
every launch. Copy the whole `dist\LocalRAG` folder to the target machine; `data\` and
`config.local.yaml` are created next to the executable on first run. For a fully built smoke
check, run `python scripts/verify_phase5.py --build`.

## Verification

```bat
python -m pytest
python scripts/verify_phase1.py
python scripts/verify_phase2.py
python scripts/smoke_api_phase2.py
python scripts/smoke_api_phase3.py
python scripts/verify_phase3.py
python scripts/verify_phase4.py
python scripts/verify_phase5.py
python scripts/smoke_desktop_gui.py
python scripts/smoke_packaged_exe.py
python scripts/eval_retrieval.py
```

The verification scripts use a separate `data\verification` root by default, so ingest, indexing,
chat, and desktop smoke checks do not modify the app's normal `data` database. Set
`LOCALRAG_DATA_ROOT` to use another verification directory.

`verify_phase1.py` generates two 20-page image-only PDFs (Arabic and English), runs the test
suite, ingests both books, prints page counts and OCR previews, then kills the ingestion process
mid-job and confirms a restart resumes from the committed pages.

`verify_phase2.py` runs the tests, calls the real `bge-m3` model, indexes both sample books and
prints child/parent counts, token statistics, BM25 hits, cosine-ranked dense hits, cache-resume
numbers and the database size. It forces CPU embeddings by default so it is safe to run while an
OCR job owns the GPU; use `set LOCALRAG_VERIFY_GPU=1` when the GPU is free.

`verify_phase3.py` loads the real cross-encoder, runs Arabic and English hybrid queries, checks
book/page filters, exercises `POST /api/retrieve`, confirms the UI panel exists, and calls the eval
script. `eval_retrieval.py` prints a four-way recall/MRR comparison and caches reranker scores in
the configured data directory so subsequent runs are fast. Both force CPU by default.

`verify_phase4.py` runs the test suite, then uses the real Ollama LLM, `bge-m3` and the cross-encoder
on the sample books. It checks Arabic and English grounded answers, the exact refusal, clickable
citations, summarize/compare modes, confirmed-memory retrieval, SSE chat persistence, memory CRUD,
settings, and the Phase 4 UI controls. Embeddings/reranking stay on CPU; if an OCR job is still
marked running, the LLM is forced to CPU too via `LOCALRAG_LLM_CPU=1`.

`verify_phase5.py` launches the real `run.bat` in headless mode on an auto-selected free port,
checks `/api/health`, `/ui/index.html` and `/api/books`, confirms a second instance is rejected,
and validates the packaging spec. `smoke_desktop_gui.py` briefly opens the real pywebview window
and closes it. Add `--build` to the Phase 5 verifier to also run the full PyInstaller build.

## Project structure

```text
localrag/
  config.py  paths.py  logging_conf.py  cli.py
  models/       domain dataclasses
  db/           SQLite schema, connections, repositories
  text/         Arabic normalization, OCR cleanup, language detection
  ocr/          backend interface, Ollama/Tesseract engines, benchmark, VRAM helpers
  ingest/       PDF reader, persistent job queue, worker, service
  index/        chunker, BM25, embeddings, vector store, indexing service/worker
  retrieval/    RRF fusion, query rewrite, rerankers, hybrid retrieval service
  llm/          Ollama chat client, grounded generation, citations, SSE events
  memory/       confirmed-fact storage and relevance lookup
  api/          FastAPI app, Phase 1–4 endpoints
  ui/static/    HTML / CSS / vanilla JS (no Node build step)
  desktop/      preflight, free-port Uvicorn host, single-instance lock, pywebview/tray
launcher.py     PyInstaller entry point
run.bat         one-command Windows developer launch
LocalRAG.spec   onedir PyInstaller build
tests/          pytest suite + sample-PDF generators
scripts/        end-to-end verification
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Desktop says "already running" | Close the existing LocalRAG window or stop the stale `python -m localrag desktop` process. |
| WebView2 / `webview` window fails to open | Install the Edge WebView2 Runtime and confirm `python -m pip show pywebview`. |
| Red "Ollama is down" pill | Start `ollama serve`. |
| "Model … is not installed" | Run the `ollama pull` command shown in the message. |
| "Tesseract unavailable" | Install Tesseract with `ara`/`eng` traineddata and set `ocr.tesseract.cmd`. The Ollama engines work without it. |
| "CUDA error / device not ready" | Another model is still resident. Wait a few seconds or restart Ollama; jobs unload models between stages. |
| Out of VRAM | Lower `ingestion.dpi` or switch to `paddleocr-vl`; contexts are already bounded in `config.yaml`. |
| Embedding fails while OCR runs | `set LOCALRAG_EMBED_CPU=1` before `localrag index` to keep bge-m3 off the GPU. |
| LLM is slow while OCR runs | `set LOCALRAG_LLM_CPU=1` so the chat model does not compete for the 8 GB GPU. |
| Answer switches to Arabic/English unexpectedly | Ask in the target language; the answer language follows the question, not the source. A one-time repair pass corrects small-model slips. |
| Citation is not clickable | Use one bracket per page, `[Book, p.N]`. Combined page brackets are split automatically during generation. |
| `sqlite_vec` import error | Expected on Python 3.14; Phase 2 stores vectors as float32 BLOBs and searches them with NumPy. |
| Reranker model not found | Run the one-time `snapshot_download('BAAI/bge-reranker-v2-m3')` command; runtime is offline by design. |
| Reranking is slow | Set `retrieval.reranker.backend: ollama` for the fallback scorer, or `retrieval.reranker.enabled: false` to keep fused order. |
| Arabic filenames look wrong | Paths use `pathlib` and logs are UTF-8; make sure the terminal is UTF-8 too. |

## Design notes

The UI follows an "ink & vellum" direction: warm near-black surfaces, saffron ochre accents, a
serif/Naskh display face, and a rotated folio **spine** on each book card. The progress bar fills
like ink across the page. It deliberately avoids a generic dashboard look — this is a reading
desk, not a control panel.

## License

Add your license here (e.g. MIT).
