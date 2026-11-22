# Task: Build "LocalRAG" — a fully offline desktop RAG app (Ollama + FastAPI)

Build a Windows desktop application that lets me chat with very large PDF books (up to ~500 pages, **mostly scanned**, Arabic and English, sometimes mixed). Everything runs locally. Work in the phases below. After each phase, run the verification step and fix failures before moving on. Do not claim something works until you have run it.

## Target machine
- Windows 64-bit, i7-10750H, 32 GB RAM, RTX 2070 Super 8 GB VRAM
- Ollama is installed. Models available locally: `bge-m3` (embeddings), `qwen3.5:latest` (6.6 GB, default LLM), `gemma4:latest` (9.6 GB, alt LLM), `llama3.2:latest` (2 GB, fast/small tasks), OCR: `glm-ocr`, `deepseek-ocr`, `seriouswebby/paddleocr-vl-1.6`
- 8 GB VRAM means **only one big model at a time**. Use `keep_alive` and explicit unloading between pipeline stages (OCR → embedding → rerank → LLM). Never assume two large models fit together.

## Stack
- Python 3.11+, FastAPI + Uvicorn backend, async where useful
- Desktop shell: **pywebview** (WebView2) opening the local FastAPI app; package later with PyInstaller. Frontend: static HTML/CSS/vanilla JS (no Node build step), **auto RTL/LTR per message**, Arabic UI by default with English toggle, clean dark/light theme
- Storage: one SQLite file (`data/localrag.db`) using **FTS5 for BM25** and **sqlite-vec for vectors** (fallback: LanceDB if sqlite-vec fails on Windows). Original PDFs, page images and OCR text cached under `data/`
- Config in `config.yaml` (models, chunk sizes, top-k values, paths), overridable from a Settings screen

## Phase 1 — Ingestion + OCR (the hard part)
1. Import PDF(s) via drag-and-drop. Create a "Book" record (title, language auto-detected, page count).
2. Per page: first try embedded text (PyMuPDF). If the page has little or no text, rasterize at ~200–300 DPI and OCR it.
3. **Pluggable OCR engine** interface with these backends: Ollama `glm-ocr`, `deepseek-ocr`, `paddleocr-vl`, and Tesseract (`ara+eng`) as fallback. Add a "Benchmark OCR" tool: run all backends on 10 sample pages I choose and show side-by-side text plus time per page, so I pick the best for Arabic. Remember the choice per book.
4. Job queue with background worker: progress bar (page X/N, ETA), pause/resume/cancel, **checkpoint every page in SQLite so a crash or restart resumes where it stopped** (500 pages must never restart from zero). Retry failed pages, and mark pages with low-confidence or empty OCR so I can re-run just those.
5. Arabic text cleanup after OCR: remove tatweel and control characters, fix broken line wraps and hyphenation, normalize presentation-form letters (NFKC), keep the original text and a normalized copy (normalized one is used for BM25 only).
6. Verification: ingest a sample 20-page scanned Arabic PDF and a 20-page English PDF. Show page counts, OCR text previews, and confirm resume works by killing the process midway.

## Phase 2 — Chunking + indexing
- Page-aware, structure-aware chunking: ~400–600 tokens, ~15% overlap, never split mid-sentence (handle Arabic punctuation `، ؛ ؟` and `.`). Store metadata: book, page range, chapter/heading if detectable.
- **Parent–child chunks**: search on small chunks, return the larger parent window to the LLM.
- Dense embeddings with Ollama `bge-m3` (multilingual, handles Arabic and cross-lingual queries). Batch requests, cache by content hash, resumable.
- BM25 over the normalized text with FTS5. Arabic normalization: strip tashkeel, normalize alef variants (أإآ→ا), ى→ي, ة→ه (for the index only), remove tatweel. Apply the same normalization to queries.
- Verification: index the sample books and print chunk counts and index size.

## Phase 3 — Hybrid retrieval + reranking
1. Query pipeline: (optional) LLM query rewrite using chat history → dense top-50 + BM25 top-50 → **Reciprocal Rank Fusion** → top-40.
2. **Reranker**: cross-encoder `BAAI/bge-reranker-v2-m3` (multilingual) via sentence-transformers or FlagEmbedding, fp16, on GPU only while reranking and then released (or on CPU if VRAM is short — make it a setting). Rerank 40 → keep top 6–8 with scores.
3. Filters: all books / selected books / single book / page range.
4. Add a **Search debug panel**: for any answer, show dense rank, BM25 rank, fused rank, reranker score per chunk. Include a small eval script (20 Q&A pairs from my sample books, recall@k and MRR) to compare dense-only vs BM25-only vs hybrid vs hybrid+rerank.

## Phase 4 — Generation + memory
- Default LLM `qwen3.5:latest` through Ollama's API with streaming (SSE to the UI). Model picker in Settings (gemma4, llama3.2, and any installed model). Context budget configurable; warn if the context exceeds what fits in VRAM.
- Answer in the **language of the question**, even when the sources are in the other language. Strictly grounded: if the sources do not contain the answer, say so. Every claim gets citations `[Book, p.N]`; clicking a citation opens the page image/text beside the chat with the cited passage highlighted.
- **Memory**:
  - Short-term: sliding window of recent turns + rolling summary stored per conversation
  - Long-term: user-confirmed facts/preferences stored in a `memories` table (embedded and retrieved when relevant), viewable/editable/deletable in a Memory screen
  - Multiple saved conversations, each can be scoped to chosen books
- Extra modes: "Summarize chapter/page range" (map-reduce over chunks) and "Compare across books".

## Phase 5 — Desktop packaging
- `pywebview` window, system-tray optional, single-instance lock, picks a free port, starts/stops FastAPI cleanly, checks Ollama is running and required models are pulled (offer `ollama pull` guidance if missing)
- `run.bat` for dev, and a PyInstaller spec for an installable build
- README with setup steps (Windows), model list, and troubleshooting

## Non-negotiables
- 100% offline after models are downloaded; no telemetry, no cloud APIs
- Unicode-safe everywhere (Arabic filenames, paths with spaces, UTF-8 logs)
- Clear error messages in the UI (Ollama down, model missing, OCR failure, out of VRAM)
- Type hints, small modules (`ingest/`, `ocr/`, `index/`, `retrieval/`, `llm/`, `memory/`, `api/`, `ui/`), pytest tests for chunking, Arabic normalization, RRF, and resume logic
- Before finishing each phase: actually run the app/tests, paste the real output, and fix errors. The final result must launch with `run.bat` and work end to end on my sample PDFs, not just pass static review.

Start by proposing the folder structure and a short plan, then begin Phase 1.