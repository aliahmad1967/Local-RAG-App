"""Real end-to-end verification for Phase 3: hybrid retrieval + reranking.

Runs pytest, loads the real cross-encoder once for a smoke score, then exercises
dense + BM25 + RRF + rerank through the service and HTTP API on the Phase 2
sample books. Embeddings and reranking default to CPU so an OCR job can keep the
GPU. Set LOCALRAG_VERIFY_GPU=1 only when the GPU is free.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import ensure_data_dirs, load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.db.repositories import BooksRepo, IndexStatusRepo  # noqa: E402
from localrag.index.service import IndexService  # noqa: E402
from localrag.logging_conf import setup_logging  # noqa: E402
from localrag.ocr.vram import ollama_status  # noqa: E402
from localrag.retrieval.query_rewrite import NoopRewriter  # noqa: E402
from localrag.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from localrag.retrieval.service import RetrievalFilters, RetrievalService  # noqa: E402
from verification_env import isolate_verification_data  # noqa: E402


def heading(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT), text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def find_book(db: Database, file_name: str) -> Any:
    return next((book for book in BooksRepo.all(db) if book.file_name == file_name), None)


def print_hits(hits: list[Any], limit: int = 5) -> None:
    for index, hit in enumerate(hits[:limit], start=1):
        print(
            f"  {index}. dense={hit.dense_rank} bm25={hit.bm25_rank} "
            f"fused={hit.fused_rank} rerank={hit.rerank_rank} "
            f"score={hit.rerank_score} {hit.book_title} p.{hit.page_start}-{hit.page_end}",
            flush=True,
        )
        print(f"     {' '.join(hit.text.split())[:150]}", flush=True)


def main() -> int:
    isolate_verification_data(ROOT)
    if os.getenv("LOCALRAG_VERIFY_GPU") != "1":
        os.environ["LOCALRAG_EMBED_CPU"] = "1"

    cfg = load_config()
    ensure_data_dirs(cfg)
    setup_logging(cfg)
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    failures: list[str] = []

    heading("1. ENVIRONMENT")
    status = ollama_status(cfg)
    embed_model = str(cfg.get("ollama", "models", "embedding", default="bge-m3"))
    model_names = [str(name) for name in status.get("models", [])]
    reranker_model = str(
        cfg.get("retrieval", "reranker", "model", default="BAAI/bge-reranker-v2-m3")
    )
    hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    reranker_dir = hub_cache / f"models--{reranker_model.replace('/', '--')}"
    weight_files = ("model.safetensors", "pytorch_model.bin")
    weights = [
        path
        for path in reranker_dir.glob("snapshots/*/*")
        if path.name in weight_files and path.is_file()
    ]
    weight_bytes = sum(path.stat().st_size for path in weights)
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    print(f"python          : {sys.version.split()[0]}")
    print(f"ollama          : {'running' if status['running'] else 'DOWN'} @ {status['host']}")
    print(f"embedding model : {embed_model} installed={any(embed_model.split(':')[0] in n for n in model_names)}")
    print(
        f"reranker model  : {reranker_model} cached={reranker_dir.exists()} "
        f"weights={weight_bytes / (1024 * 1024):.1f}MB"
    )
    print(f"reranker device : {cfg.get('retrieval', 'reranker', 'device', default='cpu')} cuda_available={cuda}")
    print(f"embedding device: {'CPU (GPU reserved for OCR)' if os.getenv('LOCALRAG_EMBED_CPU') == '1' else 'auto/GPU'}")
    if not status["running"]:
        print("Ollama is not running", flush=True)
        return 1
    if weight_bytes < 2_000_000_000:
        print(
            f"Reranker weights are missing or incomplete ({weight_bytes / (1024 * 1024):.1f}MB). "
            f"Finish the one-time download with:\n"
            f"  python -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download('{reranker_model}', 'model.safetensors')\"",
            flush=True,
        )
        return 1

    heading("2. UNIT TESTS (pytest)")
    tests = run_command(["-m", "pytest", "-q"])
    print(tests.stdout.strip(), flush=True)
    if tests.returncode != 0:
        print("pytest FAILED", flush=True)
        return 1

    heading("3. ENSURE SAMPLE BOOKS ARE INDEXED")
    books: dict[str, Any] = {}
    for key, file_name in (
        ("ar", "sample_ar_scanned_20.pdf"),
        ("en", "sample_en_scanned_20.pdf"),
    ):
        book = find_book(db, file_name)
        if book is None:
            print(f"missing sample {file_name} — run verify_phase1.py first", flush=True)
            return 1
        state = IndexStatusRepo.get(db, book.id)
        if state is None or state.status != "indexed":
            print(f"indexing {file_name} ...", flush=True)
            result = IndexService(cfg, db).index_book(book.id)
            print(f"  indexed: {result}", flush=True)
        books[key] = book
        state = IndexStatusRepo.get(db, book.id)
        print(
            f"  {key}: {book.title} chunks={state.chunk_count} "
            f"embedded={state.embedded_count} status={state.status}",
            flush=True,
        )

    heading("4. REAL CROSS-ENCODER SMOKE TEST")
    smoke = CrossEncoderReranker(cfg)
    started = time.perf_counter()
    try:
        scores = smoke.score(
            "الكتاب والمكتبة",
            ["الكتاب في المكتبة القديمة", "الطقس جميل اليوم"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"rerank load FAILED: {exc}", flush=True)
        return 1
    print(f"scores={[round(value, 4) for value in scores]} seconds={time.perf_counter() - started:.2f}")
    if len(scores) != 2 or not all(math.isfinite(value) for value in scores):
        failures.append("reranker returned invalid scores")
    smoke.unload()
    print("reranker unloaded after smoke test")

    heading("5. HYBRID RETRIEVAL — ARABIC")
    reranker = CrossEncoderReranker(cfg)
    service = RetrievalService(cfg, db, reranker=reranker, rewriter=NoopRewriter())
    ar = service.retrieve("الكتاب والمكتبة", top_k=8)
    print(
        f"counts={ar.counts} timings={ar.timings_s} rewritten={ar.rewritten_query}",
        flush=True,
    )
    print_hits(ar.hits)
    if not ar.hits:
        failures.append("Arabic retrieval returned no hits")
    if not all(hit.fused_rank is not None for hit in ar.hits):
        failures.append("Arabic hits are missing fused ranks")
    if not all(hit.rerank_score is not None for hit in ar.hits):
        failures.append("Arabic hits are missing reranker scores")
    if not all(hit.parent_text for hit in ar.hits):
        failures.append("Arabic hits are missing parent windows")
    if not any(hit.dense_rank for hit in ar.hits) or not any(hit.bm25_rank for hit in ar.hits):
        failures.append("Arabic debug payload lacks dense or BM25 ranks")

    heading("6. HYBRID RETRIEVAL — ENGLISH")
    en = service.retrieve("libraries and books", top_k=8)
    print(f"counts={en.counts} timings={en.timings_s}", flush=True)
    print_hits(en.hits)
    if not en.hits:
        failures.append("English retrieval returned no hits")
    if not all(hit.rerank_score is not None for hit in en.hits):
        failures.append("English hits are missing reranker scores")

    heading("7. FILTERS (BOOK + PAGE RANGE)")
    filtered = service.retrieve(
        "المكتبة",
        filters=RetrievalFilters(book_ids=[books["ar"].id], page_from=1, page_to=5),
        top_k=8,
    )
    print(
        f"filters={filtered.filters} hits={len(filtered.hits)} "
        f"books={sorted({hit.book_id for hit in filtered.hits})} "
        f"pages={sorted({hit.page_start for hit in filtered.hits})}",
        flush=True,
    )
    if not filtered.hits:
        failures.append("filtered retrieval returned no hits")
    if any(hit.book_id != books["ar"].id for hit in filtered.hits):
        failures.append("book filter leaked another book")
    if any(hit.page_start < 1 or hit.page_start > 5 for hit in filtered.hits):
        failures.append("page filter leaked another page")
    print(
        "query_rewrite       : disabled by config (unit-tested with a fake rewriter in test_retrieval.py)",
        flush=True,
    )

    heading("8. EVAL SCRIPT (10 questions, real models)")
    evaluation = run_command(["scripts/eval_retrieval.py", "--limit", "10"])
    print(evaluation.stdout.strip(), flush=True)
    if evaluation.returncode != 0:
        failures.append("eval_retrieval.py failed")

    heading("9. API + UI DEBUG PANEL")
    from fastapi.testclient import TestClient

    from localrag.api.app import create_app
    from localrag.ingest.service import IngestService
    from localrag.index.service import IndexService

    app = create_app()
    app.state.cfg = cfg
    app.state.db = db
    app.state.service = IngestService(cfg, db)
    app.state.index_service = IndexService(cfg, db)
    app.state.retrieval_service = service
    client = TestClient(app)
    response = client.post(
        "/api/retrieve",
        json={"query": "الكتاب والمكتبة", "top_k": 3, "book_ids": [books["ar"].id]},
    )
    payload = response.json()
    print(f"POST /api/retrieve -> {response.status_code} hits={len(payload.get('hits', []))}")
    print(f"  models={payload.get('models')} counts={payload.get('counts')}", flush=True)
    if response.status_code != 200 or not payload.get("hits"):
        failures.append("POST /api/retrieve failed")
    html = client.get("/ui/index.html")
    panel_ok = "tab-search" in html.text and "searchQuery" in html.text
    print(f"GET /ui/index.html -> {html.status_code} debug_panel={panel_ok}")
    if html.status_code != 200 or not panel_ok:
        failures.append("UI search debug panel is missing")

    service.close()

    heading("10. RESULT")
    if failures:
        for failure in failures:
            print(f"FAIL - {failure}", flush=True)
        return 1
    print("PASS - Phase 3 verified end to end", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
