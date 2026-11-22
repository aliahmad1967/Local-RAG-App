"""Real end-to-end verification for Phase 2: chunking + indexing.

Runs against the actual local database and the real Ollama bge-m3 model.
By default it forces CPU embeddings so a still-running OCR job keeps the GPU.
Set LOCALRAG_VERIFY_GPU=1 to use the GPU when no other model is resident.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import ensure_data_dirs, load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.db.repositories import (  # noqa: E402
    BooksRepo,
    ChunksRepo,
    EmbeddingRepo,
    IndexStatusRepo,
)
from localrag.index import bm25  # noqa: E402
from localrag.index.embeddings import OllamaEmbedder  # noqa: E402
from localrag.index.service import IndexService  # noqa: E402
from localrag.index.vector_store import VectorStore, cosine_scores  # noqa: E402
from localrag.logging_conf import setup_logging  # noqa: E402
from localrag.ocr.vram import ollama_status  # noqa: E402
from tests.data.make_samples import generate_all  # noqa: E402
from verification_env import isolate_verification_data  # noqa: E402


def heading(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT), text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def find_book(db: Database, file_name: str):
    for book in BooksRepo.all(db):
        if book.file_name == file_name:
            return book
    return None


def main() -> int:
    isolate_verification_data(ROOT)
    if os.getenv("LOCALRAG_VERIFY_GPU") != "1":
        os.environ["LOCALRAG_EMBED_CPU"] = "1"

    cfg = load_config()
    ensure_data_dirs(cfg)
    setup_logging(cfg)
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))

    heading("1. ENVIRONMENT")
    status = ollama_status(cfg)
    print(f"python          : {sys.version.split()[0]}")
    print(f"ollama          : {'running' if status['running'] else 'DOWN'} @ {status['host']}")
    model_names = [str(name) for name in status.get("models", [])]
    embed_model = str(cfg.get("ollama", "models", "embedding", default="bge-m3"))
    print(f"embedding model : {embed_model} installed={any(embed_model.split(':')[0] in n for n in model_names)}")
    print(f"embedding device: {'CPU (GPU reserved for OCR)' if os.getenv('LOCALRAG_EMBED_CPU') == '1' else 'auto/GPU'}")
    import sqlite3

    print(f"sqlite / FTS5   : {sqlite3.sqlite_version} / available")
    print(f"vector backend  : {cfg.get('indexing', 'vector_backend', default='numpy')} (float32 BLOB + NumPy cosine)")

    heading("2. UNIT TESTS (pytest)")
    tests = run_command(["-m", "pytest", "-q"])
    print(tests.stdout.strip(), flush=True)
    if tests.returncode != 0:
        print("pytest FAILED", flush=True)
        return 1

    heading("3. VERIFY EMBEDDING MODEL (real Ollama call)")
    embedder = OllamaEmbedder(cfg, model=embed_model)
    started = time.perf_counter()
    probe = embedder.embed(["اختبار争夺 النص", "indexing probe"])
    dim = len(probe[0])
    print(f"embed probe     : vectors={len(probe)} dim={dim} seconds={time.perf_counter() - started:.2f}")
    if dim != 1024:
        print(f"unexpected embedding dimension {dim}", flush=True)
        return 1

    heading("4. INDEX THE TWO 20-PAGE SAMPLE BOOKS")
    samples = generate_all(ROOT / "tests" / "data")
    results: dict[str, dict] = {}
    for key, label in (("ar", "Arabic"), ("en", "English")):
        path = samples[key]
        book = find_book(db, path.name)
        if book is None:
            print(f"  {key}: sample book not ingested yet — run verify_phase1.py first", flush=True)
            return 1
        service = IndexService(cfg, db, embedder=embedder)
        started = time.perf_counter()
        result = service.index_book(book.id, force=True)
        result["seconds"] = time.perf_counter() - started
        result["book"] = book
        results[key] = result
        print(
            f"  {label}: id={book.id} chunks={result['chunks']} parents={result['parents']} "
            f"embedded={result['embedded']} bm25_rows={result['bm25_rows']} "
            f"vector_rows={result['vector_rows']} size={result['index_size_bytes']:,} bytes "
            f"took={result['seconds']:.1f}s",
            flush=True,
        )

    heading("5. CHUNK / PARENT PREVIEWS")
    for key, result in results.items():
        book = result["book"]
        children = ChunksRepo.all_for_book(db, book.id, level="child")
        parents = ChunksRepo.all_for_book(db, book.id, level="parent")
        sample = children[0] if children else None
        linked = sum(1 for child in children if child.parent_id and ChunksRepo.parent_for_child(db, child))
        print(
            f"  {key}: children={len(children)} parents={len(parents)} "
            f"with_live_parent={linked} pages={min(c.page_start for c in children)}"
            f"..{max(c.page_end for c in children)}",
            flush=True,
        )
        if sample:
            print(
                f"    child[0] page={sample.page_start} tokens={sample.token_count} "
                f"heading={sample.heading!r} :: {' '.join(sample.text.split())[:180]}",
                flush=True,
            )
        token_counts = [child.token_count for child in children]
        if token_counts:
            in_target = sum(400 <= count <= 600 for count in token_counts)
            print(
                f"    child tokens min={min(token_counts)} max={max(token_counts)} "
                f"avg={sum(token_counts) / len(token_counts):.0f} "
                f"in_target={in_target}/{len(token_counts)}",
                flush=True,
            )
            if max(token_counts) > 600 or in_target < len(token_counts) - 1:
                print(f"  {key}: child token target assertion FAILED", flush=True)
                results[key]["token_target_failed"] = True

    heading("6. BM25 SEARCH (Arabic + English queries)")
    for query, key in (("الكتاب والمكتبة", "ar"), ("libraries and books", "en")):
        book = results[key]["book"]
        hits = bm25.search(db, query, limit=5, book_id=book.id, cfg=cfg)
        print(f"  {key} query={query!r} hits={len(hits)}", flush=True)
        for hit in hits[:3]:
            print(
                f"    page={hit['page_no']:>2} score={hit['rank']:.4f} "
                f":: {' '.join(hit['snippet'].split())[:130]}",
                flush=True,
            )

    heading("7. DENSE VECTOR SEARCH (real bge-m3 query)")
    for key in ("ar", "en"):
        book = results[key]["book"]
        query = "الكتاب والمكتبة" if key == "ar" else "libraries and books"
        query_vector = embedder.embed([query])[0]
        pairs = VectorStore.all_for_book(db, book.id, model=embedder.model)
        scores = cosine_scores(query_vector, [blob for _, blob in pairs])
        if len(scores):
            import numpy as np

            order = np.argsort(-scores)[:3]
            print(f"  {key}: vectors={len(pairs)} dim={len(pairs[0][1]) // 4}", flush=True)
            for index in order:
                print(
                    f"    chunk={pairs[index][0]} cosine={float(scores[index]):.4f}",
                    flush=True,
                )

    heading("8. CACHE + RESUME (re-index must not re-embed)")
    all_ok = True
    for key in ("ar", "en"):
        book = results[key]["book"]
        before = EmbeddingRepo.count_cached(db)
        started = time.perf_counter()
        repeat = IndexService(cfg, db, embedder=embedder).index_book(book.id)
        after = EmbeddingRepo.count_cached(db)
        print(
            f"  {key}: cache_before={before} cache_after={after} "
            f"chunks={repeat['chunks']} embedded={repeat['embedded']} "
            f"second_run={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if after != before or repeat["embedded"] != results[key]["embedded"]:
            all_ok = False

    heading("9. ASSERTIONS")
    for key in ("ar", "en"):
        result = results[key]
        book = result["book"]
        status = IndexStatusRepo.get(db, book.id)
        checks = {
            "status indexed": status.status == "indexed",
            "children > 0": result["chunks"] > 0,
            "parents > 0": result["parents"] > 0,
            "vectors complete": result["embedded"] == result["chunks"],
            "fts complete": result["bm25_rows"] == result["chunks"],
            "vector rows complete": result["vector_rows"] == result["chunks"],
            "all parents linked": all(
                child.parent_id for child in ChunksRepo.all_for_book(db, book.id, level="child")
            ),
            "child token target": not result.get("token_target_failed", False),
        }
        for label, ok in checks.items():
            print(f"  {key} {label:24s}: {'OK' if ok else 'FAIL'}", flush=True)
            if not ok:
                all_ok = False

    if os.getenv("LOCALRAG_VERIFY_GPU") != "1":
        embedder.unload()
        print("  embedding model unloaded (keep_alive=0)", flush=True)

    heading("10. RESULT")
    print("PASS - Phase 2 verified end to end" if all_ok else "FAIL - see output above", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
