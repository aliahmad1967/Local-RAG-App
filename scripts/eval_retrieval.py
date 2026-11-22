"""Evaluate dense-only vs BM25-only vs hybrid vs hybrid+rerank.

The gold set for each question is every child chunk of the matching sample book
that contains the answer phrase after the same Arabic normalization used by the
BM25 index. This makes the eval reproducible from the generated sample PDFs.

By default embeddings and the cross-encoder run on CPU so the script is safe to
run while OCR still owns the GPU. Use LOCALRAG_VERIFY_GPU=1 when the GPU is free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import ensure_data_dirs, load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.db.repositories import BooksRepo, ChunksRepo, IndexStatusRepo  # noqa: E402
from localrag.index import bm25  # noqa: E402
from localrag.index.embeddings import OllamaEmbedder  # noqa: E402
from localrag.index.service import IndexService  # noqa: E402
from localrag.index.vector_store import VectorStore, cosine_scores  # noqa: E402
from localrag.logging_conf import setup_logging  # noqa: E402
from localrag.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from localrag.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from localrag.text.arabic import normalize_for_bm25  # noqa: E402

# (question, answer phrase, sample key)
QA_PAIRS: list[tuple[str, str, str]] = [
    ("ماذا قرأت في المكتبة عن المدن؟", "تاريخ المدن القديمة", "ar"),
    ("كيف تُحفظ الكتب في المكتبة؟", "رفوف مرتبة", "ar"),
    ("ماذا تفعل القراءة اليومية؟", "القراءة اليومية", "ar"),
    ("ما الكتاب الذي لا يملّ من الحديث؟", "الصديق الذي لا يمل", "ar"),
    ("ماذا سنقرأ في صفحة الغد؟", "الرحلات", "ar"),
    ("ماذا تحتوي المكتبة من مخطوطات؟", "مخطوطات عربية", "ar"),
    ("أي نصوص قديمة في المكتبة؟", "نصوص لاتينية", "ar"),
    ("ما صدى المعرفة؟", "صدى التاريخ", "ar"),
    ("ما الذي شهد على مرّ الزمان؟", "شاهدة على مر الزمان", "ar"),
    ("حسب أي شيء تُربَّب الكتب؟", "الموضوع واللغة", "ar"),
    ("What does each page carry?", "trace of the hands", "en"),
    ("What does a good book ask?", "better questions", "en"),
    ("How did the archivist catalogue the manuscript?", "title, date, and language", "en"),
    ("What does reading slowly reveal?", "hurried scan", "en"),
    ("What does every chapter describe?", "different journey", "en"),
    ("What turned a thought into a lasting record?", "Paper, ink, and patience", "en"),
    ("What does the final section compare?", "Arabic and Latin sources", "en"),
    ("Who is knowledge kept safe for?", "knowledge safe", "en"),
    ("How long ago were the pages copied?", "hands that copied", "en"),
    ("What did patience help create?", "lasting record", "en"),
]

METHODS = ("dense_only", "bm25_only", "hybrid", "hybrid_rerank")


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold:
            return 1.0 / position
    return 0.0


def find_books(db: Database) -> dict[str, Any]:
    wanted = {
        "ar": "sample_ar_scanned_20.pdf",
        "en": "sample_en_scanned_20.pdf",
    }
    found: dict[str, Any] = {}
    for key, file_name in wanted.items():
        book = next((b for b in BooksRepo.all(db) if b.file_name == file_name), None)
        if book is None:
            raise RuntimeError(f"Sample book {file_name} is missing; run verify_phase1.py first")
        status = IndexStatusRepo.get(db, book.id)
        if status is None or status.status != "indexed":
            raise RuntimeError(f"Sample book {file_name} is not indexed; run verify_phase2.py first")
        found[key] = book
    return found


def load_gold(db: Database, books: dict[str, Any]) -> list[set[str]]:
    chunks = {
        key: ChunksRepo.all_for_book(db, book.id, "child", 500)
        for key, book in books.items()
    }
    normalized = {
        key: [(chunk.id, normalize_for_bm25(chunk.text)) for chunk in values]
        for key, values in chunks.items()
    }
    gold_sets: list[set[str]] = []
    for _question, phrase, key in QA_PAIRS:
        needle = normalize_for_bm25(phrase)
        gold_sets.append({chunk_id for chunk_id, text in normalized[key] if needle in text})
    return gold_sets


def load_cache(path: Path) -> dict[str, list[float]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Number of Q&A pairs")
    parser.add_argument("--rerank-candidates", type=int, default=40)
    parser.add_argument("--k", type=int, default=8, help="Recall@k")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    if os.getenv("LOCALRAG_VERIFY_GPU") != "1":
        os.environ["LOCALRAG_EMBED_CPU"] = "1"

    cfg = load_config()
    ensure_data_dirs(cfg)
    setup_logging(cfg)
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    books = find_books(db)
    gold_sets = load_gold(db, books)
    pairs = list(zip(QA_PAIRS, gold_sets))[: max(1, args.limit)]
    book_ids = [book.id for book in books.values()]

    embed_model = str(cfg.get("ollama", "models", "embedding", default="bge-m3"))
    embedder = OllamaEmbedder(cfg, model=embed_model)
    queries = [pair[0][0] for pair in pairs]
    print(f"Embedding {len(queries)} queries with {embed_model} ...", flush=True)
    query_vectors = embedder.embed(queries)
    embedder.unload()

    vector_rows = VectorStore.all_vectors(db, book_ids=book_ids, model=embed_model)
    chunk_order = [chunk_id for chunk_id, _ in vector_rows]
    matrix = [blob for _, blob in vector_rows]
    print(f"Ranking {len(vector_rows)} indexed child chunks", flush=True)

    reranker = None if args.no_rerank else CrossEncoderReranker(cfg)
    cache_path = (
        Path(args.cache)
        if args.cache
        else cfg.path("paths", "data_dir") / "retrieval_eval_cache.json"
    )
    score_cache = load_cache(cache_path)
    model_name = str(
        cfg.get("retrieval", "reranker", "model", default="BAAI/bge-reranker-v2-m3")
    )
    rankings: dict[str, list[list[str]]] = {method: [] for method in METHODS}

    for index, ((query, _phrase, _key), gold) in enumerate(pairs, start=1):
        scores = cosine_scores(query_vectors[index - 1], matrix)
        order = sorted(range(len(scores)), key=lambda pos: -float(scores[pos]))
        dense_tuples = [(chunk_order[pos], float(scores[pos])) for pos in order[:50]]
        lexical_rows = bm25.search(db, query, limit=50, cfg=cfg, book_ids=book_ids)
        lexical_tuples = [(row["chunk_id"], -float(row["rank"])) for row in lexical_rows]
        dense = [chunk_id for chunk_id, _ in dense_tuples]
        lexical = [chunk_id for chunk_id, _ in lexical_tuples]
        fused = [
            chunk_id for chunk_id, _ in reciprocal_rank_fusion(
                [dense_tuples, lexical_tuples], k=int(cfg.get("retrieval", "rrf_k", default=60))
            )[: int(cfg.get("retrieval", "fused_top_k", default=40))]
        ]
        rankings["dense_only"].append(dense)
        rankings["bm25_only"].append(lexical)
        rankings["hybrid"].append(fused)

        if reranker is None:
            reranked = fused
        else:
            candidates = fused[: args.rerank_candidates]
            texts: list[str] = []
            for chunk_id in candidates:
                chunk = ChunksRepo.get(db, chunk_id)
                texts.append(" ".join(chunk.text.split()) if chunk else "")
            reranked = list(candidates)
            uncached: list[int] = []
            for pos, chunk_id in enumerate(candidates):
                key = f"{model_name}|{query}|{chunk_id}"
                if key in score_cache:
                    continue
                uncached.append(pos)
            if uncached:
                computed = reranker.score(query, [texts[pos] for pos in uncached])
                for pos, score in zip(uncached, computed):
                    key = f"{model_name}|{query}|{candidates[pos]}"
                    score_cache[key] = float(score)
                save_cache(cache_path, score_cache)
            reranked.sort(key=lambda chunk_id: -score_cache[f"{model_name}|{query}|{chunk_id}"])
        rankings["hybrid_rerank"].append(reranked)

        first = next((pos + 1 for pos, chunk in enumerate(reranked) if chunk in gold), None)
        print(
            f"  [{index:2d}/{len(pairs)}] gold={len(gold):2d} "
            f"first_hit={first or '—'} :: {query}",
            flush=True,
        )

    if reranker is not None:
        reranker.unload()

    print("\n" + "=" * 72)
    print(f"RETRIEVAL EVAL ({len(pairs)} questions, recall@{args.k} + MRR)")
    print("=" * 72)
    print(f"{'method':<18} {'recall@5':>10} {'recall@' + str(args.k):>10} {'MRR':>10}")
    for method in METHODS:
        recalls5 = [recall_at_k(r, gold, 5) for r, gold in zip(rankings[method], [g for _, g in pairs])]
        recalls = [recall_at_k(r, gold, args.k) for r, gold in zip(rankings[method], [g for _, g in pairs])]
        mrrs = [reciprocal_rank(r, gold) for r, gold in zip(rankings[method], [g for _, g in pairs])]
        print(
            f"{method:<18} {sum(recalls5) / len(recalls5):>10.3f} "
            f"{sum(recalls) / len(recalls):>10.3f} {sum(mrrs) / len(mrrs):>10.3f}"
        )
    print(f"\nCached reranker scores: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
