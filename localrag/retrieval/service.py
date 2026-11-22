"""Hybrid retrieval: query rewrite → dense + BM25 → RRF → cross-encoder rerank."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from ..config import Config
from ..db.database import Database
from ..db.repositories import BooksRepo, ChunksRepo
from ..index import bm25
from ..index.embeddings import OllamaEmbedder
from ..index.vector_store import VectorStore, cosine_scores
from ..logging_conf import get_logger
from ..models.domain import RetrievalHit, RetrievalResult
from .fusion import rank_map, reciprocal_rank_fusion, score_map
from .query_rewrite import NoopRewriter, Rewriter, build_rewriter
from .rerank import Reranker, build_reranker

log = get_logger("retrieval.service")
_UNSET = object()


@dataclass
class RetrievalFilters:
    book_ids: list[str] | None = None
    page_from: int | None = None
    page_to: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_ids": list(self.book_ids) if self.book_ids else None,
            "page_from": self.page_from,
            "page_to": self.page_to,
        }


class EmbedQuery(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class RetrievalService:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        embedder: EmbedQuery | None = None,
        reranker: Reranker | None | object = _UNSET,
        rewriter: Rewriter | None | object = _UNSET,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self._embedder = embedder
        self._reranker: Reranker | None = (
            build_reranker(cfg) if reranker is _UNSET else cast(Reranker | None, reranker)
        )
        if rewriter is _UNSET:
            self._rewriter: Rewriter = build_rewriter(cfg)
        elif rewriter is None:
            self._rewriter = NoopRewriter()
        else:
            self._rewriter = cast(Rewriter, rewriter)

    @property
    def embedder(self) -> EmbedQuery:
        if self._embedder is None:
            model = str(self.cfg.get("ollama", "models", "embedding", default="bge-m3"))
            self._embedder = OllamaEmbedder(self.cfg, model=model)
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    def _cfg(self, *keys: str, default: Any = None) -> Any:
        return self.cfg.get("retrieval", *keys, default=default)

    # -- public API ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        explain: bool = True,
    ) -> RetrievalResult:
        started = time.perf_counter()
        filters = filters or RetrievalFilters()
        final_k = int(top_k or self._cfg("final_top_k", default=8))

        t0 = time.perf_counter()
        rewritten = self._rewriter.rewrite(query, history)
        search_query = rewritten or query
        rewrite_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense = self.dense_search(search_query, filters, int(self._cfg("dense_top_k", default=50)))
        dense_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        lexical = self.lexical_search(search_query, filters, int(self._cfg("bm25_top_k", default=50)))
        bm25_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        weights = self._cfg("rrf_weights", default=[1.0, 1.0])
        fused = reciprocal_rank_fusion(
            [dense, lexical],
            k=int(self._cfg("rrf_k", default=60)),
            weights=[float(weights[0]), float(weights[1])] if weights else None,
        )[: int(self._cfg("fused_top_k", default=40))]
        fuse_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        reranked = self.apply_rerank(search_query, fused, final_k)
        rerank_s = time.perf_counter() - t0

        dense_ranks, dense_scores = rank_map(dense), score_map(dense)
        bm25_ranks, bm25_scores = rank_map(lexical), score_map(lexical)
        fused_ranks, fused_scores = rank_map(fused), score_map(fused)

        hits: list[RetrievalHit] = []
        for position, (chunk_id, rerank_score) in enumerate(reranked, start=1):
            chunk = ChunksRepo.get(self.db, chunk_id)
            if chunk is None:
                continue
            book = BooksRepo.get(self.db, chunk.book_id)
            parent = ChunksRepo.parent_for_child(self.db, chunk) if chunk.parent_id else None
            hits.append(RetrievalHit(
                chunk_id=chunk.id,
                book_id=chunk.book_id,
                book_title=book.title if book else "",
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                heading=chunk.heading,
                text=chunk.text if explain else "",
                parent_id=chunk.parent_id,
                parent_text=parent.text if (parent and explain) else None,
                dense_rank=dense_ranks.get(chunk_id),
                dense_score=dense_scores.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
                fused_rank=fused_ranks.get(chunk_id),
                fused_score=fused_scores.get(chunk_id),
                rerank_rank=position if self._reranker else None,
                rerank_score=rerank_score if self._reranker else None,
            ))

        return RetrievalResult(
            query=query,
            rewritten_query=rewritten if rewritten != query else None,
            hits=hits,
            filters=filters.to_dict(),
            timings_s={
                "rewrite": round(rewrite_s, 4),
                "dense": round(dense_s, 4),
                "bm25": round(bm25_s, 4),
                "fuse": round(fuse_s, 4),
                "rerank": round(rerank_s, 4),
                "total": round(time.perf_counter() - started, 4),
            },
            counts={
                "dense": len(dense),
                "bm25": len(lexical),
                "fused": len(fused),
                "final": len(hits),
            },
        )

    def dense_search(
        self, query: str, filters: RetrievalFilters, limit: int
    ) -> list[tuple[str, float]]:
        vectors = VectorStore.all_vectors(
            self.db,
            book_ids=filters.book_ids,
            model=self.embedder.model,
            page_from=filters.page_from,
            page_to=filters.page_to,
        )
        if not vectors or limit <= 0:
            return []
        query_vector = self.embedder.embed([query])[0]
        scores = cosine_scores(query_vector, [blob for _, blob in vectors])
        order = np.argsort(-scores)[:limit]
        return [(vectors[int(index)][0], float(scores[index])) for index in order]

    def lexical_search(
        self, query: str, filters: RetrievalFilters, limit: int
    ) -> list[tuple[str, float]]:
        rows = bm25.search(
            self.db,
            query,
            limit=limit,
            cfg=self.cfg,
            book_ids=filters.book_ids,
            page_from=filters.page_from,
            page_to=filters.page_to,
        )
        # FTS5 bm25() is negative and more negative is better; flip it so the
        # debug panel shows a familiar "higher = better" score.
        return [(row["chunk_id"], -float(row["rank"])) for row in rows]

    def apply_rerank(
        self, query: str, fused: list[tuple[str, float]], top_k: int
    ) -> list[tuple[str, float | None]]:
        if not fused:
            return []
        if self._reranker is None:
            return [(chunk_id, None) for chunk_id, _ in fused[:top_k]]
        chunks = [(chunk_id, ChunksRepo.get(self.db, chunk_id)) for chunk_id, _ in fused]
        pairs = [(chunk_id, chunk.text) for chunk_id, chunk in chunks if chunk is not None]
        scores = self._reranker.score(query, [text for _, text in pairs])
        ranked = sorted(zip([cid for cid, _ in pairs], scores), key=lambda item: -item[1])
        try:
            unload = bool(
                self.cfg.get("retrieval", "reranker", "unload_after_search", default=False)
                or str(getattr(self._reranker, "device", "cpu")).startswith("cuda")
            )
        except Exception:  # noqa: BLE001 - fake rerankers in tests have no device
            unload = False
        if unload and hasattr(self._reranker, "unload"):
            self._reranker.unload()
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k]]

    def close(self) -> None:
        """Release the cross-encoder when the app shuts down."""
        if self._reranker is not None and hasattr(self._reranker, "unload"):
            self._reranker.unload()
