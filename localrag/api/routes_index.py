"""Phase 2 HTTP endpoints: indexing, chunk browsing and BM25 search debug."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..db.repositories import BooksRepo, ChunksRepo, EmbeddingRepo, IndexStatusRepo, JobsRepo
from ..index import bm25
from ..index.vector_store import VectorStore
from .schemas import IndexRequest, SearchRequest

router = APIRouter(prefix="/api")


def _services(request: Request) -> tuple[Any, Any, Any]:
    return request.app.state.cfg, request.app.state.db, request.app.state.index_service


@router.post("/books/{book_id}/index")
def start_index(book_id: str, request: Request, body: IndexRequest | None = None) -> dict[str, Any]:
    _, db, service = _services(request)
    if BooksRepo.get(db, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    try:
        job = service.enqueue(book_id, force=bool(body and body.force))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job": job.to_dict()}


@router.get("/books/{book_id}/index-status")
def index_status(book_id: str, request: Request) -> dict[str, Any]:
    _, db, service = _services(request)
    if BooksRepo.get(db, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    jobs = [job.to_dict() for job in JobsRepo.for_book(db, book_id) if job.kind == "index"]
    return {"status": service.progress(book_id), "jobs": jobs[:5]}


@router.get("/books/{book_id}/chunks")
def list_chunks(
    book_id: str,
    request: Request,
    level: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    _, db, _ = _services(request)
    if BooksRepo.get(db, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    if level and level not in ("child", "parent"):
        raise HTTPException(status_code=400, detail="level must be 'child' or 'parent'")
    return [
        chunk.to_dict() for chunk in ChunksRepo.all_for_book(db, book_id, level, limit)
    ]


@router.get("/books/{book_id}/chunks/{chunk_id}")
def get_chunk(book_id: str, chunk_id: str, request: Request) -> dict[str, Any]:
    _, db, _ = _services(request)
    chunk = ChunksRepo.get(db, chunk_id)
    if chunk is None or chunk.book_id != book_id:
        raise HTTPException(status_code=404, detail="Chunk not found")
    data = chunk.to_dict()
    if chunk.level == "child" and chunk.parent_id:
        parent = ChunksRepo.parent_for_child(db, chunk)
        data["parent"] = parent.to_dict() if parent else None
    data["has_vector"] = EmbeddingRepo.get_vector(db, chunk_id) is not None
    return data


@router.post("/index/search")
def bm25_search(body: SearchRequest, request: Request) -> dict[str, Any]:
    """BM25-only search — Phase 3 adds dense + RRF + reranking."""
    cfg, db, _ = _services(request)
    results = bm25.search(
        db, body.query, limit=body.top_k, book_id=body.book_id, cfg=cfg
    )
    return {"results": results, "count": len(results)}


@router.get("/index/stats")
def index_stats(request: Request) -> dict[str, Any]:
    _, db, service = _services(request)
    return {
        "bm25_rows": bm25.count(db),
        "vector_rows": VectorStore.count(db),
        "cached_embeddings": EmbeddingRepo.count_cached(db),
        "books": [status.to_dict() for status in IndexStatusRepo.all(db)],
        "size_bytes": service.index_size_bytes(),
    }
