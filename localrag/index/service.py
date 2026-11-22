"""Indexing orchestration: chunk → FTS5 → embeddings, all resumable."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ..config import Config
from ..db.database import Database
from ..db.repositories import (
    BooksRepo,
    ChunksRepo,
    EmbeddingRepo,
    IndexStatusRepo,
    JobsRepo,
    PagesRepo,
)
from ..logging_conf import get_logger
from ..models.domain import Book, Chunk, Job, utcnow
from . import bm25
from .chunker import chunk_pages
from .embeddings import OllamaEmbedder, content_hash
from .vector_store import VectorStore, pack_vector

log = get_logger("index.service")


class IndexService:
    def __init__(self, cfg: Config, db: Database, embedder: Any | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self._embedder = embedder
        self.model = str(cfg.get("ollama", "models", "embedding", default="bge-m3"))

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = OllamaEmbedder(self.cfg, model=self.model)
        return self._embedder

    # -- job creation -------------------------------------------------------
    def enqueue(self, book_id: str, force: bool = False) -> Job:
        book = BooksRepo.get(self.db, book_id)
        if book is None:
            raise ValueError(f"Book not found: {book_id}")
        job = Job(
            id=str(uuid.uuid4()),
            book_id=book_id,
            kind="index",
            status="queued",
            total_pages=book.page_count,
            params_json=json.dumps({"book_id": book_id, "force": force}),
            created_at=utcnow(),
        )
        JobsRepo.create(self.db, job)
        JobsRepo.add_event(self.db, job.id, f"Queued indexing of '{book.title}'")
        return job

    # -- main entry point ---------------------------------------------------
    def index_book(self, book_id: str, force: bool = False) -> dict[str, Any]:
        book = BooksRepo.get(self.db, book_id)
        if book is None:
            raise ValueError(f"Book not found: {book_id}")

        pages = PagesRepo.all_for_book(self.db, book_id)
        if not any(page.status in ("done", "flagged") and page.text_clean.strip() for page in pages):
            raise ValueError(
                f"Book '{book.title}' has no OCR text yet. Finish ingestion before indexing."
            )

        if force:
            ChunksRepo.delete_book(self.db, book_id)

        specs = chunk_pages(pages, self.cfg)
        chunks = self._to_chunks(book, specs)
        inserted = ChunksRepo.upsert_many(self.db, chunks)
        status = IndexStatusRepo.get(self.db, book_id)
        status.chunk_count = ChunksRepo.counts(self.db, book_id).get("child", 0)
        status.status = "chunked"
        IndexStatusRepo.save(self.db, status)
        log.info("Book %s: %d chunks inserted (%d specs)", book.title, inserted, len(specs))

        embedded = self.embed_children(book_id)
        status = IndexStatusRepo.get(self.db, book_id)
        status.chunk_count = ChunksRepo.counts(self.db, book_id).get("child", 0)
        status.embedded_count = embedded
        status.status = "indexed" if status.chunk_count else "empty"
        IndexStatusRepo.save(self.db, status)

        if self.cfg.get("indexing", "unload_embedding_after_index", default=True):
            self.embedder.unload()
        return {
            "book_id": book_id,
            "status": status.status,
            "chunks": status.chunk_count,
            "parents": ChunksRepo.counts(self.db, book_id).get("parent", 0),
            "embedded": embedded,
            "bm25_rows": bm25.count(self.db, book_id),
            "vector_rows": VectorStore.count(self.db, book_id),
            "index_size_bytes": self.index_size_bytes(),
        }

    # -- internals ----------------------------------------------------------
    def _to_chunks(self, book: Book, specs: list[Any]) -> list[Chunk]:
        ids: dict[int, str] = {}
        for spec in specs:
            if spec.level == "parent":
                ids[spec.parent_index] = (
                    f"{book.id}-p-{spec.page_start:05d}-{spec.parent_index:04d}"
                )
        chunks: list[Chunk] = []
        for spec in specs:
            if spec.level == "parent":
                chunk_id = ids[spec.parent_index]
                parent_id = None
            else:
                chunk_id = f"{book.id}-c-{spec.page_start:05d}-{spec.child_index:04d}"
                parent_id = ids.get(spec.parent_index)
            chunks.append(Chunk(
                id=chunk_id, book_id=book.id, level=spec.level, parent_id=parent_id,
                child_index=spec.child_index, page_start=spec.page_start,
                page_end=spec.page_end, heading=spec.heading, text=spec.text,
                text_normalized=self._normalize(spec.text), token_count=spec.token_count,
                char_count=len(spec.text), start_char=spec.start_char,
                end_char=spec.end_char,
            ))
        return chunks

    def _normalize(self, text: str) -> str:
        from ..text.arabic import normalize_for_bm25

        return normalize_for_bm25(
            text,
            strip_diacritics=bool(self.cfg.get("text", "bm25_strip_tashkeel", default=True)),
            fold_alef=bool(self.cfg.get("text", "bm25_normalize_alef", default=True)),
            fold_ya_marbuta=bool(self.cfg.get("text", "bm25_normalize_ya_marbuta", default=True)),
        )

    def embed_children(self, book_id: str) -> int:
        """Embed every child chunk that is not yet cached; returns new count."""
        children = ChunksRepo.all_for_book(self.db, book_id, level="child")
        if not children:
            return 0
        pairs = [(chunk.id, content_hash(self.embedder.model, chunk.text)) for chunk in children]
        text_by_hash = {
            content_hash(self.embedder.model, chunk.text): chunk.text for chunk in children
        }
        missing = EmbeddingRepo.missing_hashes(self.db, pairs, self.embedder.model)
        if not missing:
            return EmbeddingRepo.count_for_book(self.db, book_id)

        by_hash: dict[str, list[str]] = {}
        for chunk_id, digest in missing:
            by_hash.setdefault(digest, []).append(chunk_id)

        already_cached = EmbeddingRepo.cached_vectors(
            self.db, list(by_hash), self.embedder.model
        )
        to_request = [digest for digest in by_hash if digest not in already_cached]
        if to_request:
            vectors = self.embedder.embed([text_by_hash[digest] for digest in to_request])
            if len(vectors) != len(to_request):
                raise RuntimeError("Embedder returned the wrong number of vectors")
            for digest, vector in zip(to_request, vectors):
                already_cached[digest] = pack_vector(vector)
        # Copy cached vectors into any additional chunks sharing the same text.
        for digest, chunk_ids in by_hash.items():
            blob = already_cached[digest]
            entries = [
                (chunk_id, digest, self.embedder.model, len(blob) // 4, blob)
                for chunk_id in chunk_ids
            ]
            EmbeddingRepo.save_batch(self.db, entries)
        return EmbeddingRepo.count_for_book(self.db, book_id)

    def progress(self, book_id: str) -> dict[str, Any]:
        status = IndexStatusRepo.get(self.db, book_id)
        chunks = ChunksRepo.counts(self.db, book_id)
        return {
            **status.to_dict(),
            "parents": chunks.get("parent", 0),
            "embedded": EmbeddingRepo.count_for_book(self.db, book_id),
        }

    def index_size_bytes(self) -> int:
        paths = [self.cfg.path("paths", "db_path", default="data/localrag.db")]
        return sum(path.stat().st_size for path in paths if Path(path).exists())
