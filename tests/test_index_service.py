"""End-to-end Phase 2 indexing on generated sample books."""

from __future__ import annotations

import numpy as np

from localrag.db.repositories import (
    BooksRepo,
    ChunksRepo,
    EmbeddingRepo,
    IndexStatusRepo,
    PagesRepo,
)
from localrag.ingest.service import IngestService
from localrag.index import bm25
from localrag.index.service import IndexService
from localrag.models.domain import Book

from .data.make_samples import make_sample_pdf


class FakeEmbedder:
    model = "fake-bge"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def unload(self) -> None:
        pass


def _book_with_pages(tmp_path, cfg, db, pages: int = 4):
    source = make_sample_pdf(tmp_path / "book.pdf", pages=pages, language="en", dpi=80)
    book = Book(
        id="index-book", title="Index test", file_path=str(source), file_name=source.name,
        page_count=pages, language="en", ocr_engine="fake", status="ingested",
    )
    BooksRepo.create(db, book)
    for number in range(1, pages + 1):
        PagesRepo.upsert(db, __import__("localrag.models.domain", fromlist=["Page"]).Page(
            id=f"p{number}", book_id=book.id, page_no=number, status="done", source="ocr",
            text_clean=" ".join(
                f"This is sentence {i} about books and libraries in the archive."
                for i in range(35)
            ),
            text_normalized="this is sentence about books and libraries in the archive",
            char_count=1500,
        ))
    return book


def test_index_book_creates_chunks_fts_and_embeddings(tmp_path, cfg, db):
    book = _book_with_pages(tmp_path, cfg, db)
    service = IndexService(cfg, db, embedder=FakeEmbedder())
    result = service.index_book(book.id)

    assert result["status"] == "indexed"
    assert result["chunks"] > 0
    assert result["parents"] > 0
    assert result["embedded"] == result["chunks"]
    assert result["bm25_rows"] == result["chunks"]
    assert result["vector_rows"] == result["chunks"]

    counts = ChunksRepo.counts(db, book.id)
    assert counts == {"child": result["chunks"], "parent": result["parents"]}
    assert IndexStatusRepo.get(db, book.id).status == "indexed"
    assert bm25.count(db) == result["chunks"]
    assert EmbeddingRepo.count_for_book(db, book.id) == result["chunks"]

    hits = bm25.search(db, "libraries", cfg=cfg)
    assert hits


def test_reindex_is_idempotent_and_resumable(tmp_path, cfg, db):
    book = _book_with_pages(tmp_path, cfg, db)
    first = IndexService(cfg, db, embedder=FakeEmbedder()).index_book(book.id)
    second = IndexService(cfg, db, embedder=FakeEmbedder()).index_book(book.id)
    assert second["chunks"] == first["chunks"]
    assert second["embedded"] == first["embedded"]
    assert bm25.count(db) == first["chunks"]


def test_force_rebuild_replaces_chunks(tmp_path, cfg, db):
    book = _book_with_pages(tmp_path, cfg, db)
    IndexService(cfg, db, embedder=FakeEmbedder()).index_book(book.id)
    result = IndexService(cfg, db, embedder=FakeEmbedder()).index_book(book.id, force=True)
    assert result["chunks"] > 0
    assert bm25.count(db) == result["chunks"]


def test_index_requires_ocr_text(tmp_path, cfg, db):
    book = Book(
        id="empty", title="Empty", file_path="x.pdf", file_name="x.pdf",
        page_count=1, status="queued",
    )
    BooksRepo.create(db, book)
    try:
        IndexService(cfg, db, embedder=FakeEmbedder()).index_book(book.id)
    except ValueError as exc:
        assert "no OCR text" in str(exc)
    else:
        raise AssertionError("indexing an empty book must fail clearly")


def test_parent_ids_are_linked(tmp_path, cfg, db):
    book = _book_with_pages(tmp_path, cfg, db, pages=2)
    IndexService(cfg, db, embedder=FakeEmbedder()).index_book(book.id)
    children = ChunksRepo.all_for_book(db, book.id, level="child")
    assert children
    assert all(child.parent_id for child in children)
    assert all(ChunksRepo.parent_for_child(db, child) is not None for child in children)
