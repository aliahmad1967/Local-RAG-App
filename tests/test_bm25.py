"""FTS5 BM25 index and query normalization."""

from __future__ import annotations

from localrag.db.repositories import BooksRepo, ChunksRepo
from localrag.index import bm25
from localrag.models.domain import Book, Chunk

from .data.make_samples import make_sample_pdf  # noqa: F401 - keeps sample generator importable


def _book(db, book_id: str) -> None:
    if BooksRepo.get(db, book_id) is None:
        BooksRepo.create(db, Book(
            id=book_id, title=book_id, file_path="x.pdf", file_name="x.pdf", page_count=10,
        ))


def _chunk(db, book_id: str, number: int, text: str) -> Chunk:
    _book(db, book_id)
    chunk = Chunk(
        id=f"{book_id}-c-{number}", book_id=book_id, level="child",
        child_index=number, page_start=number,
        text=text,
        text_normalized=__import__(
            "localrag.text.arabic", fromlist=["normalize_for_bm25"]
        ).normalize_for_bm25(text),
    )
    ChunksRepo.upsert_many(db, [chunk])
    return chunk


def test_query_uses_the_same_normalization_as_the_index(cfg):
    from localrag.index.bm25 import build_match_query

    assert build_match_query("الْكِتَابُ", cfg) == '"الكتاب"'
    assert build_match_query("أ إ آ المدرسة", cfg) == '"ا" OR "ا" OR "ا" OR "المدرسه"'


def test_bm25_search_ranks_matching_chunk(db, cfg):
    _chunk(db, "b1", 1, "الكتابُ في المكتبةِ القديمةِ")
    _chunk(db, "b1", 2, "الطقسُ今天 جميلٌ في الصباحِ")
    _chunk(db, "b1", 3, "مكتبةٌ أخرى فيها كتبٌ كثيرةٌ")
    assert bm25.count(db) == 3

    results = bm25.search(db, "الكتاب", cfg=cfg)
    assert results
    assert results[0]["page_no"] in (1, 3)
    assert all(item["chunk_id"].startswith("b1") for item in results)


def test_bm25_search_respects_book_filter(db, cfg):
    _chunk(db, "b1", 1, "chapter about libraries and books")
    _chunk(db, "b2", 1, "chapter about libraries and books")
    results = bm25.search(db, "libraries", book_id="b2", cfg=cfg)
    assert results
    assert {item["book_id"] for item in results} == {"b2"}


def test_empty_query_returns_no_results(db, cfg):
    assert bm25.search(db, "!!!", cfg=cfg) == []
