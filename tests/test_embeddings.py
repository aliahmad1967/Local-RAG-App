"""Embedding cache, batching and resumability."""

from __future__ import annotations

import numpy as np

from localrag.db.repositories import BooksRepo, ChunksRepo, EmbeddingRepo
from localrag.index.embeddings import content_hash
from localrag.models.domain import Book, Chunk
from localrag.index.service import IndexService


class FakeEmbedder:
    model = "fake-bge"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def unload(self) -> None:
        pass


def _chunks(db, book_id: str, count: int) -> list[Chunk]:
    if BooksRepo.get(db, book_id) is None:
        BooksRepo.create(db, Book(
            id=book_id, title=book_id, file_path="x.pdf", file_name="x.pdf", page_count=count,
        ))
    chunks = [
        Chunk(
            id=f"{book_id}-c-{i}", book_id=book_id, level="child", child_index=i,
            page_start=i + 1, text=f"unique chunk content number {i}",
            text_normalized=f"unique chunk content number {i}",
        )
        for i in range(count)
    ]
    ChunksRepo.upsert_many(db, chunks)
    return chunks


def test_content_hash_is_model_specific_and_stable():
    assert content_hash("m", "text") == content_hash("m", "text")
    assert content_hash("m", "text") != content_hash("m2", "text")
    assert content_hash("m", "text") != content_hash("m", "text2")


def test_first_run_embeds_all_chunks(db, cfg):
    embedder = FakeEmbedder()
    _chunks(db, "b1", 5)
    service = IndexService(cfg, db, embedder=embedder)
    assert service.embed_children("b1") == 5
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == 5
    assert EmbeddingRepo.count_cached(db) == 5


def test_second_run_uses_cache_and_skips_ollama(db, cfg):
    embedder = FakeEmbedder()
    _chunks(db, "b1", 5)
    service = IndexService(cfg, db, embedder=embedder)
    service.embed_children("b1")
    embedder.calls.clear()
    assert service.embed_children("b1") == 5
    assert embedder.calls == []


def test_resume_only_embeds_new_chunks(db, cfg):
    embedder = FakeEmbedder()
    chunks = _chunks(db, "b1", 4)
    service = IndexService(cfg, db, embedder=embedder)
    service.embed_children("b1")
    embedder.calls.clear()

    extra = Chunk(
        id="b1-c-99", book_id="b1", level="child", child_index=99, page_start=99,
        text="a brand new chunk", text_normalized="a brand new chunk",
    )
    ChunksRepo.upsert_many(db, [extra])
    assert service.embed_children("b1") == 5
    assert embedder.calls == [["a brand new chunk"]]
    assert len(chunks) == 4


def test_duplicate_text_embedded_once_but_linked_twice(db, cfg):
    embedder = FakeEmbedder()
    BooksRepo.create(db, Book(
        id="b1", title="b1", file_path="x.pdf", file_name="x.pdf", page_count=2,
    ))
    chunks = [
        Chunk(id=f"b1-c-{i}", book_id="b1", level="child", child_index=i,
              page_start=i + 1, text="identical text", text_normalized="identical text")
        for i in range(2)
    ]
    ChunksRepo.upsert_many(db, chunks)
    service = IndexService(cfg, db, embedder=embedder)
    assert service.embed_children("b1") == 2
    assert embedder.calls == [["identical text"]]
    assert EmbeddingRepo.count_cached(db) == 1


def test_cosine_scores_are_ordered():
    from localrag.index.vector_store import cosine_scores, pack_vector

    vectors = [pack_vector([1, 0, 0]), pack_vector([0, 1, 0]), pack_vector([0.9, 0.1, 0])]
    scores = cosine_scores([1, 0, 0], vectors)
    assert np.argmax(scores) == 0
    assert scores[0] > scores[1]
