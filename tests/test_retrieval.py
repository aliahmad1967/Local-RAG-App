"""Phase 3 hybrid retrieval tests with deterministic fake models."""

from __future__ import annotations

from localrag.db.repositories import BooksRepo, PagesRepo
from localrag.index.service import IndexService
from localrag.models.domain import Book, Page
from localrag.retrieval.query_rewrite import NoopRewriter
from localrag.retrieval.rerank import CrossEncoderReranker
from localrag.retrieval.service import RetrievalFilters, RetrievalService

from .data.make_samples import make_sample_pdf  # noqa: F401 - keeps sample generator importable


class FakeEmbedder:
    model = "fake-bge"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [
                10.0 if "book" in lowered or "كتاب" in text else 0.0,
                10.0 if "librar" in lowered or "مكتبة" in text else 0.0,
                10.0 if "journey" in lowered or "رحلة" in text else 0.0,
            ]
            vector[2] += (len(text) % 3) * 0.01
            vectors.append(vector)
        return vectors

    def unload(self) -> None:
        pass


class FakeReranker:
    name = "fake-reranker"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.unloaded = False

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append(list(texts))
        return [float(len(texts) - index) for index in range(len(texts))]

    def unload(self) -> None:
        self.unloaded = True


class FakeRewriter:
    def rewrite(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        return "books and libraries"


def _seed_book(tmp_path, cfg, db, book_id: str, text: str, pages: int = 4) -> Book:
    source = make_sample_pdf(tmp_path / f"{book_id}.pdf", pages=pages, dpi=70)
    book = Book(
        id=book_id, title=f"Book {book_id}", file_path=str(source), file_name=source.name,
        page_count=pages, language="en", status="ingested",
    )
    BooksRepo.create(db, book)
    for number in range(1, pages + 1):
        PagesRepo.upsert(db, Page(
            id=f"{book_id}-p{number}", book_id=book_id, page_no=number,
            status="done", source="ocr", text_clean=text,
            text_normalized=text.lower(), char_count=len(text),
        ))
    return book


def _index(cfg, db) -> None:
    IndexService(cfg, db, embedder=FakeEmbedder()).index_book("hybrid-en")


def test_hybrid_retrieval_reports_every_rank(tmp_path, cfg, db) -> None:
    _seed_book(
        tmp_path, cfg, db, "hybrid-en",
        "Books and libraries preserve knowledge. A good book waits in the library. " * 20,
    )
    _index(cfg, db)

    service = RetrievalService(
        cfg, db, embedder=FakeEmbedder(), reranker=None, rewriter=NoopRewriter()
    )
    result = service.retrieve("books in the library", top_k=3)

    assert result.counts["dense"] >= 3
    assert result.counts["bm25"] >= 3
    assert result.counts["fused"] >= 3
    assert len(result.hits) == 3
    assert all(hit.parent_text for hit in result.hits)
    assert all(hit.rerank_score is None for hit in result.hits)
    for hit in result.hits:
        assert hit.dense_rank is not None and hit.dense_score is not None
        assert hit.bm25_rank is not None and hit.bm25_score is not None
        assert hit.fused_rank is not None and hit.fused_score is not None
    assert result.timings_s["total"] >= 0


def test_reranker_reorders_fused_results_and_respects_filters(tmp_path, cfg, db) -> None:
    _seed_book(
        tmp_path, cfg, db, "hybrid-en",
        "Books and libraries preserve knowledge. A good book waits in the library. " * 20,
    )
    _seed_book(
        tmp_path, cfg, db, "hybrid-other",
        "Journeys and distant roads are described in this other volume. " * 20,
    )
    _index(cfg, db)

    reranker = FakeReranker()
    service = RetrievalService(
        cfg, db, embedder=FakeEmbedder(), reranker=reranker, rewriter=NoopRewriter()
    )
    result = service.retrieve(
        "books library",
        filters=RetrievalFilters(book_ids=["hybrid-en"], page_from=1, page_to=2),
        top_k=2,
    )

    assert len(result.hits) == 2
    assert all(hit.book_id == "hybrid-en" for hit in result.hits)
    assert all(1 <= hit.page_start <= 2 for hit in result.hits)
    assert [hit.rerank_rank for hit in result.hits] == [1, 2]
    assert [hit.rerank_score for hit in result.hits] == [2.0, 1.0]
    assert reranker.calls

    service.close()
    assert reranker.unloaded


def test_query_rewrite_is_recorded_in_debug_payload(tmp_path, cfg, db) -> None:
    _seed_book(
        tmp_path, cfg, db, "hybrid-en",
        "Books and libraries preserve knowledge. A good book waits in the library. " * 20,
    )
    _index(cfg, db)

    service = RetrievalService(
        cfg, db, embedder=FakeEmbedder(), reranker=None, rewriter=FakeRewriter()
    )
    result = service.retrieve("what about it?", top_k=1)

    assert result.rewritten_query == "books and libraries"
    assert result.hits


def test_missing_offline_reranker_has_actionable_error(cfg, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reranker = CrossEncoderReranker(
        cfg, model="definitely/not-cached-localrag-reranker", device="cpu"
    )

    try:
        reranker.score("query", ["passage"])
    except RuntimeError as exc:
        assert "not cached" in str(exc)
        assert "snapshot_download" in str(exc)
    else:
        raise AssertionError("missing reranker cache should raise RuntimeError")
