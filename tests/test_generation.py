"""Phase 4: grounded generation, citations, language and memory behaviour."""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from localrag.db.repositories import BooksRepo, ConversationsRepo, MessagesRepo
from localrag.llm.generate import (
    GroundedGenerator,
    REFUSAL,
    has_unresolved_citation,
    normalize_citations,
    remap_citations,
)
from localrag.models.domain import (
    Book,
    Conversation,
    Message,
    RetrievalHit,
    RetrievalResult,
    utcnow,
)
from localrag.retrieval.service import RetrievalFilters


class FakeRetrieval:
    def __init__(self, hits: list[RetrievalHit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        return RetrievalResult(
            query=query,
            rewritten_query=None,
            hits=self.hits,
            filters=(kwargs.get("filters") or RetrievalFilters()).to_dict(),
            counts={"dense": 10, "bm25": 8, "fused": 12, "final": len(self.hits)},
        )


class FakeChat:
    def __init__(
        self,
        pieces: list[str] | None = None,
        complete_result: str = "partial summary",
    ) -> None:
        self.model = "fake-chat:latest"
        self.pieces = pieces or ["Answer ", "[Sample Book, p.3]"]
        self.complete_result = complete_result
        self.calls: list[dict[str, Any]] = []
        self.unloaded = False

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        self.calls.append({"prompt": prompt, "system": system or "", "history": history or []})
        yield from self.pieces

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        self.calls.append({"prompt": prompt, "system": system or "", "history": history or []})
        return self.complete_result

    def unload(self) -> None:
        self.unloaded = True


class FakeMemory:
    def relevant(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return [{"content": "User prefers concise answers.", "score": 0.9}]


def make_hit() -> RetrievalHit:
    return RetrievalHit(
        chunk_id="c1",
        book_id="b1",
        book_title="Sample Book",
        page_start=3,
        page_end=3,
        parent_text="The library keeps old books. [Sample Book, p.3]",
        dense_rank=1,
        bm25_rank=2,
        fused_rank=1,
        rerank_rank=1,
        rerank_score=0.88,
    )


def make_generator(cfg, db, hits=None, chat=None, memory=None) -> GroundedGenerator:
    cfg.data["generation"]["summary_every_messages"] = 0
    return GroundedGenerator(
        cfg, db,
        FakeRetrieval(hits if hits is not None else [make_hit()]),
        memory=memory, chat=chat or FakeChat(),
    )


def test_stream_answer_is_grounded_cited_and_language_specific(cfg, db):
    chat = FakeChat(["إجابة ", "[Sample Book, p.3]"])
    generator = make_generator(cfg, db, chat=chat, memory=FakeMemory())
    events = list(generator.stream_answer(
        "ما هو الكتاب؟", filters=RetrievalFilters(book_ids=["b1"]), language=None
    ))

    assert [event["type"] for event in events] == ["delta", "delta", "done"]
    answer = "".join(event["text"] for event in events[:-1])
    done = events[-1]
    assert answer == "إجابة [Sample Book, p.3]"
    assert done["answer"] == answer
    assert done["language"] == "ar"
    assert done["sources"][0]["label"] == "Sample Book, p.3"
    assert done["memories_used"] == ["User prefers concise answers."]
    system = chat.calls[0]["system"]
    assert "Answer ONLY in Arabic" in system
    assert REFUSAL["ar"] in system
    assert "User prefers concise answers." in system
    assert "[Sample Book, p.3]" in chat.calls[0]["prompt"]
    assert generator.retrieval.calls[0]["filters"] is not None


def test_compare_mode_asks_for_cited_differences_from_both_books(cfg, db):
    hits = [
        make_hit(),
        RetrievalHit(
            chunk_id="c2", book_id="b2", book_title="Other Book",
            page_start=5, page_end=5,
            parent_text="The other book describes different knowledge. [Other Book, p.5]",
        ),
    ]
    chat = FakeChat(["The books differ. [Sample Book, p.3] [Other Book, p.5]"])
    generator = make_generator(cfg, db, hits=hits, chat=chat)

    answer = generator.answer("Compare the books.", mode="compare", language="en")

    assert "Compare the sources explicitly" in chat.calls[0]["system"]
    assert "[Sample Book, p.3]" in chat.calls[0]["prompt"]
    assert "[Other Book, p.5]" in chat.calls[0]["prompt"]
    assert answer["answer"].startswith("The books differ.")


def test_empty_model_response_is_reported_as_error(cfg, db):
    generator = make_generator(cfg, db, chat=FakeChat([""]))

    with pytest.raises(RuntimeError, match="returned an empty answer"):
        list(generator.stream_answer("What is the book?", language="en"))


def test_english_source_arabic_question_gets_arabic_answer(cfg, db):
    chat = FakeChat(["The answer is in the source. [Sample Book, p.3]"])
    generator = make_generator(cfg, db, chat=chat)
    done = generator.answer("هل تحتوي المكتبة على كتب قديمة؟")

    assert done["language"] == "ar"
    assert "Answer ONLY in Arabic" in chat.calls[0]["system"]


def test_no_sources_returns_refusal_without_calling_model(cfg, db):
    chat = FakeChat()
    generator = make_generator(cfg, db, hits=[], chat=chat)
    done = generator.answer("Where is the library?")

    assert done["answer"] == "There are no indexed sources for this scope yet."
    assert done["language"] == "en"
    assert done["sources"] == []
    assert chat.calls == []


def test_context_warning_when_prompt_exceeds_budget(cfg, db):
    cfg.data["generation"]["context_tokens"] = 1024
    cfg.data["generation"]["context_warning_tokens"] = 1
    generator = make_generator(cfg, db)
    done = generator.answer("What is in the library?")

    assert any("Context is approximately" in warning for warning in done["warnings"])


def test_compare_mode_adds_comparison_instruction(cfg, db):
    chat = FakeChat()
    generator = make_generator(cfg, db, chat=chat)
    generator.answer("Compare these books", mode="compare")

    assert "Compare the sources explicitly" in chat.calls[0]["system"]


def test_build_history_uses_sliding_window(cfg, db):
    conversation = Conversation(id="conv-1")
    ConversationsRepo.create(db, conversation)
    for index in range(10):
        MessagesRepo.add(db, Message(
            id=f"m{index}",
            conversation_id=conversation.id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"turn {index}",
        ))
    generator = make_generator(cfg, db)
    generator.cfg.data["generation"]["history_turns"] = 1
    history = generator.build_history(ConversationsRepo.get(db, conversation.id))

    assert [item["content"] for item in history] == ["turn 8", "turn 9"]


def test_rolling_summary_updates_conversation(monkeypatch, cfg, db):
    cfg.data["generation"]["summary_every_messages"] = 2
    conversation = Conversation(id="conv-2", summary="Old summary")
    ConversationsRepo.create(db, conversation)
    for index in range(2):
        MessagesRepo.add(db, Message(
            id=f"s{index}", conversation_id=conversation.id,
            role="user", content=f"fact {index}",
        ))

    class FakeSummarizer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def complete(self, prompt: str, system: str | None = None) -> str:
            assert "Old summary" in prompt
            assert "fact 0" in prompt and "fact 1" in prompt
            return "Rolled summary"

        def unload(self) -> None:
            pass

    monkeypatch.setattr("localrag.llm.generate.OllamaChat", FakeSummarizer)
    generator = GroundedGenerator(cfg, db, FakeRetrieval([make_hit()]), chat=FakeChat())

    assert generator.maybe_summarize(conversation.id) is True
    stored = ConversationsRepo.get(db, conversation.id)
    assert stored is not None
    assert stored.summary == "Rolled summary"
    assert generator.maybe_summarize(conversation.id) is False


def test_stream_summary_is_map_reduce_with_citations(cfg, db):
    book = Book(id="b1", title="Sample Book", file_path="x.pdf", file_name="x.pdf", page_count=3)
    BooksRepo.create(db, book)
    from localrag.models.domain import Chunk

    for index in range(1, 3):
        from localrag.db.repositories import ChunksRepo

        ChunksRepo.upsert_many(db, [Chunk(
            id=f"chunk-{index}", book_id=book.id, level="child", child_index=index,
            page_start=index, page_end=index,
            text=f"Page {index} describes the library. [Sample Book, p.{index}]",
        )])
    chat = FakeChat(["A grounded summary. [Sample Book, p.1]"])
    generator = make_generator(cfg, db, chat=chat)
    events = list(generator.stream_summary("Summarize pages 1-2", book_ids=[book.id]))

    assert events[-1]["mode"] == "summarize"
    assert events[-1]["counts"]["chunks"] == 2
    assert len(events[-1]["sources"]) == 2
    assert any("PARTIAL SUMMARIES" in call["prompt"] for call in chat.calls)


def test_generic_citation_labels_are_remapped_to_sources():
    sources = [{
        "label": "Sample Book, p.3", "book_title": "Sample Book",
        "page_start": 3, "page_end": 3,
    }]
    assert remap_citations("Claim [Book, p.3]", sources) == "Claim [Sample Book, p.3]"
    assert remap_citations("Claim [Sample Book, p.3]", sources) == "Claim [Sample Book, p.3]"


def test_citation_within_mult_page_source_maps_to_range_label():
    sources = [{
        "label": "Sample Book, p.1-3", "book_title": "Sample Book",
        "page_start": 1, "page_end": 3,
    }]

    assert remap_citations("Claim [Book, p.2]", sources) == "Claim [Sample Book, p.1-3]"


def test_unresolved_citation_is_detected():
    sources = [{"label": "Sample Book, p.1-3", "page_start": 1, "page_end": 3}]

    assert has_unresolved_citation("Claim [Book Title, p.N]", sources)
    assert not has_unresolved_citation("Claim [Sample Book, p.1-3]", sources)


def test_normalize_citations_splits_combined_brackets():
    raw = "Claim [Book, p.1; Book, p.2] and [Other, p.7؛ Other, p.9]."
    assert normalize_citations(raw) == (
        "Claim [Book, p.1] [Book, p.2] and [Other, p.7] [Other, p.9]."
    )
    assert normalize_citations("Already fine [Book, p.3]") == "Already fine [Book, p.3]"


def test_multi_page_answer_is_emitted_as_clickable_citations(cfg, db):
    chat = FakeChat(["Claim [Book, p.1; Book, p.2]"])
    hits = [
        RetrievalHit(
            chunk_id=f"c{page}", book_id="b1", book_title="Book",
            page_start=page, page_end=page, parent_text=f"Source page {page}.",
        )
        for page in (1, 2)
    ]
    generator = make_generator(cfg, db, chat=chat, hits=hits)
    events = list(generator.stream_answer("What is claimed?", language="en"))

    assert events[-1]["answer"] == "Claim [Book, p.1] [Book, p.2]"
    assert events[0]["text"] == "Claim [Book, p.1] [Book, p.2]"


def test_wrong_answer_language_is_repaired_once(cfg, db):
    chat = FakeChat(
        ["الكتاب يحتوي على معرفة. [Sample Book, p.3]"],
        complete_result="The book contains knowledge. [Sample Book, p.3]",
    )
    generator = make_generator(cfg, db, chat=chat)
    done = generator.answer("What does the book contain?", language="en")

    assert done["answer"] == "The book contains knowledge. [Sample Book, p.3]"
    assert any("repaired" in warning for warning in done["warnings"])
    assert "English" in chat.calls[-1]["system"]


def test_refusal_with_sources_is_checked_once(cfg, db):
    chat = FakeChat([REFUSAL["ar"]], complete_result="العلم نور. [Sample Book, p.3]")
    generator = make_generator(cfg, db, chat=chat)
    done = generator.answer("ما هو الكتاب؟", language="ar")

    assert done["answer"] == "العلم نور. [Sample Book, p.3]"
    assert len(chat.calls) == 2  # one stream + one bounded repair
    assert "refused" in chat.calls[1]["system"]


def test_paraphrased_refusal_is_normalized(cfg, db):
    chat = FakeChat(
        ["عاصمة اليابان غير محددة في المصادر المتاحة."],
        complete_result=REFUSAL["ar"],
    )
    generator = make_generator(cfg, db, chat=chat)
    done = generator.answer("ما عاصمة اليابان؟", language="ar")

    assert done["answer"] == REFUSAL["ar"]
    assert len(chat.calls) == 2


def test_refusal_with_explanation_is_normalized_to_exact_refusal(cfg, db):
    chat = FakeChat([
        "The available sources do not contain the answer to this question. "
        "The answer is not present. [Sample Book, p.3]"
    ])
    generator = make_generator(cfg, db, chat=chat)

    assert generator.answer("What is unsupported?", language="en")["answer"] == REFUSAL["en"]


def test_missing_citations_fail_closed_after_repair(cfg, db):
    chat = FakeChat(["The library keeps books."], complete_result="The library keeps books.")
    generator = make_generator(cfg, db, chat=chat)
    done = generator.answer("What does the library keep?", language="en")

    assert done["answer"] == REFUSAL["en"]


def test_invalid_model_citation_fails_closed(cfg, db):
    chat = FakeChat(
        ["Tokyo is Japan's capital. [Book Title, p.N]"],
        complete_result="Tokyo is Japan's capital. [Book Title, p.N]",
    )
    generator = make_generator(cfg, db, chat=chat)

    assert generator.answer("What is Tokyo?", language="en")["answer"] == REFUSAL["en"]
