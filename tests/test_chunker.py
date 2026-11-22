"""Phase 2 chunker: sentence safety, overlap, page boundaries, parent windows."""

from __future__ import annotations

import uuid

from localrag.index.chunker import (
    chunk_page,
    chunk_pages,
    estimate_tokens,
    split_sentences,
)
from localrag.models.domain import Page


def _page(text: str, no: int = 1) -> Page:
    return Page(
        id=str(uuid.uuid4()), book_id="book", page_no=no,
        text_clean=text, text_normalized=text, status="done",
    )


def _sentences(count: int, language: str = "en") -> str:
    if language == "ar":
        parts = [f"هذه هي الجملة رقم {i} في النص العربي الطويل." for i in range(count)]
    else:
        parts = [f"This is sentence number {i} in a long English paragraph." for i in range(count)]
    return " ".join(parts)


def test_sentence_split_handles_arabic_punctuation():
    text = "الأولى هنا. الثانية هناك؟ الثالثة قبل الرابعة؛ الخامسة انتهت!"
    sentences = split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0].endswith(".")
    assert sentences[1].endswith("؟")


def test_estimate_tokens_counts_words_and_punctuation():
    assert estimate_tokens("one two.") == 3


def test_chunks_never_split_a_sentence():
    page = _page(_sentences(120))
    specs = chunk_page(page)
    children = [spec for spec in specs if spec.level == "child"]
    assert len(children) > 1
    for child in children:
        assert child.text.endswith(".") or child.text.endswith("؟")
        assert 1 <= child.token_count <= 600


def test_arabic_chunking_target_and_overlap():
    page = _page(_sentences(150, "ar"))
    children = [spec for spec in chunk_page(page) if spec.level == "child"]
    assert len(children) > 1
    assert all(spec.token_count <= 600 for spec in children)
    # Consecutive children share at least one sentence (15% overlap).
    first_words = set(children[0].text.split())
    second_words = set(children[1].text.split())
    assert first_words & second_words, "expected overlapping sentences between chunks"


def test_heading_is_detected_and_attached():
    page = _page("الفصل الأول: في شأن الكتب\n\n" + _sentences(60, "ar"))
    specs = chunk_page(page)
    assert any(spec.heading and spec.heading.startswith("الفصل") for spec in specs)


def test_book_chunks_reach_target_across_pages_and_keep_ranges():
    pages = [_page(_sentences(50), no=1), _page(_sentences(50), no=2)]
    specs = chunk_pages(pages)
    children = [spec for spec in specs if spec.level == "child"]
    assert len(children) > 1
    assert any(spec.page_start < spec.page_end for spec in children)
    assert all(spec.page_start <= spec.page_end for spec in children)
    assert all(spec.token_count <= 600 for spec in children)
    assert sum(400 <= spec.token_count <= 600 for spec in children) >= len(children) - 1


def test_parent_groups_children_and_links_them():
    page = _page(_sentences(200))
    specs = chunk_page(page)
    parents = [spec for spec in specs if spec.level == "parent"]
    children = [spec for spec in specs if spec.level == "child"]
    assert parents and children
    assert max(child.parent_index for child in children) == len(parents) - 1
    assert all(parent.token_count >= child.token_count for parent in parents for child in children)


def test_title_only_page_is_not_indexed():
    page = _page("الفصل الأول\n\n154")
    assert chunk_page(page) == []


def test_empty_and_failed_pages_are_skipped():
    pages = [
        _page(""),
        Page(id="x", book_id="book", page_no=2, text_clean="ignored", status="failed"),
        _page(_sentences(20), no=3),
    ]
    specs = chunk_pages(pages)
    assert specs
    assert all(spec.page_start == 3 for spec in specs)
