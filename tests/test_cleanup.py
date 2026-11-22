"""Post-OCR cleanup keeps the original and repairs wrapped lines."""

from localrag.text.cleanup import build_text_views, cleanup_ocr_text, dehyphenate


def test_dehyphenate_english():
    assert dehyphenate("exam-\nple") == "example"


def test_wrapped_line_is_joined():
    raw = "This sentence continues across\na broken line in the scan."
    assert cleanup_ocr_text(raw) == "This sentence continues across a broken line in the scan."


def test_sentence_boundary_is_preserved():
    raw = "First sentence ends here.\nSecond sentence starts here."
    assert "\n" in cleanup_ocr_text(raw)


def test_arabic_normalization_in_clean_copy():
    # Clean copy removes tatweel and presentation forms but keeps the author's
    # diacritics; folding them is the BM25 copy's job.
    raw = "الـكتابُ في المكتبةِ"
    assert cleanup_ocr_text(raw) == "الكتابُ في المكتبةِ"


def test_duplicate_markdown_fence_removed():
    body = "This is a long enough transcription to survive the cleanup."
    raw = f"{body}\n```markdown\n{body}\n```"
    assert cleanup_ocr_text(raw) == body


def test_build_text_views_keeps_raw_untouched():
    raw = "الْـكِتَابُ،  test-\nword"
    clean, normalized = build_text_views(raw)
    assert raw == "الْـكِتَابُ،  test-\nword"
    assert "\u0640" not in clean
    assert "ـ" not in normalized
    assert normalized
