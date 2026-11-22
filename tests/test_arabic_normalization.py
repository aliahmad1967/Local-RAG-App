"""Arabic normalization and BM25 folding."""

from localrag.text.arabic import (
    normalize_for_bm25,
    normalize_presentation_forms,
    remove_tatweel,
    strip_control_chars,
    strip_tashkeel,
)


def test_presentation_forms_use_nfkc():
    # U+FEFB is the presentation form LAM WITH ALEF FINAL FORM.
    assert normalize_presentation_forms("\ufefb") == "لا"


def test_tatweel_and_controls_removed():
    assert remove_tatweel("الـكتاب") == "الكتاب"
    assert strip_control_chars("أ\x00ب\x1fج\r\nد") == "أبج\nد"


def test_tashkeel_removed():
    assert strip_tashkeel("الْكِتَابُ") == "الكتاب"


def test_bm25_folds_alef_ya_and_marbuta():
    assert normalize_for_bm25("أإآ") == "ااا"
    assert normalize_for_bm25("مُدْرِسَةٌ") == "مدرسه"
    assert normalize_for_bm25("القاضى") == "القاضي"


def test_query_and_document_fold_identically():
    document = "كِتابُ الـمدرسَةِ، ١٢٣"
    query = "كتاب المدرسة 123"
    assert normalize_for_bm25(document) == normalize_for_bm25(query)


def test_punctuation_becomes_spaces():
    assert normalize_for_bm25("مرحبا،العالم!") == "مرحبا العالم"
