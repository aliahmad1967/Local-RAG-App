"""Arabic Unicode normalization.

Two distinct outputs are produced by the pipeline:

* **clean** - for display and chunking; keeps the author's letters as-is.
* **search-normalized** - for BM25 only; folds alef/ya/ta-marbuta variants and
  removes tashkeel so "الكتاب" and "الكتابَ" match the same index entry.

The same function is applied to queries at retrieval time (Phase 3).
"""

from __future__ import annotations

import re
import unicodedata

TATWEEL = "\u0640"
TASHKEEL = "\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed"
_TASHKEEL_RE = re.compile(f"[{TASHKEEL}]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

_ALEF_MAP = {ord("\u0623"): "\u0627", ord("\u0625"): "\u0627", ord("\u0622"): "\u0627"}
_YA_MAP = {ord("\u0649"): "\u064a"}
_MARBUATA_MAP = {ord("\u0629"): "\u0647"}
# Arabic-Indic and Extended Arabic-Indic digits -> ASCII, for BM25 only.
_DIGIT_MAP = {**{ord("٠") + i: str(i) for i in range(10)},
              **{ord("۰") + i: str(i) for i in range(10)}}
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


def strip_control_chars(text: str) -> str:
    """Remove C0/C1 control characters, keeping newline and tab."""
    return _CONTROL_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def normalize_presentation_forms(text: str) -> str:
    """NFKC folds Arabic Presentation Forms A/B into standard code points."""
    return unicodedata.normalize("NFKC", text)


def remove_tatweel(text: str) -> str:
    return text.replace(TATWEEL, "")


def strip_tashkeel(text: str) -> str:
    return _TASHKEEL_RE.sub("", text)


def normalize_alef_variants(text: str) -> str:
    return text.translate(_ALEF_MAP)


def normalize_for_bm25(
    text: str,
    strip_diacritics: bool = True,
    fold_alef: bool = True,
    fold_ya_marbuta: bool = True,
) -> str:
    """Canonical form used **only** for the BM25 index and BM25 queries."""
    result = normalize_presentation_forms(text)
    result = remove_tatweel(result)
    if strip_diacritics:
        result = strip_tashkeel(result)
    translation: dict[int, str] = {}
    if fold_alef:
        translation.update(_ALEF_MAP)
    if fold_ya_marbuta:
        translation.update(_YA_MAP)
        translation.update(_MARBUATA_MAP)
    if translation:
        result = result.translate(translation)
    result = result.translate(_DIGIT_MAP)
    result = result.lower()
    result = _NON_WORD_RE.sub(" ", result)
    return _SPACES_RE.sub(" ", result).strip()


def has_arabic(text: str) -> bool:
    return any(
        "\u0600" <= ch <= "\u06ff" or "\ufb50" <= ch <= "\ufeff"
        for ch in text
    )
