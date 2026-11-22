"""Script-based language detection (no model download, works offline)."""

from __future__ import annotations

import unicodedata

_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x0870, 0x089F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in _ARABIC_RANGES)


def detect_language(text: str, min_chars: int = 20) -> str:
    """Return 'ar', 'en', 'mixed' or 'unknown' from the Latin/Arabic letter ratio."""
    arabic = latin = 0
    for ch in text:
        if _is_arabic_char(ch) and unicodedata.category(ch).startswith("L"):
            arabic += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin += 1
    total = arabic + latin
    if total < min_chars:
        return "unknown"
    share_arabic = arabic / total
    if share_arabic >= 0.85:
        return "ar"
    if share_arabic <= 0.15:
        return "en"
    return "mixed"
