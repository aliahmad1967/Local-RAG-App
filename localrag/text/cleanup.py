"""Post-OCR cleanup: dehyphenation and broken-line repair.

The untouched OCR output is always preserved in `pages.text_original`.
"""

from __future__ import annotations

import re
from typing import Any

from .arabic import (
    normalize_for_bm25,
    normalize_presentation_forms,
    remove_tatweel,
    strip_control_chars,
)

_SENTENCE_END = set(".!?؟۔؛:。")
_LINE_START_MARKER = set("•●▪◦-*—")
_HYPHEN_BREAK = re.compile(r"(?<=\w)-\n(?=\w)")
_SPACES = re.compile(r"[ \t\u00a0]+")
_MANY_BLANKS = re.compile(r"\n{3,}")
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.DOTALL)


def strip_ocr_markdown(text: str) -> str:
    """Vision OCR models sometimes append the same page again inside ``` fences.

    Keep the plain transcription when one exists; otherwise unwrap the first
    fenced block so no text is lost.
    """
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        return text
    plain = _FENCE_RE.sub("", text).strip()
    if len(plain) >= 40:
        return plain
    return matches[0].group(1).strip()


def dehyphenate(text: str) -> str:
    """Join words split as `exam-\\nple` (and Arabic `-\\n` when present)."""
    return _HYPHEN_BREAK.sub("", text)


def join_wrapped_lines(text: str) -> str:
    """Rebuild lines that OCR broke in the middle of a sentence.

    A line break is kept when the previous line ends with sentence punctuation,
    when the previous line looks like a short heading, or when the next line
    starts with a list marker/number. Otherwise lines are joined with a space.
    """
    raw_lines = text.split("\n")
    out: list[str] = []
    for raw in raw_lines:
        line = _SPACES.sub(" ", raw).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if not out or out[-1] == "":
            out.append(line)
            continue

        previous = out[-1]
        keep_break = (
            previous[-1] in _SENTENCE_END
            or len(previous) < 18
            or line[0] in _LINE_START_MARKER
            or (line[0].isdigit() and len(line) < 12)
        )
        if keep_break:
            out.append(line)
        else:
            out[-1] = f"{previous} {line}"
    return _MANY_BLANKS.sub("\n\n", "\n".join(out)).strip()


def cleanup_ocr_text(raw: str, keep_paragraph_breaks: bool = True) -> str:
    """Produce the display/clean copy from raw OCR or embedded text."""
    text = strip_control_chars(raw)
    text = strip_ocr_markdown(text)
    text = normalize_presentation_forms(text)
    text = remove_tatweel(text)
    text = dehyphenate(text)
    text = join_wrapped_lines(text) if keep_paragraph_breaks else _SPACES.sub(" ", text)
    return text.strip()


def build_text_views(raw: str, cfg: Any | None = None) -> tuple[str, str]:
    """Return (clean_text, bm25_normalized_text)."""
    keep_breaks = True
    cfg_strip = True
    cfg_alef = True
    cfg_ya = True
    if cfg is not None:
        keep_breaks = bool(cfg.get("text", "keep_paragraph_breaks", default=True))
        cfg_strip = bool(cfg.get("text", "bm25_strip_tashkeel", default=True))
        cfg_alef = bool(cfg.get("text", "bm25_normalize_alef", default=True))
        cfg_ya = bool(cfg.get("text", "bm25_normalize_ya_marbuta", default=True))
    clean = cleanup_ocr_text(raw, keep_paragraph_breaks=keep_breaks)
    normalized = normalize_for_bm25(
        clean, strip_diacritics=cfg_strip, fold_alef=cfg_alef, fold_ya_marbuta=cfg_ya
    )
    return clean, normalized
