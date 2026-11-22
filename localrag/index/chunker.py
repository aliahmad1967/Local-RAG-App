"""Page-aware, structure-aware chunking with sentence-safe boundaries.

Design rules (Phase 2 spec):
  * Never split mid-sentence. Arabic sentence ends are `. ! ؟ ۔ ；` and `؛`;
    `،` is a comma and only used as a *fallback* break for over-long sentences.
  * Target 400-600 tokens with ~15% overlap between consecutive children.
  * Chunks may span pages; their inclusive page range is retained for citations.
  * Headings (chapter / section titles) are detected and attached as metadata.
  * Parent windows group consecutive children (~3x tokens) and are stored
    alongside children; retrieval searches children, generation reads parents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from ..models.domain import Page

# A token is a word or a punctuation mark. Arabic clitics are counted as part of
# the word (conservative: it keeps chunks within the 400-600 budget on GPU).
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_SENTENCE_END = ".!?؟۔؛;。！"
_SENTENCE_TAIL_RE = re.compile(rf"([{re.escape(_SENTENCE_END)}])\s+")
_COMMA_SPLIT_RE = re.compile(r"(?<=[،,؛;])\s+")

_HEADING_PREFIXES = (
    "الفصل", "الباب", "المبحث", "القسم", "الجزء", "السفر", "النصل",
    "chapter", "part", "section", "book ", "volume", "مقدمة", "خاتمة",
    "المقدمة", "الخاتمة", "فهرس", "المحتويات",
)
_HEADING_RE = re.compile(
    r"^\s*(?:\d+[\.\)،\-–—\s]*|[IVXLC]+[\.\)،\-–—\s]*)?"
    r"(?:الفصل|الباب|المبحث|القسم|الجزء|chapter|part|section|مقدمة|خاتمة)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkSpec:
    """A chunk before it is persisted (no database IDs yet)."""

    level: str
    parent_index: int
    child_index: int
    page_start: int
    page_end: int
    heading: str | None
    text: str
    token_count: int
    start_char: int
    end_char: int
    parent_id: str | None = None  # set on children once parent IDs exist


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate (words + punctuation) for budget control."""
    return len(_WORD_RE.findall(text))


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 140:
        return False
    if stripped.endswith(tuple(_SENTENCE_END)):
        return False
    lowered = stripped.lower()
    if _HEADING_RE.match(lowered):
        return True
    words = stripped.split()
    return (
        1 <= len(words) <= 12
        and any(lowered.startswith(prefix) for prefix in _HEADING_PREFIXES)
    )


def split_sentences(text: str) -> list[str]:
    """Split on Arabic/English sentence punctuation, keeping the terminator.

    Paragraph breaks are always respected. A heading line is its own unit.
    """
    result: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n", text):
        paragraph = " ".join(paragraph.split()).strip()
        if not paragraph:
            continue
        pieces = _SENTENCE_TAIL_RE.split(paragraph)
        current = ""
        for index in range(0, len(pieces), 2):
            sentence = pieces[index].strip()
            punctuation = pieces[index + 1] if index + 1 < len(pieces) else ""
            if sentence:
                candidate = f"{sentence}{punctuation}".strip()
                current = f"{current} {candidate}".strip() if current else candidate
                if punctuation:
                    result.append(current)
                    current = ""
        if current:
            result.append(current)
    return result


def split_overlong(sentence: str, max_tokens: int) -> list[str]:
    """Break a sentence that exceeds the budget: commas first, then words.

    This is the last resort — normal prose never reaches it.
    """
    if estimate_tokens(sentence) <= max_tokens:
        return [sentence]
    clauses = [part.strip() for part in _COMMA_SPLIT_RE.split(sentence) if part.strip()]
    if len(clauses) > 1:
        out: list[str] = []
        buffer = ""
        for clause in clauses:
            candidate = f"{buffer}، {clause}" if buffer else clause
            if buffer and estimate_tokens(candidate) > max_tokens:
                out.extend(split_overlong(buffer, max_tokens))
                buffer = clause
            else:
                buffer = candidate
        if buffer:
            out.extend(split_overlong(buffer, max_tokens))
        return out
    words = sentence.split()
    out, buffer = [], ""
    for word in words:
        candidate = f"{buffer} {word}".strip()
        if buffer and estimate_tokens(candidate) > max_tokens:
            out.append(buffer)
            buffer = word
        else:
            buffer = candidate
    if buffer:
        out.append(buffer)
    return out


@dataclass
class _Unit:
    text: str
    heading: str | None
    tokens: int
    page_no: int
    start_char: int
    end_char: int


def _page_units(page: Page, max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    heading: str | None = None
    cursor = 0
    for block in re.split(r"\n\s*\n", page.text_clean or ""):
        block = " ".join(block.split()).strip()
        if not block:
            continue
        block_start = page.text_clean.find(block, cursor)
        if block_start < 0:
            block_start = cursor
        cursor = block_start
        if _is_heading(block):
            heading = block
            end = block_start + len(block)
            units.append(_Unit(block, heading, estimate_tokens(block), page.page_no, block_start, end))
            cursor = end
            continue
        for sentence in split_sentences(block):
            for piece in split_overlong(sentence, max_tokens):
                if piece:
                    start = page.text_clean.find(piece, cursor)
                    if start < 0:
                        start = cursor
                    end = start + len(piece)
                    units.append(_Unit(
                        piece, heading, estimate_tokens(piece), page.page_no, start, end
                    ))
                    cursor = end
    return units


def _pack_units(
    units: list[_Unit], target: int, max_tokens: int, min_tokens: int, overlap: int
) -> list[list[_Unit]]:
    if not units:
        return []
    groups: list[list[_Unit]] = []
    buffer: list[_Unit] = []
    buffered = 0

    def emit() -> None:
        nonlocal buffer, buffered
        if not buffer:
            return
        # Merge a tiny tail into the previous group when it still fits.
        if groups and buffered < min_tokens:
            previous = groups[-1]
            if sum(u.tokens for u in previous) + buffered <= max_tokens:
                groups[-1] = previous + buffer
                buffer, buffered = [], 0
                return
        groups.append(buffer)
        buffer, buffered = [], 0

    for unit in units:
        if buffer and buffered + unit.tokens > target:
            emit()
            # Carry the tail sentences forward (15% overlap) — a real boundary.
            carry: list[_Unit] = []
            carried = 0
            for previous_unit in reversed(buffer if buffer else groups[-1]):
                if carried + previous_unit.tokens > overlap:
                    break
                carry.insert(0, previous_unit)
                carried += previous_unit.tokens
            buffer, buffered = carry, carried
        # Never let overlap push a chunk past the hard token budget.
        while buffer and buffered + unit.tokens > max_tokens:
            buffered -= buffer.pop(0).tokens
        buffer.append(unit)
        buffered += unit.tokens
    emit()
    return groups


def _parent_text(units: list[_Unit]) -> str:
    """Join child texts into a parent window, dropping duplicated overlap sentences."""
    seen: set[str] = set()
    parts: list[str] = []
    for unit in units:
        for sentence in split_sentences(unit.text):
            key = " ".join(sentence.split())
            if key not in seen:
                seen.add(key)
                parts.append(sentence)
    return "\n\n".join(parts)


def chunk_page(page: Page, cfg: Any | None = None) -> list[ChunkSpec]:
    """Chunk one page into child + parent specs."""
    target = _cfg(cfg, "target_tokens", 500)
    max_tokens = _cfg(cfg, "max_tokens", 600)
    min_tokens = _cfg(cfg, "min_tokens", 400)
    overlap_ratio = _cfg(cfg, "overlap_ratio", 0.15)
    parent_multiplier = _cfg(cfg, "parent_multiplier", 3)
    min_chunk_tokens = _cfg(cfg, "min_chunk_tokens", 10)

    overlap = max(1, round(target * overlap_ratio))
    max_tokens = max(max_tokens, target)
    min_tokens = min(min_tokens, target)
    units = _page_units(page, max_tokens)
    groups = _pack_units(units, target, max_tokens, min_tokens, overlap)
    # Scanned books have title/blank pages whose OCR yields a handful of tokens.
    # Indexing those as standalone chunks only adds retrieval noise.
    groups = [group for group in groups if sum(unit.tokens for unit in group) >= min_chunk_tokens]
    if not groups:
        return []

    children = _child_specs(groups)
    return _with_parents(children, target, parent_multiplier)


def _child_specs(groups: list[list[_Unit]]) -> list[ChunkSpec]:
    children: list[ChunkSpec] = []
    for index, group in enumerate(groups):
        children.append(ChunkSpec(
            level="child", parent_index=0, child_index=index,
            page_start=group[0].page_no, page_end=group[-1].page_no,
            heading=next((unit.heading for unit in group if unit.heading), None),
            text="\n\n".join(unit.text for unit in group),
            token_count=sum(unit.tokens for unit in group),
            start_char=group[0].start_char, end_char=group[-1].end_char,
        ))
    return children


def _with_parents(
    children: list[ChunkSpec], target: int, parent_multiplier: int
) -> list[ChunkSpec]:
    if not children:
        return []

    parent_specs: list[ChunkSpec] = []
    parent_groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for index, child in enumerate(children):
        if current and current_tokens + child.token_count > target * parent_multiplier:
            parent_groups.append(current)
            current, current_tokens = [], 0
        current.append(index)
        current_tokens += child.token_count
    if current:
        parent_groups.append(current)

    linked_children = list(children)
    for pindex, indexes in enumerate(parent_groups):
        grouped_units = [
            _Unit(
                children[i].text, children[i].heading, children[i].token_count,
                children[i].page_start, children[i].start_char, children[i].end_char,
            )
            for i in indexes
        ]
        text = _parent_text(grouped_units)
        parent_specs.append(ChunkSpec(
            level="parent", parent_index=pindex, child_index=children[indexes[0]].child_index,
            page_start=children[indexes[0]].page_start,
            page_end=children[indexes[-1]].page_end,
            heading=next((children[i].heading for i in indexes if children[i].heading), None),
            text=text, token_count=estimate_tokens(text),
            start_char=children[indexes[0]].start_char,
            end_char=children[indexes[-1]].end_char,
        ))
        for i in indexes:
            linked_children[i] = replace(children[i], parent_index=pindex)
    return parent_specs + linked_children


def chunk_pages(pages: Iterable[Page], cfg: Any | None = None) -> list[ChunkSpec]:
    """Chunk a book across page boundaries while preserving page ranges."""
    target = _cfg(cfg, "target_tokens", 500)
    max_tokens = max(_cfg(cfg, "max_tokens", 600), target)
    min_tokens = min(_cfg(cfg, "min_tokens", 400), target)
    overlap = max(1, round(target * _cfg(cfg, "overlap_ratio", 0.15)))
    parent_multiplier = _cfg(cfg, "parent_multiplier", 3)
    min_chunk_tokens = _cfg(cfg, "min_chunk_tokens", 10)

    units = [
        unit
        for page in sorted(pages, key=lambda item: item.page_no)
        if page.status in ("done", "flagged") and page.text_clean.strip()
        for unit in _page_units(page, max_tokens)
    ]
    groups = _pack_units(units, target, max_tokens, min_tokens, overlap)
    groups = [group for group in groups if sum(unit.tokens for unit in group) >= min_chunk_tokens]
    children = _child_specs(groups)
    return _with_parents(children, target, parent_multiplier)


def _cfg(cfg: Any | None, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    value = cfg.get("indexing", key, default=None)
    return default if value is None else value
