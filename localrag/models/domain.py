"""Typed domain objects shared by ingestion, OCR and the API."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> str:
    """ISO-8601 UTC timestamp, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Book:
    id: str
    title: str
    file_path: str
    file_name: str
    page_count: int = 0
    language: str = "unknown"
    ocr_engine: str | None = None
    status: str = "queued"
    file_sha256: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Book":
        return cls(**{key: row[key] for key in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Page:
    id: str
    book_id: str
    page_no: int
    source: str = "pending"
    ocr_engine: str | None = None
    text_original: str = ""
    text_clean: str = ""
    text_normalized: str = ""
    confidence: float | None = None
    status: str = "pending"
    error: str | None = None
    raster_path: str | None = None
    retry_count: int = 0
    char_count: int = 0
    duration_s: float = 0.0
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Page":
        return cls(**{key: row[key] for key in row.keys()})

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_text:
            data.pop("text_original", None)
            data.pop("text_clean", None)
            data.pop("text_normalized", None)
        return data


@dataclass
class Chunk:
    """A searchable child chunk or a larger parent window returned to the LLM."""

    id: str
    book_id: str
    level: str = "child"                 # 'child' | 'parent'
    parent_id: str | None = None
    child_index: int = 0
    page_start: int = 1
    page_end: int = 1
    heading: str | None = None
    text: str = ""
    text_normalized: str = ""
    token_count: int = 0
    char_count: int = 0
    start_char: int = 0
    end_char: int = 0
    created_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Chunk":
        return cls(**{key: row[key] for key in row.keys()})

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_text:
            data.pop("text", None)
            data.pop("text_normalized", None)
        return data


@dataclass
class IndexStatus:
    book_id: str
    status: str = "not_indexed"
    chunk_count: int = 0
    embedded_count: int = 0
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "IndexStatus":
        return cls(**{key: row[key] for key in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Job:
    id: str
    book_id: str | None = None
    kind: str = "ingest"
    status: str = "queued"
    current_page: int = 0
    total_pages: int = 0
    processed: int = 0
    failed: int = 0
    params_json: str = "{}"
    error: str | None = None
    created_at: str = field(default_factory=utcnow)
    started_at: str | None = None
    finished_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(**{key: row[key] for key in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRResult:
    """Uniform result returned by every OCR backend."""

    text: str
    engine: str
    model: str
    duration_s: float = 0.0
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BookProgress:
    book_id: str
    page_count: int
    done: int
    failed: int
    flagged: int
    pending: int
    job_status: str | None = None
    current_page: int = 0
    eta_s: float | None = None

    @property
    def percent(self) -> int:
        if self.page_count == 0:
            return 0
        return min(100, round(100 * self.done / self.page_count))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["percent"] = self.percent
        return data


@dataclass
class RetrievalHit:
    """One retrieved child chunk, with every rank/score the debug panel needs."""

    chunk_id: str
    book_id: str
    book_title: str = ""
    page_start: int = 0
    page_end: int = 0
    heading: str | None = None
    text: str = ""
    parent_id: str | None = None
    parent_text: str | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_rank: int | None = None
    fused_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    rewritten_query: str | None
    hits: list[RetrievalHit]
    filters: dict[str, Any] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "filters": self.filters,
            "timings_s": self.timings_s,
            "counts": self.counts,
            "hits": [hit.to_dict() for hit in self.hits],
        }


@dataclass
class Conversation:
    id: str
    title: str = "New conversation"
    book_ids_json: str = "[]"
    summary: str = ""
    summary_message_count: int = 0
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Conversation":
        return cls(**{key: row[key] for key in row.keys()})

    @property
    def book_ids(self) -> list[str]:
        try:
            value = json.loads(self.book_ids_json)
            return [str(item) for item in value] if isinstance(value, list) else []
        except (ValueError, TypeError):
            return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "book_ids": self.book_ids,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    sources_json: str = "[]"
    language: str = "auto"
    created_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Message":
        return cls(**{key: row[key] for key in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        try:
            sources = json.loads(self.sources_json)
        except (ValueError, TypeError):
            sources = []
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "sources": sources if isinstance(sources, list) else [],
            "language": self.language,
            "created_at": self.created_at,
        }


@dataclass
class Memory:
    id: str
    content: str
    kind: str = "fact"
    embedding_model: str = ""
    content_hash: str = ""
    embedding: bytes | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Memory":
        return cls(**{key: row[key] for key in row.keys()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "kind": self.kind,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
