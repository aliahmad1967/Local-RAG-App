"""Request bodies for the Phase 1 API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EngineChoice(BaseModel):
    engine: str


class RetryRequest(BaseModel):
    statuses: list[Literal["failed", "flagged"]] = Field(
        default_factory=lambda: ["failed", "flagged"]
    )


class BenchmarkRequest(BaseModel):
    book_id: str
    pages: list[int] = Field(..., max_length=10)
    engines: list[str] | None = None


class UISettings(BaseModel):
    language: Literal["ar", "en"] | None = None
    theme: Literal["dark", "light"] | None = None
    default_engine: str | None = None


class IndexRequest(BaseModel):
    force: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=50, ge=1, le=200)
    book_id: str | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    book_ids: list[str] | None = None
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[dict[str, str]] = Field(default_factory=list)
    explain: bool = True


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: str | None = None
    book_ids: list[str] | None = None
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    mode: Literal["chat", "summarize", "compare"] = "chat"
    top_k: int | None = Field(default=None, ge=1, le=50)
    language: Literal["auto", "ar", "en"] = "auto"
    memory_content: str | None = None
    memory_kind: Literal["fact", "preference"] = "fact"


class ConversationCreate(BaseModel):
    title: str | None = None
    book_ids: list[str] = Field(default_factory=list)


class ConversationUpdate(BaseModel):
    title: str | None = None
    book_ids: list[str] | None = None


class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1)
    kind: Literal["fact", "preference"] = "fact"


class MemoryUpdate(BaseModel):
    content: str = Field(..., min_length=1)
    kind: Literal["fact", "preference"] | None = None


class GenerationSettings(BaseModel):
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    num_predict: int | None = Field(default=None, ge=32, le=8192)
    context_tokens: int | None = Field(default=None, ge=1024, le=65536)
    history_turns: int | None = Field(default=None, ge=0, le=30)
    memory_top_k: int | None = Field(default=None, ge=0, le=20)
