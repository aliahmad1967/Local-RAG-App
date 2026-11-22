"""Phase 4 API: SSE streaming, conversations, memory and settings endpoints."""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi.testclient import TestClient

from localrag.api.app import create_app
from localrag.db.repositories import ConversationsRepo, MessagesRepo
from localrag.llm.generate import GroundedGenerator
from localrag.memory.service import MemoryService
from localrag.models.domain import RetrievalHit, RetrievalResult


class RecordingRetrieval:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        hit = RetrievalHit(
            chunk_id="c1", book_id="book-1", book_title="Sample Book",
            page_start=3, page_end=3,
            parent_text="The library keeps old books [Sample Book, p.3]",
        )
        return RetrievalResult(
            query=query, rewritten_query=None, hits=[hit],
            counts={"dense": 4, "bm25": 3, "fused": 4, "final": 1},
        )

    def close(self) -> None:
        pass


class FakeChat:
    model = "fake-chat:latest"

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        yield "Grounded answer "
        yield "[Sample Book, p.3]"

    def unload(self) -> None:
        pass


class FakeEmbedder:
    model = "fake-embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def unload(self) -> None:
        pass


def make_client(cfg, db):
    cfg.data["generation"]["summary_every_messages"] = 0
    cfg.data["generation"]["model"] = "fake-chat:latest"
    retrieval = RecordingRetrieval()
    memory = MemoryService(cfg, db, embedder=FakeEmbedder())
    generator = GroundedGenerator(cfg, db, retrieval, memory=memory, chat=FakeChat())
    app = create_app()
    app.state.cfg = cfg
    app.state.db = db
    app.state.service = None
    app.state.index_service = None
    app.state.retrieval_service = retrieval
    app.state.memory_service = memory
    app.state.generation_service = generator
    return TestClient(app), generator, retrieval, memory


def sse_events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[5:].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]


def test_chat_stream_persists_conversation_sources_and_memory(cfg, db):
    client, _, retrieval, memory = make_client(cfg, db)
    response = client.post("/api/chat", json={
        "query": "ما هو الكتاب؟",
        "book_ids": ["book-1"],
        "memory_content": "User prefers concise answers",
        "memory_kind": "preference",
        "page_from": 1,
        "page_to": 5,
    })

    assert response.status_code == 200
    events = sse_events(response.text)
    assert events[0]["type"] == "memory_saved"
    assert any(event["type"] == "delta" for event in events)
    done = next(event for event in events if event["type"] == "done")
    assert done["conversation_id"]
    assert done["sources"][0]["label"] == "Sample Book, p.3"
    assert retrieval.calls[0]["filters"].book_ids == ["book-1"]
    assert retrieval.calls[0]["filters"].page_to == 5

    messages = client.get(f"/api/conversations/{done['conversation_id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["sources"][0]["label"] == "Sample Book, p.3"
    assert messages[1]["language"] == "ar"
    assert any(item["content"] == "User prefers concise answers" for item in memory.all())


def test_conversation_crud_and_scope_update(cfg, db):
    client, _, _, _ = make_client(cfg, db)
    created = client.post("/api/conversations", json={
        "title": "Scope test", "book_ids": ["book-1"]
    })
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    patched = client.patch(f"/api/conversations/{conversation_id}", json={
        "title": "Renamed", "book_ids": ["book-1", "book-2"]
    })
    assert patched.json()["book_ids"] == ["book-1", "book-2"]

    listed = client.get("/api/conversations").json()
    assert any(item["id"] == conversation_id and item["title"] == "Renamed" for item in listed)

    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 200
    assert client.get(f"/api/conversations/{conversation_id}/messages").status_code == 404
    assert ConversationsRepo.get(db, conversation_id) is None
    assert MessagesRepo.count(db, conversation_id) == 0


def test_memories_endpoints(cfg, db):
    client, _, _, _ = make_client(cfg, db)
    created = client.post("/api/memories", json={"content": "Prefer Arabic", "kind": "preference"})
    assert created.status_code == 201
    memory_id = created.json()["id"]

    patched = client.patch(f"/api/memories/{memory_id}", json={"content": "Prefer English"})
    assert patched.json()["content"] == "Prefer English"
    assert any(item["id"] == memory_id for item in client.get("/api/memories").json())

    assert client.delete(f"/api/memories/{memory_id}").status_code == 200
    assert client.delete(f"/api/memories/{memory_id}").status_code == 404


def test_models_and_generation_settings(monkeypatch, cfg, db):
    monkeypatch.setattr(
        "localrag.api.routes_chat.list_ollama_models",
        lambda _cfg: ["fake-chat:latest", "llama3.2:latest"],
    )
    client, generator, _, _ = make_client(cfg, db)

    models = client.get("/api/models").json()
    assert "fake-chat:latest" in models["installed"]
    assert models["current"] == "fake-chat:latest"

    saved = client.put("/api/config/generation", json={
        "model": "llama3.2:latest", "temperature": 0.4,
        "context_tokens": 4096, "history_turns": 4,
    })
    assert saved.status_code == 200
    assert generator._chat is None  # next answer rebuilds with the new model
    current = client.get("/api/config/generation").json()
    assert current["model"] == "llama3.2:latest"
    assert current["temperature"] == 0.4
    assert cfg.local_path.exists()
