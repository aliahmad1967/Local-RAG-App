"""Phase 4: confirmed memory storage, updates and relevant retrieval."""

from __future__ import annotations

from typing import Any

import pytest

from localrag.db.repositories import MemoriesRepo
from localrag.index.embeddings import content_hash
from localrag.memory.service import MemoryService


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self) -> None:
        self.unloaded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        value = text.lower()
        if "cat" in value:
            return [1.0, 0.0]
        if "weather" in value:
            return [0.0, 1.0]
        return [0.2, 0.2]

    def unload(self) -> None:
        self.unloaded += 1


def make_service(cfg, db) -> tuple[MemoryService, FakeEmbedder]:
    embedder = FakeEmbedder()
    return MemoryService(cfg, db, embedder=embedder), embedder


def test_add_stores_embedding_and_hash(cfg, db):
    service, embedder = make_service(cfg, db)
    memory = service.add("I prefer concise answers", kind="preference")

    assert memory["kind"] == "preference"
    assert memory["embedding_model"] == "fake-embed"
    stored = MemoriesRepo.get(db, memory["id"])
    assert stored is not None
    assert stored.content_hash == content_hash("fake-embed", "I prefer concise answers")
    assert stored.embedding is not None and len(stored.embedding) > 0
    assert embedder.unloaded == 1


def test_relevant_returns_ranked_memories_above_threshold(cfg, db):
    service, _ = make_service(cfg, db)
    service.add("I own a cat named Mishi")
    service.add("The weather is rainy today")
    service.add("Unrelated note")

    relevant = service.relevant("cat", top_k=5)

    assert [item["content"] for item in relevant] == [
        "I own a cat named Mishi", "Unrelated note"
    ]
    assert relevant[0]["score"] > relevant[1]["score"]


def test_update_and_delete_memory(cfg, db):
    service, embedder = make_service(cfg, db)
    memory = service.add("Old fact")

    updated = service.update(memory["id"], "New fact", kind="preference")
    assert updated["content"] == "New fact"
    assert updated["kind"] == "preference"
    assert updated["id"] == memory["id"]
    assert embedder.unloaded == 2

    assert service.delete(memory["id"]) is True
    assert service.all() == []
    assert service.delete(memory["id"]) is False


def test_validation_errors(cfg, db):
    service, _ = make_service(cfg, db)

    with pytest.raises(ValueError):
        service.add("   ")
    with pytest.raises(ValueError):
        service.add("fact", kind="unknown")
    with pytest.raises(KeyError):
        service.update("missing", "content")
