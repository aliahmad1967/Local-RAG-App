"""User-confirmed facts and preferences, embedded for relevant retrieval.

Memory is never written automatically: the UI asks for confirmation, then the
API stores the fact. Retrieval is best-effort — if Ollama is down, chat still
works and simply skips long-term memory.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from ..config import Config
from ..db.database import Database
from ..db.repositories import MemoriesRepo
from ..index.embeddings import Embedder, OllamaEmbedder, content_hash
from ..index.vector_store import cosine_scores, pack_vector, unpack_vector
from ..logging_conf import get_logger
from ..models.domain import Memory, utcnow

log = get_logger("memory.service")


class MemoryService:
    def __init__(self, cfg: Config, db: Database, embedder: Embedder | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self._embedder = embedder
        self.model = str(cfg.get("ollama", "models", "embedding", default="bge-m3"))

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = OllamaEmbedder(self.cfg, model=self.model)
        return self._embedder

    def add(self, content: str, kind: str = "fact") -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        if kind not in ("fact", "preference"):
            raise ValueError("Memory kind must be 'fact' or 'preference'")
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            kind=kind,
            embedding_model=self.embedder.model,
            content_hash=content_hash(self.embedder.model, content),
            created_at=utcnow(),
        )
        try:
            vector = self.embedder.embed([content])[0]
            memory.embedding = pack_vector(vector)
            self.embedder.unload()  # free VRAM before the next pipeline stage
        except Exception as exc:  # noqa: BLE001 - store the text even if embeddings are down
            log.warning("Stored memory without embedding: %s", exc)
        return MemoriesRepo.upsert(self.db, memory).to_dict()

    def update(self, memory_id: str, content: str, kind: str | None = None) -> dict[str, Any]:
        memory = MemoriesRepo.get(self.db, memory_id)
        if memory is None:
            raise KeyError(memory_id)
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        memory.content = content
        if kind is not None:
            if kind not in ("fact", "preference"):
                raise ValueError("Memory kind must be 'fact' or 'preference'")
            memory.kind = kind
        memory.content_hash = content_hash(self.embedder.model, content)
        memory.embedding_model = self.embedder.model
        try:
            memory.embedding = pack_vector(self.embedder.embed([content])[0])
            self.embedder.unload()
        except Exception as exc:  # noqa: BLE001
            log.warning("Updated memory without embedding: %s", exc)
        return MemoriesRepo.upsert(self.db, memory).to_dict()

    def delete(self, memory_id: str) -> bool:
        return MemoriesRepo.delete(self.db, memory_id)

    def all(self) -> list[dict[str, Any]]:
        return [memory.to_dict() for memory in MemoriesRepo.all(self.db)]

    def relevant(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return memories whose embeddings are closest to the query."""
        limit = int(top_k or self.cfg.get("generation", "memory_top_k", default=5))
        memories = [m for m in MemoriesRepo.all(self.db) if m.embedding]
        if not memories or limit <= 0:
            return []
        usable = [m for m in memories if m.embedding_model == self.embedder.model]
        if not usable:
            return []
        try:
            query_vector = self.embedder.embed([query])[0]
        except Exception as exc:  # noqa: BLE001 - memory must never break chat
            log.warning("Could not embed query for memory lookup: %s", exc)
            return []
        blobs = [bytes(m.embedding or b"") for m in usable]
        scores = cosine_scores(query_vector, blobs)
        order = sorted(range(len(usable)), key=lambda index: -float(scores[index]))
        selected: list[dict[str, Any]] = []
        for index in order[:limit]:
            score = float(scores[index])
            if score < 0.25:
                continue
            item = usable[index].to_dict()
            item["score"] = round(score, 4)
            selected.append(item)
        return selected

    def close(self) -> None:
        if self._embedder is not None and hasattr(self._embedder, "unload"):
            self._embedder.unload()

    @staticmethod
    def preview(vector: bytes | None) -> list[float]:
        return [] if vector is None else unpack_vector(vector).tolist()
