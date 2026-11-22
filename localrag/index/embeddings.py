"""Ollama bge-m3 embeddings with a content-hash cache and batching.

Only one model is resident at a time: `unload()` sends keep_alive=0 after an
indexing job so the 8 GB GPU has room for the next pipeline stage.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Protocol

import httpx

from ..config import Config
from ..logging_conf import get_logger

log = get_logger("index.embeddings")


def content_hash(model: str, text: str) -> str:
    """Stable cache key: the same text + model is embedded only once."""
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


class Embedder(Protocol):
    """Minimal interface; tests inject a fake implementation."""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def unload(self) -> None: ...


class OllamaEmbedder:
    def __init__(
        self,
        cfg: Config,
        model: str | None = None,
        batch_size: int | None = None,
        keep_alive: str = "5m",
    ) -> None:
        self.cfg = cfg
        self.model = str(model or cfg.get("ollama", "models", "embedding", default="bge-m3"))
        self.host = str(cfg.get("ollama", "host", default="http://127.0.0.1:11434")).rstrip("/")
        self.timeout = float(cfg.get("ollama", "request_timeout_s", default=900))
        configured_batch = cfg.get("indexing", "embed_batch_size", default=16)
        self.batch_size = int(batch_size or configured_batch)
        self.keep_alive = keep_alive
        # VRAM discipline: when OCR is still running, set LOCALRAG_EMBED_CPU=1
        # (or indexing.embed_num_gpu: 0) to keep bge-m3 off the GPU entirely.
        configured_gpu = cfg.get("indexing", "embed_num_gpu", default=None)
        if os.getenv("LOCALRAG_EMBED_CPU") == "1":
            configured_gpu = 0
        self.num_gpu = None if configured_gpu is None else int(configured_gpu)

    def available(self) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            names = {entry.get("name") for entry in response.json().get("models", [])}
            return any(name and name.split(":")[0] == self.model.split(":")[0] for name in names)
        except Exception:  # noqa: BLE001
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            started = time.perf_counter()
            payload: dict[str, Any] = {
                "model": self.model, "input": batch, "keep_alive": self.keep_alive,
            }
            if self.num_gpu is not None:
                payload["options"] = {"num_gpu": self.num_gpu}
            response = httpx.post(f"{self.host}/api/embed", json=payload, timeout=self.timeout)
            if response.status_code == 404:
                raise RuntimeError(
                    f"Embedding endpoint not found at {self.host}. Update Ollama."
                )
            if response.status_code == 400 and "not found" in response.text.lower():
                raise RuntimeError(
                    f"Embedding model '{self.model}' is missing. Run: ollama pull {self.model}"
                )
            response.raise_for_status()
            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"Embedding failed: {data['error']}")
            batch_vectors = data.get("embeddings") or []
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(batch_vectors)} embeddings for {len(batch)} inputs"
                )
            vectors.extend([[float(value) for value in vector] for vector in batch_vectors])
            log.info(
                "Embedded %d/%d chunks in %.2fs (%s)",
                start + len(batch), len(texts), time.perf_counter() - started, self.model,
            )
        return vectors

    def unload(self) -> None:
        try:
            httpx.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
            log.debug("Unloaded embedding model %s", self.model)
        except Exception as exc:  # noqa: BLE001 - unloading must never fail a job
            log.warning("Could not unload embedding model: %s", exc)
