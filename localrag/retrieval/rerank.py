"""Pluggable cross-encoder rerankers.

The primary backend is `BAAI/bge-reranker-v2-m3`, loaded lazily and released
after each search so it never holds VRAM between pipeline stages. An Ollama
listwise scorer is available as an offline fallback when the cross-encoder
cannot be loaded, and a no-op backend supports A/B evaluation.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Protocol

from ..config import Config
from ..logging_conf import get_logger

log = get_logger("retrieval.rerank")


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...

    def unload(self) -> None: ...


class NoopReranker:
    """Keeps the fused order; used by the dense-only / hybrid eval baselines."""

    name = "none"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)

    def unload(self) -> None:
        pass


class CrossEncoderReranker:
    """Multilingual cross-encoder, fp16 on CUDA, fp32 on CPU."""

    name = "cross-encoder"

    def __init__(self, cfg: Config, model: str | None = None, device: str | None = None) -> None:
        self.cfg = cfg
        self.model_name = str(
            model or cfg.get("retrieval", "reranker", "model", default="BAAI/bge-reranker-v2-m3")
        )
        self.device = str(
            device or cfg.get("retrieval", "reranker", "device", default="cpu")
        )
        self.fp16 = bool(cfg.get("retrieval", "reranker", "fp16", default=False))
        self.batch_size = int(cfg.get("retrieval", "reranker", "batch_size", default=16))
        self.max_length = int(cfg.get("retrieval", "reranker", "max_length", default=512))
        # Once downloaded the app is fully offline; never phone Hugging Face at
        # runtime. Set local_files_only: false in config.local.yaml if needed.
        self.local_files_only = bool(
            cfg.get("retrieval", "reranker", "local_files_only", default=True)
        )
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            model_ref: str = self.model_name
            model_kwargs: dict[str, Any] = {}
            if self.local_files_only:
                model_ref = self._cached_model_path()
                model_kwargs["local_files_only"] = True
            if self.device.startswith("cuda") and self.fp16:
                import torch

                model_kwargs["torch_dtype"] = torch.float16
            log.info(
                "Loading reranker %s on %s (fp16=%s, offline=%s)",
                model_ref, self.device, self.fp16, self.local_files_only,
            )
            self._model = CrossEncoder(
                model_ref,
                device=None if self.device == "auto" else self.device,
                max_length=self.max_length,
                model_kwargs=model_kwargs or None,
                local_files_only=self.local_files_only,
            )
        return self._model

    def _cached_model_path(self) -> str:
        """Resolve the HF cache snapshot without contacting the network."""
        if Path(self.model_name).exists():
            path = Path(self.model_name)
        else:
            from huggingface_hub import snapshot_download

            try:
                path = Path(snapshot_download(self.model_name, local_files_only=True))
            except Exception as exc:  # noqa: BLE001 - HF errors vary by version
                raise RuntimeError(
                    f"Reranker model {self.model_name!r} is not cached. Download it once while "
                    "online with: python -c \"from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{self.model_name}')\""
                ) from exc
        weights = (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if not any((path / name).is_file() for name in weights):
            raise RuntimeError(
                f"Reranker cache for {self.model_name!r} has no model weights. Finish the "
                f"download with: python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('{self.model_name}', 'model.safetensors')\""
            )
        return str(path)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        pairs = [[query, text] for text in texts]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]

    def unload(self) -> None:
        if self._model is not None:
            log.debug("Unloading reranker %s", self.model_name)
            self._model = None
        gc.collect()
        if self.device.startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                log.debug("Could not clear CUDA cache: %s", exc)


class OllamaReranker:
    """Offline fallback: ask the fast local model for a 0-10 relevance score."""

    name = "ollama"

    def __init__(self, cfg: Config, model: str | None = None) -> None:
        from ..llm.ollama import OllamaChat

        self.cfg = cfg
        self.chat = OllamaChat(cfg, model=model, num_predict=32, temperature=0.0)
        self.name = f"ollama:{self.chat.model}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for index, text in enumerate(texts, start=1):
            prompt = (
                "Rate how relevant this passage is to the search query on a scale 0-10.\n"
                f"QUERY: {query}\nPASSAGE {index}: {text[:1800]}\n"
                "Reply with ONLY a JSON object like {\"score\": 7}."
            )
            try:
                reply = self.chat.complete(prompt, system="You are a search relevance grader.")
                match = re.search(r"(-?\d+(?:\.\d+)?)", reply)
                value = float(match.group(1)) if match else 0.0
            except Exception as exc:  # noqa: BLE001 - a bad score must not kill retrieval
                log.warning("Ollama rerank failed for passage %d: %s", index, exc)
                value = 0.0
            scores.append(max(0.0, min(10.0, value)))
        return scores

    def unload(self) -> None:
        self.chat.unload()


def build_reranker(cfg: Config, enabled: bool | None = None) -> Reranker | None:
    """Return the configured reranker, or ``None`` when reranking is disabled."""
    if enabled is False:
        return None
    configured = bool(cfg.get("retrieval", "reranker", "enabled", default=True))
    if enabled is None and not configured:
        return None
    backend = str(cfg.get("retrieval", "reranker", "backend", default="cross-encoder"))
    if backend in ("none", "off", "disabled"):
        return None
    if backend == "cross-encoder":
        return CrossEncoderReranker(cfg)
    if backend == "ollama":
        return OllamaReranker(cfg)
    raise ValueError(
        f"Unknown reranker backend {backend!r}; choose cross-encoder, ollama or none"
    )
