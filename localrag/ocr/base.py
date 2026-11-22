"""Common OCR interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Config
from ..models.domain import OCRResult


class OCREngine(ABC):
    """One backend = one `recognize()` call on one rasterized page image."""

    name: str = "base"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def recognize(self, image_path: Path, language: str = "unknown") -> OCRResult:
        """Transcribe a page image. Never raise for model/runtime failures: return OCRResult.error."""

    def available(self) -> bool:
        """Whether the backend can be called in this environment."""
        return True

    def unload(self) -> None:
        """Release GPU memory. No-op for backends that manage it themselves."""

    def describe(self) -> str:
        return self.name
