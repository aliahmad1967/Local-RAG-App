"""Ollama-backed OCR engines (glm-ocr, deepseek-ocr, paddleocr-vl)."""

from __future__ import annotations

import base64
import time
import unicodedata
from pathlib import Path

import httpx

from ..config import Config
from ..logging_conf import get_logger
from ..models.domain import OCRResult
from .base import OCREngine
from .vram import unload_model

log = get_logger("ocr.ollama")


def heuristic_confidence(text: str) -> float:
    """Ollama OCR models do not expose logprobs.

    This is an explicit heuristic (not a model score): mostly printable text
    scores high; very short or symbol-heavy output is flagged for review.
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    plausible = sum(
        1 for ch in stripped
        if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("P")
    )
    score = min(100.0, 40.0 + 60.0 * plausible / len(stripped))
    if len(stripped) < 20:
        score = min(score, 35.0)
    return round(score, 1)


class OllamaOCREngine(OCREngine):
    def __init__(
        self,
        cfg: Config,
        name: str,
        model: str,
        prompt: str,
        keep_alive: str = "5m",
        options: dict | None = None,
    ) -> None:
        super().__init__(cfg)
        self.name = name
        self.model = model
        self.prompt = prompt
        self.keep_alive = keep_alive
        self.options = dict(options or {"temperature": 0})
        self.host = str(cfg.get("ollama", "host", default="http://127.0.0.1:11434"))
        self.timeout = float(cfg.get("ollama", "request_timeout_s", default=900))

    def describe(self) -> str:
        return f"{self.name} ({self.model})"

    def unload(self) -> None:
        unload_model(self.cfg, self.model)

    def recognize(self, image_path: Path, language: str = "unknown") -> OCRResult:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        try:
            response = httpx.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            data: dict = response.json() if response.content else {}
            if response.status_code >= 400 or data.get("error"):
                message = str(data.get("error", response.text[:300]))
                if "not found" in message.lower() or response.status_code == 404:
                    message = (
                        f"Model '{self.model}' is not installed. "
                        f"Run: ollama pull {self.model}"
                    )
                raise RuntimeError(message)
            text = str(data.get("response", "")).strip()
            duration = time.perf_counter() - started
            log.info(
                "OCR %s on %s: %d chars in %.2fs", self.name, image_path.name, len(text), duration
            )
            return OCRResult(
                text=text,
                engine=self.name,
                model=self.model,
                duration_s=round(duration, 3),
                confidence=heuristic_confidence(text),
                raw={"eval_count": data.get("eval_count"), "language_hint": language},
            )
        except httpx.ConnectError as exc:
            return OCRResult(
                text="", engine=self.name, model=self.model,
                duration_s=round(time.perf_counter() - started, 3),
                error=f"Ollama is not running at {self.host}. Start it, then retry. ({exc})",
            )
        except Exception as exc:  # noqa: BLE001 - returned to UI/job log
            return OCRResult(
                text="", engine=self.name, model=self.model,
                duration_s=round(time.perf_counter() - started, 3),
                error=str(exc),
            )
