"""Engine factory and discovery."""

from __future__ import annotations

from ..config import Config
from .base import OCREngine
from .ollama_engine import OllamaOCREngine
from .tesseract_engine import TesseractOCREngine

ENGINE_NAMES: tuple[str, ...] = (
    "glm-ocr",
    "deepseek-ocr",
    "paddleocr-vl",
    "tesseract",
)
OLLAMA_ENGINES = ENGINE_NAMES[:3]


def get_engine(name: str, cfg: Config) -> OCREngine:
    if name in OLLAMA_ENGINES:
        spec = cfg.get("ocr", "engines", name, default=None) or {}
        model = spec.get("model")
        if not model:
            raise ValueError(f"No model configured for OCR engine '{name}'")
        options = dict(spec.get("options") or {})
        options.setdefault("temperature", 0)
        return OllamaOCREngine(
            cfg=cfg,
            name=name,
            model=str(model),
            prompt=str(spec.get("prompt", "Transcribe all text in this image. Plain text only.")),
            keep_alive=str(spec.get("keep_alive", "5m")),
            options=options,
        )
    if name == "tesseract":
        return TesseractOCREngine(cfg)
    raise ValueError(
        f"Unknown OCR engine '{name}'. Available: {', '.join(ENGINE_NAMES)}"
    )


def available_engines(cfg: Config) -> list[str]:
    return [name for name in ENGINE_NAMES if get_engine(name, cfg).available()]
