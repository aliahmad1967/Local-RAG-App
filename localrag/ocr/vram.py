"""Ollama process/model helpers.

The target GPU has 8 GB VRAM, so only one large model may be resident. Every
pipeline stage loads its model, uses it, then explicitly unloads it.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import Config
from ..logging_conf import get_logger

log = get_logger("ocr.vram")


def ollama_status(cfg: Config, timeout: float = 5.0) -> dict[str, Any]:
    """Check Ollama and list installed model names."""
    host = str(cfg.get("ollama", "host", default="http://127.0.0.1:11434"))
    try:
        response = httpx.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        return {"running": True, "host": host, "models": models, "error": None}
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a message
        return {
            "running": False,
            "host": host,
            "models": [],
            "error": f"Ollama is not reachable at {host}. Start it with `ollama serve`. ({exc})",
        }


def unload_model(cfg: Config, model: str) -> bool:
    """Ask Ollama to unload a model immediately (keep_alive=0)."""
    host = str(cfg.get("ollama", "host", default="http://127.0.0.1:11434"))
    try:
        response = httpx.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=10.0,
        )
        response.raise_for_status()
        log.debug("Unloaded Ollama model %s", model)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not unload %s: %s", model, exc)
        return False


def unload_ocr_models(cfg: Config) -> None:
    """Release every configured OCR model from VRAM."""
    for name in ("glm-ocr", "deepseek-ocr", "paddleocr-vl"):
        model = cfg.get("ocr", "engines", name, "model", default=None)
        if model:
            unload_model(cfg, str(model))
