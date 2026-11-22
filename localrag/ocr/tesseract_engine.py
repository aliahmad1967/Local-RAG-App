"""Tesseract fallback OCR engine (ara+eng)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytesseract
from PIL import Image

from ..config import Config
from ..logging_conf import get_logger
from ..models.domain import OCRResult
from .base import OCREngine

log = get_logger("ocr.tesseract")


class TesseractOCREngine(OCREngine):
    name = "tesseract"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.cmd = str(cfg.get("ocr", "tesseract", "cmd", default="") or "")
        self.lang = str(cfg.get("ocr", "tesseract", "lang", default="ara+eng"))
        self.psm = int(cfg.get("ocr", "tesseract", "psm", default=6))
        if self.cmd:
            if Path(self.cmd).exists():
                pytesseract.pytesseract.tesseract_cmd = self.cmd
            else:
                log.debug("Tesseract path from config does not exist: %s", self.cmd)

    def available(self) -> bool:
        return bool(self.cmd and Path(self.cmd).exists()) or shutil.which("tesseract") is not None

    def describe(self) -> str:
        return f"tesseract ({self.lang})"

    def recognize(self, image_path: Path, language: str = "unknown") -> OCRResult:
        started = time.perf_counter()
        if not self.available():
            return OCRResult(
                text="", engine=self.name, model=self.lang, error=(
                    f"Tesseract not found at '{self.cmd}'. Install it with ara/eng language data "
                    "or set ocr.tesseract.cmd in config.yaml."
                ),
            )
        try:
            config = f"--psm {self.psm}"
            with Image.open(image_path) as image:
                text = pytesseract.image_to_string(image, lang=self.lang, config=config)
                data = pytesseract.image_to_data(
                    image, lang=self.lang, config=config, output_type=pytesseract.Output.DICT
                )
            confidences = [
                float(conf)
                for conf, word in zip(data.get("conf", []), data.get("text", []))
                if str(word).strip() and str(conf) not in ("-1", "-1.0", "")
            ]
            confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
            duration = time.perf_counter() - started
            log.info(
                "OCR tesseract on %s: %d chars in %.2fs (conf %.1f)",
                image_path.name, len(text.strip()), duration, confidence,
            )
            return OCRResult(
                text=text.strip(), engine=self.name, model=self.lang,
                duration_s=round(duration, 3), confidence=confidence,
            )
        except Exception as exc:  # noqa: BLE001
            return OCRResult(
                text="", engine=self.name, model=self.lang,
                duration_s=round(time.perf_counter() - started, 3), error=str(exc),
            )
