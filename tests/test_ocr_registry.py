import pytest

from localrag.ocr.registry import ENGINE_NAMES, get_engine
from localrag.ocr.ollama_engine import OllamaOCREngine
from localrag.ocr.tesseract_engine import TesseractOCREngine


def test_all_four_engines_are_registered():
    assert ENGINE_NAMES == ("glm-ocr", "deepseek-ocr", "paddleocr-vl", "tesseract")


@pytest.mark.parametrize("name", ["glm-ocr", "deepseek-ocr", "paddleocr-vl"])
def test_ollama_engines_resolve_to_configured_models(name, cfg):
    engine = get_engine(name, cfg)
    assert isinstance(engine, OllamaOCREngine)
    assert engine.name == name
    assert engine.model


def test_tesseract_engine_resolves(cfg):
    engine = get_engine("tesseract", cfg)
    assert isinstance(engine, TesseractOCREngine)
    assert engine.lang == "ara+eng"
    if not engine.available():
        pytest.skip(
            "Tesseract binary is not installed here; the three Ollama OCR engines "
            "provide the primary path and the fallback reports a clear UI error."
        )


def test_unknown_engine_raises(cfg):
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        get_engine("nonexistent", cfg)
