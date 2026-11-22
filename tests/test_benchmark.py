"""Benchmark runner and VRAM-unload behaviour (no real models needed)."""

from __future__ import annotations

from localrag.db.repositories import BenchmarkRepo, BooksRepo
from localrag.ingest.pdf import inspect_pdf
from localrag.models.domain import Book, OCRResult
from localrag.ocr import benchmark as benchmark_module
from localrag.ocr import vram

from .data.make_samples import make_sample_pdf


class BenchmarkEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = name + "-model"
        self.unloaded = False

    def available(self) -> bool:
        return True

    def recognize(self, image_path, language="unknown") -> OCRResult:
        return OCRResult(
            text=f"{self.name} saw {image_path.name}",
            engine=self.name, model=self.model, confidence=88.0, duration_s=0.02,
        )

    def unload(self) -> None:
        self.unloaded = True


def test_benchmark_runs_all_engines_and_persists(tmp_path, cfg, db, monkeypatch):
    source = make_sample_pdf(tmp_path / "bench.pdf", pages=3, language="en", dpi=80)
    book = Book(
        id="bench-book", title="Bench", file_path=str(source),
        file_name=source.name, page_count=inspect_pdf(source).page_count,
        language="en", ocr_engine="glm-ocr",
    )
    BooksRepo.create(db, book)

    created: list[BenchmarkEngine] = []

    def fake_get_engine(name, config):
        engine = BenchmarkEngine(name)
        created.append(engine)
        return engine

    monkeypatch.setattr(benchmark_module, "get_engine", fake_get_engine)
    result = benchmark_module.run_benchmark(
        cfg, db, book, [1, 3], engines=["glm-ocr", "deepseek-ocr"]
    )

    assert result["page_numbers"] == [1, 3]
    assert set(result["engines"]) == {"glm-ocr", "deepseek-ocr"}
    assert all(engine.unloaded for engine in created), "each engine must release VRAM"
    for engine in result["engines"].values():
        assert engine["mean_duration_s"] == 0.02
        assert engine["total_chars"] > 0
        assert set(engine["pages"]) == {"1", "3"}

    saved = BenchmarkRepo.latest(db, book.id)
    assert saved is not None
    assert saved["page_numbers"] == [1, 3]
    assert BooksRepo.get(db, book.id).id == book.id


def test_unload_uses_keep_alive_zero(cfg, monkeypatch):
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(vram.httpx, "post", fake_post)
    vram.unload_ocr_models(cfg)
    assert len(calls) == 3
    assert all(call["json"]["keep_alive"] == 0 for call in calls)
    assert all(call["json"]["prompt"] == "" for call in calls)
