"""Side-by-side OCR benchmark on up to 10 user-selected pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..db.database import Database
from ..db.repositories import BenchmarkRepo
from ..ingest.pdf import open_document, page_image_for
from ..logging_conf import get_logger
from ..models.domain import Book
from ..text.cleanup import build_text_views
from .registry import ENGINE_NAMES, get_engine

log = get_logger("ocr.benchmark")

ProgressCallback = Callable[[str, int, int], None]


def run_benchmark(
    cfg: Config,
    db: Database,
    book: Book,
    page_numbers: list[int],
    engines: list[str] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run every engine on the same rasterized pages; unload between engines."""
    pages = sorted({int(page) for page in page_numbers if 1 <= int(page) <= book.page_count})
    pages = pages[:10]
    if not pages:
        raise ValueError("Select at least one valid page number (maximum 10).")

    engine_names = list(engines or ENGINE_NAMES)
    dpi = int(cfg.get("ingestion", "dpi", default=250))
    books_root = cfg.path("paths", "books_dir", default="data/books")
    unload_between = bool(
        cfg.get("ingestion", "unload_between_benchmark_engines", default=True)
    )
    results: dict[str, Any] = {}

    with open_document(Path(book.file_path)) as doc:
        for engine_index, engine_name in enumerate(engine_names, start=1):
            engine = get_engine(engine_name, cfg)
            engine_results: dict[str, Any] = {
                "engine": engine_name,
                "model": getattr(engine, "model", getattr(engine, "lang", engine_name)),
                "available": engine.available(),
                "pages": {},
            }
            if not engine.available():
                engine_results["error"] = "Backend is not installed on this machine."
                results[engine_name] = engine_results
                if progress:
                    progress(engine_name, engine_index, len(engine_names))
                continue

            durations: list[float] = []
            total_chars = 0
            for page_no in pages:
                raster = page_image_for(doc[page_no - 1], books_root, book.id, page_no, dpi=dpi)
                result = engine.recognize(raster, book.language)
                clean, _ = build_text_views(result.text, cfg)
                engine_results["pages"][str(page_no)] = {
                    "text": clean,
                    "raw_text": result.text,
                    "chars": len(clean),
                    "duration_s": result.duration_s,
                    "confidence": result.confidence,
                    "error": result.error,
                }
                durations.append(result.duration_s)
                total_chars += len(clean)
                if progress:
                    log.debug("Benchmark %s page %d", engine_name, page_no)

            engine_results["mean_duration_s"] = (
                round(sum(durations) / len(durations), 3) if durations else 0.0
            )
            engine_results["total_chars"] = total_chars
            results[engine_name] = engine_results
            if unload_between:
                engine.unload()
            if progress:
                progress(engine_name, engine_index, len(engine_names))
            log.info(
                "Benchmark %s: %d pages, %.2fs/page mean, %d chars",
                engine_name, len(pages), engine_results["mean_duration_s"], total_chars,
            )

    benchmark_id = BenchmarkRepo.save(db, book.id, pages, results)
    return {
        "id": benchmark_id,
        "book_id": book.id,
        "page_numbers": pages,
        "engines": results,
    }
