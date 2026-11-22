"""Phase 1 HTTP endpoints: import, progress, OCR control, benchmark."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from ..db.repositories import BenchmarkRepo, BooksRepo, JobsRepo, PagesRepo
from ..ingest.pdf import open_document, page_image_for, safe_filename
from ..ocr.benchmark import run_benchmark
from ..ocr.registry import ENGINE_NAMES, available_engines, get_engine
from ..ocr.vram import ollama_status
from .schemas import BenchmarkRequest, EngineChoice, RetryRequest, UISettings

router = APIRouter(prefix="/api")


def _services(request: Request) -> tuple[Any, Any, Any]:
    return request.app.state.cfg, request.app.state.db, request.app.state.service


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    cfg, _, _ = _services(request)
    status = ollama_status(cfg)
    return {
        "ollama": status,
        "ocr_engines": [
            {"name": name, "available": name in available_engines(cfg)}
            for name in ENGINE_NAMES
        ],
        "default_engine": cfg.get("ocr", "default_engine", default="glm-ocr"),
        "ui_language": cfg.get("ui", "language", default="ar"),
        "ui_theme": cfg.get("ui", "theme", default="dark"),
    }


@router.post("/books/import")
async def import_book(
    request: Request,
    file: UploadFile = File(...),
    engine: str | None = Form(default=None),
) -> dict[str, Any]:
    cfg, _, service = _services(request)
    name = safe_filename(file.filename or "book.pdf")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if engine and engine not in ENGINE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown OCR engine: {engine}")

    uploads: Path = cfg.path("paths", "uploads_dir", default="data/books/_uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    temp = uploads / f"{Path(name).stem}_{Path(engine or 'default').name}.pdf"
    try:
        with temp.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        book, job = service.import_pdf(temp, engine=engine)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp.unlink(missing_ok=True)
        await file.close()
    return {"book": book.to_dict(), "job": job.to_dict()}


@router.get("/books")
def list_books(request: Request) -> list[dict[str, Any]]:
    _, db, service = _services(request)
    result = []
    for book in BooksRepo.all(db):
        item = book.to_dict()
        item["progress"] = (service.progress(book.id) or BookProgressStub(book.page_count)).to_dict()
        result.append(item)
    return result


class BookProgressStub:
    """Fallback used only if a book row somehow has no progress record."""

    def __init__(self, page_count: int) -> None:
        from ..models.domain import BookProgress

        self._value = BookProgress(book_id="", page_count=page_count, done=0, failed=0,
                                   flagged=0, pending=page_count)

    def to_dict(self) -> dict[str, Any]:
        return self._value.to_dict()


@router.get("/books/{book_id}")
def get_book(book_id: str, request: Request) -> dict[str, Any]:
    _, db, service = _services(request)
    book = BooksRepo.get(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    jobs = JobsRepo.for_book(db, book_id)
    return {
        "book": book.to_dict(),
        "progress": (service.progress(book_id) or BookProgressStub(book.page_count)).to_dict(),
        "jobs": [job.to_dict() for job in jobs[:5]],
    }


@router.get("/books/{book_id}/pages")
def list_pages(
    book_id: str,
    request: Request,
    include_text: bool = Query(default=False),
) -> list[dict[str, Any]]:
    _, db, _ = _services(request)
    if BooksRepo.get(db, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return [page.to_dict(include_text=include_text) for page in PagesRepo.all_for_book(db, book_id, include_text)]


@router.get("/books/{book_id}/pages/{page_no}")
def get_page(book_id: str, page_no: int, request: Request) -> dict[str, Any]:
    _, db, _ = _services(request)
    page = PagesRepo.get(db, book_id, page_no)
    if page is None:
        raise HTTPException(status_code=404, detail="Page not processed yet")
    return page.to_dict()


@router.get("/books/{book_id}/pages/{page_no}/image")
def page_image(book_id: str, page_no: int, request: Request) -> Any:
    cfg, db, _ = _services(request)
    book = BooksRepo.get(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    page = PagesRepo.get(db, book_id, page_no)
    if page is None or not page.raster_path or not Path(page.raster_path).exists():
        with open_document(Path(book.file_path)) as doc:
            if not 1 <= page_no <= doc.page_count:
                raise HTTPException(status_code=404, detail="Page out of range")
            raster = page_image_for(
                doc, cfg.path("paths", "books_dir", default="data/books"),
                book_id, page_no, int(cfg.get("ingestion", "dpi", default=250)),
            )
    else:
        raster = Path(page.raster_path)
    from fastapi.responses import FileResponse

    return FileResponse(str(raster), media_type="image/png")


@router.put("/books/{book_id}/ocr-engine")
def set_ocr_engine(book_id: str, body: EngineChoice, request: Request) -> dict[str, Any]:
    _, db, _ = _services(request)
    if body.engine not in ENGINE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown OCR engine: {body.engine}")
    if BooksRepo.get(db, book_id) is None:
        raise HTTPException(status_code=404, detail="Book not found")
    BooksRepo.update(db, book_id, ocr_engine=body.engine)
    return {"ok": True, "engine": body.engine}


@router.post("/books/{book_id}/retry")
def retry_pages(book_id: str, body: RetryRequest, request: Request) -> dict[str, Any]:
    _, _, service = _services(request)
    job = service.retry_failed(book_id, tuple(body.statuses))
    if job is None:
        return {"ok": False, "message": "No pages to retry."}
    return {"ok": True, "job": job.to_dict()}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    _, db, _ = _services(request)
    job = JobsRepo.get(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job.to_dict(), "events": JobsRepo.events(db, job_id)}


@router.post("/jobs/{job_id}/pause")
def pause_job(job_id: str, request: Request) -> dict[str, Any]:
    return {"ok": _services(request)[2].pause(job_id)}


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str, request: Request) -> dict[str, Any]:
    return {"ok": _services(request)[2].resume(job_id)}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    return {"ok": _services(request)[2].cancel(job_id)}


@router.delete("/books/{book_id}")
def delete_book(book_id: str, request: Request) -> dict[str, Any]:
    cfg, db, service = _services(request)
    book = BooksRepo.get(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")

    for job in JobsRepo.for_book(db, book_id):
        service.cancel(job.id)

    if not BooksRepo.delete(db, book_id):
        raise HTTPException(status_code=404, detail="Book not found")

    books_dir: Path = cfg.path("paths", "books_dir", default="data/books")
    shutil.rmtree(books_dir / book_id, ignore_errors=True)
    return {"ok": True, "deleted_book_id": book_id}


@router.delete("/books")
def delete_all_books(request: Request) -> dict[str, Any]:
    cfg, db, service = _services(request)
    books = BooksRepo.all(db)
    for book in books:
        for job in JobsRepo.for_book(db, book.id):
            service.cancel(job.id)
        BooksRepo.delete(db, book.id)
        books_dir: Path = cfg.path("paths", "books_dir", default="data/books")
        shutil.rmtree(books_dir / book.id, ignore_errors=True)
    return {"ok": True, "deleted": len(books)}


@router.post("/ocr/benchmark")
def benchmark(body: BenchmarkRequest, request: Request) -> dict[str, Any]:
    cfg, db, _ = _services(request)
    book = BooksRepo.get(db, body.book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    engines = body.engines or list(ENGINE_NAMES)
    unknown = [name for name in engines if name not in ENGINE_NAMES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown OCR engine(s): {unknown}")
    try:
        return run_benchmark(cfg, db, book, body.pages, engines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ocr/benchmark/latest")
def latest_benchmark(request: Request, book_id: str = Query(...)) -> dict[str, Any]:
    _, db, _ = _services(request)
    result = BenchmarkRepo.latest(db, book_id)
    if result is None:
        return {"available": False}
    return {"available": True, **result}


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    cfg, _, _ = _services(request)
    return {
        "ocr": {
            "default_engine": cfg.get("ocr", "default_engine"),
            "engines": cfg.get("ocr", "engines"),
            "tesseract": cfg.get("ocr", "tesseract"),
        },
        "ingestion": cfg.get("ingestion"),
        "ui": cfg.get("ui"),
        "ollama": {"host": cfg.get("ollama", "host")},
    }


@router.put("/config/ui")
def update_ui_settings(body: UISettings, request: Request) -> dict[str, Any]:
    cfg, _, _ = _services(request)
    updates: dict[str, Any] = {"ui": {}}
    if body.language:
        updates["ui"]["language"] = body.language
    if body.theme:
        updates["ui"]["theme"] = body.theme
    if body.default_engine:
        if body.default_engine not in ENGINE_NAMES:
            raise HTTPException(status_code=400, detail="Unknown OCR engine")
        updates["ocr"] = {"default_engine": body.default_engine}
    cfg.save_local(updates)
    return {"ok": True, "ui": cfg.get("ui")}
