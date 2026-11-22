"""Background ingestion worker with one SQLite checkpoint per page."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import fitz

from ..db.database import Database
from ..db.repositories import BooksRepo, JobsRepo, PagesRepo
from ..logging_conf import get_logger
from ..models.domain import Book, Job, Page
from ..ocr.base import OCREngine
from ..ocr.registry import get_engine
from ..ocr.vram import unload_ocr_models
from ..text.cleanup import build_text_views
from ..text.language import detect_language
from .job_queue import JobQueue
from .pdf import (
    extract_embedded_text,
    has_usable_text,
    open_document,
    page_image_for,
)

if TYPE_CHECKING:  # pragma: no cover
    pass

log = get_logger("ingest.worker")


class IngestWorker(threading.Thread):
    def __init__(self, db: Database, cfg, queue: JobQueue, stop_event: threading.Event) -> None:
        super().__init__(name="localrag-ingest", daemon=True)
        self.db = db
        self.cfg = cfg
        self.queue = queue
        self.stop_event = stop_event
        # Verification hook: die after N committed pages to test crash recovery.
        self.crash_after = int(os.environ.get("LOCALRAG_CRASH_AFTER", "0") or 0)

    def run(self) -> None:
        log.info("Ingest worker started (pid=%d)", os.getpid())
        poll = float(self.cfg.get("ingestion", "poll_interval_s", default=0.4))
        while not self.stop_event.is_set():
            job = self.queue.claim_next()
            if job is None:
                self.queue.wait(poll)
                continue
            try:
                self.process_job(job)
            except Exception as exc:  # noqa: BLE001 - keep the worker alive
                log.exception("Job %s failed: %s", job.id, exc)
                JobsRepo.update(
                    self.db, job.id, status="failed", error=str(exc), finished_at=_now()
                )
                JobsRepo.add_event(self.db, job.id, f"Job failed: {exc}", level="error")
        log.info("Ingest worker stopped")

    # ------------------------------------------------------------------
    def process_job(self, job: Job) -> None:
        book = BooksRepo.get(self.db, job.book_id or "")
        if book is None:
            raise ValueError(f"Book {job.book_id} no longer exists")

        params = json.loads(job.params_json or "{}")
        engine_name = str(
            params.get("engine") or book.ocr_engine
            or self.cfg.get("ocr", "default_engine", default="glm-ocr")
        )
        retry_statuses = tuple(params.get("retry_statuses", []))
        engine: OCREngine | None = None

        dpi = int(self.cfg.get("ingestion", "dpi", default=250))
        min_chars = int(self.cfg.get("ingestion", "min_embedded_chars", default=80))
        threshold = float(
            self.cfg.get("ingestion", "low_confidence_threshold", default=45.0)
        )
        books_root = self.cfg.path("paths", "books_dir", default="data/books")

        JobsRepo.update(self.db, job.id, status="running", total_pages=book.page_count, error=None)
        JobsRepo.add_event(self.db, job.id, f"Processing embedded text with OCR fallback: {engine_name}")
        BooksRepo.update(self.db, book.id, status="processing", ocr_engine=engine_name)

        with open_document(Path(book.file_path)) as doc:
            for page_no in range(1, book.page_count + 1):
                live = self.queue.get(job.id)
                if live is None or live.status == "cancelled":
                    log.info("Job %s cancelled at page %d", job.id, page_no)
                    return
                if live.status == "paused":
                    log.info("Job %s paused at page %d", job.id, page_no)
                    return

                existing = PagesRepo.get(self.db, book.id, page_no)
                if retry_statuses:
                    if existing is None or existing.status not in retry_statuses:
                        continue
                elif existing is not None and existing.status in ("done", "flagged") and existing.source != "pending":
                    # Durable checkpoint: never reprocess a completed page.
                    continue

                if engine is None:
                    first_page = doc[page_no - 1]
                    embedded = extract_embedded_text(first_page)
                    if not has_usable_text(embedded, min_chars):
                        engine = get_engine(engine_name, self.cfg)
                        if not engine.available():
                            raise RuntimeError(f"OCR engine '{engine_name}' is not available on this machine")

                status = self.process_page(
                    doc=doc, book=book, engine=engine, page_no=page_no,
                    dpi=dpi, min_chars=min_chars, threshold=threshold,
                    books_root=books_root, existing=existing,
                )
                counts = PagesRepo.counts(self.db, book.id)
                processed = counts.get("done", 0) + counts.get("flagged", 0)
                self.queue.advance(
                    job.id, page_no, processed=processed, failed=counts.get("failed", 0)
                )
                if page_no % 10 == 0 or page_no == book.page_count:
                    JobsRepo.add_event(
                        self.db, job.id, f"Page {page_no}/{book.page_count} ({status})"
                    )

                if self.crash_after and processed >= self.crash_after:
                    print(f"[CRASH-HOOK] committed {processed} pages; simulating hard kill", flush=True)
                    os._exit(9)  # no cleanup: exactly what a real crash looks like

        live = self.queue.get(job.id)
        if live is None or live.status != "running":
            return  # paused / cancelled during the last page

        if bool(self.cfg.get("ingestion", "unload_ocr_after_job", default=True)):
            unload_ocr_models(self.cfg)
        counts = PagesRepo.counts(self.db, book.id)
        JobsRepo.update(
            self.db, job.id,
            status="done",
            current_page=book.page_count,
            processed=counts.get("done", 0) + counts.get("flagged", 0),
            failed=counts.get("failed", 0),
            finished_at=_now(),
        )
        BooksRepo.update(self.db, book.id, status="ingested")
        JobsRepo.add_event(
            self.db, job.id,
            f"Finished: {counts.get('done', 0)} done, "
            f"{counts.get('flagged', 0)} flagged, {counts.get('failed', 0)} failed",
        )
        log.info("Job %s finished for book %s", job.id, book.title)

    # ------------------------------------------------------------------
    def process_page(
        self,
        doc: fitz.Document,
        book: Book,
        engine: OCREngine | None,
        page_no: int,
        dpi: int,
        min_chars: int,
        threshold: float,
        books_root: Path,
        existing: Page | None,
    ) -> str:
        fitz_page = doc[page_no - 1]
        embedded = extract_embedded_text(fitz_page)

        if has_usable_text(embedded, min_chars):
            raw = embedded
            source = "embedded"
            engine_used: str | None = None
            confidence: float | None = None
            raster: str | None = None
            duration = 0.0
            status = "done"
            error = None
        else:
            raster_path = page_image_for(fitz_page, books_root, book.id, page_no, dpi=dpi)
            raster = str(raster_path)
            source = "ocr"
            if engine is None:
                engine = get_engine(book.ocr_engine, self.cfg)
                if not engine.available():
                    raise RuntimeError(f"OCR engine '{engine.name}' is not available on this machine")
            engine_used = engine.name
            raw = ""
            confidence = None
            duration = 0.0
            error = None
            status = "done"

            retry_limit = int(self.cfg.get("ingestion", "retry_limit", default=2))
            for attempt in range(retry_limit + 1):
                result = engine.recognize(raster_path, book.language)
                duration += result.duration_s
                if result.ok and result.text.strip():
                    raw = result.text
                    confidence = result.confidence
                    error = None
                    break
                error = result.error or "OCR returned empty text"
                if attempt < retry_limit:
                    engine.unload()
                    log.warning("Retry %d/%d for page %d: %s", attempt + 1, retry_limit, page_no, error)

            if not raw.strip():
                status = "failed"
            elif confidence is not None and confidence < threshold:
                status = "flagged"

        clean, normalized = build_text_views(raw, self.cfg)
        if source == "ocr" and clean and book.language == "unknown":
            detected = detect_language(raw)
            if detected != "unknown":
                BooksRepo.update(self.db, book.id, language=detected)
                book.language = detected

        page = existing or Page(
            id=str(uuid.uuid4()), book_id=book.id, page_no=page_no
        )
        page.source = source
        page.ocr_engine = engine_used
        page.text_original = raw
        page.text_clean = clean
        page.text_normalized = normalized
        page.confidence = confidence
        page.status = status
        page.error = error
        page.raster_path = raster
        page.char_count = len(clean)
        page.duration_s = round(duration, 3)
        if status == "failed":
            page.retry_count += 1
        PagesRepo.upsert(self.db, page)  # <-- the per-page commit / checkpoint
        return status


def _now() -> str:
    from ..models.domain import utcnow

    return utcnow()
