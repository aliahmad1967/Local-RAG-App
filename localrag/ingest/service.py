"""Ingestion orchestration used by both the CLI and the HTTP API."""

from __future__ import annotations

import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..db.database import Database
from ..db.repositories import BooksRepo, JobsRepo, PagesRepo
from ..logging_conf import get_logger
from ..models.domain import Book, BookProgress, Job
from ..paths import book_dir
from ..text.language import detect_language
from .job_queue import JobQueue
from .pdf import extract_embedded_text, inspect_pdf, open_document, safe_filename, file_sha256
from .worker import IngestWorker

log = get_logger("ingest.service")


class IngestService:
    def __init__(self, cfg: Config, db: Database) -> None:
        self.cfg = cfg
        self.db = db
        self.queue = JobQueue(db, cfg)
        self.stop_event = threading.Event()
        self.worker: IngestWorker | None = None

    # -- worker lifecycle --------------------------------------------------
    def start(self) -> list[str]:
        """Recover orphaned jobs and start the background worker thread."""
        recovered = self.queue.recover_orphans()
        if self.worker is None or not self.worker.is_alive():
            self.stop_event = threading.Event()
            self.worker = IngestWorker(self.db, self.cfg, self.queue, self.stop_event)
            self.worker.start()
        if recovered:
            log.warning("Recovered %d orphaned job(s) after restart", len(recovered))
        return recovered

    def stop(self, timeout: float = 10.0) -> None:
        self.stop_event.set()
        self.queue.wake()
        if self.worker is not None:
            self.worker.join(timeout=timeout)
            self.worker = None

    # -- commands ----------------------------------------------------------
    def import_pdf(self, source: Path, engine: str | None = None) -> tuple[Book, Job]:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"PDF not found: {source}")
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {source.name}")

        metadata = inspect_pdf(source)
        book_id = str(uuid.uuid4())
        destination_dir = book_dir(self.cfg.path("paths", "books_dir", default="data/books"), book_id) / "source"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / safe_filename(source.name)
        shutil.copy2(source, destination)

        preview_parts: list[str] = []
        with open_document(destination) as doc:
            for index in range(min(5, doc.page_count)):
                preview_parts.append(extract_embedded_text(doc[index]))
        language = detect_language("\n".join(preview_parts))

        book = Book(
            id=book_id,
            title=metadata.title or destination.stem,
            file_path=str(destination),
            file_name=destination.name,
            file_sha256=file_sha256(destination),
            page_count=metadata.page_count,
            language=language,
            ocr_engine=engine or str(self.cfg.get("ocr", "default_engine", default="glm-ocr")),
            status="queued",
        )
        BooksRepo.create(self.db, book)
        job = self.queue.enqueue_ingest(book, engine=book.ocr_engine)
        log.info("Imported '%s' (%d pages) as %s", book.title, book.page_count, book.id)
        return book, job

    def retry_failed(self, book_id: str, statuses: tuple[str, ...] = ("failed", "flagged")) -> Job | None:
        book = BooksRepo.get(self.db, book_id)
        if book is None:
            return None
        count = PagesRepo.reset(self.db, book_id, statuses)
        if not count:
            return None
        return self.queue.enqueue_ingest(book, engine=book.ocr_engine, retry_statuses=statuses)

    def pause(self, job_id: str) -> bool:
        return self.queue.pause(job_id)

    def resume(self, job_id: str) -> bool:
        return self.queue.resume(job_id)

    def cancel(self, job_id: str) -> bool:
        return self.queue.cancel(job_id)

    # -- read models -------------------------------------------------------
    def progress(self, book_id: str) -> BookProgress | None:
        book = BooksRepo.get(self.db, book_id)
        if book is None:
            return None
        counts = PagesRepo.counts(self.db, book_id)
        jobs = JobsRepo.for_book(self.db, book_id)
        latest = jobs[0] if jobs else None
        eta: float | None = None
        if (
            latest is not None
            and latest.status == "running"
            and latest.started_at
            and latest.processed > 0
        ):
            try:
                start = datetime.fromisoformat(latest.started_at)
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                remaining = book.page_count - counts.get("done", 0) - counts.get("flagged", 0)
                eta = round(max(0.0, elapsed / latest.processed) * remaining, 1)
            except ValueError:
                eta = None
        return BookProgress(
            book_id=book_id,
            page_count=book.page_count,
            done=counts.get("done", 0),
            failed=counts.get("failed", 0),
            flagged=counts.get("flagged", 0),
            pending=book.page_count - sum(counts.values()),
            job_status=latest.status if latest else None,
            current_page=latest.current_page if latest else 0,
            eta_s=eta,
        )
