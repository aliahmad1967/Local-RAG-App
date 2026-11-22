"""Persistent job queue backed by SQLite jobs/job_events tables."""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any

from ..config import Config
from ..db.database import Database
from ..db.repositories import JobsRepo
from ..models.domain import Book, Job, utcnow


class JobQueue:
    def __init__(self, db: Database, cfg: Config) -> None:
        self.db = db
        self.cfg = cfg
        self._lock = threading.RLock()
        self._wake = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def recover_orphans(self) -> list[str]:
        """Requeue jobs left 'running' by a crashed process."""
        return JobsRepo.recover_orphans(self.db)

    def claim_next(self, kinds: tuple[str, ...] = ("ingest",)) -> Job | None:
        with self._lock:
            return JobsRepo.claim_next(self.db, kinds)

    def wait(self, timeout: float) -> None:
        self._wake.wait(timeout)
        self._wake.clear()

    def wake(self) -> None:
        self._wake.set()

    # -- creation ----------------------------------------------------------
    def enqueue_ingest(
        self,
        book: Book,
        engine: str | None = None,
        retry_statuses: tuple[str, ...] | None = None,
    ) -> Job:
        params: dict[str, Any] = {"engine": engine or book.ocr_engine}
        if retry_statuses:
            params["retry_statuses"] = list(retry_statuses)
        job = Job(
            id=str(uuid.uuid4()),
            book_id=book.id,
            kind="ingest",
            status="queued",
            total_pages=book.page_count,
            params_json=json.dumps(params, ensure_ascii=False),
            created_at=utcnow(),
        )
        JobsRepo.create(self.db, job)
        JobsRepo.add_event(
            self.db, job.id,
            f"Queued {book.page_count} pages with {params['engine'] or 'default OCR'}",
        )
        self.wake()
        return job

    # -- control -----------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return JobsRepo.get(self.db, job_id)

    def set_status(self, job_id: str, status: str, **fields: Any) -> None:
        JobsRepo.update(self.db, job_id, status=status, **fields)
        JobsRepo.add_event(self.db, job_id, f"Status → {status}")
        self.wake()

    def pause(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job and job.status in ("queued", "running"):
            self.set_status(job_id, "paused")
            return True
        return False

    def resume(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job and job.status == "paused":
            self.set_status(job_id, "queued")
            return True
        return False

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job and job.status in ("queued", "running", "paused"):
            self.set_status(job_id, "cancelled")
            return True
        return False

    def is_paused(self, job_id: str) -> bool:
        job = self.get(job_id)
        return job is not None and job.status == "paused"

    def is_cancelled(self, job_id: str) -> bool:
        job = self.get(job_id)
        return job is not None and job.status == "cancelled"

    def advance(self, job_id: str, page_no: int, processed: int, failed: int) -> None:
        JobsRepo.update(
            self.db, job_id,
            current_page=page_no, processed=processed, failed=failed,
        )
