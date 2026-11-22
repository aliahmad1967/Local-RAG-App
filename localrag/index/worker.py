"""Background worker for Phase 2 index jobs."""

from __future__ import annotations

import json
import threading

from ..db.database import Database
from ..db.repositories import JobsRepo
from ..ingest.job_queue import JobQueue
from ..logging_conf import get_logger
from ..models.domain import Job, utcnow
from .service import IndexService

log = get_logger("index.worker")


class IndexWorker(threading.Thread):
    def __init__(
        self, db: Database, queue: JobQueue, stop_event: threading.Event, service: IndexService
    ) -> None:
        super().__init__(name="localrag-index", daemon=True)
        self.db = db
        self.queue = queue
        self.stop_event = stop_event
        self.service = service
        self.poll = 0.5

    def run(self) -> None:
        log.info("Index worker started")
        while not self.stop_event.is_set():
            job = self.queue.claim_next(kinds=("index",))
            if job is None:
                self.queue.wait(self.poll)
                continue
            try:
                self.process(job)
            except Exception as exc:  # noqa: BLE001
                log.exception("Index job %s failed: %s", job.id, exc)
                JobsRepo.update(self.db, job.id, status="failed", error=str(exc))
                JobsRepo.add_event(self.db, job.id, f"Index failed: {exc}", level="error")
        log.info("Index worker stopped")

    def process(self, job: Job) -> None:
        params = json.loads(job.params_json or "{}")
        book_id = str(params.get("book_id") or job.book_id or "")
        JobsRepo.update(self.db, job.id, status="running")
        JobsRepo.add_event(self.db, job.id, "Indexing started")
        result = self.service.index_book(book_id, force=bool(params.get("force", False)))
        JobsRepo.update(
            self.db, job.id, status="done", processed=int(result.get("chunks", 0)),
            finished_at=utcnow(),
        )
        JobsRepo.add_event(
            self.db, job.id,
            f"Indexed {result['chunks']} children / {result['parents']} parents, "
            f"{result['embedded']} vectors",
        )
        log.info("Index job %s finished: %s", job.id, result)
