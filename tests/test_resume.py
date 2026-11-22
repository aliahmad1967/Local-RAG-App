"""Checkpoint / crash-resume logic, exercised with a fake OCR engine."""

from __future__ import annotations

import time

from localrag.db.repositories import BooksRepo, JobsRepo, PagesRepo
from localrag.ingest.service import IngestService
from localrag.ingest import worker as worker_module
from localrag.models.domain import OCRResult

from .data.make_samples import make_sample_pdf


class FakeEngine:
    name = "fake-ocr"

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "fake-ocr"

    def recognize(self, image_path, language="unknown") -> OCRResult:
        self.calls.append(image_path.name)
        return OCRResult(
            text="This is a recognisable line of fake OCR text for testing.",
            engine=self.name, model="fake", confidence=95.0, duration_s=0.01,
        )

    def unload(self) -> None:
        pass


def _wait_for_job(db, job_id, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = JobsRepo.get(db, job_id)
        if job and job.status in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_checkpoint_then_resume_after_crash(tmp_path, cfg, db, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(worker_module, "get_engine", lambda name, config: FakeEngine(calls))

    source = make_sample_pdf(tmp_path / "book.pdf", pages=6, language="en", dpi=80)
    service = IngestService(cfg, db)
    service.start()
    book, job = service.import_pdf(source, engine="glm-ocr")
    finished = _wait_for_job(db, job.id)
    assert finished.status == "done"
    assert len(calls) == 6, "all six pages should have been OCR'd once"
    assert PagesRepo.counts(db, book.id).get("done") == 6

    # Simulate a hard kill: the job row is still 'running', pages 4-6 are undone.
    with db.transaction() as conn:
        conn.execute(
            "UPDATE pages SET status='pending', text_clean='' WHERE book_id=? AND page_no>=4",
            (book.id,),
        )
        conn.execute("UPDATE jobs SET status='running' WHERE id=?", (job.id,))
    calls.clear()
    service.stop()

    recovered = service.start()  # restart of the application
    assert recovered == [job.id], "orphaned running job must be requeued"
    resumed = _wait_for_job(db, job.id)
    assert resumed.status == "done"
    assert calls == ["p0004.png", "p0005.png", "p0006.png"], (
        "resume must skip the three committed pages"
    )
    assert PagesRepo.counts(db, book.id).get("done") == 6
    service.stop()


def test_retry_only_processes_failed_and_flagged_pages(tmp_path, cfg, db, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(worker_module, "get_engine", lambda name, config: FakeEngine(calls))

    source = make_sample_pdf(tmp_path / "book.pdf", pages=6, language="en", dpi=80)
    service = IngestService(cfg, db)
    service.start()
    book, job = service.import_pdf(source, engine="glm-ocr")
    assert _wait_for_job(db, job.id).status == "done"

    with db.transaction() as conn:
        conn.execute(
            "UPDATE pages SET status='failed' WHERE book_id=? AND page_no=2", (book.id,)
        )
        conn.execute(
            "UPDATE pages SET status='flagged' WHERE book_id=? AND page_no=5", (book.id,)
        )
    calls.clear()
    retry = service.retry_failed(book.id, ("failed", "flagged"))
    assert retry is not None
    assert _wait_for_job(db, retry.id).status == "done"
    assert calls == ["p0002.png", "p0005.png"]
    assert PagesRepo.counts(db, book.id).get("done") == 6
    service.stop()


def test_pause_resume_keeps_completed_pages(tmp_path, cfg, db, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(worker_module, "get_engine", lambda name, config: FakeEngine(calls))

    source = make_sample_pdf(tmp_path / "book.pdf", pages=8, language="en", dpi=80)
    service = IngestService(cfg, db)
    service.start()
    book, job = service.import_pdf(source, engine="glm-ocr")

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        live = JobsRepo.get(db, job.id)
        if live and live.current_page >= 2:
            break
        time.sleep(0.02)
    assert service.pause(job.id)
    time.sleep(0.4)
    paused = JobsRepo.get(db, job.id)
    assert paused.status == "paused"
    done_before = PagesRepo.counts(db, book.id).get("done", 0)
    assert 0 < done_before < 8

    assert service.resume(job.id)
    finished = _wait_for_job(db, job.id)
    assert finished.status == "done"
    assert PagesRepo.counts(db, book.id).get("done") == 8
    assert BooksRepo.get(db, book.id).status == "ingested"
    service.stop()
