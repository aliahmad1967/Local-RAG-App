"""Resume an existing book's OCR job from its last committed page.

Usage:
    python scripts/resume_ingest.py <book-id> [--timeout 7200]

No new book/job is created. Running jobs left behind by a crash are requeued,
paused jobs are resumed, and already committed pages are skipped by the worker.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.db.repositories import BooksRepo, JobsRepo, PagesRepo  # noqa: E402
from localrag.ingest.service import IngestService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("book_id")
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()

    cfg = load_config()
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    book = BooksRepo.get(db, args.book_id)
    if book is None:
        print(f"ERROR: no book with id {args.book_id}", flush=True)
        return 1

    service = IngestService(cfg, db)
    service.start()
    try:
        jobs = JobsRepo.for_book(db, book.id)
        if not jobs:
            print(f"ERROR: book {book.title!r} has no ingest job", flush=True)
            return 1
        job = jobs[0]
        if job.status == "paused":
            service.resume(job.id)
        print(
            f"RESUME book={book.id} pages={book.page_count} "
            f"job={job.id} status={job.status}",
            flush=True,
        )
        last_page = -1
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            live = JobsRepo.get(db, job.id)
            progress = service.progress(book.id)
            page = live.current_page if live else 0
            if page != last_page:
                counts = PagesRepo.counts(db, book.id)
                print(
                    f"PROGRESS page={page}/{book.page_count} "
                    f"done={counts.get('done', 0)} flagged={counts.get('flagged', 0)} "
                    f"failed={counts.get('failed', 0)} status={live.status if live else 'missing'}",
                    flush=True,
                )
                last_page = page
            if live is None or live.status in ("done", "failed", "cancelled"):
                break
            time.sleep(0.5)

        live = JobsRepo.get(db, job.id)
        counts = PagesRepo.counts(db, book.id)
        print(
            f"FINAL job={live.status if live else 'missing'} "
            f"counts={dict(sorted(counts.items()))}",
            flush=True,
        )
        return 0 if live is not None and live.status == "done" else 1
    finally:
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
