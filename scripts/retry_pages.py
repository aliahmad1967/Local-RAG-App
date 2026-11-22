"""Retry only failed/flagged OCR pages for an existing book.

Usage:
    python scripts/retry_pages.py <book-id> [--statuses failed flagged]

Committed good pages are untouched; only the selected statuses are reset to
pending and processed by a new retry job.
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
    parser.add_argument("--statuses", nargs="+", default=["failed", "flagged"])
    parser.add_argument("--timeout", type=float, default=3600.0)
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
        before = PagesRepo.counts(db, book.id)
        job = service.retry_failed(book.id, tuple(args.statuses))
        if job is None:
            print(f"RETRY skipped; counts={dict(sorted(before.items()))}", flush=True)
            return 0
        print(
            f"RETRY book={book.id} job={job.id} statuses={args.statuses} "
            f"before={dict(sorted(before.items()))}",
            flush=True,
        )
        deadline = time.monotonic() + args.timeout
        last_page = -1
        while time.monotonic() < deadline:
            live = JobsRepo.get(db, job.id)
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
