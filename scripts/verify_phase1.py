"""Real end-to-end verification for Phase 1. Prints everything it checks."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import ensure_data_dirs, load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.db.repositories import BooksRepo, JobsRepo, PagesRepo  # noqa: E402
from localrag.ingest.pdf import extract_embedded_text, open_document  # noqa: E402
from localrag.logging_conf import setup_logging  # noqa: E402
from localrag.ocr.vram import ollama_status  # noqa: E402
from tests.data.make_samples import generate_all, make_sample_pdf  # noqa: E402
from verification_env import isolate_verification_data  # noqa: E402


def heading(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def run_command(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT), env=merged, text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def preview(pages: list, indices: list[int], limit: int = 220) -> None:
    wanted = set(indices)
    for page in pages:
        if page.page_no in wanted:
            sample = " ".join(page.text_clean.split())[:limit]
            print(
                f"  p{page.page_no:03d} [{page.source}/{page.status}] "
                f"chars={page.char_count} conf={page.confidence} :: {sample}",
                flush=True,
            )


def main() -> int:
    isolate_verification_data(ROOT)
    cfg = load_config()
    ensure_data_dirs(cfg)
    setup_logging(cfg)
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))

    heading("1. ENVIRONMENT")
    print(f"python          : {sys.version.split()[0]}")
    status = ollama_status(cfg)
    print(f"ollama          : {'running' if status['running'] else 'DOWN'} @ {status['host']}")
    print(f"ollama models   : {len(status['models'])} installed")
    import sqlite3

    probe = sqlite3.connect(":memory:")
    probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    print(f"sqlite / FTS5   : {sqlite3.sqlite_version} / available")
    probe.close()

    heading("2. UNIT TESTS (pytest)")
    result = run_command(["-m", "pytest"])
    print(result.stdout.strip(), flush=True)
    if result.returncode != 0:
        print("pytest FAILED", flush=True)
        return 1

    heading("3. SAMPLE PDFs")
    samples = generate_all(ROOT / "tests" / "data")
    for key, path in samples.items():
        with open_document(path) as doc:
            layer = " ".join(extract_embedded_text(doc[i]) for i in range(min(3, doc.page_count)))
        print(
            f"  {key:2s}: {path.name} pages={_page_count(path)} "
            f"bytes={path.stat().st_size} embedded_text_chars={len(layer.strip())}",
            flush=True,
        )

    heading("4. INGEST SCANNED ARABIC SAMPLE (paddleocr-vl)")
    arabic_result = run_command(
        ["-m", "localrag", "ingest", str(samples["ar"]), "--engine", "paddleocr-vl", "--timeout", "1800"]
    )
    print(arabic_result.stdout.strip(), flush=True)
    if arabic_result.returncode != 0:
        print("Arabic ingest FAILED", flush=True)
        return 1

    heading("5. INGEST SCANNED ENGLISH SAMPLE (paddleocr-vl)")
    english_result = run_command(
        ["-m", "localrag", "ingest", str(samples["en"]), "--engine", "paddleocr-vl", "--timeout", "1800"]
    )
    print(english_result.stdout.strip(), flush=True)
    if english_result.returncode != 0:
        print("English ingest FAILED", flush=True)
        return 1

    heading("6. PAGES, PREVIEWS AND CHECKPOINTS")
    all_ok = True
    for key in ("ar", "en"):
        book = _latest_book_for(db, samples[key].name)
        assert book is not None
        pages = PagesRepo.all_for_book(db, book.id)
        counts = PagesRepo.counts(db, book.id)
        print(f"  {key}: book='{book.title}' language={book.language} pages={len(pages)} counts={dict(sorted(counts.items()))}", flush=True)
        preview(pages, [1, 2, 10, 20])
        if len(pages) != book.page_count:
            all_ok = False

    heading("7. KILL MIDWAY + RESUME")
    resume_pdf = make_sample_pdf(ROOT / "tests" / "data" / "sample_resume_8.pdf", 8, "en", dpi=120)
    crash = run_command(
        ["-m", "localrag", "ingest", str(resume_pdf), "--engine", "paddleocr-vl", "--timeout", "600"],
        env={"LOCALRAG_CRASH_AFTER": "3"},
    )
    print(crash.stdout.strip(), flush=True)
    print(f"  killed process exit code: {crash.returncode} (expected 9)", flush=True)
    killed_book = _latest_book_for(db, resume_pdf.name)
    assert killed_book is not None
    before = PagesRepo.counts(db, killed_book.id)
    print(f"  pages committed before kill: {dict(sorted(before.items()))}", flush=True)

    jobs = JobsRepo.for_book(db, killed_book.id)
    print(f"  jobs for killed book (before recovery): {[job.status for job in jobs]}", flush=True)

    # A fresh application start recovers the orphaned job; committed pages stay done.
    from localrag.ingest.service import IngestService

    service = IngestService(cfg, db)
    recovered = service.start()
    print(f"  recovered orphaned jobs: {recovered}", flush=True)
    deadline = time.monotonic() + 600
    killed_job = next((job for job in jobs if job.status == "running"), jobs[0] if jobs else None)
    while killed_job is not None and time.monotonic() < deadline:
        live = JobsRepo.get(db, killed_job.id)
        if live and live.status in ("done", "failed", "cancelled"):
            killed_job = live
            break
        time.sleep(0.5)
    after = PagesRepo.counts(db, killed_book.id)
    print(f"  pages after resume: {dict(sorted(after.items()))}", flush=True)
    recovered_job = JobsRepo.get(db, killed_job.id) if killed_job else None
    print(f"  job status after resume: {recovered_job.status if recovered_job else 'missing'}", flush=True)
    service.stop()
    if after.get("done", 0) + after.get("flagged", 0) != 8 or not recovered_job or recovered_job.status != "done":
        all_ok = False

    heading("8. RESULT")
    print("PASS - Phase 1 verified end to end" if all_ok else "FAIL - see output above", flush=True)
    return 0 if all_ok else 1


def _page_count(path: Path) -> int:
    from localrag.ingest.pdf import inspect_pdf

    return inspect_pdf(path).page_count


def _latest_book_for(db: Database, file_name: str):
    for book in BooksRepo.all(db):
        if book.file_name == file_name:
            return book
    return None


if __name__ == "__main__":
    raise SystemExit(main())
