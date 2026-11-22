"""Command-line entry points (also used by verification scripts)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import ensure_data_dirs, load_config
from .db.database import Database
from .db.repositories import BooksRepo, JobsRepo, PagesRepo
from .ingest.service import IngestService
from .index.service import IndexService
from .logging_conf import setup_logging
from .paths import ensure_utf8_console


def _bootstrap():
    cfg = load_config()
    ensure_data_dirs(cfg)
    log = setup_logging(cfg)
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    return cfg, db, log


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    cfg, _, _ = _bootstrap()
    uvicorn.run(
        "localrag.api.app:app",
        host=str(cfg.get("server", "host", default="127.0.0.1")),
        port=int(cfg.get("server", "port", default=8765)),
        reload=False,
        log_level="info",
    )
    return 0


def cmd_desktop(args: argparse.Namespace) -> int:
    from .desktop.app import run_desktop

    return run_desktop(headless=args.headless, port=args.port)


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg, db, log = _bootstrap()
    service = IngestService(cfg, db)
    service.start()
    try:
        book, job = service.import_pdf(Path(args.pdf), engine=args.engine)
        print(f"BOOK id={book.id} pages={book.page_count} lang={book.language} engine={book.ocr_engine}", flush=True)
        last_page = -1
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            live = JobsRepo.get(db, job.id)
            progress = service.progress(book.id)
            if live is None:
                break
            page = live.current_page
            if page != last_page:
                print(
                    f"PROGRESS page={page}/{book.page_count} "
                    f"done={progress.done} flagged={progress.flagged} "
                    f"failed={progress.failed} status={live.status}",
                    flush=True,
                )
                last_page = page
            if live.status in ("done", "failed", "cancelled"):
                break
            time.sleep(0.2)

        live = JobsRepo.get(db, job.id)
        counts = PagesRepo.counts(db, book.id)
        print(f"FINAL job={live.status if live else 'missing'} counts={dict(sorted(counts.items()))}", flush=True)
        return 0 if live is not None and live.status == "done" else 1
    finally:
        service.stop()


def cmd_index(args: argparse.Namespace) -> int:
    cfg, db, _ = _bootstrap()
    service = IndexService(cfg, db)
    try:
        result = service.index_book(args.book_id, force=args.force)
    except ValueError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(
        f"INDEX book={result['book_id']} status={result['status']} "
        f"chunks={result['chunks']} parents={result['parents']} "
        f"embedded={result['embedded']} bm25_rows={result['bm25_rows']} "
        f"vector_rows={result['vector_rows']} size_bytes={result['index_size_bytes']}",
        flush=True,
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .config import load_config
    from .index import bm25

    cfg = load_config()
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    results = bm25.search(db, args.query, limit=args.limit, book_id=args.book_id, cfg=cfg)
    print(f"BM25 query={args.query!r} hits={len(results)}", flush=True)
    for index, hit in enumerate(results[:10], start=1):
        print(
            f"  {index:2d}. page={hit['page_no']} score={hit['rank']:.4f} "
            f"{' '.join(hit['snippet'].split())[:160]}",
            flush=True,
        )
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    from .retrieval.query_rewrite import NoopRewriter
    from .retrieval.rerank import build_reranker
    from .retrieval.service import RetrievalFilters, RetrievalService

    cfg, db, _ = _bootstrap()
    service = RetrievalService(
        cfg,
        db,
        reranker=None if args.no_rerank else build_reranker(cfg),
        rewriter=NoopRewriter(),
    )
    try:
        result = service.retrieve(
            args.query,
            filters=RetrievalFilters(
                book_ids=[item.strip() for item in (args.book_ids or "").split(",") if item.strip()] or None,
                page_from=args.page_from,
                page_to=args.page_to,
            ),
            top_k=args.top_k,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    finally:
        service.close()

    print(
        f"RETRIEVE query={result.query!r} rewritten={result.rewritten_query!r} "
        f"counts={result.counts} timings={result.timings_s}",
        flush=True,
    )
    for index, hit in enumerate(result.hits, start=1):
        print(
            f"  {index:2d}. dense={hit.dense_rank} bm25={hit.bm25_rank} "
            f"fused={hit.fused_rank} rerank={hit.rerank_rank} "
            f"score={hit.rerank_score} {hit.book_title} p.{hit.page_start}-{hit.page_end}",
            flush=True,
        )
        print(f"       {' '.join(hit.text.split())[:180]}", flush=True)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _, db, _ = _bootstrap()
    books = BooksRepo.all(db)
    print(f"BOOKS={len(books)}")
    for book in books:
        counts = PagesRepo.counts(db, book.id)
        jobs = JobsRepo.for_book(db, book.id)
        print(
            f"  {book.title!r} id={book.id} pages={book.page_count} "
            f"lang={book.language} ocr={book.ocr_engine} counts={dict(sorted(counts.items()))} "
            f"jobs={[j.status for j in jobs]}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localrag", description="Offline book RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI server")
    serve.set_defaults(func=cmd_serve)

    desktop = sub.add_parser("desktop", help="Run the pywebview desktop app")
    desktop.add_argument(
        "--headless", action="store_true", help="Start only the local server (verification)"
    )
    desktop.add_argument(
        "--port", type=int, default=None, help="Override the desktop port (0 = auto)"
    )
    desktop.set_defaults(func=cmd_desktop)

    ingest = sub.add_parser("ingest", help="Import and OCR a PDF")
    ingest.add_argument("pdf", help="Path to a PDF file (Unicode paths supported)")
    ingest.add_argument("--engine", default=None, help="OCR engine name")
    ingest.add_argument("--timeout", type=float, default=3600.0, help="Max wait seconds")
    ingest.set_defaults(func=cmd_ingest)

    index = sub.add_parser("index", help="Chunk, BM25-index and embed a book")
    index.add_argument("book_id", help="Book ID (see 'status')")
    index.add_argument("--force", action="store_true", help="Rebuild chunks from scratch")
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="BM25 search (Phase 2 debug)")
    search.add_argument("query", help="Query text (Arabic or English)")
    search.add_argument("--book-id", default=None)
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(func=cmd_search)

    retrieve = sub.add_parser("retrieve", help="Hybrid search with rank debug (Phase 3)")
    retrieve.add_argument("query", help="Query text (Arabic or English)")
    retrieve.add_argument("--book-ids", default=None, help="Comma-separated book IDs")
    retrieve.add_argument("--page-from", type=int, default=None)
    retrieve.add_argument("--page-to", type=int, default=None)
    retrieve.add_argument("--top-k", type=int, default=None)
    retrieve.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder")
    retrieve.set_defaults(func=cmd_retrieve)

    status = sub.add_parser("status", help="Show books, pages and jobs")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_console()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
