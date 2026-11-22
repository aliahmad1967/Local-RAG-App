"""Repositories: the only place that writes SQL for books, pages and jobs."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from ..models.domain import (
    Book, Chunk, Conversation, IndexStatus, Job, Memory, Message, Page, utcnow,
)
from .database import Database


class BooksRepo:
    @staticmethod
    def create(db: Database, book: Book) -> Book:
        with db.transaction() as conn:
            conn.execute(
                """INSERT INTO books
                   (id, title, file_path, file_name, file_sha256, page_count,
                    language, ocr_engine, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    book.id, book.title, book.file_path, book.file_name,
                    book.file_sha256, book.page_count, book.language,
                    book.ocr_engine, book.status, book.created_at, book.updated_at,
                ),
            )
        return book

    @staticmethod
    def get(db: Database, book_id: str) -> Book | None:
        row = db.connect().execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        return Book.from_row(row) if row else None

    @staticmethod
    def all(db: Database) -> list[Book]:
        rows = db.connect().execute("SELECT * FROM books ORDER BY created_at DESC").fetchall()
        return [Book.from_row(row) for row in rows]

    @staticmethod
    def update(db: Database, book_id: str, **fields: Any) -> None:
        allowed = {
            "title", "file_path", "file_name", "file_sha256", "page_count",
            "language", "ocr_engine", "status",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = utcnow()
        assignments = ", ".join(f"{k} = ?" for k in updates)
        with db.transaction() as conn:
            conn.execute(
                f"UPDATE books SET {assignments} WHERE id = ?",
                (*updates.values(), book_id),
            )

    @staticmethod
    def delete(db: Database, book_id: str) -> bool:
        with db.transaction() as conn:
            cursor = conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
            return cursor.rowcount > 0


class PagesRepo:
    _COLUMNS = (
        "id", "book_id", "page_no", "source", "ocr_engine", "text_original",
        "text_clean", "text_normalized", "confidence", "status", "error",
        "raster_path", "retry_count", "char_count", "duration_s", "updated_at",
    )

    @staticmethod
    def upsert(db: Database, page: Page) -> Page:
        """Insert or replace a single page. Called once per page = durable checkpoint."""
        page.updated_at = utcnow()
        values = tuple(getattr(page, col) for col in PagesRepo._COLUMNS)
        placeholders = ", ".join("?" for _ in values)
        updates = ", ".join(
            f"{col} = excluded.{col}" for col in PagesRepo._COLUMNS if col not in ("id", "book_id", "page_no")
        )
        with db.transaction() as conn:
            conn.execute(
                f"INSERT INTO pages ({', '.join(PagesRepo._COLUMNS)}) VALUES ({placeholders}) "
                f"ON CONFLICT (book_id, page_no) DO UPDATE SET {updates}",
                values,
            )
        return page

    @staticmethod
    def get(db: Database, book_id: str, page_no: int) -> Page | None:
        row = db.connect().execute(
            "SELECT * FROM pages WHERE book_id = ? AND page_no = ?", (book_id, page_no)
        ).fetchone()
        return Page.from_row(row) if row else None

    @staticmethod
    def all_for_book(db: Database, book_id: str, include_text: bool = True) -> list[Page]:
        columns = "*" if include_text else (
            "id, book_id, page_no, source, ocr_engine, confidence, status, error, "
            "raster_path, retry_count, char_count, duration_s, updated_at"
        )
        rows = db.connect().execute(
            f"SELECT {columns} FROM pages WHERE book_id = ? ORDER BY page_no", (book_id,)
        ).fetchall()
        pages = []
        for row in rows:
            page = Page.from_row(row)
            if not include_text:
                page.text_original = page.text_clean = page.text_normalized = ""
            pages.append(page)
        return pages

    @staticmethod
    def counts(db: Database, book_id: str) -> dict[str, int]:
        rows = db.connect().execute(
            "SELECT status, COUNT(*) AS n FROM pages WHERE book_id = ? GROUP BY status",
            (book_id,),
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    @staticmethod
    def reset(db: Database, book_id: str, statuses: Iterable[str]) -> int:
        statuses = tuple(statuses)
        if not statuses:
            return 0
        marks = ", ".join("?" for _ in statuses)
        with db.transaction() as conn:
            # Keep the original status: the retry job carries `retry_statuses`
            # and the worker selects exactly those pages. Flipping them to
            # 'pending' here made the worker skip every page and finish empty.
            cursor = conn.execute(
                f"UPDATE pages SET error = NULL "
                f"WHERE book_id = ? AND status IN ({marks})",
                (book_id, *statuses),
            )
            return cursor.rowcount


class ChunksRepo:
    _COLUMNS = (
        "id", "book_id", "level", "parent_id", "child_index", "page_start",
        "page_end", "heading", "text", "text_normalized", "token_count",
        "char_count", "start_char", "end_char", "created_at",
    )

    @staticmethod
    def upsert_many(db: Database, chunks: list[Chunk]) -> int:
        """Insert chunks; existing (book, level, page, index) rows are left alone.

        Skipping existing rows is what makes indexing resumable.
        """
        if not chunks:
            return 0
        inserted = 0
        with db.transaction() as conn:
            for chunk in chunks:
                chunk.created_at = utcnow()
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO chunks "
                    f"({', '.join(ChunksRepo._COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in ChunksRepo._COLUMNS)})",
                    tuple(getattr(chunk, col) for col in ChunksRepo._COLUMNS),
                )
                inserted += cursor.rowcount
        return inserted

    @staticmethod
    def all_for_book(
        db: Database, book_id: str, level: str | None = None, limit: int | None = None
    ) -> list[Chunk]:
        sql = "SELECT * FROM chunks WHERE book_id = ?"
        params: list[Any] = [book_id]
        if level:
            sql += " AND level = ?"
            params.append(level)
        sql += " ORDER BY page_start, child_index, level DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [Chunk.from_row(row) for row in db.connect().execute(sql, params).fetchall()]

    @staticmethod
    def get(db: Database, chunk_id: str) -> Chunk | None:
        row = db.connect().execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return Chunk.from_row(row) if row else None

    @staticmethod
    def counts(db: Database, book_id: str) -> dict[str, int]:
        rows = db.connect().execute(
            "SELECT level, COUNT(*) AS n FROM chunks WHERE book_id = ? GROUP BY level",
            (book_id,),
        ).fetchall()
        return {row["level"]: row["n"] for row in rows}

    @staticmethod
    def delete_book(db: Database, book_id: str) -> int:
        with db.transaction() as conn:
            return conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,)).rowcount

    @staticmethod
    def parent_for_child(db: Database, child: Chunk) -> Chunk | None:
        if not child.parent_id:
            return None
        return ChunksRepo.get(db, child.parent_id)


class IndexStatusRepo:
    @staticmethod
    def get(db: Database, book_id: str) -> IndexStatus:
        row = db.connect().execute(
            "SELECT * FROM book_index_status WHERE book_id = ?", (book_id,)
        ).fetchone()
        return IndexStatus.from_row(row) if row else IndexStatus(book_id=book_id)

    @staticmethod
    def save(db: Database, status: IndexStatus) -> None:
        status.updated_at = utcnow()
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO book_index_status "
                "(book_id, status, chunk_count, embedded_count, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (book_id) DO UPDATE SET "
                "status = excluded.status, chunk_count = excluded.chunk_count, "
                "embedded_count = excluded.embedded_count, updated_at = excluded.updated_at",
                (status.book_id, status.status, status.chunk_count,
                 status.embedded_count, status.updated_at),
            )

    @staticmethod
    def all(db: Database) -> list[IndexStatus]:
        rows = db.connect().execute("SELECT * FROM book_index_status").fetchall()
        return [IndexStatus.from_row(row) for row in rows]


class EmbeddingRepo:
    @staticmethod
    def get_vector(db: Database, chunk_id: str) -> bytes | None:
        row = db.connect().execute(
            "SELECT vector FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return bytes(row["vector"]) if row else None

    @staticmethod
    def missing_hashes(db: Database, pairs: list[tuple[str, str]], model: str) -> list[tuple[str, str]]:
        """Return (chunk_id, hash) pairs whose hash is absent from the cache."""
        if not pairs:
            return []
        conn = db.connect()
        missing = []
        for chunk_id, digest in pairs:
            row = conn.execute(
                "SELECT 1 FROM embedding_cache WHERE content_hash = ? AND model = ?",
                (digest, model),
            ).fetchone()
            if row is None:
                missing.append((chunk_id, digest))
        return missing

    @staticmethod
    def cached_vectors(
        db: Database, hashes: list[str], model: str
    ) -> dict[str, bytes]:
        """Fetch vectors already cached for these content hashes."""
        if not hashes:
            return {}
        conn = db.connect()
        found: dict[str, bytes] = {}
        for digest in set(hashes):
            row = conn.execute(
                "SELECT vector FROM embedding_cache WHERE content_hash = ? AND model = ?",
                (digest, model),
            ).fetchone()
            if row is not None:
                found[digest] = bytes(row["vector"])
        return found

    @staticmethod
    def save_batch(
        db: Database,
        entries: list[tuple[str, str, str, int, bytes]],
    ) -> None:
        """Save (chunk_id, hash, model, dim, blob) rows atomically."""
        if not entries:
            return
        with db.transaction() as conn:
            for chunk_id, digest, model, dim, blob in entries:
                conn.execute(
                    "INSERT OR IGNORE INTO embedding_cache "
                    "(content_hash, model, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                    (digest, model, dim, blob, utcnow()),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO chunk_embeddings "
                    "(chunk_id, content_hash, model, dim, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, digest, model, dim, blob, utcnow()),
                )

    @staticmethod
    def count_for_book(db: Database, book_id: str) -> int:
        row = db.connect().execute(
            "SELECT COUNT(*) AS n FROM chunk_embeddings e "
            "JOIN chunks c ON c.id = e.chunk_id WHERE c.book_id = ?",
            (book_id,),
        ).fetchone()
        return int(row["n"])

    @staticmethod
    def count_cached(db: Database) -> int:
        row = db.connect().execute("SELECT COUNT(*) AS n FROM embedding_cache").fetchone()
        return int(row["n"])


class JobsRepo:
    @staticmethod
    def create(db: Database, job: Job) -> Job:
        with db.transaction() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, book_id, kind, status, current_page, total_pages, processed,
                    failed, params_json, error, created_at, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id, job.book_id, job.kind, job.status, job.current_page,
                    job.total_pages, job.processed, job.failed, job.params_json,
                    job.error, job.created_at, job.started_at, job.finished_at,
                ),
            )
        return job

    @staticmethod
    def get(db: Database, job_id: str) -> Job | None:
        row = db.connect().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    @staticmethod
    def all(db: Database, limit: int = 50) -> list[Job]:
        rows = db.connect().execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Job.from_row(row) for row in rows]

    @staticmethod
    def for_book(db: Database, book_id: str) -> list[Job]:
        rows = db.connect().execute(
            "SELECT * FROM jobs WHERE book_id = ? ORDER BY created_at DESC", (book_id,)
        ).fetchall()
        return [Job.from_row(row) for row in rows]

    @staticmethod
    def update(db: Database, job_id: str, **fields: Any) -> None:
        allowed = {
            "book_id", "kind", "status", "current_page", "total_pages",
            "processed", "failed", "params_json", "error", "started_at", "finished_at",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{k} = ?" for k in updates)
        with db.transaction() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*updates.values(), job_id))

    @staticmethod
    def add_event(db: Database, job_id: str, message: str, level: str = "info") -> None:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO job_events (job_id, ts, level, message) VALUES (?, ?, ?, ?)",
                (job_id, utcnow(), level, message),
            )

    @staticmethod
    def events(db: Database, job_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = db.connect().execute(
            "SELECT ts, level, message FROM job_events WHERE job_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    @staticmethod
    def claim_next(db: Database, kinds: tuple[str, ...] = ("ingest",)) -> Job | None:
        """Atomically move the oldest queued job to `running`."""
        marks = ", ".join("?" for _ in kinds)
        with db.transaction() as conn:
            row = conn.execute(
                f"SELECT * FROM jobs WHERE status = 'queued' AND kind IN ({marks}) "
                "ORDER BY created_at, id LIMIT 1",
                kinds,
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = COALESCE(started_at, ?) WHERE id = ?",
                (utcnow(), row["id"]),
            )
            return Job.from_row(row)

    @staticmethod
    def recover_orphans(db: Database) -> list[str]:
        """After a crash, running jobs have no live worker: requeue them."""
        with db.transaction() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status = 'running'"
            ).fetchall()
            ids = [row["id"] for row in rows]
            for job_id in ids:
                conn.execute(
                    "UPDATE jobs SET status = 'queued' WHERE id = ?", (job_id,)
                )
                conn.execute(
                    "INSERT INTO job_events (job_id, ts, level, message) VALUES (?, ?, 'warn', ?)",
                    (job_id, utcnow(), "Recovered after restart; completed pages are kept."),
                )
        return ids


class BenchmarkRepo:
    @staticmethod
    def save(db: Database, book_id: str, page_numbers: list[int], results: dict[str, Any]) -> str:
        import uuid

        benchmark_id = str(uuid.uuid4())
        with db.transaction() as conn:
            conn.execute(
                """INSERT INTO ocr_benchmarks (id, book_id, page_numbers, results_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    benchmark_id, book_id, json.dumps(page_numbers),
                    json.dumps(results, ensure_ascii=False), utcnow(),
                ),
            )
        return benchmark_id

    @staticmethod
    def latest(db: Database, book_id: str) -> dict[str, Any] | None:
        row = db.connect().execute(
            "SELECT * FROM ocr_benchmarks WHERE book_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (book_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "book_id": row["book_id"],
            "page_numbers": json.loads(row["page_numbers"]),
            "results": json.loads(row["results_json"]),
            "created_at": row["created_at"],
        }


class ConversationsRepo:
    @staticmethod
    def create(db: Database, conversation: Conversation) -> Conversation:
        with db.transaction() as conn:
            conn.execute(
                """INSERT INTO conversations
                   (id, title, book_ids_json, summary, summary_message_count,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    conversation.id, conversation.title, conversation.book_ids_json,
                    conversation.summary, conversation.summary_message_count,
                    conversation.created_at, conversation.updated_at,
                ),
            )
        return conversation

    @staticmethod
    def get(db: Database, conversation_id: str) -> Conversation | None:
        row = db.connect().execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return Conversation.from_row(row) if row else None

    @staticmethod
    def all(db: Database, limit: int = 100) -> list[Conversation]:
        rows = db.connect().execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Conversation.from_row(row) for row in rows]

    @staticmethod
    def update(db: Database, conversation_id: str, **fields: Any) -> None:
        allowed = {"title", "book_ids_json", "summary", "summary_message_count"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = utcnow()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with db.transaction() as conn:
            conn.execute(
                f"UPDATE conversations SET {assignments} WHERE id = ?",
                (*updates.values(), conversation_id),
            )

    @staticmethod
    def delete(db: Database, conversation_id: str) -> bool:
        with db.transaction() as conn:
            return bool(conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            ).rowcount)


class MessagesRepo:
    @staticmethod
    def add(db: Database, message: Message) -> Message:
        with db.transaction() as conn:
            conn.execute(
                """INSERT INTO messages
                   (id, conversation_id, role, content, sources_json, language, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.id, message.conversation_id, message.role, message.content,
                    message.sources_json, message.language, message.created_at,
                ),
            )
        ConversationsRepo.update(db, message.conversation_id)
        return message

    @staticmethod
    def all_for_conversation(
        db: Database, conversation_id: str, limit: int | None = None
    ) -> list[Message]:
        sql = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid"
        params: list[Any] = [conversation_id]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [
            Message.from_row(row)
            for row in db.connect().execute(sql, params).fetchall()
        ]

    @staticmethod
    def recent(db: Database, conversation_id: str, limit: int = 8) -> list[Message]:
        rows = db.connect().execute(
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [Message.from_row(row) for row in reversed(rows)]

    @staticmethod
    def count(db: Database, conversation_id: str) -> int:
        row = db.connect().execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row["n"]) if row else 0


class MemoriesRepo:
    @staticmethod
    def upsert(db: Database, memory: Memory) -> Memory:
        memory.updated_at = utcnow()
        with db.transaction() as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, content, kind, embedding_model, content_hash, embedding,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (id) DO UPDATE SET
                     content = excluded.content, kind = excluded.kind,
                     embedding_model = excluded.embedding_model,
                     content_hash = excluded.content_hash, embedding = excluded.embedding,
                     updated_at = excluded.updated_at""",
                (
                    memory.id, memory.content, memory.kind, memory.embedding_model,
                    memory.content_hash, memory.embedding, memory.created_at,
                    memory.updated_at,
                ),
            )
        return memory

    @staticmethod
    def get(db: Database, memory_id: str) -> Memory | None:
        row = db.connect().execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return Memory.from_row(row) if row else None

    @staticmethod
    def all(db: Database, limit: int = 200) -> list[Memory]:
        rows = db.connect().execute(
            "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Memory.from_row(row) for row in rows]

    @staticmethod
    def delete(db: Database, memory_id: str) -> bool:
        with db.transaction() as conn:
            return bool(conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            ).rowcount)
