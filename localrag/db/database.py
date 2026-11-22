"""SQLite connection management.

One connection per thread (thread-local), WAL mode, and short explicit
transactions so every committed page is a durable checkpoint.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """Thread-local SQLite connections around a single database file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,  # explicit transactions only
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 30000")
            self._local.conn = conn
        return conn

    def initialize(self) -> None:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            # CREATE TABLE IF NOT EXISTS does not add new columns to existing DBs.
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(conversations)")
            }
            if "summary_message_count" not in columns:
                conn.execute(
                    "ALTER TABLE conversations "
                    "ADD COLUMN summary_message_count INTEGER NOT NULL DEFAULT 0"
                )
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE / COMMIT, with rollback on any exception."""
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close_current_thread(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
