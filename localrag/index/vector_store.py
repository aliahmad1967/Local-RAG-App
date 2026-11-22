"""Vector storage and cosine search.

sqlite-vec has no Python 3.14 wheel on Windows, and LanceDB would add a large
native dependency. We store float32 vectors as SQLite BLOBs in the *same* db
file and use NumPy for exact cosine similarity — fast enough for tens of
thousands of chunks and completely offline.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from ..db.database import Database


def pack_vector(vector: list[float] | np.ndarray) -> bytes:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("Vector must be one-dimensional")
    return array.tobytes()


def unpack_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_scores(query: list[float] | np.ndarray, vectors: list[bytes]) -> np.ndarray:
    """Cosine similarity of one query against a list of packed vectors."""
    if not vectors:
        return np.empty(0, dtype=np.float32)
    query_array = np.asarray(query, dtype=np.float32)
    matrix = np.stack([unpack_vector(blob) for blob in vectors])
    norm = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_array)
    norm[norm == 0] = 1.0
    return (matrix @ query_array) / norm


class VectorStore:
    @staticmethod
    def count(db: Database, book_id: str | None = None) -> int:
        if book_id:
            sql = (
                "SELECT COUNT(*) AS n FROM chunk_embeddings e "
                "JOIN chunks c ON c.id = e.chunk_id WHERE c.book_id = ?"
            )
            row = db.connect().execute(sql, (book_id,)).fetchone()
        else:
            row = db.connect().execute(
                "SELECT COUNT(*) AS n FROM chunk_embeddings"
            ).fetchone()
        return int(row["n"])

    @staticmethod
    def all_vectors(
        db: Database,
        book_ids: list[str] | None = None,
        model: str | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> list[tuple[str, bytes]]:
        """All child vectors in scope, in a stable order for dense search."""
        sql = (
            "SELECT e.chunk_id, e.vector FROM chunk_embeddings e "
            "JOIN chunks c ON c.id = e.chunk_id WHERE c.level = 'child'"
        )
        params: list[object] = []
        if book_ids:
            marks = ", ".join("?" for _ in book_ids)
            sql += f" AND c.book_id IN ({marks})"
            params.extend(book_ids)
        if model:
            sql += " AND e.model = ?"
            params.append(model)
        if page_from is not None:
            sql += " AND c.page_end >= ?"
            params.append(page_from)
        if page_to is not None:
            sql += " AND c.page_start <= ?"
            params.append(page_to)
        sql += " ORDER BY c.book_id, c.page_start, c.child_index"
        rows: list[sqlite3.Row] = db.connect().execute(sql, params).fetchall()
        return [(row["chunk_id"], bytes(row["vector"])) for row in rows]

    @staticmethod
    def all_for_book(db: Database, book_id: str, model: str | None = None) -> list[tuple[str, bytes]]:
        return VectorStore.all_vectors(db, book_ids=[book_id], model=model)
