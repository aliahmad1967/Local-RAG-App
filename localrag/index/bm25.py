"""FTS5 BM25 helpers. Query text goes through the very same Arabic
normalization used when building the index, so `المدرسة` matches `المدرسَة`.
"""

from __future__ import annotations

from typing import Any

from ..db.database import Database
from ..text.arabic import normalize_for_bm25


def build_match_query(text: str, cfg: Any | None = None) -> str:
    """Turn a user query into an FTS5 MATCH expression of quoted terms."""
    strip = True
    alef = True
    ya = True
    if cfg is not None:
        strip = bool(cfg.get("text", "bm25_strip_tashkeel", default=True))
        alef = bool(cfg.get("text", "bm25_normalize_alef", default=True))
        ya = bool(cfg.get("text", "bm25_normalize_ya_marbuta", default=True))
    normalized = normalize_for_bm25(
        text, strip_diacritics=strip, fold_alef=alef, fold_ya_marbuta=ya
    )
    terms = [term for term in normalized.split() if term]
    return " OR ".join(f'"{term}"' for term in terms)


def search(
    db: Database,
    query: str,
    limit: int = 50,
    book_id: str | None = None,
    cfg: Any | None = None,
    book_ids: list[str] | None = None,
    page_from: int | None = None,
    page_to: int | None = None,
) -> list[dict[str, Any]]:
    """Return BM25-ranked child chunks: [{chunk_id, book_id, page_no, rank, snippet}]."""
    match = build_match_query(query, cfg)
    if not match:
        return []
    sql = (
        "SELECT c.id AS chunk_id, c.book_id, c.page_start, c.page_end, "
        "bm25(chunks_fts) AS rank, snippet(chunks_fts, 0, '[', ']', '…', 24) AS snippet "
        "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? AND c.level = 'child'"
    )
    params: list[Any] = [match]
    if book_id:
        sql += " AND c.book_id = ?"
        params.append(book_id)
    scope = list(book_ids or [])
    if book_id and book_id not in scope:
        scope.append(book_id)
    if scope:
        marks = ", ".join("?" for _ in scope)
        sql += f" AND c.book_id IN ({marks})"
        params.extend(scope)
    if page_from is not None:
        sql += " AND c.page_end >= ?"
        params.append(page_from)
    if page_to is not None:
        sql += " AND c.page_start <= ?"
        params.append(page_to)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    rows = db.connect().execute(sql, params).fetchall()
    return [
        {
            "chunk_id": row["chunk_id"],
            "book_id": row["book_id"],
            "page_no": row["page_start"],
            "rank": float(row["rank"]),
            "snippet": row["snippet"],
        }
        for row in rows
    ]


def count(db: Database, book_id: str | None = None) -> int:
    """Number of indexed child chunks (optionally for one book).

    FTS5 external-content tables do not store a row count; `COUNT(*) FROM
    chunks_fts` would scan the content table and include parent windows too.
    Only child chunks are indexed (see the insert/update/delete triggers).
    """
    sql = "SELECT COUNT(*) AS n FROM chunks WHERE level = 'child'"
    params: list[Any] = []
    if book_id:
        sql += " AND book_id = ?"
        params.append(book_id)
    row = db.connect().execute(sql, params).fetchone()
    return int(row["n"])
