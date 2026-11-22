-- LocalRAG schema. Phase 1: books/pages/jobs. Phase 2: chunks + FTS5 + vectors.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS books (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    file_name    TEXT NOT NULL,
    file_sha256  TEXT,
    page_count   INTEGER NOT NULL DEFAULT 0,
    language     TEXT NOT NULL DEFAULT 'unknown',
    ocr_engine   TEXT,
    status       TEXT NOT NULL DEFAULT 'queued',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id              TEXT PRIMARY KEY,
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page_no         INTEGER NOT NULL,
    source          TEXT NOT NULL DEFAULT 'pending',
    ocr_engine      TEXT,
    text_original   TEXT NOT NULL DEFAULT '',
    text_clean      TEXT NOT NULL DEFAULT '',
    text_normalized TEXT NOT NULL DEFAULT '',
    confidence      REAL,
    status          TEXT NOT NULL DEFAULT 'pending',
    error           TEXT,
    raster_path     TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    char_count      INTEGER NOT NULL DEFAULT 0,
    duration_s      REAL NOT NULL DEFAULT 0.0,
    updated_at      TEXT NOT NULL,
    UNIQUE (book_id, page_no)
);

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    book_id      TEXT REFERENCES books(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL DEFAULT 'ingest',
    status       TEXT NOT NULL DEFAULT 'queued',
    current_page INTEGER NOT NULL DEFAULT 0,
    total_pages  INTEGER NOT NULL DEFAULT 0,
    processed    INTEGER NOT NULL DEFAULT 0,
    failed       INTEGER NOT NULL DEFAULT 0,
    params_json  TEXT NOT NULL DEFAULT '{}',
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS job_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts      TEXT NOT NULL,
    level   TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ocr_benchmarks (
    id           TEXT PRIMARY KEY,
    book_id      TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page_numbers TEXT NOT NULL,
    results_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Phase 2: chunking + indexing
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    level           TEXT NOT NULL DEFAULT 'child',      -- 'child' | 'parent'
    parent_id       TEXT REFERENCES chunks(id) ON DELETE CASCADE,
    child_index     INTEGER NOT NULL DEFAULT 0,
    page_start      INTEGER NOT NULL,
    page_end        INTEGER NOT NULL,
    heading         TEXT,
    text            TEXT NOT NULL,
    text_normalized TEXT NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0,
    char_count      INTEGER NOT NULL DEFAULT 0,
    start_char      INTEGER NOT NULL DEFAULT 0,
    end_char        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    UNIQUE (book_id, level, page_start, child_index)
);

-- BM25 index. External-content FTS5 keeps one source of truth (chunks).
-- Only child chunks are indexed (see triggers) — they are the search units;
-- parent windows are returned to the LLM after retrieval.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text_normalized,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks
WHEN new.level = 'child' BEGIN
    INSERT INTO chunks_fts(rowid, text_normalized) VALUES (new.rowid, new.text_normalized);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks
WHEN old.level = 'child' BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text_normalized)
    VALUES ('delete', old.rowid, old.text_normalized);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text_normalized ON chunks
WHEN new.level = 'child' BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text_normalized)
    VALUES ('delete', old.rowid, old.text_normalized);
    INSERT INTO chunks_fts(rowid, text_normalized) VALUES (new.rowid, new.text_normalized);
END;

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id     TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL
);

-- Content-addressed cache: the same text is embedded once across all books.
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS book_index_status (
    book_id        TEXT PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,
    status         TEXT NOT NULL DEFAULT 'not_indexed',
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    embedded_count INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pages_book_status ON pages(book_id, status);
CREATE INDEX IF NOT EXISTS idx_pages_book_page   ON pages(book_id, page_no);
CREATE INDEX IF NOT EXISTS idx_jobs_status       ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_events_job        ON job_events(job_id, id);
CREATE INDEX IF NOT EXISTS idx_chunks_book_level ON chunks(book_id, level);
CREATE INDEX IF NOT EXISTS idx_chunks_parent     ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page       ON chunks(book_id, page_start, page_end);
CREATE INDEX IF NOT EXISTS idx_embeddings_model  ON chunk_embeddings(model);

-- ---------------------------------------------------------------------------
-- Phase 4: conversations, grounded answers and confirmed long-term memory
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT 'New conversation',
    book_ids_json TEXT NOT NULL DEFAULT '[]',
    summary      TEXT NOT NULL DEFAULT '',
    summary_message_count INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role         TEXT NOT NULL,                 -- 'user' | 'assistant'
    content      TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',
    language     TEXT NOT NULL DEFAULT 'auto',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'fact', -- 'fact' | 'preference'
    embedding_model TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    embedding    BLOB,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
