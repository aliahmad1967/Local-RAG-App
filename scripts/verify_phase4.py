"""Real end-to-end verification for Phase 4: grounded generation and memory.

Runs pytest, then exercises the real Ollama LLM, bge-m3 embeddings and the
cross-encoder on the Phase 2 sample books. Embeddings and reranking stay on CPU;
the LLM uses GPU only when no OCR job is running (set LOCALRAG_LLM_CPU=1 to
force CPU). No cloud service is contacted.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localrag.config import ensure_data_dirs, load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.db.repositories import BooksRepo, IndexStatusRepo, JobsRepo  # noqa: E402
from localrag.index.service import IndexService  # noqa: E402
from localrag.llm.generate import (  # noqa: E402
    REFUSAL,
    GroundedGenerator,
    detect_answer_language,
)
from localrag.logging_conf import setup_logging  # noqa: E402
from localrag.memory.service import MemoryService  # noqa: E402
from localrag.ocr.vram import ollama_status  # noqa: E402
from localrag.retrieval.query_rewrite import NoopRewriter  # noqa: E402
from localrag.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from localrag.retrieval.service import RetrievalFilters, RetrievalService  # noqa: E402
from verification_env import isolate_verification_data  # noqa: E402

CITATION = re.compile(r"\[[^\]\n]+,\s*p\.\d+(?:-\d+)?\]")
SOURCE_CITATION = re.compile(r"\[([^\]\n]+?,\s*p\.\d+(?:-\d+)?)\]")


def unresolved_citations(answer: str, sources: list[dict[str, Any]]) -> list[str]:
    labels = {source["label"] for source in sources}
    return [label for label in SOURCE_CITATION.findall(answer or "") if label not in labels]


def heading(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT), text=True, encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def find_book(db: Database, file_name: str) -> Any:
    return next((book for book in BooksRepo.all(db) if book.file_name == file_name), None)


def short(text: str, limit: int = 220) -> str:
    return " ".join(text.split())[:limit]


def has_model(names: list[str], wanted: str) -> bool:
    prefix = wanted.split(":")[0]
    return any(name.split(":")[0] == prefix for name in names)


def preview(done: dict[str, Any]) -> None:
    print(f"  language={done.get('language')} mode={done.get('mode')} "
          f"counts={done.get('counts')} models={done.get('models')}", flush=True)
    print(f"  answer: {short(done.get('answer', ''))}", flush=True)
    for source in done.get("sources", [])[:3]:
        print(f"    source: {source['label']} :: {short(source['text'], 120)}", flush=True)


def main() -> int:
    isolate_verification_data(ROOT)
    os.environ["LOCALRAG_EMBED_CPU"] = "1"  # reranker is CPU by config; GPU reserved for LLM/OCR
    cfg = load_config()
    ensure_data_dirs(cfg)
    setup_logging(cfg)
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    failures: list[str] = []

    heading("1. ENVIRONMENT")
    status = ollama_status(cfg)
    model_names = [str(name) for name in status.get("models", [])]
    embed_model = str(cfg.get("ollama", "models", "embedding", default="bge-m3"))
    default_llm = str(cfg.get("ollama", "models", "llm", default="qwen3.5:latest"))
    running_jobs = [job for job in JobsRepo.all(db, 50) if job.status == "running"]
    if running_jobs:
        # OCR owns the GPU: force the LLM to CPU so only one large model is active.
        os.environ["LOCALRAG_LLM_CPU"] = "1"
    llm = default_llm
    cfg.data["generation"]["model"] = llm
    print(f"python          : {sys.version.split()[0]}")
    print(f"ollama          : {'running' if status['running'] else 'DOWN'} @ {status['host']}")
    print(f"models          : {', '.join(model_names)}")
    print(f"embedding       : {embed_model} installed={has_model(model_names, embed_model)} CPU")
    print(f"default LLM     : {default_llm} installed={has_model(model_names, default_llm)}")
    print(f"verification LLM: {llm} device={'CPU (OCR active)' if running_jobs else 'GPU'}")
    print(f"running jobs    : {[job.id for job in running_jobs]}")
    if not status["running"]:
        print("Ollama is not running", flush=True)
        return 1
    if not has_model(model_names, embed_model) or not has_model(model_names, llm):
        print("Required Ollama models are missing", flush=True)
        return 1

    heading("2. UNIT TESTS")
    tests = run_command(["-m", "pytest", "-q"])
    print(tests.stdout.strip(), flush=True)
    if tests.returncode != 0:
        print("pytest FAILED", flush=True)
        return 1

    heading("3. ENSURE SAMPLE BOOKS ARE INDEXED")
    books: dict[str, Any] = {}
    for key, file_name in (
        ("ar", "sample_ar_scanned_20.pdf"),
        ("en", "sample_en_scanned_20.pdf"),
    ):
        book = find_book(db, file_name)
        if book is None:
            print(f"missing sample {file_name} — run verify_phase1.py first", flush=True)
            return 1
        state = IndexStatusRepo.get(db, book.id)
        if state is None or state.status != "indexed":
            print(f"indexing {file_name} ...", flush=True)
            print(f"  {IndexService(cfg, db).index_book(book.id)}", flush=True)
        state = IndexStatusRepo.get(db, book.id)
        print(f"  {key}: {book.title} chunks={state.chunk_count} "
              f"embedded={state.embedded_count} status={state.status}", flush=True)
        books[key] = book

    heading("4. SERVICES + CONFIRMED MEMORY")
    retrieval = RetrievalService(
        cfg, db, reranker=CrossEncoderReranker(cfg), rewriter=NoopRewriter()
    )
    memory = MemoryService(cfg, db, embedder=retrieval.embedder)
    generator = GroundedGenerator(cfg, db, retrieval, memory=memory)
    saved = memory.add("المكتبة تحتوي على كتب قديمة", kind="fact")
    print(f"  memory added: {saved['id']} kind={saved['kind']}", flush=True)
    relevant = memory.relevant("الكتاب والمكتبة")
    print(f"  relevant memories: {[item['content'] for item in relevant]}", flush=True)
    if not relevant:
        failures.append("confirmed memory was not retrieved")

    heading("5. REAL GROUNDED ANSWER — ARABIC")
    ar_done = generator.answer(
        "ماذا تحتوي المكتبة؟",
        filters=RetrievalFilters(book_ids=[books["ar"].id]),
        language="ar",
    )
    preview(ar_done)
    assert_lang = detect_answer_language(ar_done.get("answer", ""))
    print(f"  detected answer language: {assert_lang}", flush=True)
    if not CITATION.search(ar_done.get("answer", "")):
        failures.append("Arabic answer has no [Book, p.N] citation")
    unresolved = unresolved_citations(ar_done.get("answer", ""), ar_done.get("sources", []))
    if unresolved:
        failures.append(f"Arabic answer has unresolved citations: {unresolved[:3]}")
    if assert_lang != "ar":
        failures.append("Arabic question did not produce an Arabic answer")
    if not ar_done.get("memories_used"):
        failures.append("confirmed memory was not injected into the answer")

    heading("6. REAL GROUNDED ANSWER — ENGLISH")
    en_done = generator.answer(
        "What does each page carry?",
        filters=RetrievalFilters(book_ids=[books["en"].id]),
        language="en",
    )
    preview(en_done)
    if not CITATION.search(en_done.get("answer", "")):
        failures.append("English answer has no [Book, p.N] citation")
    if detect_answer_language(en_done.get("answer", "")) != "en":
        failures.append("English question did not produce an English answer")
    unresolved = unresolved_citations(en_done.get("answer", ""), en_done.get("sources", []))
    if unresolved:
        failures.append(f"English answer has unresolved citations: {unresolved[:3]}")

    heading("7. STRICT GROUNDING — UNSUPPORTED QUESTION")
    refusal = generator.answer(
        "ما عاصمة اليابان؟",
        filters=RetrievalFilters(book_ids=[books["ar"].id]),
        language="ar",
    )
    print(f"  answer: {short(refusal.get('answer', ''))}", flush=True)
    if REFUSAL["ar"] not in refusal.get("answer", ""):
        failures.append("unsupported question did not use the exact refusal")

    heading("8. SUMMARIZE + COMPARE MODES")
    summary_events = list(generator.stream_summary(
        "Summarize these pages", book_ids=[books["en"].id],
        page_from=1, page_to=2, language="en",
    ))
    summary = next(event for event in summary_events if event["type"] == "done")
    preview(summary)
    if summary.get("counts", {}).get("chunks", 0) < 1:
        failures.append("summarize mode found no chunks")

    compare = generator.answer(
        "Which labeled book mentions Arabic manuscripts, and which labeled book mentions paper and ink?",
        filters=RetrievalFilters(book_ids=[books["ar"].id, books["en"].id]),
        mode="compare", language="en",
    )
    preview(compare)
    compare_book_ids = {source["book_id"] for source in compare.get("sources", [])}
    if not {books["ar"].id, books["en"].id}.issubset(compare_book_ids):
        failures.append("compare mode did not retrieve sources from both books")
    compare_answer = compare.get("answer", "")
    if compare_answer != REFUSAL["en"] and not CITATION.search(compare_answer):
        failures.append("compare answer has no citation")
    if detect_answer_language(compare_answer) != "en":
        failures.append("compare answer did not use the question's language")
    unresolved = unresolved_citations(compare_answer, compare.get("sources", []))
    if unresolved:
        failures.append(f"compare answer has unresolved citations: {unresolved[:3]}")

    heading("9. HTTP API + UI")
    from fastapi.testclient import TestClient

    from localrag.api.app import create_app
    from localrag.ingest.service import IngestService
    from localrag.index.service import IndexService as APIIndexService

    app = create_app()
    app.state.cfg = cfg
    app.state.db = db
    app.state.service = IngestService(cfg, db)
    app.state.index_service = APIIndexService(cfg, db)
    app.state.retrieval_service = retrieval
    app.state.memory_service = memory
    app.state.generation_service = generator
    client = TestClient(app)

    response = client.post("/api/chat", json={
        "query": "ماذا تحتوي المكتبة؟", "book_ids": [books["ar"].id], "language": "ar",
    })
    body = response.text
    events = [
        json.loads(line[5:].strip())
        for line in body.splitlines() if line.startswith("data:")
    ]
    done = next((event for event in events if event["type"] == "done"), {})
    print(f"POST /api/chat -> {response.status_code} events={[e['type'] for e in events]}", flush=True)
    print(f"  conversation_id={done.get('conversation_id')} answer={short(done.get('answer', ''), 120)}",
          flush=True)
    if response.status_code != 200 or not done.get("conversation_id"):
        failures.append("POST /api/chat failed")
    else:
        api_answer = done.get("answer", "")
        if api_answer == REFUSAL["ar"] or not CITATION.search(api_answer):
            failures.append("HTTP chat refused or omitted a citation despite indexed sources")
        unresolved = unresolved_citations(api_answer, done.get("sources", []))
        if unresolved:
            failures.append(f"HTTP chat has unresolved citations: {unresolved[:3]}")
        messages = client.get(f"/api/conversations/{done['conversation_id']}/messages").json()
        print(f"  persisted messages={[message['role'] for message in messages]}", flush=True)
        if len(messages) != 2:
            failures.append("chat messages were not persisted")

    models = client.get("/api/models")
    print(f"GET /api/models -> {models.status_code} current={models.json().get('current')}",
          flush=True)
    if models.status_code != 200:
        failures.append("GET /api/models failed")

    created = client.post("/api/memories", json={
        "content": "The user reads Arabic books", "kind": "preference",
    })
    memory_id = created.json().get("id")
    print(f"POST /api/memories -> {created.status_code} id={memory_id}", flush=True)
    if created.status_code != 201 or not memory_id:
        failures.append("POST /api/memories failed")
    else:
        patched = client.patch(f"/api/memories/{memory_id}", json={"content": "User reads Arabic and English"})
        deleted = client.delete(f"/api/memories/{memory_id}")
        print(f"PATCH /api/memories/{memory_id} -> {patched.status_code}; "
              f"DELETE -> {deleted.status_code}", flush=True)
        if patched.status_code != 200 or deleted.status_code != 200:
            failures.append("memory update/delete failed")
    if not memory.delete(saved["id"]):
        failures.append("could not clean verification memory")

    # Settings are persisted into a scratch file so verification never edits
    # the user's config.local.yaml.
    original_path = cfg.local_path
    original_generation = copy.deepcopy(cfg.data.get("generation", {}))
    scratch = ROOT / "data" / "phase4_settings_check.yaml"
    cfg.local_path = scratch
    try:
        saved_settings = client.put("/api/config/generation", json={
            "model": llm, "temperature": 0.3, "context_tokens": 4096, "history_turns": 4,
        })
        current_settings = client.get("/api/config/generation").json()
        print(f"PUT /api/config/generation -> {saved_settings.status_code} "
              f"model={current_settings.get('model')} ctx={current_settings.get('context_tokens')}",
              flush=True)
        if saved_settings.status_code != 200 or current_settings.get("context_tokens") != 4096:
            failures.append("generation settings endpoint failed")
    finally:
        cfg.local_path = original_path
        cfg.data["generation"] = original_generation
        scratch.unlink(missing_ok=True)

    html = client.get("/ui/index.html")
    ui_ok = all(token in html.text for token in (
        "tab-chat", "chatQuery", "citationPanel", "memoryList", "modelSelect"
    ))
    print(f"GET /ui/index.html -> {html.status_code} phase4_ui={ui_ok}", flush=True)
    if html.status_code != 200 or not ui_ok:
        failures.append("Phase 4 UI controls are missing")

    retrieval.close()

    heading("10. RESULT")
    if failures:
        for failure in failures:
            print(f"FAIL - {failure}", flush=True)
        return 1
    print("PASS - Phase 4 verified end to end", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
