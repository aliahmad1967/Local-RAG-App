"""Smoke-test the Phase 2 HTTP endpoints without starting background workers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from localrag.api.app import create_app  # noqa: E402
from localrag.config import load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.ingest.service import IngestService  # noqa: E402
from localrag.index.service import IndexService  # noqa: E402
from verification_env import isolate_verification_data  # noqa: E402


def main() -> int:
    isolate_verification_data(ROOT)
    cfg = load_config()
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    app = create_app()
    app.state.cfg = cfg
    app.state.db = db
    app.state.service = IngestService(cfg, db)
    app.state.index_service = IndexService(cfg, db)
    client = TestClient(app)

    book_id = "800c4c78-dcc5-468f-94f2-ff657d66e153"
    checks = [
        ("GET", "/api/index/stats", None),
        ("GET", f"/api/books/{book_id}/index-status", None),
        ("GET", f"/api/books/{book_id}/chunks?limit=2", None),
        ("POST", "/api/index/search", {"query": "الكتاب", "top_k": 3, "book_id": book_id}),
    ]
    ok = True
    for method, url, payload in checks:
        response = client.get(url) if method == "GET" else client.post(url, json=payload)
        body = json.dumps(response.json(), ensure_ascii=False)
        print(f"{method} {url} -> {response.status_code} {body[:240]}", flush=True)
        ok = ok and response.status_code == 200

    chunks = client.get(f"/api/books/{book_id}/chunks?limit=1").json()
    if chunks:
        detail = client.get(f"/api/books/{book_id}/chunks/{chunks[0]['id']}")
        data = detail.json()
        print(
            f"GET chunk detail -> {detail.status_code} "
            f"has_vector={data.get('has_vector')} parent={bool(data.get('parent'))}",
            flush=True,
        )
        ok = ok and detail.status_code == 200
    print("PASS - Phase 2 API smoke test" if ok else "FAIL - API smoke test", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
