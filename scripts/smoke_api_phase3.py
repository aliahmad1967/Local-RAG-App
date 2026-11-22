"""Fast API/UI smoke test for Phase 3 (skips the cross-encoder)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOCALRAG_EMBED_CPU", "1")

from fastapi.testclient import TestClient  # noqa: E402

from localrag.api.app import create_app  # noqa: E402
from localrag.config import load_config  # noqa: E402
from localrag.db.database import Database  # noqa: E402
from localrag.ingest.service import IngestService  # noqa: E402
from localrag.index.service import IndexService  # noqa: E402
from localrag.retrieval.query_rewrite import NoopRewriter  # noqa: E402
from localrag.retrieval.service import RetrievalService  # noqa: E402
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
    app.state.retrieval_service = RetrievalService(
        cfg, db, reranker=None, rewriter=NoopRewriter()
    )
    client = TestClient(app)

    response = client.post(
        "/api/retrieve",
        json={"query": "الكتاب والمكتبة", "top_k": 3},
    )
    payload = response.json()
    print(f"POST /api/retrieve -> {response.status_code} hits={len(payload.get('hits', []))}")
    print(f"  models={payload.get('models')} counts={payload.get('counts')}")
    for hit in payload.get("hits", [])[:3]:
        print(
            f"  dense={hit.get('dense_rank')} bm25={hit.get('bm25_rank')} "
            f"fused={hit.get('fused_rank')} rerank={hit.get('rerank_score')} "
            f"p.{hit.get('page_start')} {hit.get('book_title')}"
        )

    html = client.get("/ui/index.html")
    panel = "tab-search" in html.text and "searchQuery" in html.text
    print(f"GET /ui/index.html -> {html.status_code} debug_panel={panel}")
    app.state.retrieval_service.close()
    ok = response.status_code == 200 and bool(payload.get("hits")) and html.status_code == 200 and panel
    print("PASS - Phase 3 API/UI smoke test" if ok else "FAIL - Phase 3 API/UI smoke test")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
