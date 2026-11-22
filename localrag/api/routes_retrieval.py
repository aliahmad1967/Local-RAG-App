"""Phase 3 HTTP endpoint: hybrid retrieval with a rank-debug payload."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..retrieval.service import RetrievalFilters, RetrievalService
from .schemas import RetrieveRequest

router = APIRouter(prefix="/api")


def _retrieval_service(request: Request) -> RetrievalService:
    """Use the lifespan-owned service, or create one lazily for lightweight clients."""
    service: RetrievalService | None = getattr(request.app.state, "retrieval_service", None)
    if service is None:
        service = RetrievalService(request.app.state.cfg, request.app.state.db)
        request.app.state.retrieval_service = service
    return service


@router.post("/retrieve")
def retrieve(body: RetrieveRequest, request: Request) -> dict[str, Any]:
    if body.page_from and body.page_to and body.page_from > body.page_to:
        raise HTTPException(status_code=400, detail="page_from must be <= page_to")
    filters = RetrievalFilters(
        book_ids=body.book_ids or None,
        page_from=body.page_from,
        page_to=body.page_to,
    )
    try:
        result = _retrieval_service(request).retrieve(
            body.query,
            filters=filters,
            history=body.history,
            top_k=body.top_k,
            explain=body.explain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - missing local models must show a UI error
        # Missing local models / Ollama failures should be actionable in the UI.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = result.to_dict()
    service = _retrieval_service(request)
    reranker = service.reranker
    payload["models"] = {
        "embedding": service.embedder.model,
        "reranker": getattr(reranker, "name", None) or getattr(reranker, "model_name", None),
    }
    return payload
