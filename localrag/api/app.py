"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from .. import __version__
from ..config import ensure_data_dirs, load_config
from ..db.database import Database
from ..ingest.service import IngestService
from ..index.service import IndexService
from ..index.worker import IndexWorker
from ..llm.generate import GroundedGenerator
from ..logging_conf import setup_logging
from ..memory.service import MemoryService
from ..retrieval.service import RetrievalService
from .routes_chat import router as chat_router
from .routes_index import router as index_router
from .routes_retrieval import router as retrieval_router
from .routes_ingest import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = load_config()
    ensure_data_dirs(cfg)
    log = setup_logging(cfg)
    db = Database(cfg.path("paths", "db_path", default="data/localrag.db"))
    service = IngestService(cfg, db)
    index_service = IndexService(cfg, db)
    app.state.cfg = cfg
    app.state.db = db
    app.state.service = service
    app.state.index_service = index_service
    recovered = service.start()
    index_worker = IndexWorker(db, service.queue, service.stop_event, index_service)
    index_worker.start()
    app.state.index_worker = index_worker
    retrieval_service = RetrievalService(cfg, db)
    app.state.retrieval_service = retrieval_service
    memory_service = MemoryService(cfg, db, embedder=retrieval_service.embedder)
    app.state.memory_service = memory_service
    generation_service = GroundedGenerator(cfg, db, retrieval_service, memory_service)
    app.state.generation_service = generation_service
    if recovered:
        log.warning("Recovered jobs after restart: %s", ", ".join(recovered))
    try:
        yield
    finally:
        service.stop()
        index_worker.join(timeout=10)
        generation_service.close()
        memory_service.close()
        retrieval_service.close()
        db.close_current_thread()


class UTF8StaticFiles(StaticFiles):
    """Serve text assets with an explicit UTF-8 charset (Arabic UI strings)."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        media = response.media_type or ""
        if media.startswith("text/") or media in ("application/javascript", "image/svg+xml"):
            response.headers["content-type"] = f"{media}; charset=utf-8"
        return response


def create_app() -> FastAPI:
    app = FastAPI(title="LocalRAG", version=__version__, lifespan=lifespan)
    app.include_router(router)
    app.include_router(index_router)
    app.include_router(retrieval_router)
    app.include_router(chat_router)
    static_dir = Path(__file__).resolve().parents[1] / "ui" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/ui", UTF8StaticFiles(directory=str(static_dir)), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/ui/index.html")

    return app


app = create_app()
