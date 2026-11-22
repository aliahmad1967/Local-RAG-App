"""Phase 4 API: grounded chat (SSE), conversations, memory and settings."""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..db.repositories import ConversationsRepo, MessagesRepo
from ..llm.generate import GroundedGenerator
from ..llm.ollama import list_ollama_models
from ..memory.service import MemoryService
from ..models.domain import Conversation, Message, utcnow
from ..retrieval.service import RetrievalFilters
from .schemas import (
    ChatRequest,
    ConversationCreate,
    ConversationUpdate,
    GenerationSettings,
    MemoryCreate,
    MemoryUpdate,
)

router = APIRouter(prefix="/api")


def _services(request: Request) -> tuple[Any, Any, GroundedGenerator, MemoryService]:
    state = request.app.state
    return state.cfg, state.db, state.generation_service, state.memory_service


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _get_or_create_conversation(request: Request, body: ChatRequest) -> Conversation:
    _, db, _, _ = _services(request)
    conversation = (
        ConversationsRepo.get(db, body.conversation_id) if body.conversation_id else None
    )
    if body.conversation_id and conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation is None:
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=" ".join(body.query.split())[:48] or "New conversation",
            book_ids_json=json.dumps(body.book_ids or []),
        )
        ConversationsRepo.create(db, conversation)
    elif body.book_ids is not None:
        ConversationsRepo.update(
            db, conversation.id, book_ids_json=json.dumps(body.book_ids)
        )
        conversation.book_ids_json = json.dumps(body.book_ids)
    return conversation


@router.post("/chat")
def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    cfg, db, generator, memory_service = _services(request)
    conversation = _get_or_create_conversation(request, body)
    book_ids = body.book_ids if body.book_ids is not None else conversation.book_ids
    filters = RetrievalFilters(
        book_ids=book_ids or None,
        page_from=body.page_from,
        page_to=body.page_to,
    )
    language = None if body.language == "auto" else body.language
    history = generator.build_history(conversation)
    saved_memory: dict[str, Any] | None = None
    if body.memory_content and body.memory_content.strip():
        try:
            saved_memory = memory_service.add(body.memory_content.strip(), body.memory_kind)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    MessagesRepo.add(db, Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation.id,
        role="user",
        content=body.query,
        created_at=utcnow(),
    ))

    if body.mode == "summarize":
        events: Iterator[dict[str, Any]] = generator.stream_summary(
            body.query, book_ids=book_ids, page_from=body.page_from,
            page_to=body.page_to, language=language,
        )
    else:
        events = generator.stream_answer(
            body.query,
            filters=filters,
            history=history,
            conversation=conversation,
            language=language,
            mode=body.mode,
            top_k=body.top_k,
        )

    def stream() -> Iterator[str]:
        answer_pieces: list[str] = []
        final: dict[str, Any] = {}
        if saved_memory:
            yield _sse({"type": "memory_saved", "memory": saved_memory})
        try:
            for event in events:
                if event.get("type") == "delta":
                    answer_pieces.append(str(event.get("text", "")))
                if event.get("type") == "done":
                    final = event
                    event = {**event, "conversation_id": conversation.id}
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - stream errors must reach the UI
            yield _sse({"type": "error", "message": str(exc), "conversation_id": conversation.id})
        finally:
            answer = "".join(answer_pieces).strip()
            if answer:
                MessagesRepo.add(db, Message(
                    id=str(uuid.uuid4()),
                    conversation_id=conversation.id,
                    role="assistant",
                    content=answer,
                    sources_json=json.dumps(final.get("sources", []), ensure_ascii=False),
                    language=str(final.get("language", "auto")),
                    created_at=utcnow(),
                ))
                generator.maybe_summarize(conversation.id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations")
def list_conversations(request: Request) -> list[dict[str, Any]]:
    _, db, _, _ = _services(request)
    return [item.to_dict() for item in ConversationsRepo.all(db)]


@router.post("/conversations", status_code=201)
def create_conversation(body: ConversationCreate, request: Request) -> dict[str, Any]:
    _, db, _, _ = _services(request)
    conversation = Conversation(
        id=str(uuid.uuid4()),
        title=(body.title or "New conversation").strip() or "New conversation",
        book_ids_json=json.dumps(body.book_ids),
    )
    return ConversationsRepo.create(db, conversation).to_dict()


@router.get("/conversations/{conversation_id}/messages")
def list_messages(conversation_id: str, request: Request) -> list[dict[str, Any]]:
    _, db, _, _ = _services(request)
    if ConversationsRepo.get(db, conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [item.to_dict() for item in MessagesRepo.all_for_conversation(db, conversation_id)]


@router.patch("/conversations/{conversation_id}")
def update_conversation(
    conversation_id: str, body: ConversationUpdate, request: Request
) -> dict[str, Any]:
    _, db, _, _ = _services(request)
    if ConversationsRepo.get(db, conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    updates: dict[str, Any] = {}
    if body.title is not None:
        updates["title"] = body.title.strip() or "New conversation"
    if body.book_ids is not None:
        updates["book_ids_json"] = json.dumps(body.book_ids)
    ConversationsRepo.update(db, conversation_id, **updates)
    return ConversationsRepo.get(db, conversation_id).to_dict()


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    _, db, _, _ = _services(request)
    ok = ConversationsRepo.delete(db, conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@router.get("/memories")
def list_memories(request: Request) -> list[dict[str, Any]]:
    _, _, _, memory_service = _services(request)
    return memory_service.all()


@router.post("/memories", status_code=201)
def create_memory(body: MemoryCreate, request: Request) -> dict[str, Any]:
    _, _, _, memory_service = _services(request)
    try:
        return memory_service.add(body.content, body.kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/memories/{memory_id}")
def update_memory(
    memory_id: str, body: MemoryUpdate, request: Request
) -> dict[str, Any]:
    _, _, _, memory_service = _services(request)
    try:
        return memory_service.update(memory_id, body.content, body.kind)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Memory not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, request: Request) -> dict[str, Any]:
    _, _, _, memory_service = _services(request)
    if not memory_service.delete(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"ok": True}


@router.get("/models")
def models(request: Request) -> dict[str, Any]:
    cfg, _, generator, _ = _services(request)
    try:
        installed = list_ollama_models(cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"current": generator.model, "installed": installed}


@router.get("/config/generation")
def get_generation_config(request: Request) -> dict[str, Any]:
    cfg, _, generator, _ = _services(request)
    section = dict(cfg.get("generation") or {})
    section["model"] = generator.model
    return section


@router.put("/config/generation")
def update_generation_config(body: GenerationSettings, request: Request) -> dict[str, Any]:
    cfg, _, generator, _ = _services(request)
    updates = {
        key: value for key, value in body.model_dump(exclude_none=True).items()
    }
    if updates:
        cfg.save_local({"generation": updates})
        if any(key in updates for key in ("model", "temperature", "num_predict")):
            generator._chat = None  # rebuild with the new model on the next answer
    section = dict(cfg.get("generation") or {})
    section["model"] = updates.get("model") or cfg.get(
        "ollama", "models", "llm", default="qwen3.5:latest"
    )
    return {"ok": True, "generation": section}
