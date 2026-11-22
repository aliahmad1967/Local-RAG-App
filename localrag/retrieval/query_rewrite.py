"""Optional LLM query rewrite using the recent conversation."""

from __future__ import annotations

import re
from typing import Protocol

from ..config import Config
from ..llm.ollama import OllamaChat
from ..logging_conf import get_logger

log = get_logger("retrieval.rewrite")

SYSTEM = (
    "You rewrite a user's last question into a short standalone search query for a "
    "book index. Resolve pronouns and references using the conversation. Keep names "
    "and technical terms. Answer with the rewritten query only — no explanation, "
    "no quotes, no markdown."
)


class Rewriter(Protocol):
    def rewrite(self, query: str, history: list[dict[str, str]] | None = None) -> str: ...


class NoopRewriter:
    def rewrite(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        return query


class LLMQueryRewriter:
    def __init__(self, cfg: Config, model: str | None = None) -> None:
        self.cfg = cfg
        configured = cfg.get("retrieval", "query_rewrite_model", default=None)
        self.chat = OllamaChat(cfg, model=model or configured, num_predict=96)

    def rewrite(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        recent = (history or [])[-6:]
        if not recent:
            return query
        transcript = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in recent
        )
        prompt = f"CONVERSATION:\n{transcript}\n\nLAST QUESTION: {query}\n\nSTANDALONE QUERY:"
        try:
            answer = self.chat.complete(prompt, system=SYSTEM)
        except Exception as exc:  # noqa: BLE001 - rewrite is best-effort
            log.warning("Query rewrite failed, using the original query: %s", exc)
            return query
        answer = re.sub(r"^[\"'\s]+|[\"'\s]+$", "", answer).strip()
        return answer or query


def build_rewriter(cfg: Config, enabled: bool | None = None) -> Rewriter:
    if enabled is False:
        return NoopRewriter()
    configured = bool(cfg.get("retrieval", "query_rewrite", default=False))
    if enabled is None and not configured:
        return NoopRewriter()
    return LLMQueryRewriter(cfg)
