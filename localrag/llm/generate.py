"""Grounded, citation-aware answer generation on top of Phase 3 retrieval.

Answers are streamed as small dictionaries so the API can expose them as SSE.
The model only sees retrieved parent windows and confirmed memories; if there
are no sources it must refuse instead of guessing.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Protocol

from ..config import Config
from ..db.database import Database
from ..db.repositories import BooksRepo, ChunksRepo, ConversationsRepo, MessagesRepo
from ..logging_conf import get_logger
from ..models.domain import Conversation, Message, RetrievalHit
from ..retrieval.service import RetrievalFilters, RetrievalService
from ..text.language import detect_language
from .ollama import OllamaChat

log = get_logger("llm.generate")

LANGUAGE_NAMES = {"ar": "Arabic", "en": "English"}
REFUSAL = {
    "ar": "المصادر المتاحة لا تحتوي على الإجابة عن هذا السؤال.",
    "en": "The available sources do not contain the answer to this question.",
}
NO_SOURCES = {
    "ar": "لا توجد مصادر مفهرسة بعد لهذا النطاق.",
    "en": "There are no indexed sources for this scope yet.",
}

# One bracket must contain exactly one page so the UI can open it as a link.
_SINGLE_CITATION = re.compile(r"^[^\[\]\n]+,\s*p\.\d+(?:-\d+)?$")
_BRACKETED = re.compile(r"\[([^\[\]\n]+)\]")
_CITATION_SEPARATOR = re.compile(r"[;؛]")


def normalize_citations(text: str) -> str:
    """Split ``[Book, p.1; Book, p.2]`` into two clickable page citations."""
    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if not _CITATION_SEPARATOR.search(content):
            return match.group(0)
        parts = [part.strip() for part in _CITATION_SEPARATOR.split(content) if part.strip()]
        if len(parts) < 2 or not all(_SINGLE_CITATION.fullmatch(part) for part in parts):
            return match.group(0)
        return " ".join(f"[{part}]" for part in parts)

    return _BRACKETED.sub(replace, text or "")


def has_citation(text: str) -> bool:
    """True when the text contains at least one clickable ``[Book, p.N]`` label."""
    return any(
        _SINGLE_CITATION.fullmatch(match.group(1).strip())
        for match in _BRACKETED.finditer(text or "")
    )


_REFUSAL_MARKERS = {
    "ar": (
        "المصادر المتاحة", "المصادر", "غير محدد", "غير مذكور", "لا تحتوي",
        "لا توجد إجابة", "لا يمكن", "غير متوفر", "لم يرد",
    ),
    "en": (
        "available sources", "sources do not", "sources don't", "not mentioned",
        "no information", "not available", "cannot be found", "does not contain",
    ),
}


def looks_like_refusal(text: str, language: str) -> bool:
    """Catch small-model paraphrases of the strict no-answer sentence."""
    value = (text or "").strip()
    if value.startswith(REFUSAL.get(language, "")):
        return True
    if not value or len(value) > 300 or has_citation(value):
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS.get(language, ()))


_CITATION_PARTS = re.compile(r"^(.+?),\s*p\.(\d+)(?:-(\d+))?$")


def remap_citations(text: str, sources: list[dict[str, Any]]) -> str:
    """Point every ``[label, p.N]`` at a real retrieved source label."""
    if not sources:
        return text
    valid = {source["label"] for source in sources}

    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        parts = _CITATION_PARTS.fullmatch(content)
        if parts is None or content in valid:
            return match.group(0)
        label, page = parts.group(1).strip(), int(parts.group(2))
        candidates = [
            source for source in sources
            if source["page_start"] <= page <= (
                source.get("page_end") or source["page_start"]
            )
        ]
        if not candidates:
            return match.group(0)
        lowered = label.lower()
        best = next((
            source for source in candidates
            if any(
                len(token) > 3 and (
                    token.lower() in lowered or lowered in token.lower()
                )
                for token in str(source.get("book_title") or "").split()
            )
        ), None) or candidates[0]
        target = str(best["label"])
        return f"[{target}]"

    return _BRACKETED.sub(replace, text or "")


def has_unresolved_citation(text: str, sources: list[dict[str, Any]]) -> bool:
    """Reject citation-like labels that do not resolve to a retrieved source."""
    valid = {source["label"] for source in sources}
    return any(
        re.search(r",\s*p\.", match.group(1), re.IGNORECASE)
        and match.group(1).strip() not in valid
        for match in _BRACKETED.finditer(text or "")
    )


class MemoryLookup(Protocol):
    def relevant(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]: ...


class ChatClient(Protocol):
    model: str

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]: ...

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str: ...

    def unload(self) -> None: ...


def detect_answer_language(text: str) -> str:
    """Detect prose language, ignoring citation brackets (often in source script)."""
    prose = _BRACKETED.sub(" ", text or "")
    detected = detect_language(prose, min_chars=1)
    if detected in LANGUAGE_NAMES:
        return detected
    if detected == "mixed":
        arabic = sum(1 for char in prose if "\u0600" <= char <= "\u06ff")
        latin = sum(1 for char in prose if char.isascii() and char.isalpha())
        return "ar" if arabic >= latin else "en"
    return "en"


class GroundedGenerator:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        retrieval: RetrievalService,
        memory: MemoryLookup | None = None,
        chat: ChatClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.retrieval = retrieval
        self.memory = memory
        self._chat = chat

    def _cfg(self, *keys: str, default: Any = None) -> Any:
        return self.cfg.get("generation", *keys, default=default)

    @property
    def model(self) -> str:
        configured = self._cfg("model", default=None)
        if configured:
            return str(configured)
        return str(self.cfg.get("ollama", "models", "llm", default="qwen3.5:latest"))

    @property
    def chat(self) -> ChatClient:
        if self._chat is None:
            self._chat = OllamaChat(
                self.cfg,
                model=self.model,
                num_predict=int(self._cfg("num_predict", default=1024)),
                temperature=float(self._cfg("temperature", default=0.2)),
                num_ctx=int(self._cfg("context_tokens", default=8192)),
            )
        return self._chat

    # -- public streaming API ----------------------------------------------
    def stream_answer(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        history: list[dict[str, str]] | None = None,
        conversation: Conversation | None = None,
        language: str | None = None,
        mode: str = "chat",
        top_k: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        lang = language if language in LANGUAGE_NAMES else detect_answer_language(query)
        if lang not in LANGUAGE_NAMES:
            lang = "en"
        memories = self._memories(query)
        base_k = int(self.cfg.get("retrieval", "final_top_k", default=8))
        retrieval_k = top_k or (base_k if mode == "chat" else base_k * 2)
        result = self.retrieval.retrieve(query, filters=filters, top_k=retrieval_k)
        sources = self._sources(result.hits)

        if not sources:
            text = NO_SOURCES.get(lang, NO_SOURCES["en"])
            yield {"type": "delta", "text": text}
            yield self._done(
                answer=text, language=lang, sources=[], warnings=[], result=result,
                mode=mode, memories=memories,
            )
            return

        system = self._system(lang, conversation, memories, mode)
        prompt, used_sources, truncated = self._prompt(query, sources, lang, mode)
        warnings: list[str] = []
        approx_tokens = max(1, len(prompt) // 4)
        context_limit = int(self._cfg("context_tokens", default=8192))
        warning_at = int(self._cfg("context_warning_tokens", default=6500))
        if approx_tokens > warning_at:
            warnings.append(
                f"Context is approximately {approx_tokens} tokens "
                f"(configured budget {context_limit}); answer quality may drop."
            )
        if truncated:
            warnings.append(
                "Some retrieved passages were trimmed to fit the configured context budget."
            )

        raw_pieces: list[str] = []
        for delta in self.chat.stream(prompt, system=system, history=history or []):
            if delta:
                raw_pieces.append(delta)
        raw_answer = "".join(raw_pieces).strip()
        if not raw_answer:
            raise RuntimeError(
                "The local language model returned an empty answer. "
                "Disable thinking or increase the generation token budget, then retry."
            )
        answer = normalize_citations(raw_answer)

        # Small local models paraphrase refusals, switch to the source language
        # or forget citations. One bounded repair pass fixes those slips without
        # inventing new facts; deterministic fallbacks guarantee the contract.
        refusal_like = looks_like_refusal(answer, lang)
        if refusal_like:
            answer = REFUSAL[lang]

        answer_lang = detect_answer_language(answer) if answer else lang
        reasons: list[str] = []
        if refusal_like or answer == REFUSAL[lang]:
            reasons.append(
                "The previous attempt refused; re-read SOURCES. Answer only if the "
                "answer is covered there; otherwise repeat the exact refusal."
            )
        if answer_lang in LANGUAGE_NAMES and answer_lang != lang:
            reasons.append(
                f"It was written in {LANGUAGE_NAMES[answer_lang]}, "
                f"but the question is in {LANGUAGE_NAMES[lang]}."
            )
        if answer and not has_citation(answer):
            reasons.append("It has no [Book Title, p.N] citation; add one after each claim.")
        if answer and reasons:
            repair_system = (
                "You are revising a grounded answer. " + " ".join(reasons) + "\n"
                f"Write the entire response in {LANGUAGE_NAMES[lang]}. "
                "Keep every [Book Title, p.N] citation unchanged; never add facts "
                "or page numbers. Repeat the exact refusal only if the sources "
                "genuinely lack the answer."
            )
            try:
                revised = self.chat.complete(
                    f"QUESTION ({LANGUAGE_NAMES[lang]}): {query}\n\n"
                    f"PREVIOUS ANSWER:\n{answer}\n\nREVISED ANSWER:",
                    system=repair_system,
                )
                revised = normalize_citations((revised or "").strip())
                if revised and revised != answer:
                    answer = revised
                    warnings.append("Answer language/citations were repaired once.")
            except Exception as exc:  # noqa: BLE001 - repair is best effort
                log.warning("Could not repair answer: %s", exc)

        if looks_like_refusal(answer, lang):
            answer = REFUSAL[lang]
        answer = normalize_citations(remap_citations(answer, used_sources))
        if answer and not looks_like_refusal(answer, lang) and (
            has_unresolved_citation(answer, used_sources) or not has_citation(answer)
        ):
            answer = REFUSAL[lang]

        if answer == raw_answer:
            for delta in raw_pieces:
                yield {"type": "delta", "text": delta}
        else:
            yield {"type": "delta", "text": answer}
        yield self._done(
            answer=answer, language=lang, sources=used_sources, warnings=warnings,
            result=result, mode=mode, memories=memories, approx_tokens=approx_tokens,
        )

    def answer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Non-streaming convenience used by CLI/tests and map-reduce mode."""
        done: dict[str, Any] = {}
        for event in self.stream_answer(*args, **kwargs):
            if event.get("type") == "done":
                done = event
        return done

    def stream_summary(
        self,
        query: str,
        book_ids: list[str] | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        language: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Map-reduce over every child chunk in the requested scope."""
        lang = language if language in LANGUAGE_NAMES else detect_answer_language(query)
        if lang not in LANGUAGE_NAMES:
            lang = "en"
        ids = book_ids or [book.id for book in BooksRepo.all(self.db)]
        chunks: list[tuple[str, Any]] = []
        for book_id in ids:
            book = BooksRepo.get(self.db, book_id)
            if book is None:
                continue
            for chunk in ChunksRepo.all_for_book(self.db, book_id, "child"):
                if page_from is not None and chunk.page_end < page_from:
                    continue
                if page_to is not None and chunk.page_start > page_to:
                    continue
                chunks.append((book.title, chunk))
        if not chunks:
            text = NO_SOURCES.get(lang, NO_SOURCES["en"])
            yield {"type": "delta", "text": text}
            yield {
                "type": "done", "answer": text, "language": lang, "sources": [],
                "warnings": [], "mode": "summarize", "models": {"llm": self.chat.model},
            }
            return

        budget_chars = max(2000, int(self._cfg("context_tokens", default=8192)) * 3)
        batches: list[list[tuple[str, Any]]] = []
        current: list[tuple[str, Any]] = []
        size = 0
        for item in chunks:
            item_size = len(item[1].text) + 120
            if current and size + item_size > budget_chars:
                batches.append(current)
                current, size = [], 0
            current.append(item)
            size += item_size
        if current:
            batches.append(current)

        partials: list[str] = []
        for batch in batches:
            body = "\n\n".join(
                f"[{self._label(title, chunk.page_start, chunk.page_end)}] {chunk.text}"
                for title, chunk in batch
            )
            partial = self.chat.complete(
                body,
                system=(
                    "Summarize this passage faithfully in the question's language. "
                    "Keep every [Book, p.N] citation exactly as shown. Do not add facts."
                ),
            )
            if partial:
                partials.append(partial)

        final_prompt = (
            f"QUESTION: {query}\n\n"
            + "PARTIAL SUMMARIES:\n" + "\n\n".join(partials)
            + "\n\nWrite one coherent summary in the question's language. "
              "Keep citations and do not add uncited facts."
        )
        pieces: list[str] = []
        for delta in self.chat.stream(
            final_prompt,
            system="You write grounded summaries. Never invent details.",
        ):
            if delta:
                pieces.append(delta)
                yield {"type": "delta", "text": delta}
        sources = self._chunk_sources(chunks)
        answer = normalize_citations(remap_citations("".join(pieces).strip(), sources))
        yield {
            "type": "done", "answer": answer, "language": lang,
            "sources": sources, "warnings": [], "mode": "summarize",
            "models": {"llm": self.chat.model}, "counts": {"chunks": len(chunks)},
        }

    # -- short- and long-term memory helpers -------------------------------
    def build_history(
        self, conversation: Conversation | None, recent: list[Message] | None = None
    ) -> list[dict[str, str]]:
        limit = int(self._cfg("history_turns", default=6)) * 2
        messages = recent or (
            MessagesRepo.recent(self.db, conversation.id, limit) if conversation else []
        )
        return [{"role": message.role, "content": message.content} for message in messages[-limit:]]

    def maybe_summarize(self, conversation_id: str | None) -> bool:
        """Roll old turns into a short summary every N messages."""
        if not conversation_id:
            return False
        every = int(self._cfg("summary_every_messages", default=8))
        if every <= 0:
            return False
        count = MessagesRepo.count(self.db, conversation_id)
        if count == 0 or count % every != 0:
            return False
        conversation = ConversationsRepo.get(self.db, conversation_id)
        if conversation is None or conversation.summary_message_count == count:
            return False
        recent = MessagesRepo.recent(self.db, conversation_id, every)
        transcript = "\n".join(f"{m.role}: {m.content}" for m in recent)
        system = "Compress the conversation into a short factual summary. No citations."
        try:
            summarizer = OllamaChat(
                self.cfg,
                model=str(self.cfg.get("ollama", "models", "llm_fast", default="llama3.2:latest")),
                num_predict=220,
                temperature=0.0,
            )
            summary = summarizer.complete(
                f"Previous summary:\n{conversation.summary}\n\nNew turns:\n{transcript}",
                system=system,
            )
            summarizer.unload()
        except Exception as exc:  # noqa: BLE001 - summary is optional
            log.warning("Could not roll conversation summary: %s", exc)
            return False
        ConversationsRepo.update(self.db, conversation_id, summary=summary[:4000],
                                 summary_message_count=count)
        return True

    # -- internals ----------------------------------------------------------
    def _memories(self, query: str) -> list[dict[str, Any]]:
        if self.memory is None:
            return []
        try:
            return self.memory.relevant(query)
        except Exception as exc:  # noqa: BLE001 - memory must never break chat
            log.warning("Memory lookup failed: %s", exc)
            return []

    @staticmethod
    def _label(book_title: str, page_start: int, page_end: int | None = None) -> str:
        title = " ".join(book_title.split())[:48] or "Book"
        page_label = (
            f"{page_start}-{page_end}" if page_end and page_end > page_start else str(page_start)
        )
        return f"{title}, p.{page_label}"

    def _sources(self, hits: list[RetrievalHit]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        for hit in hits:
            key = (hit.book_id, hit.page_start, hit.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "label": self._label(hit.book_title, hit.page_start, hit.page_end),
                "book_id": hit.book_id,
                "book_title": hit.book_title,
                "page_start": hit.page_start,
                "page_end": hit.page_end,
                "chunk_id": hit.chunk_id,
                "text": " ".join((hit.parent_text or hit.text).split()),
            })
        return sources

    @staticmethod
    def _chunk_sources(chunks: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for title, chunk in chunks:
            key = (chunk.book_id, chunk.page_start)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "label": GroundedGenerator._label(title, chunk.page_start, chunk.page_end),
                "book_id": chunk.book_id,
                "book_title": title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chunk_id": chunk.id,
                "text": " ".join(chunk.text.split())[:1200],
            })
        return sources

    def _system(
        self,
        language: str,
        conversation: Conversation | None,
        memories: list[dict[str, Any]],
        mode: str,
    ) -> str:
        language_name = LANGUAGE_NAMES[language]
        parts = [
            "You are a private offline research assistant.",
            f"Answer ONLY in {language_name}, even if sources are in another language.",
            f"All prose must be {language_name}; do not switch to the language of a source.",
            "Use ONLY the SOURCES below. Do not use outside knowledge.",
            "Cite every claim immediately after the claim.",
            "Use one bracket per page: [Book Title, p.N]. "
            "Never combine pages in one bracket; write [Book, p.3] [Book, p.5], "
            "not [Book, p.3; p.5].",
            f'If the answer is absent, reply exactly: "{REFUSAL[language]}"',
            "Never invent a citation or a page number.",
        ]
        if mode == "compare":
            parts.append(
                "Compare the sources explicitly: similarities, differences and contradictions, "
                f"with a citation for each point. Write the comparison ONLY in {language_name}."
                " The book title in each source label identifies which book the passage comes "
                "from; use different source labels to distinguish the books."
            )
        if conversation and conversation.summary:
            parts.append(
                "Conversation summary (context only, never cited):\n" + conversation.summary
            )
        if memories:
            parts.append(
                "Confirmed user context (context only, never cited):\n"
                + "\n".join(f"- {m['content']}" for m in memories)
            )
        return "\n".join(parts)

    def _prompt(
        self, query: str, sources: list[dict[str, Any]], language: str, mode: str
    ) -> tuple[str, list[dict[str, Any]], bool]:
        budget_chars = max(2000, int(self._cfg("context_tokens", default=8192)) * 3)
        header = (
            f"QUESTION ({LANGUAGE_NAMES[language]}): {query}\n\n"
            "SOURCES:\n"
        )
        blocks: list[str] = []
        used: list[dict[str, Any]] = []
        size = len(header)
        truncated = False
        for source in sources:
            text = source["text"]
            block = f"[{source['label']}]\n{text}\n\n"
            if size + len(block) > budget_chars and blocks:
                truncated = True
                break
            if size + len(block) > budget_chars:
                block = f"[{source['label']}]\n{text[:budget_chars - size - 80]}\n\n"
                truncated = True
            blocks.append(block)
            used.append(source)
            size += len(block)
        suffix = (
            "Write the answer now in the question's language. "
            "Cite every claim; if unsupported, use the exact refusal sentence."
        )
        return header + "".join(blocks) + suffix, used, truncated

    def _done(
        self,
        answer: str,
        language: str,
        sources: list[dict[str, Any]],
        warnings: list[str],
        result: Any,
        mode: str,
        memories: list[dict[str, Any]],
        approx_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "done",
            "answer": answer,
            "language": language,
            "sources": sources,
            "warnings": warnings,
            "mode": mode,
            "models": {"llm": self.chat.model},
            "counts": result.counts,
            "retrieval_query": result.query,
            "memories_used": [m["content"] for m in memories],
        }
        if approx_tokens is not None:
            payload["approx_tokens"] = approx_tokens
        return payload

    def close(self) -> None:
        if self._chat is not None and bool(self._cfg("unload_llm_after_answer", default=False)):
            self._chat.unload()
