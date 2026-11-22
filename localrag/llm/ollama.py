"""Ollama chat client for query rewriting, reranking and grounded generation.

`LOCALRAG_LLM_CPU=1` forces the LLM onto CPU while OCR holds the GPU; retrieval
has its own `LOCALRAG_EMBED_CPU` switch. No cloud calls, ever.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx

from ..config import Config
from ..logging_conf import get_logger

log = get_logger("llm.ollama")


class OllamaChat:
    def __init__(
        self,
        cfg: Config,
        model: str | None = None,
        num_predict: int = 256,
        temperature: float = 0.0,
        keep_alive: str = "5m",
        num_ctx: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = str(model or cfg.get("ollama", "models", "llm_fast", default="llama3.2:latest"))
        self.host = str(cfg.get("ollama", "host", default="http://127.0.0.1:11434")).rstrip("/")
        self.timeout = float(cfg.get("ollama", "request_timeout_s", default=900))
        self.num_predict = int(num_predict)
        self.temperature = float(temperature)
        self.keep_alive = keep_alive
        options: dict[str, Any] = {
            "num_predict": self.num_predict,
            "temperature": self.temperature,
        }
        context = int(num_ctx or cfg.get("generation", "context_tokens", default=8192))
        if context > 0:
            options["num_ctx"] = context
        if os.getenv("LOCALRAG_LLM_CPU") == "1":
            options["num_gpu"] = 0
        self.options = options

    def _messages(
        self,
        prompt: str,
        system: str | None,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(
                {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                for item in history
            )
        messages.append({"role": "user", "content": prompt})
        return messages

    def _payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": bool(self.cfg.get("generation", "think", default=False)),
            "keep_alive": self.keep_alive,
            "options": self.options,
        }

    def _check_error(self, response: httpx.Response) -> None:
        if response.status_code == 400 and "not found" in response.text.lower():
            raise RuntimeError(
                f"Chat model '{self.model}' is missing. Run: ollama pull {self.model}"
            )
        response.raise_for_status()

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        response = httpx.post(
            f"{self.host}/api/chat",
            json=self._payload(self._messages(prompt, system, history), stream=False),
            timeout=self.timeout,
        )
        self._check_error(response)
        data = response.json()
        if data.get("error"):
            raise RuntimeError(f"Ollama chat failed: {data['error']}")
        return str((data.get("message") or {}).get("content", "")).strip()

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """Yield answer fragments as Ollama emits them (SSE on the API side)."""
        with httpx.stream(
            "POST",
            f"{self.host}/api/chat",
            json=self._payload(self._messages(prompt, system, history), stream=True),
            timeout=self.timeout,
        ) as response:
            self._check_error(response)
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if data.get("error"):
                    raise RuntimeError(f"Ollama chat failed: {data['error']}")
                delta = str((data.get("message") or {}).get("content", ""))
                if delta:
                    yield delta
                if data.get("done"):
                    break

    def unload(self) -> None:
        try:
            httpx.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001 - unloading must never fail a job
            log.warning("Could not unload chat model %s: %s", self.model, exc)


def list_ollama_models(cfg: Config) -> list[str]:
    """Names of locally installed chat models; never contacts a cloud service."""
    host = str(cfg.get("ollama", "host", default="http://127.0.0.1:11434")).rstrip("/")
    response = httpx.get(f"{host}/api/tags", timeout=15)
    response.raise_for_status()
    return sorted({str(item.get("name", "")) for item in response.json().get("models", []) if item.get("name")})
