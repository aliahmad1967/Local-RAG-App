from __future__ import annotations

from localrag.llm.ollama import OllamaChat


def test_chat_payload_disables_thinking_by_default(cfg) -> None:
    chat = OllamaChat(cfg)

    assert chat._payload([], stream=False)["think"] is False
