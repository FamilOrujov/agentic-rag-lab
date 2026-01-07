"""
LLM Factory (Ollama only).

This project intentionally supports only local Ollama models.
Any non-ollama provider is rejected to keep the stack simple and offline-first.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from agentic_rag.config import settings


Provider = Literal["ollama"]
ApiKey = SecretStr | str | None


def get_chat_model(
    provider: Provider = "ollama",
    model: str | None = None,
    api_key: ApiKey = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Return the Ollama chat model.

    api_key is ignored and kept only for compatibility with call sites.
    """
    if provider != "ollama":
        raise ValueError("Only the Ollama provider is supported in this project.")

    return ChatOllama(
        model=model or settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
    )
