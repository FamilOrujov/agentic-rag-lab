from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from agentic_rag.rag.llm import get_chat_model, ApiKey




@dataclass
class Source:
    source_id: str
    chunk_id: str
    score: float | None
    text: str
    metadata: dict[str, Any]


def get_llm(
    provider: str = "ollama",
    model: str | None = None,
    api_key: ApiKey = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Get a chat model for the specified provider.
    
    This is a wrapper around get_chat_model for backwards compatibility.
    """
    return get_chat_model(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def build_context(sources: list[Source], max_chars: int) -> tuple[str, list[Source]]:
    """
    Builds a single context string and enforces a character budget.

    We keep sources in order until adding one would exceed max_chars.
    Returns:
      context_text, kept_sources
    """
    parts: list[str] = []
    kept: list[Source] = []
    total = 0

    for s in sources:
        block = f"[{s.source_id}] {s.text}".strip()
        if not block:
            continue

        # +2 for spacing
        if total + len(block) + 2 > max_chars:
            break

        parts.append(block)
        kept.append(s)
        total += len(block) + 2

    return "\n\n".join(parts), kept


def answer_question(
    *,
    query: str,
    sources: list[Source],
    max_context_chars: int,
    config: RunnableConfig | None = None,
    provider: str = "ollama",
    model: str | None = None,
    api_key: ApiKey = None,
    temperature: float = 0.0,
) -> tuple[str, list[Source]]:
    """
    Calls the LLM with a grounded RAG prompt.

    The model must cite sources like [S1], [S2].
    We return the answer text and the final list of used sources.
    """
    
    llm = get_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )

    context_text, kept_sources = build_context(sources, max_context_chars)

    system = (
        "You are an Agentic RAG Assistant - a document-aware AI system.\n"
        "Your primary purpose is to answer and analyze using the uploaded documents only.\n\n"
        "CURRENT MODE: Document Retrieval Mode\n"
        "Grounding rules:\n"
        "- Use only the SOURCES provided below.\n"
        "- If the sources do not contain the answer, say the documents do not contain it.\n"
        "- Cite every factual claim with [S1], [S2], etc.\n"
        "- Do not invent citations or add external knowledge.\n\n"
        "Quality rules:\n"
        "- Synthesize across multiple sources when helpful.\n"
        "- For summaries, comparisons, critiques, or recommendations, use only the sources.\n"
        "- If you include analysis, label it as Analysis and tie it to cited sources.\n\n"
        "Formatting rules:\n"
        "- Provide a direct answer first with inline citations.\n"
        "- Do not ask the user any questions.\n"
        "- End with a declarative sentence."
    )

    human = (
        f"QUESTION:\n{query}\n\n"
        f"SOURCES:\n{context_text}\n\n"
        "Write a helpful answer. Include citations."
    )
    
    # LangChain docs show invoke(messages) with tuple style messages.
    msg = llm.invoke(
        [("system", system), ("human", human)],
        config=config,
    )

    # msg.content can be str or list, but for text models it's always str
    answer = msg.content if isinstance(msg.content, str) else str(msg.content)
    return answer, kept_sources

    
