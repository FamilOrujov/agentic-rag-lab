from __future__ import annotations

import re
from typing import Any

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr

from agentic_rag.rag.llm import get_chat_model
from agentic_rag.rag.retrieval import retrieve
from agentic_rag.rag.answering import Source, answer_question
from agentic_rag.agents.state import AgentState, Route


def _coerce_api_key(value: str | SecretStr | None) -> SecretStr | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value
    value = str(value).strip()
    return SecretStr(value) if value else None


def _get_llm_from_state(state: AgentState) -> BaseChatModel:
    """
    Get the appropriate LLM based on provider settings in state.
    
    Falls back to Ollama if no provider is specified.
    """
    provider = state.get("provider", "ollama")
    model = state.get("model")
    api_key_str = state.get("api_key")
    temperature = state.get("temperature", 0.0)
    
    # Convert string api_key to SecretStr if provided
    api_key = _coerce_api_key(api_key_str)
    
    return get_chat_model(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )



def _safe_text(content: Any) -> str:
    """
    LangChain AIMessage.content can be a string or a list of provider-native blocks. :contentReference[oaicite:8]{index=8}
    This makes sure we always return a string for the API.
    """
    return content if isinstance(content, str) else str(content)


def route_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Decide whether retrieval is needed.

    Output: {"route": "direct"} or {"route": "retrieve"}
    """
    llm = _get_llm_from_state(state)

    system = SystemMessage(
        "You are a router for a RAG system.\n"
        "Return exactly one token: direct or retrieve.\n"
        "Use retrieve if the user asks about uploaded documents or needs factual grounding.\n"
        "Use direct for greetings, chit-chat, or questions about the assistant itself."
    )
    human = HumanMessage(state["query"])

    msg = llm.invoke([system, human], config=config)
    raw = _safe_text(msg.content).strip().lower()

    route: Route = "retrieve" if "retrieve" in raw else "direct"
    return {"route": route}


def retrieve_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Retrieve top-k chunks using your existing retrieval function.
    Store serializable source dicts.
    """
    # Retrieval here is your custom function, not a LangChain retriever runnable.
    # Langfuse will still trace the node execution via LangGraph.
    rows, _applied_filter = retrieve(
        query=state["query"],
        k=state.get("k", 6),
        doc_ids=state.get("doc_ids"),
    )

    sources: list[dict[str, Any]] = []
    for i, (chunk_id, score, text, metadata) in enumerate(rows, start=1):
        sources.append(
            {
                "source_id": f"S{i}",
                "chunk_id": chunk_id,
                "score": score,
                "text": text,
                "metadata": metadata,
            }
        )

    return {"sources": sources}


def answer_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Generate the final answer.

    If route=direct, answer without sources.
    If route=retrieve, answer with sources and citations.
    Enforce: do not end with a question.
    """
    route = state.get("route", "retrieve")
    query = state["query"]

    if route == "direct":
        llm = _get_llm_from_state(state)
        system = SystemMessage(
            "You are an Agentic RAG Assistant - an intelligent document-aware AI system.\n"
            "Your primary purpose is to help users understand and query their uploaded documents.\n\n"
            "CAPABILITIES:\n"
            "- Answer questions about uploaded documents with citations\n"
            "- Retrieve relevant information from the document knowledge base\n"
            "- Maintain conversation context and memory across messages\n"
            "- Route queries intelligently (direct response vs. document retrieval)\n\n"
            "CURRENT MODE: Direct Response (no document retrieval needed)\n"
            "For this conversational query, respond naturally but stay in character as the Agentic RAG Assistant.\n"
            "If the user asks what you can do, explain your document-aware capabilities.\n"
            "Use the conversation history for context.\n"
            "Do not ask the user any questions. End with a declarative sentence."
        )
        
        # Build messages: system + history + current query
        # Include previous messages for multi-turn context
        history = state.get("messages", [])
        messages = [system] + list(history) + [HumanMessage(query)]
        
        msg = llm.invoke(messages, config=config)
        answer = _safe_text(msg.content).strip()
        return {
            "answer": answer,
            "citations": [],    # Empty list for direct mode
            "messages": [AIMessage(content=answer)],
        }

    # route == "retrieve"
    raw_sources = state.get("sources", [])
    sources = [
        Source(
            source_id=s["source_id"],
            chunk_id=s["chunk_id"],
            score=s.get("score"),
            text=s["text"],
            metadata=s.get("metadata", {}),
        )
        for s in raw_sources
    ]

    answer, used_sources = answer_question(
        query=query,
        sources=sources,
        max_context_chars=state.get("max_context_chars", 12000),
        config=config,
        provider=state.get("provider", "ollama"),
        model=state.get("model"),
        api_key=_coerce_api_key(state.get("api_key")),
        temperature=state.get("temperature", 0.0),
    )

    citations = [
        {
            "source_id": s.source_id,
            "chunk_id": s.chunk_id,
            "score": s.score,
            "metadata": s.metadata,
        }
        for s in used_sources
    ]

    answer = answer.strip()
    return {
        "answer": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """
    Last safety pass:
    - remove trailing whitespace
    - if it ends with '?', rewrite the ending to a statement
    - remove common 'Any questions?' patterns

    This enforces your rule even if the model slips.
    """
    text = (state.get("answer") or "").strip()

    # Remove common ending questions
    text = re.sub(r"\b(any questions\?|any other questions\?)\s*$", "", text, flags=re.I).strip()

    # If still ends with a question mark, replace final '?' with '.'
    if text.endswith("?"):
        text = text[:-1].rstrip() + "."

    return {"answer": text}
