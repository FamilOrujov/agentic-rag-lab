from __future__ import annotations

import uuid
from typing import cast, Any

from fastapi import APIRouter
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agentic_rag.agents.graph import get_agentic_rag_graph, get_agentic_rag_graph_with_memory
from agentic_rag.agents.state import AgentState
from agentic_rag.db.checkpoint import get_checkpointer
from agentic_rag.ops.langfuse import client as lf_client, new_handler, make_trace_id
from agentic_rag.rag.answering import Source, answer_question
from agentic_rag.rag.retrieval import retrieve
from agentic_rag.rag.schemas import (
    AgenticAskRequest,
    AgenticAskResponse,
    AnswerOnlyResponse,
    AskRequest,
    AskResponse,
    Citation,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
)


def _truncate_text(s: str, limit: int) -> str:
    """Prevent huge trace payloads. Keeps Langfuse and RAM stable."""
    s = s or ""
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def _build_ragas_trace_io(
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    retrieved_chunk_ids: list[str],
    k: int,
    doc_ids: list[str] | None,
    applied_filter: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return (trace_input, trace_output) in a schema that matches Ragas expectations:
    - user_input
    - response
    - retrieved_contexts

    I also keep extra fields (chunk ids, k, doc_ids) for debugging.
    Ragas metrics commonly require 'response' and 'retrieved_contexts'.
    """
    safe_contexts = [_truncate_text(c, 2000) for c in retrieved_contexts]
    safe_response = _truncate_text(response, 4000)

    trace_input = {
        "user_input": user_input,
        "k": k,
        "doc_ids": doc_ids,
        "applied_filter": applied_filter,
    }
    trace_output = {
        "response": safe_response,
        "retrieved_context": safe_contexts,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }
    return trace_input, trace_output



router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest) -> RetrieveResponse:
    rows, applied_filter = retrieve(query=req.query, k=req.k, doc_ids=req.doc_ids)

    results: list[RetrievedChunk] = []
    for chunk_id, score, text, metadata in rows:
        results.append(
            RetrievedChunk(
                id=chunk_id,
                score=score,
                text=text,
                metadata=metadata,
            )
        )

    return RetrieveResponse(
        query=req.query,
        k=req.k,
        applied_filter=applied_filter,
        results=results,
    )


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    """
    Classic RAG endpoint that returns answer plus citations.
    Tracing is optional. If Langfuse is enabled, we wrap the LLM call in a span
    and pass the Langfuse CallbackHandler via RunnableConfig callbacks.
    """
    rows, applied_filter = retrieve(query=req.query, k=req.k, doc_ids=req.doc_ids)

    sources: list[Source] = []
    for i, (chunk_id, score, text, metadata) in enumerate(rows, start=1):
        sources.append(
            Source(
                source_id=f"S{i}",
                chunk_id=chunk_id,
                score=score,
                text=text,
                metadata=metadata,
            )
        )

    # Build Langfuse config if enabled
    handler = new_handler()
    lf = lf_client()

    if handler is not None and lf is not None:
        request_id = f"req_{uuid.uuid4().hex}"
        trace_id = make_trace_id(seed=request_id)

        with lf.start_as_current_observation(
            as_type="span",
            name="http:/chat/ask",
            trace_context={"trace_id": trace_id},
        ) as span:
            cfg: RunnableConfig = {
                "callbacks": [handler],
                "metadata": {
                    "langfuse_session_id": request_id,
                    "langfuse_tags": ["fastapi", "rag", "ask", "ragas_eval"],
                },
                "run_name": "rag_ask",
            }

            response, used_sources = answer_question(
                query=req.query,
                sources=sources,
                max_context_chars=req.max_context_chars,
                config=cfg,
            )

            retrieved_contexts = [s.text for s in used_sources]
            retrieved_chunk_ids = [s.chunk_id for s in used_sources]

            trace_input, trace_output = _build_ragas_trace_io(
                user_input=req.query,
                response=response,
                retrieved_contexts=retrieved_contexts,
                retrieved_chunk_ids=retrieved_chunk_ids,
                k=req.k,
                doc_ids=req.doc_ids,
                applied_filter=applied_filter,
            )        
            
            # Write eval-ready payload at TRACE level
            span.update_trace(input=trace_input, output=trace_output)

    else:
        # Tracing disabled
        response, used_sources = answer_question(
            query=req.query,
            sources=sources,
            max_context_chars=req.max_context_chars,
            config=None,
        )

    citations = [
        Citation(
            source_id=s.source_id,
            chunk_id=s.chunk_id,
            score=s.score,
            metadata=s.metadata,
        )
        for s in used_sources
    ]

    return AskResponse(
        query=req.query,
        answer=response,
        citations=citations,
        used_k=len(used_sources),
        applied_filter=applied_filter,
    )


@router.post("/ask_agentic", response_model=AgenticAskResponse)
def ask_agentic(req: AgenticAskRequest) -> AgenticAskResponse:
    """
    Agentic LangGraph endpoint with optional conversation memory.

    Features:
    - Route decision: skips retrieval for greetings/chit-chat
    - Optional memory: pass session_id to enable multi-turn conversations
    - Langfuse tracing: traces are logged for evaluation

    Memory requires PostgreSQL to be configured (POSTGRES_DSN in .env).
    """
    # Check if memory should be used
    checkpointer = None
    memory_enabled = False
    
    if req.session_id:
        checkpointer = get_checkpointer()
        if checkpointer is not None:
            memory_enabled = True

    # Get the appropriate graph
    if memory_enabled and checkpointer is not None:
        graph = get_agentic_rag_graph_with_memory(checkpointer)
    else:
        graph = get_agentic_rag_graph()

    handler = new_handler()
    lf = lf_client()

    # Build the graph input
    graph_input: AgentState = cast(
        AgentState,
        {
            "query": req.query,
            "k": req.k,
            "doc_ids": req.doc_ids,
            "max_context_chars": req.max_context_chars,
            "messages": [HumanMessage(content=req.query)],
            # LLM Provider settings
            "provider": req.provider,
            "model": req.model,
            "api_key": req.api_key,
            "temperature": req.temperature,
        },
    )

    # Build config with thread_id for memory
    configurable: dict[str, str] = {}
    if memory_enabled and req.session_id:
        configurable["thread_id"] = req.session_id

    # If tracing is disabled, run normally
    if handler is None or lf is None:
        invoke_config: RunnableConfig | None = None
        if configurable:
            invoke_config = cast(RunnableConfig, {"configurable": configurable})
        final_state = graph.invoke(graph_input, config=invoke_config)
        return AgenticAskResponse(
            answer=final_state.get("answer", ""),
            route=final_state.get("route"),
            session_id=req.session_id if memory_enabled else None,
            memory_enabled=memory_enabled,
        )

    # If tracing is enabled, bind this request to a deterministic trace_id
    request_id = f"req_{uuid.uuid4().hex}"
    trace_id = make_trace_id(seed=request_id)

    # Langfuse docs show using trace_context={"trace_id": ...} to force the trace id
    with lf.start_as_current_observation(
        as_type="span",
        name="http:/chat/ask_agentic",
        trace_context={"trace_id": trace_id},
    ) as span:
        cfg_dict: dict[str, Any] = {
            "callbacks": [handler],
            "metadata": {
                "langfuse_session_id": request_id,
                "langfuse_tags": ["fastapi", "agentic_rag", "langgraph", "ragas_eval"],
            },
            "run_name": "agentic_rag_graph",
        }
        if configurable:
            cfg_dict["configurable"] = configurable
        cfg = cast(RunnableConfig, cfg_dict)

        # Important: pass config into graph.invoke so it propagates to nodes and nested LLM calls.
        final_state = graph.invoke(graph_input, config=cfg)

        response = final_state.get("answer", "")
        route = final_state.get("route")

        raw_sources = final_state.get("sources") or []
        retrieved_contexts: list[str] = []
        retrieved_chunk_ids: list[str] = []

        for s in raw_sources:
            if isinstance(s, dict):
                retrieved_contexts.append(str(s.get("text", "")))
                retrieved_chunk_ids.append(str(s.get("chunk_id") or s.get("id") or ""))
            else:
                # fallback if the structure changes
                retrieved_contexts.append(str(s))
                retrieved_chunk_ids.append("")

        trace_input, trace_output = _build_ragas_trace_io(
            user_input=req.query,
            response=response,
            retrieved_contexts=retrieved_contexts,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=req.k,
            doc_ids=req.doc_ids,
            applied_filter=None,
        )

        # helpful for later filtering, eg skip direct answers
        trace_output["route"] = route
        trace_output["memory_enabled"] = memory_enabled

        # write eval-ready payload at TRACE level
        span.update_trace(input=trace_input, output=trace_output)

    return AgenticAskResponse(
        answer=response,
        route=route,
        session_id=req.session_id if memory_enabled else None,
        memory_enabled=memory_enabled,
    )

