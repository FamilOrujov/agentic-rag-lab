"""
LangGraph agent graph for agentic RAG.

The graph implements a routing pattern:
- Route node decides if retrieval is needed (direct vs retrieve)
- Retrieve node fetches relevant chunks from Chroma
- Answer node generates the response (with or without sources)
- Finalize node cleans up the answer

With checkpointing enabled, the graph maintains conversation memory
across multiple invocations within the same thread.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agents.state import AgentState
from agentic_rag.agents.nodes import route_node, retrieve_node, answer_node, finalize_node


def _route_to_next(state: AgentState) -> Literal["retrieve", "answer"]:
    """
    Conditional router function.
    LangGraph supports conditional edges via add_conditional_edges.
    """
    return "retrieve" if state.get("route") == "retrieve" else "answer"


def _build_graph() -> StateGraph:
    """Build the state graph (without compiling)."""
    builder = StateGraph(AgentState)

    builder.add_node("route", route_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        _route_to_next,
        {"retrieve": "retrieve", "answer": "answer"},
    )
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "finalize")
    builder.add_edge("finalize", END)

    return builder


@lru_cache(maxsize=1)
def get_agentic_rag_graph() -> CompiledStateGraph:
    """
    Get the compiled graph WITHOUT checkpointing (stateless).

    Use this for simple single-turn queries where memory isn't needed.
    """
    return _build_graph().compile()


def get_agentic_rag_graph_with_memory(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """
    Get the compiled graph WITH checkpointing (stateful).

    This enables conversation memory by persisting state between invocations.
    Pass a thread_id in the config to maintain separate conversations:

        config = {"configurable": {"thread_id": "user-123-session-456"}}
        result = graph.invoke(input, config=config)

    The checkpointer stores:
    - Message history (messages channel with add_messages reducer)
    - All state fields from previous turns
    """
    return _build_graph().compile(checkpointer=checkpointer)
