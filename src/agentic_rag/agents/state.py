from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages


Route = Literal["direct", "retrieve"]


class _RequiredAgentState(TypedDict):
    """Required keys that must always be present."""
    query: str


class AgentState(_RequiredAgentState, total=False):
    """
    AgentState with required and optional keys.
    
    query is required (inherited from _RequiredAgentState).
    All other keys are optional (total=False), allowing nodes to return partial updates.

    messages: chat history channel. We annotate it with add_messages reducer so
    each node can append messages without overwriting prior history.
    LangGraph docs describe this reducer and the MessagesState pattern. :contentReference[oaicite:4]{index=4}
    """
    messages: Annotated[list[AnyMessage], add_messages]

    # Optional request inputs
    k: int
    doc_ids: list[str] | None
    max_context_chars: int

    # LLM Provider settings
    provider: str
    model: str | None
    api_key: str | None
    temperature: float

    # Control
    route: Route

    # Retrieval outputs
    sources: list[dict[str, Any]]   # I store serializable dicts for API friendliness

    # Final outputs
    answer: str
    citations: list[dict[str, Any]]
    