from __future__ import annotations
from types import NoneType

from pydantic import BaseModel, Field
from typing import Any

class IngestStats(BaseModel):
    file_bytes: int = Field(ge=0)
    extracted_units: int = Field(ge=0)
    extracted_chars: int = Field(ge=0)
    avg_chunk_chars: float = Field(ge=0.0)


class IngestedDocument(BaseModel):
    doc_id: str
    filename: str
    stored_path: str
    chunks_added: int = Field(ge=0)
    stats: IngestStats
    metadata: dict[str, Any] = Field(default_factory=dict)


class UploadSummary(BaseModel):
    files_received: int = Field(ge=0)
    documents_ingested: int = Field(ge=0)
    total_file_bytes: int = Field(ge=0)
    total_extracted_chars: int = Field(ge=0)
    total_chunks_added: int = Field(ge=0)
    chunk_size: int = Field(ge=0)
    chunk_overlap: int = Field(ge=0)    


class UploadResponse(BaseModel):
    summary: UploadSummary
    documents: list[IngestedDocument]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=50)
    doc_ids: list[str] | None = None


class RetrievedChunk(BaseModel):
    id: str
    score: float | None = None
    text: str
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    k: int
    applied_filter: dict[str, Any] | None
    results: list[RetrievedChunk]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=50)
    doc_ids: list[str] | None = None
    max_context_chars: int = Field(default=12000, ge=1000, le=80000)


class AgenticAskRequest(BaseModel):
    """Request for the agentic RAG endpoint with optional session memory."""
    
    query: str = Field(min_length=1)
    k: int = Field(default=6, ge=1, le=50)
    doc_ids: list[str] | None = None
    max_context_chars: int = Field(default=12000, ge=1000, le=80000)
    
    # Session ID for conversation memory (optional)
    # If provided and PostgreSQL is configured, enables multi-turn conversations
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation memory. If provided, the agent will remember previous turns.",
    )
    
    # LLM Provider settings
    provider: str = Field(
        default="ollama",
        description="LLM provider: ollama, openai, anthropic, or google",
    )
    model: str | None = Field(
        default=None,
        description="Model name. If not specified, uses provider default.",
    )
    api_key: str | None = Field(
        default=None,
        description="Runtime API key for the provider. If not specified, uses config.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Model temperature (0.0 = deterministic, higher = more creative)",
    )


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    score: float | None = None
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    used_k: int
    applied_filter: dict[str, Any] | None


class AnswerOnlyResponse(BaseModel):
    answer: str


class AgenticAskResponse(BaseModel):
    """Response from the agentic RAG endpoint with session info."""
    
    answer: str
    route: str | None = Field(
        default=None,
        description="The route taken: 'direct' (no retrieval) or 'retrieve' (with retrieval)",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID if memory was used",
    )
    memory_enabled: bool = Field(
        default=False,
        description="Whether conversation memory was used for this request",
    )

