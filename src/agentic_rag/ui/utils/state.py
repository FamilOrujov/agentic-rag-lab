"""Session state management for Streamlit."""
from __future__ import annotations

import uuid
import streamlit as st
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    """A chat message."""
    role: Literal["user", "assistant"]
    content: str
    route: str | None = None
    memory_enabled: bool = False
    citations: list[dict] | None = None


def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        # Theme
        "theme": "dark",
        
        # Model settings
        "model_provider": "ollama",
        "api_key": "",
        "model_name": "gemma3:4b",
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 2048,
        
        # Chat
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "memory_enabled": True,
        
        # Documents
        "uploaded_docs": [],  # List of {"doc_id": ..., "filename": ...}
        "selected_doc_ids": [],
        
        # UI state
        "backend_status": False,
        "is_processing": False,
        "k_value": 6,
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_chat() -> None:
    """Clear chat history and create new session."""
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


def add_message(role: Literal["user", "assistant"], content: str, **kwargs) -> None:
    """Add a message to the chat history."""
    msg = Message(role=role, content=content, **kwargs)
    st.session_state.messages.append(msg)


def toggle_theme() -> None:
    """Toggle between dark and light theme."""
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def get_session_id() -> str | None:
    """Get session ID if memory is enabled."""
    if st.session_state.memory_enabled:
        return st.session_state.session_id
    return None


def add_document(doc_id: str, filename: str) -> None:
    """Add a document to the uploaded list."""
    st.session_state.uploaded_docs.append({
        "doc_id": doc_id,
        "filename": filename,
    })


def get_doc_id_by_name(filename: str) -> str | None:
    """Get document ID by filename."""
    for doc in st.session_state.uploaded_docs:
        if doc["filename"].lower() == filename.lower():
            return doc["doc_id"]
    return None


def parse_file_tags(query: str) -> tuple[str, list[str]]:
    """
    Parse @file tags from query and return clean query and doc_ids.
    
    Example: "What is in @document1.pdf?" -> ("What is in?", ["doc-id-1"])
    """
    import re
    
    # Find all @filename patterns
    pattern = r"@(\S+)"
    matches = re.findall(pattern, query)
    
    doc_ids = []
    for match in matches:
        doc_id = get_doc_id_by_name(match)
        if doc_id:
            doc_ids.append(doc_id)
    
    # Remove @tags from query
    clean_query = re.sub(pattern, "", query).strip()
    clean_query = " ".join(clean_query.split())  # Normalize whitespace
    
    return clean_query, doc_ids

