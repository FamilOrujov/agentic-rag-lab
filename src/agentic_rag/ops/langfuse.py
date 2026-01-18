from __future__ import annotations

import os
from typing import Optional

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from pydantic import SecretStr

from agentic_rag.config.config import settings


def _normalize_secret(value: SecretStr | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        raw = value.get_secret_value()
    else:
        raw = str(value)
    raw = raw.strip()
    return raw or None


def _resolve_langfuse_config() -> tuple[str | None, str | None, str | None]:
    public_key = _normalize_secret(settings.LANGFUSE_PUBLIC_KEY) or os.getenv(
        "LANGFUSE_PUBLIC_KEY"
    )
    secret_key = _normalize_secret(settings.LANGFUSE_SECRET_KEY) or os.getenv(
        "LANGFUSE_SECRET_KEY"
    )
    host = (
        settings.LANGFUSE_BASE_URL
        or os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
    )
    if host:
        host = str(host).strip() or None
    return public_key, secret_key, host


def _ensure_langfuse_env() -> None:
    public_key, secret_key, host = _resolve_langfuse_config()
    if public_key and not os.getenv("LANGFUSE_PUBLIC_KEY"):
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    if secret_key and not os.getenv("LANGFUSE_SECRET_KEY"):
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    if host and not (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")):
        os.environ["LANGFUSE_HOST"] = host
        os.environ["LANGFUSE_BASE_URL"] = host


def langfuse_enabled() -> bool:
    public_key, secret_key, _ = _resolve_langfuse_config()
    return bool(settings.LANGFUSE_ENABLED and public_key and secret_key)


def client() -> Langfuse | None:
    """
    Returns the singleton Langfuse client.
    get_client() initializes from LANGFUSE_* env vars. :contentReference[oaicite:1]{index=1}
    """
    if not langfuse_enabled():
        return None
    _ensure_langfuse_env()
    return get_client()


def new_handler() -> Optional[CallbackHandler]:
    """
    Creates a new Langfuse CallbackHandler for LangChain and LangGraph tracing. :contentReference[oaicite:2]{index=2}
    Create one per request to avoid mixing trace contexts in concurrent execution.
    """
    if not langfuse_enabled():
        return None
    _ensure_langfuse_env()
    return CallbackHandler()


def make_trace_id(seed: str) -> str:
    """
    Create a deterministic W3C compatible trace id from any external seed.
    Langfuse recommends this for distributed tracing and correlation. :contentReference[oaicite:3]{index=3}
    """
    return Langfuse.create_trace_id(seed=seed)


def flush() -> None:
    """
    Flush the Langfuse client to ensure all traces are sent.
    Call this after completing a trace to ensure it reaches the server
    before the request response is returned.
    """
    lf = client()
    if lf is not None:
        lf.flush()


def shutdown() -> None:
    """
    Shutdown the Langfuse client gracefully.
    Call on application shutdown to flush remaining traces.
    """
    lf = client()
    if lf is not None:
        lf.flush()
        lf.shutdown()
