"""
PostgreSQL-based checkpointing for LangGraph conversation memory.

This module provides the checkpoint saver that persists conversation
state to PostgreSQL, enabling multi-turn conversations with memory.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from agentic_rag.config.config import settings


# Module-level pool (initialized lazily)
_pool: ConnectionPool | None = None


def _get_dsn() -> str | None:
    """
    Get PostgreSQL DSN from settings.
    
    Handles SQLAlchemy-style URLs (postgresql+psycopg://) and converts
    them to standard libpq format (postgresql://) for psycopg.
    """
    if settings.POSTGRES_DSN is None:
        return None
    
    dsn = settings.POSTGRES_DSN.get_secret_value()
    
    # Convert SQLAlchemy URL format to standard libpq format
    # postgresql+psycopg://... -> postgresql://...
    if dsn.startswith("postgresql+"):
        dsn = "postgresql://" + dsn.split("://", 1)[1]
    
    return dsn


def _get_pool() -> ConnectionPool | None:
    """
    Get or create the connection pool.
    
    Returns None if PostgreSQL is not configured or connection fails.
    Uses lazy connection opening to avoid blocking on startup.
    """
    global _pool
    import logging
    logger = logging.getLogger(__name__)

    dsn = _get_dsn()
    if dsn is None:
        return None

    if _pool is None:
        try:
            _pool = ConnectionPool(
                conninfo=dsn,
                min_size=0,  # Don't require connections at startup
                max_size=10,
                open=False,  # Lazy opening - don't block startup
                timeout=5.0,  # Short timeout for getting connections
            )
            _pool.open(wait=False)  # Start opening in background
        except Exception as e:
            # Connection failed - PostgreSQL might not be running
            logger.warning(f"Failed to create PostgreSQL pool: {e}")
            return None

    return _pool


def get_checkpointer() -> PostgresSaver | None:
    """
    Get a PostgreSQL checkpointer for LangGraph.

    Returns None if PostgreSQL is not configured or unavailable.
    The checkpointer enables conversation memory by persisting
    graph state between invocations.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        pool = _get_pool()
        if pool is None:
            return None

        return PostgresSaver(pool)
    except Exception as e:
        logger.warning(f"Failed to get checkpointer: {e}")
        return None


def setup_checkpointer_tables() -> bool:
    """
    Create the checkpoint tables in PostgreSQL if they don't exist.

    Returns True if setup succeeded, False if PostgreSQL is not configured
    or connection failed.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        pool = _get_pool()
        if pool is None:
            return False

        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        return True
    except Exception as e:
        logger.warning(f"Failed to setup checkpoint tables: {e}")
        return False


@contextmanager
def checkpointer_context() -> Generator[PostgresSaver | None, None, None]:
    """
    Context manager for checkpointer with proper cleanup.

    Usage:
        with checkpointer_context() as checkpointer:
            graph = get_graph(checkpointer)
            graph.invoke(...)
    """
    checkpointer = get_checkpointer()
    try:
        yield checkpointer
    finally:
        # Pool cleanup is handled at shutdown, not per-request
        pass


def shutdown_pool() -> None:
    """Shutdown the connection pool. Call on app shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None

