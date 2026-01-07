import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentic_rag.config.config import settings
from agentic_rag.api.routes.health import router as health_router
from agentic_rag.api.routes.documents import router as documents_router
from agentic_rag.api.routes.chat import router as chat_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize PostgreSQL checkpoint tables if configured
    - Shutdown: Close database connection pool
    """
    # Startup
    from agentic_rag.db.checkpoint import setup_checkpointer_tables, shutdown_pool
    
    if setup_checkpointer_tables():
        logger.info("PostgreSQL checkpoint tables initialized")
    else:
        logger.info("PostgreSQL not configured, memory disabled")
    
    yield
    
    # Shutdown
    shutdown_pool()
    logger.info("Database pool closed")


def create_app() -> FastAPI:
    """
    Application factory.

    This pattern is production-friendly:
    - tests can create isolated app instances
    - startup configuration is explicit
    """
    app = FastAPI(
        title="Agentic RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    
    @app.get("/meta")
    def meta() -> dict:
        # show non-secret runtime config for debugging
        return {
            "llm_model": settings.LLM_MODEL,
            "embed_model": settings.EMBED_MODEL,
            "chroma_path": settings.CHROMA_DIR,
            "postgres_configured": settings.POSTGRES_DSN is not None,
        }

    return app

app = create_app()
