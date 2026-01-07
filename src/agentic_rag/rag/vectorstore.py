from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from agentic_rag.config import settings



@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:

    return OllamaEmbeddings(
        model=settings.EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )



@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:

    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_DIR,
    )

