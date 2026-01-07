from __future__ import annotations

from typing import Any

from agentic_rag.rag.vectorstore import get_vectorstore


def build_metadata_filter(doc_ids: list[str] | None) -> dict[str, Any] | None:
    """
    Convert an optional doc_id list into a Chroma where filter.

    Chroma supports MongoDB-like operators such as $in. :contentReference[oaicite:3]{index=3}
    """
    if not doc_ids:
        return None
 
    # wehere={"doc_id": {"$in": ["id1", "id2"]}}
    return {"doc_id": {"$in": doc_ids}}


def retrieve(
    *,
    query: str,
    k: int = 6,
    doc_ids: list[str] | None = None,
) -> tuple[list[tuple[str, float | None, str, dict[str, Any]]], dict[str, Any] | None]:
    """
    Returns:
      - list of tuples: (chunk_id, score, text, metadata)
      - applied filter dict (or None)

    We prefer similarity_search_with_score so you can see how strong each match was.
    If the current vectorstore backend does not provide scores, we fall back to plain retrieval.
    """
    vectorstore = get_vectorstore()
    filt = build_metadata_filter(doc_ids)

    # Try score-based retrieval first
    try:
        pairs = vectorstore.similarity_search_with_score(query, k=k, filter=filt)
        out: list[tuple[str, float | None, str, dict[str, Any]]] = []
        for doc, score in pairs:
            # doc.id is not guaranteed. LangChain stores ids in metadata sometimes.
            # For my ingestion I explicitly set ids when adding documents, so Chroma has them.
            chunk_id = doc.metadata.get("chunk_id", "")
            out.append((chunk_id, float(score), doc.page_content, dict(doc.metadata)))
        return out, filt
    except Exception:
        # Fallback: no scores
        retriever = vectorstore.as_retriever(search_kwargs={"k": k, "filter": filt})
        docs = retriever.invoke(query)

        out2: list[tuple[str, float | None, str, dict[str, Any]]] = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", "")
            out2.append((chunk_id, None, doc.page_content, dict(doc.metadata)))
        return out2, filt
