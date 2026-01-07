from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from langchain_core.documents import Document

from agentic_rag.rag.extractors import extract_text_units
from agentic_rag.rag.splitter import get_splitter
from agentic_rag.rag.vectorstore import get_vectorstore


@dataclass
class IngestResult:
    doc_id: str
    extracted_units: int
    extracted_chars: int
    chunks_added: int
    avg_chunk_chars: float


def ingest_file(
    *,
    path: Path,
    source_name: str,
    doc_id: str | None = None,
    batch_size: int = 64,
) -> IngestResult:
    """
    Ingest a file into Chroma using LangChain.

    Pipeline:
    1) Extract text units (pages for PDF, whole doc otherwise).
    2) Split into chunks using RecursiveCharacterTextSplitter.
    3) Add chunks in batches using vectorstore.add_documents.

    This is RAM safe because at any moment we only keep up to batch_size chunks.
    """
    doc_id = doc_id or str(uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat()
    
    splitter = get_splitter()
    vectorstore = get_vectorstore()

    buffer_docs: list[Document] = []
    buffer_ids: list[str] = []


    extracted_units = 0
    extracted_chars = 0
    chunks_added = 0
    total_chunk_chars = 0
    global_chunk_index = 0


    for unit in extract_text_units(path):
        text = unit.text.strip()
        if not text:
            continue

        extracted_units += 1
        extracted_chars += len(text)

        base_doc = Document(
            page_content=text,
            metadata={
                "doc_id": doc_id,
                "source_name": source_name,
                "unit_index": unit.unit_index,
                "uploaded_at": uploaded_at,
                "file_ext": path.suffix.lower(),
            },
        )

        chunks = splitter.split_documents([base_doc])

        for c in chunks:
            chunk_id = f"{doc_id}:{unit.unit_index}:{global_chunk_index}"
            
            c.metadata = dict(c.metadata)
            c.metadata["chunk_index"] = global_chunk_index
            c.metadata["chunk_id"] = chunk_id
            
            global_chunk_index += 1

            buffer_docs.append(c)
            buffer_ids.append(chunk_id)
            total_chunk_chars += len(c.page_content)

            if len(buffer_docs) >= batch_size:
                vectorstore.add_documents(buffer_docs, ids=buffer_ids)
                chunks_added += len(buffer_docs)
                buffer_docs.clear()
                buffer_ids.clear()


    if buffer_docs:
        vectorstore.add_documents(buffer_docs, ids=buffer_ids)
        chunks_added += len(buffer_docs)

    
    avg_chunk_chars = (total_chunk_chars / chunks_added) if chunks_added else 0.0

    return IngestResult(
        doc_id=doc_id,
        extracted_units=extracted_units,
        extracted_chars=extracted_chars,
        chunks_added=chunks_added,
        avg_chunk_chars=avg_chunk_chars,
    )



