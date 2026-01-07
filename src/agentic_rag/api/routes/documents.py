from __future__ import annotations

import shutil
from pathlib import Path
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from agentic_rag.config import settings
from agentic_rag.rag.ingestion import ingest_file
from agentic_rag.rag.schemas import UploadResponse, IngestedDocument, UploadSummary, IngestStats
from agentic_rag.rag.splitter import CHUNK_SIZE, CHUNK_OVERLAP

router = APIRouter(prefix="/documents", tags=["documents"])



def _ensure_dirs() -> None:
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)



@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    _ensure_dirs()

    docs_out: list[IngestedDocument] = []

    total_file_bytes = 0
    total_extracted_chars = 0
    total_chunks_added = 0

    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        safe_name = Path(f.filename).name
        doc_id = str(uuid.uuid4())

        stored_path = Path(settings.UPLOAD_DIR) / f"{doc_id}__{safe_name}"
        
        with stored_path.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        file_bytes = stored_path.stat().st_size
        total_file_bytes += file_bytes

        result = await run_in_threadpool(
            ingest_file,
            path=stored_path,
            source_name=safe_name,
            doc_id=doc_id,
        )

        total_extracted_chars += result.extracted_chars
        total_chunks_added += result.chunks_added

        docs_out.append(
            IngestedDocument(
                doc_id=result.doc_id,
                filename=safe_name,
                stored_path=str(stored_path),
                chunks_added=result.chunks_added,
                stats=IngestStats(
                    file_bytes=file_bytes,
                    extracted_units=result.extracted_units,
                    extracted_chars=result.extracted_chars,
                    avg_chunk_chars=result.avg_chunk_chars,
                ),
                metadata={"collection": settings.CHROMA_COLLECTION},
            )
        )


    summary = UploadSummary(
        files_received=len(files),
        documents_ingested=len(docs_out),
        total_file_bytes=total_file_bytes,
        total_extracted_chars=total_extracted_chars,
        total_chunks_added=total_chunks_added,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return UploadResponse(summary=summary, documents=docs_out)

