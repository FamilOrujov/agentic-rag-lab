from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def get_splitter() -> RecursiveCharacterTextSplitter:

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
