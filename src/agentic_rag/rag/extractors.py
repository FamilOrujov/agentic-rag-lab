from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from docx import Document as DocxDocument


@dataclass(frozen=True)
class ExtractedUnit:
    """
    A small unit of extracted text.

    For PDF we use real pages.
    For non-PDF we treat the whole file as unit 0.
    """
    unit_index: int
    text: str


def extract_pdf_pages(path: Path) -> Iterable[ExtractedUnit]:
    """
    Page-by-page text extraction from the PDF text layer.
    This is RAM-safe because we never hold all pages at once.
    """
    doc = fitz.open(str(path))
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text() or ""
            if isinstance(text, str):
                text = text.strip()
                if text:
                    yield ExtractedUnit(unit_index=i, text=text)
    finally:
        doc.close()


def extract_docx(path: Path) -> Iterable[ExtractedUnit]:
    doc = DocxDocument(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    text = "\n".join(parts).strip()
    if text:
        yield ExtractedUnit(unit_index=0, text=text)


def extract_txt(path: Path) -> Iterable[ExtractedUnit]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield ExtractedUnit(unit_index=0, text=text)


def extract_html(path: Path) -> Iterable[ExtractedUnit]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if text:
        yield ExtractedUnit(unit_index=0, text=text)


def extract_text_units(path: Path) -> Iterable[ExtractedUnit]:
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf_pages(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext in {".txt", ".md"}:
        return extract_txt(path)
    if ext in {".html", ".htm"}:
        return extract_html(path)

    raise ValueError(f"Unsupported file type: {ext}")


