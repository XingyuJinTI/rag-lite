"""
Document loaders for ingestion.

Parses a file into (text, page) segments plus document-level metadata, ready for
`chunking.chunk_segments`. All parsing is local/on-prem — no cloud or OCR services.

Supported: PDF (.pdf), Word (.docx), Markdown (.md), plain text (.txt).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# segments: list of (text, page). page is 1-based for PDFs, None otherwise.
Segments = List[Tuple[str, Optional[int]]]


class UnsupportedFormatError(ValueError):
    """Raised when a file's extension has no registered loader."""


def _load_pdf(path: Path) -> Tuple[Segments, Dict]:
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(str(path))
    segments: Segments = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            segments.append((text, i))

    title = None
    try:
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title
    except Exception:  # malformed PDF metadata shouldn't fail ingestion
        pass

    return segments, {"title": title or path.stem}


def _load_docx(path: Path) -> Tuple[Segments, Dict]:
    import docx  # python-docx, lazy import

    document = docx.Document(str(path))
    text = "\n\n".join(p.text for p in document.paragraphs if p.text and p.text.strip())
    # Word has no fixed page model at the XML level → page is None.
    segments: Segments = [(text, None)] if text.strip() else []

    title = None
    try:
        title = document.core_properties.title or None
    except Exception:
        pass

    return segments, {"title": title or path.stem}


def _load_text(path: Path) -> Tuple[Segments, Dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    segments: Segments = [(text, None)] if text.strip() else []
    return segments, {"title": path.stem}


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".md": _load_text,
    ".txt": _load_text,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS.keys())


def load_document(path: str, *, uri: Optional[str] = None) -> Tuple[Segments, Dict]:
    """
    Parse a document into (segments, doc_meta).

    Returns:
        segments: list of (text, page)
        doc_meta: {"title", "source", "uri"}

    Raises:
        FileNotFoundError: path does not exist
        UnsupportedFormatError: extension has no loader
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFormatError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    segments, meta = loader(p)
    meta = {
        "title": meta.get("title") or p.stem,
        "source": p.name,
        "uri": uri or str(p.resolve()),
    }
    logger.info(f"Loaded '{p.name}' → {len(segments)} segment(s)")
    return segments, meta
