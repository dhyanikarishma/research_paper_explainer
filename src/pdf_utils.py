"""PDF text extraction using PyMuPDF (imported as `fitz`).

A research paper PDF is just a container of text laid out on pages. Here we
pull that text out, page by page, so the rest of the pipeline can work with
plain strings.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF given as raw bytes.

    We accept bytes (not a path) because Streamlit hands us an uploaded file
    in memory, which is cleaner than writing a temp file to disk.

    Returns the full document text with page markers, or raises ValueError
    if the PDF has no extractable text (e.g. it's a scanned image).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"[Page {page_number}]\n{text}")
    doc.close()

    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        raise ValueError(
            "No selectable text found in this PDF. It may be a scanned "
            "image. Try a text-based PDF (most arXiv papers work)."
        )
    return full_text


def get_pdf_metadata(file_bytes: bytes) -> dict:
    """Return basic metadata (title, author, page count) for display."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    meta = doc.metadata or {}
    info = {
        "title": meta.get("title") or "Untitled",
        "author": meta.get("author") or "Unknown",
        "pages": doc.page_count,
    }
    doc.close()
    return info
