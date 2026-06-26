"""Split long document text into overlapping chunks.

Why chunk? LLMs and embedding models work best on focused pieces of text.
If we embedded a whole 30-page paper as one vector, "meaning" would be
blurred. Instead we cut it into ~1200-character chunks. We overlap chunks
slightly so an idea that straddles a boundary isn't lost.

This is a simple, dependency-free splitter that prefers to break on
paragraph/sentence boundaries for cleaner chunks.
"""

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Return a list of text chunks of roughly `chunk_size` characters.

    We walk through the text in steps of (chunk_size - overlap). For each
    chunk we try to end on a paragraph or sentence boundary so chunks read
    naturally instead of cutting mid-word.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # If we're not at the very end, try to back up to a clean boundary
        # (paragraph break, then sentence break) within the last part of
        # the chunk so we don't slice a sentence in half.
        if end < text_length:
            window = text[start:end]
            boundary = max(window.rfind("\n\n"), window.rfind(". "))
            # Only honour the boundary if it's reasonably far in, otherwise
            # we'd create tiny chunks.
            if boundary > chunk_size * 0.5:
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, keeping `overlap` characters of context.
        next_start = end - overlap
        # Guard against not making progress (can happen with tiny overlaps).
        start = next_start if next_start > start else end

    return chunks
