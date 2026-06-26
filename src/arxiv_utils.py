"""Fetch papers directly from arXiv by ID or URL.

Instead of forcing the user to download a PDF and re-upload it, they can
paste an arXiv link (or just the ID like 1706.03762) and we fetch the paper
for them. Returns the same (bytes, metadata) shape the upload path uses, so
the rest of the pipeline doesn't care where the PDF came from.
"""

import re

import arxiv


def _normalize_arxiv_id(text: str) -> str:
    """Extract a bare arXiv ID from an ID or any arXiv URL.

    Accepts things like:
      - "1706.03762"
      - "arXiv:1706.03762"
      - "https://arxiv.org/abs/1706.03762"
      - "https://arxiv.org/pdf/1706.03762v5"
    """
    text = text.strip()
    # Look for the standard new-style ID (NNNN.NNNNN, optional version).
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text)
    if match:
        return match.group(1) + (match.group(2) or "")

    # Fall back to old-style IDs like "hep-th/9901001".
    match = re.search(r"([a-z\-]+/\d{7})(v\d+)?", text)
    if match:
        return match.group(1) + (match.group(2) or "")

    raise ValueError(
        "Could not find a valid arXiv ID in that input. Try an ID like "
        "1706.03762 or a link like https://arxiv.org/abs/1706.03762"
    )


def fetch_arxiv_paper(id_or_url: str) -> tuple[bytes, dict]:
    """Download an arXiv paper's PDF and return (pdf_bytes, metadata)."""
    arxiv_id = _normalize_arxiv_id(id_or_url)

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    try:
        result = next(client.results(search))
    except StopIteration as exc:
        raise ValueError(f"No arXiv paper found for ID '{arxiv_id}'.") from exc

    # Download the PDF into memory (no temp files needed by the caller).
    import urllib.request

    req = urllib.request.Request(
        result.pdf_url, headers={"User-Agent": "research-paper-explainer/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        pdf_bytes = response.read()

    metadata = {
        "title": result.title,
        "author": ", ".join(a.name for a in result.authors[:5]),
        "pages": 0,  # filled in later by pdf_utils after we read the bytes
        "arxiv_id": arxiv_id,
        "summary": result.summary,
        "published": str(result.published.date()) if result.published else "",
    }
    return pdf_bytes, metadata
