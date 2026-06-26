"""Convert Markdown notes into a downloadable, styled PDF.

Pipeline: Markdown -> HTML (python-markdown) -> PDF (xhtml2pdf/pisa).
Both libraries are pure Python, so this works on free hosting with no
system packages to install. Returns raw PDF bytes for Streamlit's
download_button.
"""

import io

import markdown as md
from xhtml2pdf import pisa

# Simple print stylesheet so the PDF looks clean, not like raw text.
_CSS = """
<style>
  body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt;
         color: #222; line-height: 1.5; }
  h1 { font-size: 20pt; color: #4B3FBE; border-bottom: 2px solid #6C5CE7;
       padding-bottom: 4px; }
  h2 { font-size: 15pt; color: #4B3FBE; margin-top: 18px; }
  h3 { font-size: 12.5pt; color: #333; margin-top: 12px; }
  ul { margin-left: 14px; }
  li { margin-bottom: 4px; }
  strong { color: #111; }
  em { color: #555; }
  hr { border: none; border-top: 1px solid #ccc; }
</style>
"""


def markdown_to_pdf_bytes(markdown_text: str, title: str = "Notes") -> bytes:
    """Render Markdown into PDF bytes.

    Raises RuntimeError if the PDF engine reports an error so the UI can
    show a friendly message instead of silently producing a broken file.
    """
    html_body = md.markdown(
        markdown_text,
        extensions=["extra", "sane_lists", "nl2br"],
    )
    full_html = f"<html><head><meta charset='utf-8'>{_CSS}</head><body>{html_body}</body></html>"

    buffer = io.BytesIO()
    result = pisa.CreatePDF(src=full_html, dest=buffer, encoding="utf-8")
    if result.err:
        raise RuntimeError("Failed to render the PDF from the notes.")
    return buffer.getvalue()
