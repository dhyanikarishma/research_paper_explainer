"""Thin wrapper around the Google Gemini API.

Keeping every Gemini call in one file means if the SDK ever changes, you fix
it in exactly one place. Two jobs live here:
  1. generate_text() - send a prompt, get a text answer (summaries, quiz...).
  2. embed_texts()   - turn text into vectors for similarity search (RAG).
"""

import google.generativeai as genai

from src.config import EMBEDDING_MODEL, GENERATION_MODEL, get_api_key


class GeminiError(RuntimeError):
    """Raised when Gemini is misconfigured or a call fails."""


def _ensure_configured() -> None:
    """Configure the SDK with the API key, or raise a friendly error."""
    api_key = get_api_key()
    if not api_key:
        raise GeminiError(
            "No Gemini API key found. Set GEMINI_API_KEY in a .env file "
            "(local) or in Streamlit secrets (deployed). See the README."
        )
    genai.configure(api_key=api_key)


def generate_text(prompt: str, temperature: float = 0.3) -> str:
    """Send `prompt` to the generation model and return the text response.

    `temperature` controls creativity: low (0.2-0.3) for factual summaries,
    higher (0.7) for more varied output. We default low to stay grounded.
    """
    _ensure_configured()
    model = genai.GenerativeModel(GENERATION_MODEL)
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
    except Exception as exc:  # network / quota / bad key
        raise GeminiError(f"Gemini generation failed: {exc}") from exc

    if not getattr(response, "text", None):
        raise GeminiError(
            "Gemini returned an empty response. This often means the content "
            "was blocked by safety filters or the model name is invalid."
        )
    return response.text.strip()


def embed_texts(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """Turn a list of strings into a list of embedding vectors.

    `task_type` tells Gemini how the text will be used:
      - "retrieval_document" for the paper chunks we store, and
      - "retrieval_query" for the user's question.
    Using the right task_type measurably improves retrieval quality.
    """
    _ensure_configured()
    vectors: list[list[float]] = []
    # The SDK embeds one item per call here for simplicity and clear errors.
    for text in texts:
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type=task_type,
            )
        except Exception as exc:
            raise GeminiError(f"Gemini embedding failed: {exc}") from exc
        vectors.append(result["embedding"])
    return vectors
