"""Central configuration for the Research Paper Explainer app.

Everything tunable lives here so you never hunt through the codebase to
change a model name or a chunk size. The API key is read from either:
  1. Streamlit secrets (used when deployed on Streamlit Cloud), or
  2. an environment variable / local .env file (used on your laptop).
"""

import os

from dotenv import load_dotenv

# Load variables from a local .env file if one exists (development only).
load_dotenv()

# --- Model names -----------------------------------------------------------
# The text-generation model (summaries, quizzes, chat answers).
# If you hit free-tier quota limits, switch to "gemini-2.5-flash-lite",
# which has more generous free limits (set GENERATION_MODEL in .env).
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")

# The embedding model: turns text into vectors for similarity search.
# (Google retired text-embedding-004; gemini-embedding-001 is the current one.)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# --- RAG tuning ------------------------------------------------------------
# How many characters per chunk, and how much chunks overlap so we don't
# cut a sentence's meaning in half at a boundary.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# How many of the most relevant chunks to feed the model when answering.
TOP_K = int(os.getenv("TOP_K", "5"))

# --- Vector backend --------------------------------------------------------
# "numpy" (default, zero-setup, deploy-safe) or "chroma" (ChromaDB).
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "numpy")

# --- Study sessions --------------------------------------------------------
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")

# --- Security / abuse-prevention limits ------------------------------------
# Cap the model's output length to prevent runaway cost (AI/LLM rule).
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

# Reject oversized uploads server-side (defense in depth on top of the
# Streamlit uploader limit). Documents default to 25 MB.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

# Maximum characters accepted from a user's chat question (input validation).
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))

# Per-session rate limit on AI calls (cost-attack / abuse protection).
# Allow at most RATE_LIMIT_MAX_CALLS AI calls per RATE_LIMIT_WINDOW_SEC.
RATE_LIMIT_MAX_CALLS = int(os.getenv("RATE_LIMIT_MAX_CALLS", "20"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))


def get_api_key() -> str | None:
    """Return the Gemini API key, or None if it isn't configured.

    Checks the environment / local .env first (this also covers Streamlit
    Cloud, which exposes secrets as environment variables). Only if that's
    empty do we touch st.secrets, which avoids Streamlit's noisy
    "No secrets found" message during local development.
    """
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return None
