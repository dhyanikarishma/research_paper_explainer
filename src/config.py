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
# If Google deprecates this name, change it here. To see what your key can
# use, run the helper in README ("List available models").
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
# The UI lets the user switch at runtime; this is just the default.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "numpy")

# --- Study sessions --------------------------------------------------------
# Where saved study sessions (summary, flashcards, quiz, etc.) are stored.
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")


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

    # Import here so the module still works in plain-Python scripts/tests
    # where Streamlit may not be running.
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return None
