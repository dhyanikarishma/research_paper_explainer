"""Security helpers: input sanitization, output escaping, rate limiting.

These functions implement the parts of a standard app-security checklist
that actually apply to a Streamlit + LLM app:
  - sanitize_user_input(): clean and length-cap untrusted user text before
    it ever reaches the LLM (input validation + light prompt-injection
    hardening).
  - escape_label(): HTML-escape any model/paper-derived text before it is
    embedded into the interactive concept-map HTML (XSS prevention).
  - check_rate_limit(): a per-session budget on AI calls so a single
    session can't run up cost or hammer the API (rate limiting).
"""

import html
import time

from src.config import (
    MAX_QUESTION_CHARS,
    RATE_LIMIT_MAX_CALLS,
    RATE_LIMIT_WINDOW_SEC,
)

# Phrases commonly used in prompt-injection attempts. We don't try to be
# exhaustive (that's impossible); we just defang the most obvious ones by
# neutralizing them so they read as plain text to the model.
_INJECTION_MARKERS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the above",
    "system prompt",
    "you are now",
]


def sanitize_user_input(text: str | None, max_len: int = MAX_QUESTION_CHARS) -> str:
    """Clean untrusted user text before sending it to the LLM.

    - Drops null bytes and surrounding whitespace.
    - Caps the length to bound token cost and abuse.
    - Softly neutralizes blatant prompt-injection markers.
    """
    if not text:
        return ""

    cleaned = text.replace("\x00", "").strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]

    lowered = cleaned.lower()
    for marker in _INJECTION_MARKERS:
        if marker in lowered:
            # Insert a zero-width break so the phrase loses its imperative
            # punch but the user's words are still visible/answerable.
            cleaned = cleaned.replace(marker, marker.replace(" ", " \u200b"))
            cleaned = cleaned.replace(marker.title(), marker.title().replace(" ", " \u200b"))

    return cleaned


def escape_label(text: object, max_len: int = 80) -> str:
    """HTML-escape and length-cap a label for safe embedding in HTML.

    Used for concept-map node/edge labels, which come from the LLM/paper and
    are injected into a pyvis HTML page. Escaping prevents stored-XSS via
    crafted labels.
    """
    return html.escape(str(text))[:max_len]


def check_rate_limit() -> tuple[bool, int]:
    """Enforce a per-session limit on AI calls.

    Returns (allowed, retry_after_seconds). When not allowed, the caller
    should refuse the action and tell the user when to retry. State is kept
    in Streamlit's per-session memory, so it resets per user session.
    """
    import streamlit as st

    now = time.time()
    key = "_ai_call_times"
    window = RATE_LIMIT_WINDOW_SEC

    # Keep only timestamps within the current window.
    times = [t for t in st.session_state.get(key, []) if now - t < window]

    if len(times) >= RATE_LIMIT_MAX_CALLS:
        retry_after = int(window - (now - times[0])) + 1
        st.session_state[key] = times
        return False, retry_after

    times.append(now)
    st.session_state[key] = times
    return True, 0
