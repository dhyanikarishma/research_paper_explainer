"""Save and load study sessions to local JSON files.

A "session" is everything the user generated for a paper: the extracted
text, summary, flashcards, quiz, gaps, citations, chat history, etc. Saving
lets them close the app and come back later without re-running the AI (which
also saves on API quota).

We deliberately store only plain data (strings/lists/dicts) - never the
VectorStore object - because that can be rebuilt cheaply from the text.
"""

import json
import os
import re
from datetime import datetime

from src.config import SESSIONS_DIR


def _safe_name(name: str) -> str:
    """Turn a paper name into a safe filename."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return stem or "session"


def save_session(session: dict) -> str:
    """Write a session dict to disk and return the file path.

    `session` should contain at least a "paper_name". A timestamp is added
    so multiple saves of the same paper don't overwrite each other.
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    paper_name = session.get("paper_name", "session")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_name(paper_name)}__{timestamp}.json"
    path = os.path.join(SESSIONS_DIR, filename)

    session = {**session, "saved_at": datetime.now().isoformat()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(session, fh, ensure_ascii=False, indent=2)
    return path


def list_sessions() -> list[dict]:
    """Return metadata for all saved sessions, newest first."""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, filename)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            sessions.append(
                {
                    "path": path,
                    "filename": filename,
                    "paper_name": data.get("paper_name", filename),
                    "saved_at": data.get("saved_at", ""),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(sessions, key=lambda s: s["saved_at"], reverse=True)


def load_session(path: str) -> dict:
    """Load a session dict from disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
