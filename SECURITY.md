# Security

This document describes the security posture of **Research Paper Explainer
AI**. The app is a Streamlit + Google Gemini application with **no user
accounts, no database, and no custom backend server**, so several classic
web-app risks do not apply. The controls that *do* apply are implemented and
listed below.

## Implemented controls

| Area | Control | Where |
|---|---|---|
| Secrets | API key read server-side only; never in client code; `.env` is gitignored; `.env.example` provided | `src/config.py`, `.gitignore` |
| Rate limiting | Per-session budget on AI calls to prevent cost/abuse | `src/security.py` (`check_rate_limit`), `app.py` (`run_guarded`) |
| Input validation | User chat input is sanitized and length-capped before reaching the LLM | `src/security.py` (`sanitize_user_input`) |
| File uploads | Type restricted to PDF; size checked server-side; parsed defensively | `app.py` (`load_paper`), `src/pdf_utils.py` |
| Output cost control | `max_output_tokens` cap on every generation call | `src/config.py`, `src/llm.py` |
| XSS | All LLM/paper-derived labels are HTML-escaped before being embedded in the concept-map HTML | `src/security.py` (`escape_label`), `src/concept_map.py` |
| Error handling | Unexpected errors are logged server-side; users see a generic message (no stack traces) | `app.py` (`run_guarded`) |
| Usage logging | Token usage logged per call to detect abnormal usage | `src/llm.py` |
| Dependencies | Versions pinned in `requirements.txt`; audit with `pip-audit` | `requirements.txt` |
| Prompt injection | User input lightly defanged; RAG answers are instructed to use only retrieved context | `src/security.py`, `src/features.py` |

## Not applicable (and why)

- **Authentication / authorization** — the app has no login or per-user data;
  it's a public, stateless tool.
- **SQL / database security** — there is no SQL database. The optional vector
  store (NumPy or in-memory ChromaDB) holds only the current paper's chunks.
- **CORS / HTTP security headers** — these are managed by the hosting
  platform (Streamlit Community Cloud), which serves the app over HTTPS with
  XSRF protection enabled by default; the app does not expose a custom API.

## Operational notes

- Rotate the Gemini API key if it is ever exposed (Google AI Studio →
  delete + create new key), then update the `.env` / Streamlit secret.
- Run `pip install pip-audit && pip-audit` periodically and address high/
  critical findings.

## Reporting

This is a personal portfolio project. To report an issue, open a GitHub issue
on the repository.
