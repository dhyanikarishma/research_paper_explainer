# Project security rules (apply to all code generated in this project)

These rules keep the app secure by default. Follow them for every change.

## Secrets
- Never hardcode API keys/tokens. Read them server-side via src/config.get_api_key().
- Keep secrets in .env (local) or Streamlit secrets (deployed). Never commit .env.
- Keep .env, .env.local, .env.*.local, and .streamlit/secrets.toml in .gitignore.

## LLM / AI
- Route all LLM calls through src/llm.py. Never call the model from the browser.
- Always pass max_output_tokens (config.MAX_OUTPUT_TOKENS) to cap cost.
- Sanitize user input with src/security.sanitize_user_input() before sending to the LLM.
- Apply src/security.check_rate_limit() (via app.run_guarded) on every AI action.
- Treat LLM output as untrusted: HTML-escape it (src/security.escape_label) before
  embedding in any HTML (e.g. the concept map). Never use unsafe_allow_html with raw model text.

## Input validation
- Validate file uploads server-side: restrict to PDF, enforce config.MAX_UPLOAD_MB.
- Cap and clean all free-text input; never trust client-side checks.

## Error handling
- Never show stack traces or internal errors to users. Log details server-side
  (logging) and show a generic message (see app.run_guarded).

## Dependencies
- Pin versions in requirements.txt. Run pip-audit after adding packages.

## Not applicable here (do not add unless the architecture changes)
- No auth, no SQL/ORM, no custom CORS/headers: the app has no login, no database,
  and no custom backend server. Revisit these if that changes.
