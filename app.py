"""Research Paper Explainer AI - Streamlit entry point.

Run locally with:  streamlit run app.py

This file is ONLY about the user interface and wiring. All the real logic
lives in the `src/` package, which keeps this file readable.

Security wiring in this file:
  - run_guarded(): every AI action goes through a per-session rate limit and
    a single error handler that logs details server-side but shows the user
    a generic message (no stack traces leaked).
  - Uploads are size-checked server-side before parsing.
  - Chat input is sanitized/length-capped before reaching the LLM.
"""

import logging

import streamlit as st
import streamlit.components.v1 as components

from src.agents import run_multi_agent_analysis
from src.arxiv_utils import fetch_arxiv_paper
from src.chunking import split_text
from src.concept_map import build_concept_map_html
from src.config import MAX_UPLOAD_MB, get_api_key
from src.features import (
    answer_question,
    build_markdown_notes,
    compare_papers,
    extract_citations,
    generate_concept_map,
    generate_flashcards,
    generate_literature_review,
    generate_quiz,
    generate_research_gaps,
    generate_summary,
)
from src.llm import GeminiError
from src.pdf_export import markdown_to_pdf_bytes
from src.pdf_utils import extract_text_from_pdf, get_pdf_metadata
from src.security import check_rate_limit, sanitize_user_input
from src.sessions import list_sessions, load_session, save_session
from src.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper_explainer")

st.set_page_config(
    page_title="Research Paper Explainer AI",
    page_icon="📄",
    layout="wide",
)


def run_guarded(label: str, fn, *args, **kwargs):
    """Run an AI-backed action with rate limiting + safe error handling.

    Returns the function result, or None if the call was rate-limited or
    failed. Full error detail is logged server-side; the user only sees a
    short, safe message.
    """
    allowed, retry_after = check_rate_limit()
    if not allowed:
        st.warning(
            f"⏳ You've hit the per-session limit. Please wait ~{retry_after}s "
            "and try again."
        )
        return None
    try:
        return fn(*args, **kwargs)
    except GeminiError as exc:
        # Operational AI errors (quota, bad key) are safe and useful to show.
        logger.warning("AI action '%s' failed: %s", label, exc)
        st.error(f"AI service error: {exc}")
        return None
    except Exception:  # anything unexpected
        logger.exception("Unexpected error in AI action '%s'", label)
        st.error("Something went wrong while processing that. Please try again.")
        return None


def init_state():
    """Create the keys we keep in Streamlit's session memory."""
    defaults = {
        "paper_text": None,
        "paper_name": None,
        "metadata": None,
        "vector_store": None,
        "summary": None,
        "flashcards": None,
        "quiz": None,
        "gaps": None,
        "citations": None,
        "concept_graph": None,
        "agent_result": None,
        "chat_history": [],
        "paper_b_text": None,
        "paper_b_name": None,
        "compare_result": None,
        "litreview_result": None,
        "backend": "numpy",
        "last_loaded_key": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


def load_paper(file_bytes: bytes, name: str, metadata: dict | None = None):
    """Parse a PDF, build the vector index, and reset prior results.

    Enforces a server-side upload size limit before doing any work.
    """
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(
            f"File is too large ({len(file_bytes) // (1024 * 1024)} MB). "
            f"The limit is {MAX_UPLOAD_MB} MB."
        )

    text = extract_text_from_pdf(file_bytes)
    meta = metadata or {}
    pdf_meta = get_pdf_metadata(file_bytes)
    meta = {**pdf_meta, **{k: v for k, v in meta.items() if v}}

    store = get_vector_store(st.session_state.backend)
    store.build(split_text(text))

    st.session_state.paper_text = text
    st.session_state.paper_name = name
    st.session_state.metadata = meta
    st.session_state.vector_store = store
    for k in [
        "summary", "flashcards", "quiz", "gaps", "citations",
        "concept_graph", "agent_result", "compare_result", "litreview_result",
    ]:
        st.session_state[k] = None
    st.session_state.chat_history = []


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("📄 Paper Explainer")
    st.caption("Upload or fetch a research paper and let AI do the reading.")

    if get_api_key():
        st.success("Gemini API key detected ✅")
    else:
        st.error(
            "No Gemini API key found. Add `GEMINI_API_KEY` to a `.env` file "
            "(local) or Streamlit secrets (deployed). See the README."
        )

    st.divider()
    st.subheader("1. Load a paper")

    source = st.radio(
        "Source", ["Upload PDF", "Fetch from arXiv"], horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Upload PDF":
        uploaded = st.file_uploader("Upload a paper (PDF)", type=["pdf"])
        if uploaded is not None and uploaded.name != st.session_state.last_loaded_key:
            with st.spinner("Reading PDF and building the search index..."):
                try:
                    load_paper(uploaded.read(), uploaded.name)
                    st.session_state.last_loaded_key = uploaded.name
                    st.success(f"Loaded: {uploaded.name}")
                except ValueError as exc:
                    st.error(str(exc))
                except GeminiError as exc:
                    st.error(f"AI service error: {exc}")
                except Exception:
                    logger.exception("Failed to load uploaded PDF")
                    st.error("Could not process that PDF. Please try another file.")
    else:
        arxiv_input = st.text_input(
            "arXiv ID or URL", placeholder="e.g. 1706.03762"
        )
        if st.button("Fetch from arXiv", use_container_width=True) and arxiv_input:
            key = f"arxiv:{sanitize_user_input(arxiv_input, max_len=200)}"
            with st.spinner("Fetching from arXiv and indexing..."):
                try:
                    pdf_bytes, meta = fetch_arxiv_paper(arxiv_input)
                    name = meta.get("title") or arxiv_input
                    load_paper(pdf_bytes, name, meta)
                    st.session_state.last_loaded_key = key
                    st.success(f"Loaded: {name}")
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    logger.exception("Failed to fetch arXiv paper")
                    st.error("Could not fetch that paper. Check the ID/URL and try again.")

    st.divider()
    st.subheader("2. Settings")
    backend_choice = st.selectbox(
        "Vector search backend",
        ["numpy", "chroma"],
        index=0 if st.session_state.backend == "numpy" else 1,
        help="NumPy is zero-setup and deploy-safe. ChromaDB is a real vector "
             "DB. Changing this applies to the NEXT paper you load.",
    )
    st.session_state.backend = backend_choice

    st.divider()
    st.subheader("3. Study sessions")
    if st.session_state.paper_text and st.button(
        "💾 Save current session", use_container_width=True
    ):
        path = save_session(
            {
                "paper_name": st.session_state.paper_name,
                "paper_text": st.session_state.paper_text,
                "metadata": st.session_state.metadata,
                "summary": st.session_state.summary,
                "flashcards": st.session_state.flashcards,
                "quiz": st.session_state.quiz,
                "gaps": st.session_state.gaps,
                "citations": st.session_state.citations,
                "concept_graph": st.session_state.concept_graph,
                "chat_history": st.session_state.chat_history,
            }
        )
        st.success(f"Saved to {path}")

    saved = list_sessions()
    if saved:
        labels = {s["path"]: f"{s['paper_name'][:30]} · {s['saved_at'][:16]}"
                  for s in saved}
        chosen = st.selectbox(
            "Load a saved session",
            options=["—"] + list(labels.keys()),
            format_func=lambda p: "—" if p == "—" else labels[p],
        )
        if chosen != "—" and st.button("📂 Load session", use_container_width=True):
            with st.spinner("Restoring session and rebuilding index..."):
                try:
                    data = load_session(chosen)
                    store = get_vector_store(st.session_state.backend)
                    store.build(split_text(data["paper_text"]))
                    st.session_state.update(
                        {
                            "paper_text": data["paper_text"],
                            "paper_name": data.get("paper_name"),
                            "metadata": data.get("metadata"),
                            "vector_store": store,
                            "summary": data.get("summary"),
                            "flashcards": data.get("flashcards"),
                            "quiz": data.get("quiz"),
                            "gaps": data.get("gaps"),
                            "citations": data.get("citations"),
                            "concept_graph": data.get("concept_graph"),
                            "chat_history": data.get("chat_history", []),
                            "last_loaded_key": chosen,
                        }
                    )
                    st.success("Session restored.")
                except Exception:
                    logger.exception("Failed to load session")
                    st.error("Could not load that session.")

    if st.session_state.metadata:
        meta = st.session_state.metadata
        st.divider()
        st.markdown(f"**Pages:** {meta.get('pages', '?')}")
        st.markdown(f"**Title:** {meta.get('title', 'Untitled')}")
        st.markdown(f"**Author:** {meta.get('author', 'Unknown')}")


# --- Main area -------------------------------------------------------------
st.title("Research Paper Explainer AI")

if st.session_state.paper_text is None:
    st.info(
        "👈 Load a research paper in the sidebar (upload a PDF or fetch from "
        "arXiv) to get started.\n\nYou'll get a summary, flashcards, a quiz, "
        "research-gap analysis, citations, an interactive concept map, a "
        "multi-agent review, paper comparison, a literature review, and a "
        "grounded Q&A chat."
    )
    st.stop()


tabs = st.tabs([
    "📝 Summary", "🃏 Flashcards", "❓ Quiz", "🔬 Gaps", "💬 Chat",
    "🔗 Citations", "🕸️ Concept Map", "🤖 Multi-Agent",
    "⚖️ Compare", "📚 Lit Review", "📥 Export",
])
(tab_summary, tab_cards, tab_quiz, tab_gaps, tab_chat, tab_cite,
 tab_map, tab_agents, tab_compare, tab_lit, tab_export) = tabs


# --- Summary ---------------------------------------------------------------
with tab_summary:
    st.subheader("Summary")
    st.caption("Explain Like I'm 15 + a section-by-section breakdown.")
    if st.button("Generate summary", key="btn_summary"):
        with st.spinner("Summarizing..."):
            result = run_guarded("summary", generate_summary, st.session_state.paper_text)
            if result is not None:
                st.session_state.summary = result
    if st.session_state.summary:
        st.markdown(st.session_state.summary)


# --- Flashcards ------------------------------------------------------------
with tab_cards:
    st.subheader("Flashcards")
    num_cards = st.slider("How many cards?", 4, 15, 8, key="num_cards")
    if st.button("Generate flashcards", key="btn_cards"):
        with st.spinner("Creating flashcards..."):
            result = run_guarded(
                "flashcards", generate_flashcards,
                st.session_state.paper_text, count=num_cards,
            )
            if result is not None:
                st.session_state.flashcards = result
    if st.session_state.flashcards:
        for i, card in enumerate(st.session_state.flashcards, 1):
            with st.expander(f"{i}. {card['front']}"):
                st.write(card["back"])


# --- Quiz ------------------------------------------------------------------
with tab_quiz:
    st.subheader("Quiz")
    num_q = st.slider("How many questions?", 3, 10, 5, key="num_q")
    if st.button("Generate quiz", key="btn_quiz"):
        with st.spinner("Writing quiz..."):
            result = run_guarded(
                "quiz", generate_quiz,
                st.session_state.paper_text, count=num_q,
            )
            if result is not None:
                st.session_state.quiz = result

    if st.session_state.quiz:
        with st.form("quiz_form"):
            user_answers = []
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                choice = st.radio(
                    "Choose one:",
                    options=list(range(4)),
                    format_func=lambda idx, opts=q["options"]: opts[idx],
                    key=f"quiz_{i}",
                )
                user_answers.append(choice)
            submitted = st.form_submit_button("Submit answers")

        if submitted:
            correct = 0
            for i, q in enumerate(st.session_state.quiz):
                if user_answers[i] == q["answer_index"]:
                    correct += 1
                    st.success(f"Q{i + 1}: Correct! {q['explanation']}")
                else:
                    right = q["options"][q["answer_index"]]
                    st.error(
                        f"Q{i + 1}: Not quite. Correct answer: **{right}**. "
                        f"{q['explanation']}"
                    )
            st.info(f"Score: {correct} / {len(st.session_state.quiz)}")


# --- Research gaps ---------------------------------------------------------
with tab_gaps:
    st.subheader("Research Gaps")
    if st.button("Analyze research gaps", key="btn_gaps"):
        with st.spinner("Thinking like a peer reviewer..."):
            result = run_guarded(
                "gaps", generate_research_gaps, st.session_state.paper_text
            )
            if result is not None:
                st.session_state.gaps = result
    if st.session_state.gaps:
        st.markdown(st.session_state.gaps)


# --- Chat (RAG) ------------------------------------------------------------
with tab_chat:
    st.subheader("Ask the paper")
    st.caption("Answers are grounded in the paper's text via vector search.")
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
    raw_question = st.chat_input("Ask a question about this paper...")
    if raw_question:
        question = sanitize_user_input(raw_question)
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the paper..."):
                answer = run_guarded(
                    "chat", answer_question, question, st.session_state.vector_store
                )
            if answer is None:
                answer = "⚠️ I couldn't answer that right now. Please try again."
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))


# --- Citations -------------------------------------------------------------
with tab_cite:
    st.subheader("Citation Extraction")
    st.caption("Pulls the paper's reference list into a structured table.")
    if st.button("Extract citations", key="btn_cite"):
        with st.spinner("Reading the references section..."):
            result = run_guarded(
                "citations", extract_citations, st.session_state.paper_text
            )
            if result is not None:
                st.session_state.citations = result
    if st.session_state.citations is not None:
        if st.session_state.citations:
            st.dataframe(st.session_state.citations, use_container_width=True)
            st.caption(f"Found {len(st.session_state.citations)} references.")
        else:
            st.warning("No references could be extracted from this paper.")


# --- Concept map -----------------------------------------------------------
with tab_map:
    st.subheader("Interactive Concept Map")
    st.caption("Drag nodes, zoom, and hover edges to explore relationships.")
    if st.button("Generate concept map", key="btn_map"):
        with st.spinner("Mapping the key concepts..."):
            result = run_guarded(
                "concept_map", generate_concept_map, st.session_state.paper_text
            )
            if result is not None:
                st.session_state.concept_graph = result
    if st.session_state.concept_graph:
        graph = st.session_state.concept_graph
        if graph.get("nodes"):
            html = build_concept_map_html(graph)
            components.html(html, height=620, scrolling=True)
        else:
            st.warning("No concepts were extracted to map.")


# --- Multi-agent -----------------------------------------------------------
with tab_agents:
    st.subheader("Multi-Agent Analysis")
    st.caption(
        "Three specialized agents collaborate: Summarizer → Critic → "
        "Gap-Finder, then a Synthesizer writes the verdict."
    )
    if st.button("Run multi-agent analysis", key="btn_agents"):
        status = st.empty()
        result = run_guarded(
            "multi_agent",
            run_multi_agent_analysis,
            st.session_state.paper_text,
            progress=lambda label: status.info(f"🤖 {label}"),
        )
        if result is not None:
            st.session_state.agent_result = result
            status.success("Analysis complete.")

    result = st.session_state.agent_result
    if result:
        st.markdown("### 🧭 Synthesis (final verdict)")
        st.markdown(result.synthesis)
        with st.expander("🧑‍🏫 Summarizer agent"):
            st.markdown(result.summarizer)
        with st.expander("🔍 Critic agent"):
            st.markdown(result.critic)
        with st.expander("💡 Gap-Finder agent"):
            st.markdown(result.gap_finder)


# --- Compare two papers ----------------------------------------------------
with tab_compare:
    st.subheader("Compare Two Papers")
    st.caption(
        f"Paper A is the loaded paper: **{st.session_state.paper_name}**. "
        "Upload a second paper to compare."
    )
    paper_b = st.file_uploader("Upload Paper B (PDF)", type=["pdf"], key="upload_b")
    if paper_b is not None and paper_b.name != st.session_state.paper_b_name:
        with st.spinner("Reading Paper B..."):
            try:
                b_bytes = paper_b.read()
                if len(b_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
                st.session_state.paper_b_text = extract_text_from_pdf(b_bytes)
                st.session_state.paper_b_name = paper_b.name
                st.success(f"Paper B loaded: {paper_b.name}")
            except ValueError as exc:
                st.error(str(exc))

    if st.session_state.paper_b_text and st.button("Compare papers", key="btn_compare"):
        with st.spinner("Comparing..."):
            result = run_guarded(
                "compare", compare_papers,
                st.session_state.paper_text,
                st.session_state.paper_name or "Paper A",
                st.session_state.paper_b_text,
                st.session_state.paper_b_name or "Paper B",
            )
            if result is not None:
                st.session_state.compare_result = result
    if st.session_state.compare_result:
        st.markdown(st.session_state.compare_result)


# --- Literature review -----------------------------------------------------
with tab_lit:
    st.subheader("Literature Review")
    st.caption(
        "Synthesizes the loaded paper (and Paper B from the Compare tab, if "
        "loaded) into a literature-review-style narrative."
    )
    if st.button("Generate literature review", key="btn_lit"):
        papers = [
            {"name": st.session_state.paper_name or "Paper A",
             "text": st.session_state.paper_text}
        ]
        if st.session_state.paper_b_text:
            papers.append(
                {"name": st.session_state.paper_b_name or "Paper B",
                 "text": st.session_state.paper_b_text}
            )
        with st.spinner("Writing the literature review..."):
            result = run_guarded("lit_review", generate_literature_review, papers)
            if result is not None:
                st.session_state.litreview_result = result
    if st.session_state.litreview_result:
        st.markdown(st.session_state.litreview_result)


# --- Export ----------------------------------------------------------------
with tab_export:
    st.subheader("Export notes")
    st.write("Download everything you've generated as Markdown or PDF.")

    notes = build_markdown_notes(
        paper_name=st.session_state.paper_name or "paper",
        summary=st.session_state.summary,
        flashcards=st.session_state.flashcards,
        quiz=st.session_state.quiz,
        gaps=st.session_state.gaps,
        citations=st.session_state.citations,
    )
    stem = (st.session_state.paper_name or "paper").rsplit(".", 1)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Markdown (.md)",
            data=notes,
            file_name=f"{stem}_notes.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        try:
            pdf_bytes = markdown_to_pdf_bytes(notes, title=stem)
            st.download_button(
                "📄 Download PDF (.pdf)",
                data=pdf_bytes,
                file_name=f"{stem}_notes.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            logger.exception("PDF export failed")
            st.warning("PDF export is unavailable right now.")

    with st.expander("Preview notes"):
        st.markdown(notes)
