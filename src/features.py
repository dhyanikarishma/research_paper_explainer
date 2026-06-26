"""The product features, each built on top of the LLM + vector store.

Every feature is a function with a clear input and output. The "magic" is
mostly in the prompts: precise instructions to Gemini produce reliable,
well-structured results. We ask for JSON where we need structured data
(flashcards, quiz) and parse it defensively.
"""

import json
import re

from src.llm import generate_text
from src.vector_store import VectorStore

# To keep prompts within limits and costs low, cap how much paper text we
# feed to "whole document" features. ~30k characters ≈ a long paper's worth
# of signal for summaries/gaps without sending the entire thing.
_MAX_CONTEXT_CHARS = 30_000


def _trim(text: str) -> str:
    return text[:_MAX_CONTEXT_CHARS]


def _extract_json(raw: str):
    """Pull a JSON object/array out of an LLM response.

    Models sometimes wrap JSON in ```json fences or add stray prose. We
    strip fences and grab the first {...} or [...] block, then parse it.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: find the outermost array or object.
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


# --- 1. Summary ------------------------------------------------------------
def generate_summary(paper_text: str) -> str:
    """Return a Markdown summary: 'Explain Like I'm 15' + section-wise breakdown.

    Instead of one blob, we ask the model to detect the paper's real sections
    (Abstract, Introduction, Methods, Results, etc.) and summarize each one.
    This matches how researchers actually read papers.
    """
    prompt = f"""You are an expert research assistant. Summarize the academic
paper below. Respond in clean Markdown with EXACTLY this structure:

## Explain Like I'm 15
A 4-6 sentence plain-English explanation a smart 15-year-old could follow.
No jargon; use a simple analogy if it helps.

## Section-by-Section Summary
Identify the paper's actual sections (e.g. Abstract, Introduction, Related
Work, Method/Approach, Experiments, Results, Discussion, Conclusion). For
EACH section that exists in the paper, add a `### <Section name>` header
followed by 2-4 bullet points capturing its key points. Include concrete
numbers/results where present. Only include sections that actually appear.

## Key Contribution
One short paragraph: the single most important thing this paper adds.

PAPER TEXT:
\"\"\"
{_trim(paper_text)}
\"\"\""""
    return generate_text(prompt, temperature=0.3)


# --- 2. Flashcards ---------------------------------------------------------
def generate_flashcards(paper_text: str, count: int = 8) -> list[dict]:
    """Return a list of {"front": ..., "back": ...} flashcard dicts."""
    prompt = f"""Create {count} study flashcards from the research paper below.
Each flashcard tests one important concept, term, method, or result.

Return ONLY valid JSON: an array of objects with keys "front" (a question or
term) and "back" (a concise answer, 1-3 sentences). No prose, no code fences.

PAPER TEXT:
\"\"\"
{_trim(paper_text)}
\"\"\""""
    data = _extract_json(generate_text(prompt, temperature=0.4))
    # Keep only well-formed cards.
    return [
        {"front": str(c["front"]), "back": str(c["back"])}
        for c in data
        if isinstance(c, dict) and "front" in c and "back" in c
    ]


# --- 3. Quiz ---------------------------------------------------------------
def generate_quiz(paper_text: str, count: int = 5) -> list[dict]:
    """Return multiple-choice questions.

    Each item: {"question", "options": [4 strings], "answer_index", "explanation"}.
    """
    prompt = f"""Create a {count}-question multiple-choice quiz from the paper
below. Test understanding, not trivia.

Return ONLY valid JSON: an array of objects with keys:
  "question": string,
  "options": array of exactly 4 strings,
  "answer_index": integer 0-3 (the index of the correct option),
  "explanation": string (why that answer is correct).
No prose, no code fences.

PAPER TEXT:
\"\"\"
{_trim(paper_text)}
\"\"\""""
    data = _extract_json(generate_text(prompt, temperature=0.4))
    quiz = []
    for q in data:
        if (
            isinstance(q, dict)
            and isinstance(q.get("options"), list)
            and len(q["options"]) == 4
            and isinstance(q.get("answer_index"), int)
        ):
            quiz.append(
                {
                    "question": str(q["question"]),
                    "options": [str(o) for o in q["options"]],
                    "answer_index": q["answer_index"] % 4,
                    "explanation": str(q.get("explanation", "")),
                }
            )
    return quiz


# --- 4. Research gaps ------------------------------------------------------
def generate_research_gaps(paper_text: str) -> str:
    """Return a Markdown analysis of limitations and future-work directions."""
    prompt = f"""You are a critical peer reviewer. Read the paper below and
identify its RESEARCH GAPS. Respond in Markdown with these sections:

## Limitations
What the paper does not address or where its method is weak.

## Open Questions
Important questions the paper leaves unanswered.

## Future Work Ideas
Concrete, promising directions a researcher could pursue next.

Be specific and reference the paper's actual content. PAPER TEXT:
\"\"\"
{_trim(paper_text)}
\"\"\""""
    return generate_text(prompt, temperature=0.5)


# --- 5. Q&A (the RAG chat) -------------------------------------------------
def answer_question(question: str, store: VectorStore) -> str:
    """Answer a question using ONLY retrieved chunks (grounded RAG)."""
    relevant_chunks = store.search(question)
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""Answer the user's question using ONLY the context from the
research paper below. If the answer is not in the context, say so honestly
instead of guessing. Be precise and cite specific details.

CONTEXT FROM THE PAPER:
\"\"\"
{context}
\"\"\"

QUESTION: {question}

ANSWER:"""
    return generate_text(prompt, temperature=0.2)


# --- 7. Citation extraction (V2) ------------------------------------------
def extract_citations(paper_text: str) -> list[dict]:
    """Extract the paper's references as structured citation dicts.

    Each item: {"authors", "title", "year", "venue"}. We ask for JSON and
    parse defensively. We feed the *end* of the paper because references
    almost always live there.
    """
    # References are at the end, so look at the tail of the document.
    tail = paper_text[-_MAX_CONTEXT_CHARS:]
    prompt = f"""Extract the bibliography / reference list from the research
paper text below. Return ONLY valid JSON: an array of objects with keys
"authors" (string), "title" (string), "year" (string), and "venue" (journal
or conference, string; use "" if unknown). Ignore in-text citations; only
list entries from the References/Bibliography section. No prose, no fences.

PAPER TEXT (tail):
\"\"\"
{tail}
\"\"\""""
    try:
        data = _extract_json(generate_text(prompt, temperature=0.1))
    except Exception:
        return []
    citations = []
    for c in data:
        if isinstance(c, dict) and (c.get("title") or c.get("authors")):
            citations.append(
                {
                    "authors": str(c.get("authors", "")),
                    "title": str(c.get("title", "")),
                    "year": str(c.get("year", "")),
                    "venue": str(c.get("venue", "")),
                }
            )
    return citations


# --- 8. Concept map data (V2) ---------------------------------------------
def generate_concept_map(paper_text: str, max_nodes: int = 12) -> dict:
    """Return {"nodes": [...], "edges": [...]} describing key concepts.

    nodes: list of {"id", "label"}.
    edges: list of {"source", "target", "label"} (target ids must exist).
    This data drives the interactive concept-map graph in the UI.
    """
    prompt = f"""Read the research paper below and build a CONCEPT MAP of its
{max_nodes} most important concepts and how they relate.

Return ONLY valid JSON with this exact shape (no prose, no fences):
{{
  "nodes": [{{"id": "short_id", "label": "Concept name"}}],
  "edges": [{{"source": "short_id", "target": "short_id", "label": "relationship"}}]
}}
Rules: ids are short slugs (e.g. "attention"); every edge's source and
target MUST be ids that appear in nodes; keep labels concise.

PAPER TEXT:
\"\"\"
{_trim(paper_text)}
\"\"\""""
    data = _extract_json(generate_text(prompt, temperature=0.3))
    nodes = [
        {"id": str(n["id"]), "label": str(n.get("label", n["id"]))}
        for n in data.get("nodes", [])
        if isinstance(n, dict) and "id" in n
    ]
    valid_ids = {n["id"] for n in nodes}
    edges = [
        {
            "source": str(e["source"]),
            "target": str(e["target"]),
            "label": str(e.get("label", "")),
        }
        for e in data.get("edges", [])
        if isinstance(e, dict)
        and e.get("source") in valid_ids
        and e.get("target") in valid_ids
    ]
    return {"nodes": nodes, "edges": edges}


# --- 9. Compare two papers (V2) -------------------------------------------
def compare_papers(text_a: str, name_a: str, text_b: str, name_b: str) -> str:
    """Return a Markdown comparison of two papers."""
    half = _MAX_CONTEXT_CHARS // 2
    prompt = f"""You are a research analyst. Compare the TWO papers below.
Respond in Markdown with these sections:

## TL;DR
Two sentences: how these papers relate.

## Side-by-Side
A Markdown table comparing them on: Problem, Method, Dataset/Setup, Key
Results, Limitations. Use "{name_a}" and "{name_b}" as the column headers.

## Key Differences
Bullet points of the most important differences.

## Which to read when
Practical guidance on when each paper is more useful.

=== PAPER A: {name_a} ===
\"\"\"
{text_a[:half]}
\"\"\"

=== PAPER B: {name_b} ===
\"\"\"
{text_b[:half]}
\"\"\""""
    return generate_text(prompt, temperature=0.3)


# --- 10. Literature review (V2) -------------------------------------------
def generate_literature_review(papers: list[dict]) -> str:
    """Generate a literature-review-style synthesis of one or more papers.

    `papers` is a list of {"name": str, "text": str}.
    """
    budget = _MAX_CONTEXT_CHARS // max(len(papers), 1)
    blocks = []
    for i, p in enumerate(papers, 1):
        blocks.append(
            f"=== PAPER {i}: {p['name']} ===\n\"\"\"\n{p['text'][:budget]}\n\"\"\""
        )
    joined = "\n\n".join(blocks)

    prompt = f"""You are writing a mini LITERATURE REVIEW that synthesizes the
paper(s) below into a coherent narrative (not separate summaries). Respond in
Markdown with these sections:

## Introduction
The shared research area and why it matters.

## Themes & Approaches
Group the work by theme; compare and contrast methods. Cite papers by name.

## Findings & Trends
What the body of work collectively shows.

## Gaps & Future Directions
What remains open across these works.

## Conclusion
A short synthesis paragraph.

{joined}"""
    return generate_text(prompt, temperature=0.4)


# --- 6. Export to Markdown -------------------------------------------------
def build_markdown_notes(
    paper_name: str,
    summary: str | None,
    flashcards: list[dict] | None,
    quiz: list[dict] | None,
    gaps: str | None,
    citations: list[dict] | None = None,
) -> str:
    """Assemble everything generated so far into one Markdown notes file."""
    parts: list[str] = [f"# Notes: {paper_name}\n"]

    if summary:
        parts.append("## Summary\n\n" + summary + "\n")

    if flashcards:
        parts.append("## Flashcards\n")
        for i, card in enumerate(flashcards, 1):
            parts.append(f"**{i}. {card['front']}**\n\n{card['back']}\n")

    if quiz:
        parts.append("## Quiz\n")
        for i, q in enumerate(quiz, 1):
            parts.append(f"**Q{i}. {q['question']}**\n")
            for j, opt in enumerate(q["options"]):
                marker = " ✅" if j == q["answer_index"] else ""
                parts.append(f"- {chr(65 + j)}. {opt}{marker}")
            if q.get("explanation"):
                parts.append(f"\n_Explanation: {q['explanation']}_\n")

    if gaps:
        parts.append("## Research Gaps\n\n" + gaps + "\n")

    if citations:
        parts.append("## References\n")
        for i, c in enumerate(citations, 1):
            line = f"{i}. {c.get('authors', '')} ({c.get('year', '')}). " \
                   f"*{c.get('title', '')}*. {c.get('venue', '')}".strip()
            parts.append(line)

    return "\n".join(parts)
