"""Multi-agent analysis: three specialized AI agents collaborate on a paper.

This mirrors how a real research group works:
  1. Summarizer  - explains what the paper says (neutral, faithful).
  2. Critic      - stress-tests it: flaws, weak claims, unsupported results.
  3. Gap-Finder  - proposes concrete future work and open problems.
  4. Synthesizer - merges all three into one balanced verdict.

Each "agent" is just an LLM call with a distinct role/persona and the right
inputs. The orchestrator runs them in sequence, feeding earlier outputs into
later agents (the Critic reads the Summarizer's summary, etc.). This is a
simple, transparent agent pipeline - easy to explain in an interview.
"""

from dataclasses import dataclass

from src.features import _trim
from src.llm import generate_text


@dataclass
class AgentResult:
    """Holds each agent's labeled output for display."""

    summarizer: str
    critic: str
    gap_finder: str
    synthesis: str


def _run_summarizer(paper_text: str) -> str:
    prompt = f"""You are the SUMMARIZER agent. Your only job is to faithfully
explain what this paper claims and does - no opinions, no criticism.

Produce a tight Markdown brief: the problem, the approach, and the main
results (with numbers where given). 150-220 words.

PAPER:
\"\"\"
{_trim(paper_text)}
\"\"\""""
    return generate_text(prompt, temperature=0.2)


def _run_critic(paper_text: str, summary: str) -> str:
    prompt = f"""You are the CRITIC agent: a skeptical, fair peer reviewer.
Using BOTH the paper and the Summarizer's brief, critically evaluate the
work. Cover in Markdown bullets:
- Methodological weaknesses or questionable assumptions
- Claims that seem unsupported or over-stated
- Threats to validity / reproducibility concerns
Be specific and constructive. Do NOT just re-summarize.

SUMMARIZER'S BRIEF:
\"\"\"
{summary}
\"\"\"

PAPER:
\"\"\"
{_trim(paper_text)}
\"\"\""""
    return generate_text(prompt, temperature=0.5)


def _run_gap_finder(paper_text: str, summary: str, critique: str) -> str:
    prompt = f"""You are the GAP-FINDER agent. Using the paper, the summary,
and the critique, propose concrete, promising FUTURE WORK. In Markdown:
- 3-5 specific open problems or extensions
- For each, one line on why it matters and how one might start
Prioritize ideas that address the critique's concerns.

SUMMARY:
\"\"\"
{summary}
\"\"\"

CRITIQUE:
\"\"\"
{critique}
\"\"\""""
    return generate_text(prompt, temperature=0.6)


def _run_synthesizer(summary: str, critique: str, gaps: str) -> str:
    prompt = f"""You are the SYNTHESIZER. Merge the three agents' outputs into
one balanced verdict in Markdown with these sections:

## Verdict
2-3 sentences: is this paper important, and how solid is it?

## Strengths
Bullets.

## Weaknesses
Bullets (from the critique).

## Most Promising Next Step
One concrete recommendation (from the gaps).

SUMMARY:
\"\"\"
{summary}
\"\"\"
CRITIQUE:
\"\"\"
{critique}
\"\"\"
GAPS:
\"\"\"
{gaps}
\"\"\""""
    return generate_text(prompt, temperature=0.3)


def run_multi_agent_analysis(paper_text: str, progress=None) -> AgentResult:
    """Run the full agent pipeline.

    `progress` is an optional callback(step_label: str) so the UI can show
    which agent is currently working.
    """
    def _step(label: str) -> None:
        if progress:
            progress(label)

    _step("Summarizer is reading the paper...")
    summary = _run_summarizer(paper_text)

    _step("Critic is stress-testing the claims...")
    critique = _run_critic(paper_text, summary)

    _step("Gap-Finder is proposing future work...")
    gaps = _run_gap_finder(paper_text, summary, critique)

    _step("Synthesizer is writing the final verdict...")
    synthesis = _run_synthesizer(summary, critique, gaps)

    return AgentResult(
        summarizer=summary,
        critic=critique,
        gap_finder=gaps,
        synthesis=synthesis,
    )
