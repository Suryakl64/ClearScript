"""
RAG Engine — Retrieval-Augmented Generation for medical report Q&A.

Flow:
  1. User asks a question about their report
  2. ChromaDB retrieves the most relevant findings via similarity search
  3. Retrieved context + question are sent to Ollama/Mistral
  4. LLM generates a grounded answer using only the report's data
  5. Medical disclaimer is always included

Falls back to a context-only answer (no LLM) when Ollama is offline.
"""

import logging
from typing import Optional

from backend.constants import DISCLAIMER_SHORT, MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "You are a helpful medical report assistant. Answer the patient's question "
    "using ONLY the lab findings provided below. Rules:\n"
    "- Be concise (3-5 sentences)\n"
    "- Use simple, non-technical language\n"
    "- Do NOT diagnose or prescribe treatment\n"
    "- If the findings don't answer the question, say so honestly\n"
    "- Always remind the patient to consult their doctor\n"
    "- Include relevant values and reference ranges in your answer"
)


def ask_question(
    report_id: str,
    question: str,
    top_k: int = 5,
) -> dict:
    """
    Answer a user question about a specific medical report using RAG.

    Parameters
    ----------
    report_id : str
        The report to search within.
    question : str
        User's question in natural language.
    top_k : int
        Number of relevant findings to retrieve.

    Returns
    -------
    dict
        {
            "answer": str,
            "llm_available": bool,
            "relevant_findings": list[dict],
            "report_id": str,
            "disclaimer": str,
        }
    """
    # Step 1: Retrieve relevant findings from ChromaDB
    from backend.rag.chroma_store import query_report

    relevant = query_report(report_id, question, top_k=top_k)

    if not relevant:
        return {
            "answer": (
                "I couldn't find any relevant findings for your question in this report. "
                "This may mean the report hasn't been indexed yet, or the question "
                "isn't related to the tests in this report. "
                "Please consult your doctor for further guidance."
            ),
            "llm_available": False,
            "relevant_findings": [],
            "report_id": report_id,
            "disclaimer": DISCLAIMER_SHORT,
            "disclaimer_full": MEDICAL_DISCLAIMER,
        }

    # Step 2: Build context from retrieved findings
    context_lines = []
    for i, chunk in enumerate(relevant, 1):
        context_lines.append(f"{i}. {chunk['text']}")
    context_text = "\n".join(context_lines)

    # Step 3: Generate answer via Ollama/Mistral
    prompt = (
        f"Patient's question: {question}\n\n"
        f"Relevant lab findings from their report:\n{context_text}\n\n"
        f"Answer the question using only the findings above. "
        f"Be clear and concise. Do not diagnose."
    )

    try:
        from backend.llm.ollama_client import generate, OllamaError

        answer = generate(
            prompt,
            system=RAG_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=300,
        )

        return {
            "answer": answer,
            "llm_available": True,
            "relevant_findings": _simplify_findings(relevant),
            "report_id": report_id,
            "disclaimer": DISCLAIMER_SHORT,
            "disclaimer_full": MEDICAL_DISCLAIMER,
        }

    except Exception as exc:
        logger.warning("RAG LLM failed (Ollama offline): %s", exc)

        # Fallback: return the retrieved context directly
        fallback = _build_fallback_answer(question, relevant)

        return {
            "answer": fallback,
            "llm_available": False,
            "relevant_findings": _simplify_findings(relevant),
            "report_id": report_id,
            "disclaimer": DISCLAIMER_SHORT,
            "disclaimer_full": MEDICAL_DISCLAIMER,
            "note": "Ollama is offline. Showing retrieved findings directly.",
        }


def _simplify_findings(chunks: list[dict]) -> list[dict]:
    """Simplify ChromaDB results for the API response."""
    simplified = []
    for c in chunks:
        meta = c.get("metadata", {})
        simplified.append({
            "test": meta.get("test", ""),
            "full_name": meta.get("full_name", ""),
            "value": meta.get("value", ""),
            "unit": meta.get("unit", ""),
            "flag": meta.get("flag", ""),
            "category": meta.get("category", ""),
            "relevance": c.get("relevance", 0),
            "context": c.get("text", ""),
        })
    return simplified


def _build_fallback_answer(question: str, relevant: list[dict]) -> str:
    """Build a rule-based answer when Ollama is offline."""
    parts = ["Based on your report, here are the most relevant findings:\n"]

    for i, chunk in enumerate(relevant[:3], 1):
        meta = chunk.get("metadata", {})
        name = meta.get("full_name") or meta.get("test", "Test")
        value = meta.get("value", "N/A")
        unit = meta.get("unit", "")
        flag = meta.get("flag", "")
        parts.append(f"{i}. {name}: {value} {unit} [{flag}]")

    parts.append(
        "\nPlease discuss these results with your doctor for proper interpretation."
    )
    return "\n".join(parts)
