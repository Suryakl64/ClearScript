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
    "You are a medical lab report assistant.\n\n"
    "Answer the patient's question using ONLY the laboratory findings provided in the context below.\n\n"
    "STRICT SOURCE RULES:\n"
    "1. The provided laboratory findings are the ONLY source of patient-specific information.\n"
    "2. Do not invent, assume, estimate, or infer patient-specific results that are not explicitly present.\n"
    "3. Do not introduce tests, values, diagnoses, symptoms, medications, or medical history that are not present in the provided findings.\n"
    "4. Do not use a 'typical' or 'expected' lab value as a substitute for a missing value.\n"
    "5. If the provided findings do not contain enough information to answer the question, explicitly say that the available report does not provide enough information.\n\n"
    "INTERPRETATION:\n"
    "6. Use the reported result, unit, reference range, and abnormal/borderline status exactly as provided.\n"
    "7. If the report explicitly labels a result as LOW, HIGH, NORMAL, or BORDERLINE, preserve that interpretation.\n"
    "8. Do not silently change or override the laboratory's reference range or interpretation.\n"
    "9. When explaining a result, clearly distinguish between:\n"
    "   - what the report shows\n"
    "   - what that result may indicate\n"
    "   - what cannot be concluded from the available data\n\n"
    "MEDICAL SAFETY:\n"
    "10. Do not diagnose a disease solely from a single laboratory result.\n"
    "11. Do not claim that a result definitely means the patient has or does not have a condition unless the provided findings explicitly establish this.\n"
    "12. For potentially concerning results, recommend discussing them with a qualified healthcare professional rather than giving a definitive diagnosis.\n"
    "13. Do not provide treatment or medication recommendations unless the user specifically asks and the available information supports a safe, appropriately qualified response.\n\n"
    "ANSWERING STYLE:\n"
    "14. Answer the patient's actual question directly and concisely.\n"
    "15. Use simple, patient-friendly language and explain medical terminology when necessary.\n"
    "16. Mention the relevant test name, result, unit, and reference range when useful.\n"
    "17. Do not overwhelm the patient with unrelated findings.\n"
    "18. If multiple findings are relevant, organize them clearly using bullets.\n"
    "19. Never pretend to have information that is not present in the provided findings.\n\n"
    "IMPORTANT: The laboratory report is the source of truth for patient-specific values. "
    "If information is missing, say it is missing instead of guessing."
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
