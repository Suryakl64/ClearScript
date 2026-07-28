"""Overall report summary generator via Mistral 7B (Ollama).

Takes ALL findings from a parsed medical report and produces a short
plain-English summary covering:
  - What's normal
  - What needs attention
  - General guidance to see a doctor

Includes a rule-based fallback when Ollama is offline.
"""

import logging

from backend.constants import DISCLAIMER_SHORT, MEDICAL_DISCLAIMER
from backend.llm.ollama_client import OllamaError, generate

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = (
    "You are a medical report summariser for patients. "
    "Write a brief overall summary of a medical report in plain English.\n"
    "Rules:\n"
    "- 3-5 sentences maximum\n"
    "- Mention how many results are normal vs abnormal\n"
    "- Highlight the most important abnormal findings\n"
    "- Recommend seeing a doctor if anything is flagged\n"
    "- Do NOT diagnose, prescribe, or speculate on conditions\n"
    "- End with a reassuring but honest tone"
)


def generate_report_summary(
    findings: list[dict],
    report_type: str = "structured",
) -> dict:
    """
    Generate an overall plain-English summary for a complete medical report.

    Parameters
    ----------
    findings : list[dict]
        List of structured findings in the Phase 2 unified schema.
    report_type : str
        The detected report type ("structured", "narrative", "mixed").

    Returns
    -------
    dict
        {
            "summary": str,
            "summary_available": bool,
            "normal_count": int,
            "abnormal_count": int,
            "abnormal_tests": [str],
            "disclaimer": str,
            "disclaimer_full": str,
        }
    """
    if not findings:
        return {
            "summary": "No findings were extracted from this report.",
            "summary_available": True,
            "normal_count": 0,
            "abnormal_count": 0,
            "abnormal_tests": [],
            "disclaimer": DISCLAIMER_SHORT,
            "disclaimer_full": MEDICAL_DISCLAIMER,
        }

    # Categorise findings
    abnormal = []
    normal_count = 0
    for f in findings:
        flag = (f.get("flag") or "UNKNOWN").upper()
        if flag in ("HIGH", "LOW", "CRITICAL_HIGH", "CRITICAL_LOW"):
            abnormal.append(f)
        elif flag == "NORMAL":
            normal_count += 1

    abnormal_names = [
        f.get("full_name") or f.get("test", "Unknown") for f in abnormal
    ]

    # Build a compact findings list for the prompt (cap at 25 to save tokens)
    lines = []
    for f in findings[:25]:
        name = f.get("full_name") or f.get("test", "?")
        value = f.get("value", "N/A")
        unit = f.get("unit", "")
        flag = f.get("flag", "?")
        lines.append(f"- {name}: {value} {unit} [{flag}]")
    findings_text = "\n".join(lines)

    prompt = (
        f"Summarise this medical report for a patient.\n\n"
        f"Report type: {report_type}\n"
        f"Total findings: {len(findings)}\n"
        f"Normal: {normal_count}, Abnormal: {len(abnormal)}\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"Write a 3-5 sentence overview. Do not diagnose."
    )

    try:
        summary = generate(
            prompt,
            system=SUMMARY_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=300,
        )
        return {
            "summary": summary,
            "summary_available": True,
            "normal_count": normal_count,
            "abnormal_count": len(abnormal),
            "abnormal_tests": abnormal_names,
            "disclaimer": DISCLAIMER_SHORT,
            "disclaimer_full": MEDICAL_DISCLAIMER,
        }
    except OllamaError as exc:
        logger.warning("Report summary failed (Ollama offline): %s", exc)
        return {
            "summary": _fallback_summary(findings, abnormal, normal_count),
            "summary_available": False,
            "normal_count": normal_count,
            "abnormal_count": len(abnormal),
            "abnormal_tests": abnormal_names,
            "error": str(exc),
            "disclaimer": DISCLAIMER_SHORT,
            "disclaimer_full": MEDICAL_DISCLAIMER,
        }


def _fallback_summary(
    findings: list[dict],
    abnormal: list[dict],
    normal_count: int,
) -> str:
    """Rule-based summary when Ollama is offline."""
    total = len(findings)

    if not abnormal:
        return (
            f"Your report contains {total} test results, and all of them "
            f"appear to be within the normal reference ranges. This is a "
            f"positive sign. Please share this report with your doctor "
            f"during your next visit for a complete assessment."
        )

    names = ", ".join(
        f.get("full_name") or f.get("test", "Unknown")
        for f in abnormal[:5]
    )
    extra = f" and {len(abnormal) - 5} more" if len(abnormal) > 5 else ""

    return (
        f"Your report contains {total} test results. "
        f"{normal_count} are within normal range, but {len(abnormal)} "
        f"result(s) need attention: {names}{extra}. "
        f"Please consult your doctor to review these findings and "
        f"determine if any follow-up is needed."
    )
