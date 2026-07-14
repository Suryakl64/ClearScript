"""Plain-English explanations for lab findings via Mistral 7B (Ollama)."""
from backend.constants import DISCLAIMER_SHORT
from backend.llm.ollama_client import OllamaError, generate

SYSTEM_PROMPT = """You are a medical report explainer for Indian patients.
Explain lab results in simple, plain English that a non-medical person can understand.
Rules:
- Be concise (2-4 sentences per finding)
- Do NOT diagnose or prescribe treatment
- Mention if a value is slightly or significantly out of range
- Use everyday language, avoid jargon unless you explain it
- Always remind the reader this is informational only"""


def explain_finding(finding: dict) -> dict:
    """Generate plain-English explanation for a single lab finding."""
    name = finding.get("test_name", "Unknown test")
    value = finding.get("value_raw") or finding.get("value", "N/A")
    unit = finding.get("unit", "")
    ref = finding.get("reference_range", "not provided")
    flag = finding.get("flag", "unknown")

    prompt = f"""Explain this lab test result to a patient in plain English:

Test: {name}
Value: {value} {unit}
Reference Range: {ref}
Status: {flag}

Write a clear, reassuring explanation. Do not diagnose."""

    try:
        explanation = generate(prompt, system=SYSTEM_PROMPT, temperature=0.2)
        return {
            "explanation": explanation,
            "explanation_available": True,
            "disclaimer": DISCLAIMER_SHORT,
        }
    except OllamaError as exc:
        return {
            "explanation": _fallback_explanation(finding),
            "explanation_available": False,
            "error": str(exc),
            "disclaimer": DISCLAIMER_SHORT,
        }


def explain_all_findings(findings: list[dict]) -> list[dict]:
    """Add explanation field to each finding."""
    enriched = []
    for finding in findings:
        result = explain_finding(finding)
        enriched.append({**finding, **result})
    return enriched


def explain_report_summary(findings: list[dict], report_type: str) -> dict:
    """Generate an overall report summary."""
    abnormal = [f for f in findings if f.get("flag") not in ("normal", "unknown", None)]
    normal_count = len(findings) - len(abnormal)

    findings_text = "\n".join(
        f"- {f.get('test_name')}: {f.get('value_raw', f.get('value'))} "
        f"{f.get('unit', '')} [{f.get('flag', '?')}]"
        for f in findings[:20]
    )

    prompt = f"""Summarize this medical report for a patient in plain English.

Report type: {report_type}
Total findings: {len(findings)}
Normal: {normal_count}, Abnormal/flagged: {len(abnormal)}

Findings:
{findings_text}

Write a 3-5 sentence overview. Highlight anything that needs doctor follow-up.
Do not diagnose."""

    try:
        summary = generate(prompt, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=300)
        return {"summary": summary, "summary_available": True}
    except OllamaError as exc:
        return {
            "summary": _fallback_summary(findings, abnormal),
            "summary_available": False,
            "error": str(exc),
        }


def _fallback_explanation(finding: dict) -> str:
    name = finding.get("test_name", "This test")
    flag = finding.get("flag", "unknown")
    if flag == "normal":
        return f"{name} appears to be within the normal range based on available reference values."
    if flag in ("low", "high"):
        return (
            f"{name} is {flag} compared to the reference range. "
            "Please discuss this result with your doctor for proper interpretation."
        )
    if flag.startswith("critical"):
        return (
            f"{name} shows a potentially critical value. "
            "Please contact your healthcare provider promptly."
        )
    return f"{name} result recorded. Please consult your doctor for interpretation."


def _fallback_summary(findings: list, abnormal: list) -> str:
    if not findings:
        return "No structured findings were extracted from this report."
    if not abnormal:
        return f"All {len(findings)} extracted values appear within normal ranges."
    names = ", ".join(f["test_name"] for f in abnormal[:5])
    return (
        f"Found {len(abnormal)} result(s) outside normal range including: {names}. "
        "Please review these with your doctor."
    )
