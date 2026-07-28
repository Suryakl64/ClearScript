"""Plain-English explanations for lab findings via Mistral 7B (Ollama).

Supports two modes:
  1. Single-finding explanation — `explain_finding(finding)`
  2. Batch explanation — `explain_all_findings(findings)` (single LLM call,
     ~10x faster on CPU)

When Ollama is offline, rule-based fallback explanations are generated
automatically so the API never returns an error.
"""

import json
import logging
import re
from typing import Optional

from backend.constants import DISCLAIMER_SHORT
from backend.llm.ollama_client import OllamaError, generate

logger = logging.getLogger(__name__)

# ── System prompt — kept short for CPU efficiency ─────────────────────────────
SYSTEM_PROMPT = (
    "You are a medical report explainer for patients. "
    "Explain lab results in simple, plain English that a non-medical person "
    "can understand.\n"
    "Rules:\n"
    "- Be concise: 2-3 sentences per finding\n"
    "- Do NOT diagnose or prescribe treatment\n"
    "- Say whether the value is slightly or significantly out of range\n"
    "- Use everyday language; if you must use a medical term, explain it\n"
    "- This is informational only — always remind to consult a doctor"
)


# ---------------------------------------------------------------------------
# Single-finding explanation
# ---------------------------------------------------------------------------

def _format_range(finding: dict) -> str:
    """Build a human-readable reference range string from range_low/high."""
    low = finding.get("range_low")
    high = finding.get("range_high")
    if low is not None and high is not None:
        return f"{low}-{high}"
    if low is not None:
        return f"> {low}"
    if high is not None:
        return f"< {high}"
    return "not provided"


def explain_finding(finding: dict) -> dict:
    """
    Generate a plain-English explanation for a single lab finding.

    Uses the Phase 2 unified schema fields:
      test, value, unit, range_low, range_high, flag, full_name

    Returns a dict with 'explanation', 'explanation_available', 'disclaimer'.
    """
    name = finding.get("full_name") or finding.get("test", "Unknown test")
    value = finding.get("value", "N/A")
    unit = finding.get("unit", "")
    ref = _format_range(finding)
    flag = finding.get("flag", "UNKNOWN")

    prompt = (
        f"Explain this lab result to a patient in 2-3 sentences:\n\n"
        f"Test: {name}\n"
        f"Value: {value} {unit}\n"
        f"Reference Range: {ref}\n"
        f"Status: {flag}\n\n"
        f"Be clear and reassuring. Do not diagnose."
    )

    try:
        explanation = generate(
            prompt, system=SYSTEM_PROMPT, temperature=0.2, max_tokens=150
        )
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


# ---------------------------------------------------------------------------
# Batch explanation — single LLM call for all findings (CPU-efficient)
# ---------------------------------------------------------------------------

def explain_all_findings(findings: list[dict]) -> list[dict]:
    """
    Add an explanation to every finding using a **single** LLM call.

    This is ~10x faster than calling explain_finding() per item because
    Mistral only loads context once.

    Falls back to per-finding rule-based explanations if Ollama is offline.
    """
    if not findings:
        return []

    # Build a numbered list of findings for the prompt
    lines = []
    for i, f in enumerate(findings, 1):
        name = f.get("full_name") or f.get("test", "Unknown")
        value = f.get("value", "N/A")
        unit = f.get("unit", "")
        ref = _format_range(f)
        flag = f.get("flag", "UNKNOWN")
        lines.append(
            f"{i}. {name}: {value} {unit} (range: {ref}, status: {flag})"
        )

    findings_text = "\n".join(lines)

    prompt = (
        f"Here are {len(findings)} lab results. For each one, write a brief "
        f"2-3 sentence explanation in plain English. Number your responses "
        f"to match.\n\n{findings_text}\n\n"
        f"Keep each explanation concise. Do not diagnose."
    )

    try:
        raw = generate(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=min(150 * len(findings), 4096),
        )
        explanations = _parse_numbered_response(raw, len(findings))
    except OllamaError as exc:
        logger.warning("Batch explain failed (Ollama offline): %s", exc)
        explanations = None

    enriched = []
    for i, f in enumerate(findings):
        if explanations and i < len(explanations) and explanations[i]:
            enriched.append({
                **f,
                "explanation": explanations[i],
                "explanation_available": True,
                "disclaimer": DISCLAIMER_SHORT,
            })
        else:
            enriched.append({
                **f,
                "explanation": _fallback_explanation(f),
                "explanation_available": explanations is None and False or False,
                "error": "Ollama offline" if explanations is None else None,
                "disclaimer": DISCLAIMER_SHORT,
            })

    return enriched


def _parse_numbered_response(text: str, count: int) -> list[Optional[str]]:
    """
    Parse a numbered response like:
        1. Hemoglobin is...
        2. WBC count is...
    into a list of strings indexed by finding number.
    """
    results: list[Optional[str]] = [None] * count

    # Split on numbered patterns like "1.", "2.", etc.
    parts = re.split(r"\n(?=\d+[\.\)]\s)", text.strip())

    for part in parts:
        match = re.match(r"(\d+)[\.\)]\s*(.*)", part, re.DOTALL)
        if match:
            idx = int(match.group(1)) - 1  # 0-indexed
            if 0 <= idx < count:
                results[idx] = match.group(2).strip()

    return results


# ---------------------------------------------------------------------------
# Rule-based fallback (always works, no LLM needed)
# ---------------------------------------------------------------------------

def _fallback_explanation(finding: dict) -> str:
    """Generate a simple rule-based explanation when Ollama is offline."""
    name = finding.get("full_name") or finding.get("test", "This test")
    flag = (finding.get("flag") or "UNKNOWN").upper()
    value = finding.get("value")
    unit = finding.get("unit", "")
    ref = _format_range(finding)

    if flag == "NORMAL":
        return (
            f"Your {name} result ({value} {unit}) is within the normal "
            f"reference range ({ref}). This is a good sign."
        )
    if flag == "HIGH":
        return (
            f"Your {name} result ({value} {unit}) is above the normal "
            f"reference range ({ref}). Please discuss this with your doctor "
            f"to understand what it means for your health."
        )
    if flag == "LOW":
        return (
            f"Your {name} result ({value} {unit}) is below the normal "
            f"reference range ({ref}). Please discuss this with your doctor "
            f"to understand what it means for your health."
        )
    if flag.startswith("CRITICAL"):
        return (
            f"Your {name} result ({value} {unit}) is significantly outside "
            f"the normal range. Please contact your healthcare provider promptly."
        )
    return (
        f"Your {name} result is {value} {unit}. "
        f"Please consult your doctor for proper interpretation."
    )
