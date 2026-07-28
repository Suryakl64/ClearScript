"""
Phase 3 — Explanation Engine Test Script

Tests:
  1. Batch explanation of sample findings (single LLM call)
  2. Report summary generation
  3. Fallback mode (rule-based) when Ollama is offline

Usage:
    python backend/llm/test_explainer.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.llm.explainer import explain_finding, explain_all_findings, _fallback_explanation
from backend.llm.report_summary import generate_report_summary
from backend.llm.ollama_client import check_ollama_available
from backend.constants import DISCLAIMER_SHORT

# ── Sample findings (Phase 2 unified schema) ─────────────────────────────────
SAMPLE_FINDINGS = [
    {
        "test": "Hb",
        "full_name": "Hemoglobin",
        "value": 9.2,
        "unit": "g/dL",
        "range_low": 13.0,
        "range_high": 17.0,
        "flag": "LOW",
        "status": "LOW",
        "loinc": "718-7",
        "category": "hematology",
        "source": "rule_parser",
    },
    {
        "test": "FBS",
        "full_name": "Fasting Blood Sugar",
        "value": 158.0,
        "unit": "mg/dL",
        "range_low": 70.0,
        "range_high": 100.0,
        "flag": "HIGH",
        "status": "HIGH",
        "loinc": "1558-6",
        "category": "glucose",
        "source": "rule_parser",
    },
    {
        "test": "TSH",
        "full_name": "Thyroid Stimulating Hormone",
        "value": 6.8,
        "unit": "mIU/L",
        "range_low": 0.27,
        "range_high": 4.2,
        "flag": "HIGH",
        "status": "HIGH",
        "loinc": "3016-3",
        "category": "thyroid",
        "source": "rule_parser",
    },
    {
        "test": "PLT",
        "full_name": "Platelet Count",
        "value": 185000.0,
        "unit": "/cumm",
        "range_low": 150000.0,
        "range_high": 410000.0,
        "flag": "NORMAL",
        "status": "NORMAL",
        "loinc": "777-3",
        "category": "hematology",
        "source": "rule_parser",
    },
    {
        "test": "Vitamin D",
        "full_name": "Vitamin D",
        "value": 12.5,
        "unit": "ng/mL",
        "range_low": 30.0,
        "range_high": 100.0,
        "flag": "LOW",
        "status": "LOW",
        "loinc": "1989-3",
        "category": "vitamins",
        "source": "rule_parser",
    },
]


def print_separator(title: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def test_ollama_status():
    """Check if Ollama is running."""
    print_separator("Ollama Status Check")
    status = check_ollama_available()
    print(f"  Available: {status.get('available')}")
    print(f"  Model:     {status.get('target_model')}")
    print(f"  Ready:     {status.get('model_ready')}")
    if status.get("error"):
        print(f"  Error:     {status['error']}")
    return status.get("available", False)


def test_fallback_explanations():
    """Test rule-based fallback explanations (no LLM needed)."""
    print_separator("Fallback Explanations (rule-based, no LLM)")
    for f in SAMPLE_FINDINGS:
        name = f["full_name"]
        explanation = _fallback_explanation(f)
        print(f"\n  [{f['flag']:>6}] {name}: {f['value']} {f['unit']}")
        print(f"          -> {explanation}")
    print(f"\n  Disclaimer: {DISCLAIMER_SHORT}")


def test_batch_explain():
    """Test batch explanation (single LLM call)."""
    print_separator("Batch Explanation (single LLM call)")
    results = explain_all_findings(SAMPLE_FINDINGS)

    for r in results:
        name = r.get("full_name", r.get("test"))
        avail = "[OK] LLM" if r.get("explanation_available") else "[--] Fallback"
        print(f"\n  [{r['flag']:>6}] {name}: {r['value']} {r['unit']}")
        print(f"          [{avail}] {r['explanation']}")

    print(f"\n  Disclaimer: {DISCLAIMER_SHORT}")


def test_report_summary():
    """Test overall report summary."""
    print_separator("Report Summary")
    result = generate_report_summary(SAMPLE_FINDINGS, "structured")

    avail = "[OK] LLM" if result["summary_available"] else "[--] Fallback"
    print(f"\n  [{avail}] Summary:")
    print(f"  {result['summary']}")
    print(f"\n  Normal:   {result['normal_count']}")
    print(f"  Abnormal: {result['abnormal_count']}")
    print(f"  Flagged:  {', '.join(result['abnormal_tests'])}")
    print(f"\n  Disclaimer: {result['disclaimer']}")
    print(f"  Full:       {result['disclaimer_full'][:80]}...")


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  ClearScript Phase 3 — Explanation Engine Test")
    print("=" * 72)

    ollama_running = test_ollama_status()

    # Always test fallback (works without LLM)
    test_fallback_explanations()

    # Test batch explain (uses LLM if available, otherwise fallback)
    test_batch_explain()

    # Test report summary
    test_report_summary()

    print_separator("Test Complete")
    if ollama_running:
        print("  [OK] Ollama is running -- LLM explanations were generated")
    else:
        print("  [--] Ollama is offline -- fallback explanations were used")
        print("    To enable LLM: ollama serve && ollama pull mistral")
    print()
