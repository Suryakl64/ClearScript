"""
ClearScript — End-to-End Test (Phase 1 + 2 + 3)

Chains all three phases together:
  Phase 1: Upload image -> OCR -> raw text
  Phase 2: Raw text -> NER -> structured findings
  Phase 3: Structured findings -> Explanations + Summary

Usage:
    python backend/test_e2e.py
    python backend/test_e2e.py path/to/your/report.jpg
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def phase1_ocr(image_path):
    """Phase 1: Extract raw text from image via OCR."""
    print_header("PHASE 1 -- OCR Text Extraction")

    from backend.ocr.extractor import extract_text

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = extract_text(image_bytes, os.path.basename(image_path))

    if "error" in result:
        print(f"  [ERROR] {result['error']}")
        return None

    text = result.get("text", "")
    method = result.get("method", "unknown")
    print(f"  Method:     {method}")
    print(f"  Characters: {len(text)}")
    print(f"  Preview:    {text[:200]}...")
    return text


def phase2_ner(raw_text):
    """Phase 2: Extract structured findings from raw text."""
    print_header("PHASE 2 -- NER Structured Extraction")

    from backend.ner.pipeline import run_ner_pipeline

    result = run_ner_pipeline(raw_text, skip_biobert=True)

    findings = result.get("findings", [])
    report_type = result.get("report_type", "unknown")
    parsers = result.get("parsers_used", [])

    print(f"  Report type:  {report_type}")
    print(f"  Parsers used: {', '.join(parsers)}")
    print(f"  Findings:     {len(findings)}")

    if findings:
        print(f"\n  {'Test':<25} {'Value':>8}  {'Unit':<10} {'Range':<16} {'Flag':<8}")
        print(f"  {'-'*25} {'-'*8}  {'-'*10} {'-'*16} {'-'*8}")
        for f in findings[:15]:
            name = (f.get("full_name") or f.get("test", "?"))[:25]
            val = f.get("value", "")
            unit = (f.get("unit") or "")[:10]
            low = f.get("range_low")
            high = f.get("range_high")
            rng = ""
            if low is not None and high is not None:
                rng = f"{low}-{high}"
            elif high is not None:
                rng = f"<{high}"
            elif low is not None:
                rng = f">{low}"
            flag = f.get("flag", "?")
            print(f"  {name:<25} {str(val):>8}  {unit:<10} {rng:<16} {flag:<8}")
        if len(findings) > 15:
            print(f"  ... and {len(findings) - 15} more findings")

    return findings, report_type


def phase3_explain(findings, report_type):
    """Phase 3: Generate plain-English explanations."""
    print_header("PHASE 3 -- AI Explanations")

    from backend.llm.explainer import explain_all_findings
    from backend.llm.report_summary import generate_report_summary
    from backend.llm.ollama_client import check_ollama_available

    # Check Ollama status
    status = check_ollama_available()
    mode = "[LLM]" if status.get("available") else "[Fallback]"
    print(f"  Mode: {mode}")
    if not status.get("available"):
        print(f"  (Ollama offline -- using rule-based explanations)")
        print(f"  (To enable AI: ollama serve && ollama pull phi3)\n")

    # Explain each finding
    explained = explain_all_findings(findings)

    print(f"\n  --- Individual Explanations ---")
    for e in explained[:10]:
        name = e.get("full_name") or e.get("test", "?")
        flag = e.get("flag", "?")
        avail = "LLM" if e.get("explanation_available") else "Fallback"
        print(f"\n  [{flag:>6}] {name}")
        print(f"  [{avail}] {e.get('explanation', 'N/A')}")

    if len(explained) > 10:
        print(f"\n  ... and {len(explained) - 10} more explanations")

    # Report summary
    summary = generate_report_summary(findings, report_type)
    avail = "LLM" if summary.get("summary_available") else "Fallback"

    print(f"\n  --- Report Summary ---")
    print(f"  [{avail}] {summary['summary']}")
    print(f"\n  Normal:   {summary['normal_count']}")
    print(f"  Abnormal: {summary['abnormal_count']}")
    if summary['abnormal_tests']:
        print(f"  Flagged:  {', '.join(summary['abnormal_tests'][:8])}")
    print(f"\n  Disclaimer: {summary['disclaimer']}")

    return explained, summary


def run_with_sample_data():
    """Run Phase 2+3 with built-in sample data (no image needed)."""
    print_header("USING BUILT-IN SAMPLE DATA (no image)")
    print("  Skipping Phase 1 (OCR) -- using pre-built sample text\n")

    sample_text = """
    COMPLETE BLOOD COUNT (CBC)
    Test                Result      Unit        Reference Range

    Hemoglobin          9.2         g/dL        13.0 - 17.0
    TLC                 12500       /cumm       4000 - 11000
    RBC                 4.1         mill/cumm   4.5 - 5.5
    Platelet Count      185000      /cumm       150000 - 410000
    ESR                 35          mm/hr       0 - 10

    LIVER FUNCTION TEST
    SGPT (ALT)          68          U/L         7 - 56
    SGOT (AST)          52          U/L         5 - 40

    THYROID PROFILE
    TSH                 6.8         mIU/L       0.27 - 4.2

    BLOOD SUGAR
    Fasting Blood Sugar 158         mg/dL       70 - 100

    VITAMINS
    Vitamin D           12.5        ng/mL       30.0 - 100.0
    Vitamin B12         180         pg/mL       211 - 946
    """

    findings, report_type = phase2_ner(sample_text)

    if findings:
        phase3_explain(findings, report_type)
    else:
        print("\n  [!] No findings extracted -- cannot run Phase 3")


def run_with_image(image_path):
    """Run full Phase 1+2+3 pipeline with an image file."""
    if not os.path.exists(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        return

    print(f"  Image: {image_path}")
    print(f"  Size:  {os.path.getsize(image_path) / 1024:.0f} KB")

    # Phase 1
    raw_text = phase1_ocr(image_path)
    if not raw_text or len(raw_text.strip()) < 10:
        print("\n  [!] OCR returned too little text -- cannot continue")
        return

    # Phase 2
    findings, report_type = phase2_ner(raw_text)
    if not findings:
        print("\n  [!] No findings extracted -- cannot run Phase 3")
        return

    # Phase 3
    phase3_explain(findings, report_type)


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  ClearScript -- End-to-End Test (Phase 1 + 2 + 3)")
    print("=" * 72)

    if len(sys.argv) > 1:
        # User provided an image path
        run_with_image(sys.argv[1])
    else:
        # Use built-in sample data
        run_with_sample_data()

    print_header("END-TO-END TEST COMPLETE")
    print()
