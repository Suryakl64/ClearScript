"""
Phase 4 -- End-to-End Test Script

Tests the full pipeline:
  Phase 2: NER (sample text -> structured findings)
  Phase 3: Explain (findings -> plain-English explanations)
  Phase 4a: Store (save report to SQLite + index in ChromaDB)
  Phase 4b: Translate (English explanations -> Hindi/Tamil)
  Phase 4c: Chat/RAG (ask a question -> get grounded answer)

Usage:
    python backend/test_phase4.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# ── Sample lab text ──────────────────────────────────────────────────────────
SAMPLE_TEXT = """
COMPLETE BLOOD COUNT (CBC)
Hemoglobin          9.2         g/dL        13.0 - 17.0
Platelet Count      185000      /cumm       150000 - 410000
ESR                 35          mm/hr       0 - 10

LIVER FUNCTION TEST
SGPT (ALT)          68          U/L         7 - 56

THYROID PROFILE
TSH                 6.8         mIU/L       0.27 - 4.2

BLOOD SUGAR
Fasting Blood Sugar 158         mg/dL       70 - 100

VITAMINS
Vitamin D           12.5        ng/mL       30.0 - 100.0
Vitamin B12         180         pg/mL       211 - 946
"""


def test_phase2_ner():
    """Phase 2: Extract structured findings."""
    print_header("PHASE 2 -- NER Extraction")
    from backend.ner.pipeline import run_ner_pipeline

    result = run_ner_pipeline(SAMPLE_TEXT, skip_biobert=True)
    findings = result["findings"]

    print(f"  Report type:  {result['report_type']}")
    print(f"  Findings:     {len(findings)}")
    for f in findings[:5]:
        name = (f.get("full_name") or f["test"])[:30]
        print(f"    {name:<30} {f['value']:>8} {f['unit']:<8} [{f['flag']}]")
    if len(findings) > 5:
        print(f"    ... and {len(findings) - 5} more")
    return findings, result["report_type"]


def test_phase3_explain(findings):
    """Phase 3: Generate explanations."""
    print_header("PHASE 3 -- Explanations")
    from backend.llm.explainer import explain_all_findings

    explained = explain_all_findings(findings)
    for e in explained[:3]:
        name = e.get("full_name") or e["test"]
        avail = "LLM" if e.get("explanation_available") else "Fallback"
        print(f"  [{avail}] {name}: {e['explanation'][:80]}...")
    return explained


def test_phase4a_store(findings, report_type):
    """Phase 4a: Store report in SQLite + index in ChromaDB."""
    print_header("PHASE 4a -- Store Report (SQLite + ChromaDB)")
    from backend.data.db import generate_report_id, save_report
    from backend.rag.chroma_store import index_findings

    report_id = generate_report_id("test_report.jpg")
    print(f"  Report ID: {report_id}")

    # Save to SQLite
    db_result = save_report(
        report_id=report_id,
        filename="test_report.jpg",
        report_type=report_type,
        findings=findings,
        summary="Test report with multiple abnormal findings.",
    )
    print(f"  SQLite:    {db_result.get('status', 'error')}")

    # Index in ChromaDB
    idx_result = index_findings(report_id, findings)
    print(f"  ChromaDB:  {idx_result['findings_indexed']} findings indexed")

    # Verify DB
    from backend.data.db import list_reports
    reports = list_reports()
    print(f"  Total reports in DB: {len(reports)}")

    return report_id


def test_phase4b_translate(findings):
    """Phase 4b: Translate explanations."""
    print_header("PHASE 4b -- Translation (NLLB-200)")
    from backend.translation.translator import check_translator_available

    status = check_translator_available()
    print(f"  Model:  {status['model']}")
    print(f"  Loaded: {status['loaded']}")
    print(f"  Languages: {status['languages']}")

    # Only test translation if not in CI / quick mode
    # The model is 1.2GB so we skip the actual download in quick tests
    print("\n  [SKIP] Actual translation skipped in quick test (1.2GB model download).")
    print("  To test manually: POST /translate with text + target_lang='hi'")
    print("  Or run: python -c \"from backend.translation.translator import translate_text; print(translate_text('Your hemoglobin is low', 'hi'))\"")


def test_phase4c_chat(report_id):
    """Phase 4c: RAG Chat -- ask questions about the report."""
    print_header("PHASE 4c -- RAG Chat")
    from backend.chat.rag_engine import ask_question

    questions = [
        "Is my hemoglobin normal?",
        "What does my TSH level mean?",
        "Which of my results need attention?",
    ]

    for q in questions:
        print(f"\n  Q: {q}")
        result = ask_question(report_id, q)
        mode = "LLM" if result["llm_available"] else "Fallback"
        print(f"  [{mode}] A: {result['answer'][:120]}...")
        print(f"  Relevant findings: {len(result['relevant_findings'])}")

    print(f"\n  Disclaimer: {result['disclaimer']}")


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  ClearScript Phase 4 -- Translation + RAG Chat Test")
    print("=" * 72)

    # Phase 2
    findings, report_type = test_phase2_ner()

    # Phase 3
    explained = test_phase3_explain(findings)

    # Phase 4a: Store
    report_id = test_phase4a_store(explained, report_type)

    # Phase 4b: Translate (status check only -- model download is large)
    test_phase4b_translate(explained)

    # Phase 4c: RAG Chat
    test_phase4c_chat(report_id)

    print_header("PHASE 4 TEST COMPLETE")
    print()
