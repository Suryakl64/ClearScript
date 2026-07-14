"""
Test script for the ClearScript NER pipeline.

Runs the NER pipeline on:
  1. Synthetic structured lab report text (simulates Phase 1 OCR output)
  2. Synthetic narrative discharge summary text
  3. Any real files found in reports_test/ (reuses Phase 1 OCR → NER)

Usage:
    python -m backend.ner.test_ner_pipeline
    # or from project root:
    python backend/ner/test_ner_pipeline.py
"""

import json
import os
import sys

# ── Ensure project root is on sys.path ────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.ner.pipeline import run_ner_pipeline


# ── Colour helpers for terminal output ────────────────────────────────────────

def _c(text: str, code: str) -> str:
    """Wrap *text* in ANSI colour *code*. Falls back to plain on Windows < 10."""
    try:
        if os.name == "nt":
            os.system("")  # enable ANSI on Windows 10+
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text

def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _red(t):    return _c(t, "31")
def _cyan(t):   return _c(t, "36")
def _bold(t):   return _c(t, "1")


def _flag_colour(flag: str) -> str:
    """Return a coloured flag string."""
    f = flag.upper()
    if f in ("HIGH", "CRITICAL_HIGH"):
        return _red(f"^ {f}")
    if f in ("LOW", "CRITICAL_LOW"):
        return _yellow(f"v {f}")
    if f == "NORMAL":
        return _green(f"* {f}")
    return f"? {f}"


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_findings(result: dict, title: str = "NER Pipeline Results"):
    """Pretty-print the NER pipeline result to stdout."""
    print(f"\n{'=' * 72}")
    print(_bold(f"  {title}"))
    print(f"{'=' * 72}")
    print(f"  Report type   : {_cyan(result['report_type'])}")
    print(f"  Parsers used  : {', '.join(result['parsers_used']) or 'none'}")
    print(f"  Findings      : {result['finding_count']}")

    det = result.get("detection", {})
    if det:
        print(f"  Lab score     : {det.get('lab_score', '?')}")
        print(f"  Narrative     : {det.get('narrative_score', '?')}")
        print(f"  Numeric rows  : {det.get('numeric_row_count', '?')}")
        print(f"  Confidence    : {det.get('confidence', '?')}")

    print(f"{'-' * 72}")

    if not result["findings"]:
        print("  (no findings extracted)")
        print(f"{'=' * 72}\n")
        return

    # Table header
    print(f"  {'Test':<28} {'Value':>8}  {'Unit':<10} {'Range':<16} {'Flag':<16} {'Source'}")
    print(f"  {'-' * 28} {'-' * 8}  {'-' * 10} {'-' * 16} {'-' * 16} {'-' * 14}")

    for f in result["findings"]:
        test = f.get("test", "?")[:28]
        val = str(f.get("value", "")) if f.get("value") is not None else "—"
        unit = f.get("unit", "")[:10]
        low = f.get("range_low")
        high = f.get("range_high")
        if low is not None and high is not None:
            ref = f"{low}-{high}"
        elif high is not None:
            ref = f"<{high}"
        elif low is not None:
            ref = f">{low}"
        else:
            ref = "-"
        flag = _flag_colour(f.get("flag", "UNKNOWN"))
        source = f.get("source", "?")

        print(f"  {test:<28} {val:>8}  {unit:<10} {ref:<16} {flag:<16} {source}")

    print(f"{'=' * 72}\n")


# ── Sample texts ──────────────────────────────────────────────────────────────

SAMPLE_STRUCTURED_LAB = """
--- Page 1 ---
PATHOLOGY LAB REPORT
Patient: Demo Patient    Age: 45 yrs   Sex: Male
Date: 01/07/2026          Ref Dr: Dr. Sharma

COMPLETE BLOOD COUNT (CBC)
Test Name                Result      Unit       Reference Range

Haemoglobin              9.2         g/dL       13.0 - 17.0           LOW
Total Leukocyte Count    12500       /cumm      4000 - 11000          HIGH
RBC Count                4.1         mill/cumm  4.5 - 5.5             LOW
PCV                      32          %          40 - 50               LOW
MCV                      78          fL         83 - 101
MCH                      26.5        pg         27 - 31
MCHC                     33.2        g/dL       31.5 - 34.5
Platelet Count           185000      /cumm      150000 - 410000
ESR                      35          mm/hr      0 - 10                HIGH

DIFFERENTIAL COUNT
Neutrophils              72          %          40 - 80
Lymphocytes              20          %          20 - 40
Monocytes                5           %          2 - 10
Eosinophils              2           %          1 - 6
Basophils                1           %          0 - 2

LIVER FUNCTION TESTS (LFT)
SGPT/ALT                 68          U/L        7 - 56                HIGH
SGOT/AST                 52          U/L        5 - 40                HIGH
ALP                      95          U/L        44 - 147
T.Bil                    1.8         mg/dL      0.1 - 1.2             HIGH
D.Bil                    0.5         mg/dL      0.0 - 0.3             HIGH
Albumin                  3.8         g/dL       3.5 - 5.0
Total Protein            7.2         g/dL       6.0 - 8.3

KIDNEY FUNCTION TESTS (KFT)
S.Creat                  1.5         mg/dL      0.7 - 1.3             HIGH
Urea                     48          mg/dL      17 - 43               HIGH
Uric Acid                7.8         mg/dL      3.4 - 7.0             HIGH

THYROID PROFILE
TSH                      6.8         mIU/L      0.27 - 4.2            HIGH

LIPID PROFILE
Total Cholesterol        245         mg/dL      <200                  HIGH
HDL Cholesterol          38          mg/dL      >40                   LOW
LDL Cholesterol          165         mg/dL      <100                  HIGH
Triglycerides            280         mg/dL      <150                  HIGH
VLDL                     42          mg/dL      <30                   HIGH

DIABETES
HbA1c                    8.2         %          <5.7                  HIGH
FBS                      158         mg/dL      70 - 100              HIGH
PPBS                     245         mg/dL      <140                  HIGH

OTHER
Vitamin D                12.5        ng/mL      30 - 100              LOW
Vitamin B12              180         pg/mL      211 - 946             LOW
CRP                      15.2        mg/L       <5.0                  HIGH
Ferritin                 18          ng/mL      20 - 250              LOW
"""

SAMPLE_NARRATIVE = """
DISCHARGE SUMMARY
Patient was admitted with chief complaint of persistent fatigue, shortness
of breath on exertion, and generalised weakness for the past 2 weeks.
History of present illness reveals the patient has type 2 diabetes mellitus
on metformin 500 mg BD and hypothyroidism on levothyroxine 50 mcg OD.

On examination: BP: 130/80 mmHg, Pulse: 88 bpm, SpO2: 96%
Temperature: 98.6°F. Pallor present. No icterus, cyanosis, or oedema.

Investigations revealed Haemoglobin of 9.2 g/dL (low), elevated SGPT at
68 U/L, and TSH of 6.8 mIU/L suggesting suboptimal thyroid control.
HbA1c was 8.2% indicating poor glycaemic control.

Diagnosis: Iron deficiency anaemia, uncontrolled Type 2 DM, subclinical
hypothyroidism.

Treatment: Ferrous sulphate 200 mg TDS, Levothyroxine dose increased to
75 mcg OD, Metformin continued. Advised dietary modifications and follow up
in 4 weeks with repeat CBC, LFT, TSH, and HbA1c.
"""


# ── Test runner ───────────────────────────────────────────────────────────────

def test_structured():
    """Test the pipeline on synthetic structured lab text."""
    result = run_ner_pipeline(SAMPLE_STRUCTURED_LAB, skip_biobert=True)
    print_findings(result, "Structured Lab Report (rule-based)")
    return result


def test_narrative():
    """Test the pipeline on synthetic narrative text (with BioBERT if available)."""
    result = run_ner_pipeline(SAMPLE_NARRATIVE, skip_biobert=True)
    print_findings(result, "Narrative Discharge Summary (skip BioBERT)")
    return result


def test_narrative_with_biobert():
    """Test with BioBERT enabled (slow, downloads model on first run)."""
    result = run_ner_pipeline(SAMPLE_NARRATIVE, skip_biobert=False)
    print_findings(result, "Narrative Discharge Summary (BioBERT)")
    return result


def test_mixed():
    """Test with combined structured + narrative text."""
    mixed = SAMPLE_STRUCTURED_LAB + "\n\n" + SAMPLE_NARRATIVE
    result = run_ner_pipeline(mixed, skip_biobert=True)
    print_findings(result, "Mixed Report (structured + narrative)")
    return result


def test_real_files():
    """Run on real report files from reports_test/ using Phase 1 OCR."""
    test_dir = os.path.join(_project_root, "reports_test")
    if not os.path.isdir(test_dir):
        print(f"\n  reports_test/ not found — skipping real-file tests.\n")
        return

    files = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))
    ]

    if not files:
        print(f"\n  No test files in reports_test/ — skipping.\n")
        return

    try:
        from backend.ocr.extractor import extract_text
    except ImportError as e:
        print(f"\n  OCR dependencies not available ({e}) -- skipping real-file tests.\n")
        return

    for fname in files:
        filepath = os.path.join(test_dir, fname)
        print(f"\n  Processing: {fname} …")

        with open(filepath, "rb") as f:
            file_bytes = f.read()

        ocr_result = extract_text(file_bytes, fname)
        raw_text = ocr_result.get("text", "")

        if not raw_text:
            print(f"  WARNING: No text extracted from {fname}")
            continue

        print(f"  OCR method: {ocr_result['method']}  |  chars: {len(raw_text)}")

        ner_result = run_ner_pipeline(raw_text, skip_biobert=True)
        print_findings(ner_result, f"Real Report: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_bold("\n>> ClearScript NER Pipeline -- Test Suite\n"))

    test_structured()
    test_narrative()
    test_mixed()

    # Uncomment to test BioBERT (requires transformers + model download):
    # test_narrative_with_biobert()

    test_real_files()

    print(_bold("[OK] All NER pipeline tests complete.\n"))
