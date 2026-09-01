"""
CBC Regression Test — verifies extraction accuracy against the Drlogy CBC sample report.

Run with:
    python -m pytest tests/test_cbc_regression.py -v

The test uses the raw OCR text that EasyOCR produces from sample1.jpg.
This is the gold standard for the OCR path. Gemini path is tested separately.
"""

import sys
import os

# Ensure backend is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ner.pipeline import run_ner_pipeline

# ── The exact OCR text EasyOCR produces from sample1.jpg ─────────────────────
SAMPLE_CBC_OCR = """DRLOGY PATHOLOGY LAB   0123456789   0912345678
Accurate   Caring   Instant   drlogypathlab@drlogy.com
105 -108, SMART VISION COMPLEX, HEALTHCARE ROAD, OPPOSITE HEALTHCARE COMPLEX. MUMBAI   689578
WWW.drlogy com
Yash M. Patel   Sample Collected At:
Age   21 Years   125, Shivam Bungalow, S G Road,
Mumbai   Registered on: 02.31 PM 02 Dec, 2X
Sex   Male   Collected on: 03.11 PM 02 Dec, 2X
PID   555   Ref. By: Dr. Hiren Shah   Reported on: 04.35 PM 02 Dec, 2X
Complete Blood Count (CBC)
Investigation   Result   Reference Value   Unit
Primary Sample Type   Blood
HEMOGLOBIN
Hemoglobin (Hb)   12.5   Low   13.0 - 17.0   g/dL
RBC COUNT
Total RBC count   5.2   4.5   5.5   mill/cumm
BLOOD INDICES
Packed Cell Volume (PCV)   57.5   High   40 - 50   %
Mean Corpuscular Volume (MCV)   87.75   83   101   fL
Calculated
MCH   27.2   27   32   pg
Calculated
MCHC   32.8   32.5 - 34.5   g/dL
Calculated
RDW   13.6   11.6   14.0   %
count
4000-11000
DIFFERENTIAL WBC COUNT
Neutrophils   60   50 - 62   %
Lymphocytes   31   20 - 40   %
Eosinophils
00 - 06   %
Monocytes   7   00   10   %
Basophils
00   02   %
PLATELET COUNT
Platelet Count   150000   Borderline   150000   410000   cumm
Instruments: Fully automated cell counter   Mindray 300
Interpretation: Further confirm for Anemia
Thanks for Reference
Medical Lab Technician   Dr. Payal Shah   Dr. Vimal Shah
(DMLT, BMLT)   (MD, Pathologist)   (MD, Pathologist)
Generated on   02 Dec, 202X 05.00 PM   Page 1 of 1
"""

# ── Tests that must NOT appear (hallucinations) ───────────────────────────────
HALLUCINATED_TESTS = {
    "lft", "kft", "rft", "esr", "alt", "ast", "sgpt", "sgot",
    "hba1c", "fbs", "ppbs", "rbs", "fasting blood sugar",
    "post prandial blood sugar", "random blood sugar",
    "overall fitness", "fitness status", "malaria", "mp",
}


def get_findings():
    result = run_ner_pipeline(SAMPLE_CBC_OCR)
    return result["findings"]


def test_no_hallucinations():
    """No hallucinated tests should appear in CBC output."""
    import re
    findings = get_findings()
    test_names_lower = {f["test"].lower() for f in findings}
    test_names_lower |= {f.get("full_name", "").lower() for f in findings}

    for hallucinated in HALLUCINATED_TESTS:
        # Use word boundaries to avoid false positives like 'mp' inside 'lymphocytes'
        pattern = re.compile(rf"\b{re.escape(hallucinated)}\b")
        matching = [n for n in test_names_lower if pattern.search(n)]
        assert not matching, (
            f"Hallucinated test '{hallucinated}' found in output: {matching}"
        )


def test_hemoglobin_low():
    """Hemoglobin should be 12.5 with flag=LOW."""
    findings = get_findings()
    hb = next((f for f in findings if f["test"] in ("Hb", "HGB", "Hemoglobin")), None)
    assert hb is not None, "Hemoglobin finding missing"
    assert hb["value"] == 12.5, f"Hb value wrong: {hb['value']}"
    assert hb["flag"] == "LOW", f"Hb flag wrong: {hb['flag']}"
    assert hb["range_low"] == 13.0, f"Hb range_low wrong: {hb['range_low']}"
    assert hb["range_high"] == 17.0, f"Hb range_high wrong: {hb['range_high']}"


def test_pcv_high():
    """PCV should be 57.5 with flag=HIGH."""
    findings = get_findings()
    pcv = next((f for f in findings if f["test"] == "PCV"), None)
    assert pcv is not None, "PCV finding missing"
    assert pcv["value"] == 57.5, f"PCV value wrong: {pcv['value']}"
    assert pcv["flag"] == "HIGH", f"PCV flag wrong: {pcv['flag']}"


def test_platelet_borderline():
    """Platelet Count should be 150000 with flag=BORDERLINE."""
    findings = get_findings()
    plt = next((f for f in findings if f["test"] in ("PLT", "Platelet", "Platelet Count")), None)
    assert plt is not None, "Platelet Count finding missing"
    assert plt["value"] == 150000.0, f"PLT value wrong: {plt['value']}"
    assert plt["flag"] == "BORDERLINE", (
        f"PLT flag wrong: expected BORDERLINE got {plt['flag']}"
    )


def test_rbc_normal():
    findings = get_findings()
    rbc = next((f for f in findings if f["test"] in ("RBC",)), None)
    assert rbc is not None, "RBC finding missing"
    assert rbc["value"] == 5.2
    assert rbc["flag"] == "NORMAL"


def test_mcv_normal():
    findings = get_findings()
    mcv = next((f for f in findings if f["test"] == "MCV"), None)
    assert mcv is not None, "MCV finding missing"
    assert mcv["value"] == 87.75


def test_mch():
    findings = get_findings()
    mch = next((f for f in findings if f["test"] == "MCH"), None)
    assert mch is not None, "MCH finding missing"
    assert mch["value"] == 27.2


def test_mchc():
    findings = get_findings()
    mchc = next((f for f in findings if f["test"] == "MCHC"), None)
    assert mchc is not None, "MCHC finding missing"
    assert mchc["value"] == 32.8


def test_rdw():
    findings = get_findings()
    rdw = next((f for f in findings if f["test"] == "RDW"), None)
    assert rdw is not None, "RDW finding missing"
    assert rdw["value"] == 13.6


def test_neutrophils():
    findings = get_findings()
    n = next((f for f in findings if f["test"] == "Neutrophils"), None)
    assert n is not None, "Neutrophils finding missing"
    assert n["value"] == 60.0
    assert n["flag"] == "NORMAL"


def test_lymphocytes():
    findings = get_findings()
    n = next((f for f in findings if f["test"] == "Lymphocytes"), None)
    assert n is not None, "Lymphocytes finding missing"
    assert n["value"] == 31.0


def test_monocytes():
    findings = get_findings()
    n = next((f for f in findings if f["test"] == "Monocytes"), None)
    assert n is not None, "Monocytes finding missing"
    assert n["value"] == 7.0


def test_eosinophils_present():
    """Eosinophils must appear in output (OCR drops value, but test should still appear with value=None)."""
    findings = get_findings()
    eos = next((f for f in findings if "eosinophil" in f["test"].lower() or
                "eosinophil" in f.get("full_name", "").lower()), None)
    assert eos is not None, "Eosinophils completely missing from output"
    # Value may be None (OCR limitation) — that's acceptable
    print(f"Eosinophils: value={eos['value']}, flag={eos['flag']}")


def test_basophils_present():
    """Basophils must appear in output."""
    findings = get_findings()
    baso = next((f for f in findings if "basophil" in f["test"].lower() or
                 "basophil" in f.get("full_name", "").lower()), None)
    assert baso is not None, "Basophils completely missing from output"
    print(f"Basophils: value={baso['value']}, flag={baso['flag']}")


def test_minimum_finding_count():
    """At least 11 findings should be extracted from the OCR path."""
    findings = get_findings()
    assert len(findings) >= 11, (
        f"Too few findings: expected at least 11, got {len(findings)}.\n"
        f"Tests found: {[f['test'] for f in findings]}"
    )


def test_no_phone_numbers_as_values():
    """No finding should have a value >= 10_000_000 (phone number length)."""
    findings = get_findings()
    for f in findings:
        v = f.get("value")
        if v is not None:
            assert v < 10_000_000, (
                f"Suspicious value {v} for test '{f['test']}' — looks like a phone number"
            )


if __name__ == "__main__":
    # Quick manual runner
    findings = get_findings()
    print(f"\nTotal findings: {len(findings)}\n")
    for f in findings:
        print(f"  {f['test']:20s} | value={str(f['value']):10s} | flag={f['flag']:12s} | unit={f['unit']}")
