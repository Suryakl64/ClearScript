"""
Report type detector for the NER pipeline.

Examines the raw OCR text and classifies the report as one of:
    • ``structured``   – tabular lab report (use rule-based parser)
    • ``narrative``    – free-text (discharge summary, notes — use BioBERT)
    • ``mixed``        – contains both structured tables and narrative prose
                         (run both parsers and merge)

The detection is heuristic-based, using keyword counts, row patterns, and
simple structural signals found in typical Indian pathology lab reports.
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

_LAB_KEYWORDS = re.compile(
    r"\b(CBC|Complete Blood Count|LFT|Liver Function|KFT|Kidney Function|RFT|"
    r"Renal Function|Lipid Profile|Thyroid Profile|HbA1c|Biochemistry|"
    r"Haematology|Hematology|Pathology|Investigation|Test Name|Result|"
    r"Reference|Reference Range|Normal Range|Biological Ref|"
    r"Ref\.? Range|Unit|Lab Report|Blood Report|Test Report|"
    r"Haemogram|Blood Count)\b",
    re.IGNORECASE,
)

_NARRATIVE_KEYWORDS = re.compile(
    r"\b(discharge summary|diagnosis|chief complaint|history of present|"
    r"impression|admitted|discharged|patient was|presented with|"
    r"clinical findings|upon examination|on examination|final diagnosis|"
    r"clinical history|provisional diagnosis|advice on discharge|"
    r"follow up|treatment given|course in hospital)\b",
    re.IGNORECASE,
)

# A line that looks like a lab row:  <text>  <number>  (with 2+ spaces)
_NUMERIC_ROW = re.compile(
    r"^[\w\s./():-]{2,40}\s{2,}[\d.<>]+", re.MULTILINE
)

# Lines containing colon-separated test:value pairs
_COLON_VALUE = re.compile(
    r"^.{2,40}\s*:\s*[\d.<>]+", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class ReportTypeResult:
    """Encapsulates the detection verdict and diagnostic scores."""
    report_type: str          # "structured" | "narrative" | "mixed"
    lab_score: int            # keyword hits for structured lab
    narrative_score: int      # keyword hits for narrative
    numeric_row_count: int    # lines that look like lab rows
    confidence: float         # 0.0–1.0 heuristic confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_report_type(text: str) -> ReportTypeResult:
    """
    Analyse *text* and decide which parser strategy to use.

    Returns a :class:`ReportTypeResult` with the verdict and diagnostics.
    """
    if not text or len(text.strip()) < 20:
        return ReportTypeResult("mixed", 0, 0, 0, 0.0)

    lab_hits = _LAB_KEYWORDS.findall(text)
    narrative_hits = _NARRATIVE_KEYWORDS.findall(text)
    numeric_rows = _NUMERIC_ROW.findall(text)
    colon_rows = _COLON_VALUE.findall(text)

    lab_score = len(lab_hits) + len(colon_rows) // 2
    narrative_score = len(narrative_hits)
    numeric_row_count = len(numeric_rows) + len(colon_rows)

    # Heuristic decision tree
    if narrative_score >= 3 and lab_score <= 1 and numeric_row_count < 3:
        report_type = "narrative"
        confidence = min(1.0, narrative_score / 5)
    elif lab_score >= 2 or numeric_row_count >= 5:
        if narrative_score >= 2:
            report_type = "mixed"
            confidence = 0.6
        else:
            report_type = "structured"
            confidence = min(1.0, (lab_score + numeric_row_count) / 10)
    elif numeric_row_count >= 3:
        report_type = "structured"
        confidence = 0.5
    elif narrative_score >= 1 and numeric_row_count < 2:
        report_type = "narrative"
        confidence = 0.4
    else:
        report_type = "mixed"
        confidence = 0.3

    return ReportTypeResult(
        report_type=report_type,
        lab_score=lab_score,
        narrative_score=narrative_score,
        numeric_row_count=numeric_row_count,
        confidence=round(confidence, 2),
    )
