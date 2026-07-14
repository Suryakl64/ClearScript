"""
Rule-based parser for structured Indian lab reports.
Uses regex patterns + abbreviation normalisation + LOINC lookup.
"""
import re
from typing import Optional

from backend.parser.abbreviations import normalize_test_name
from backend.parser.loinc_lookup import (
    determine_flag,
    flag_color,
    get_loinc_range,
    parse_numeric,
    parse_reference_range,
)


# Common lab row patterns in Indian pathology reports
LAB_ROW_PATTERNS = [
    # Test Name    Value    Unit    Ref Range    Flag
    re.compile(
        r"^(.{2,50}?)\s+([\d.<>]+)\s+(\S+)\s+([\d.\s\-–—<>]+(?:\s*[-–—to]+\s*[\d.]+)?)\s*(H|L|HIGH|LOW)?$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Test Name : Value Unit (Ref: X-Y)
    re.compile(
        r"^(.{2,50}?)\s*[:]\s*([\d.<>]+)\s*(\S+)?\s*(?:\(?\s*(?:Ref\.?|Reference|Normal)\s*[:\s]?\s*([\d.\s\-–—<>]+))?\)?\s*(H|L)?$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Test Name    Value    (Ref Range)
    re.compile(
        r"^(.{2,50}?)\s{2,}([\d.<>]+)\s*(?:\(([\d.\s\-–—]+)\))?\s*(H|L|HIGH|LOW)?$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Simple: Name - Value Unit
    re.compile(
        r"^(.{2,40}?)\s*[-–]\s*([\d.<>]+)\s+(\S+)$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

# Detect if text looks like a structured lab report vs narrative
LAB_SECTION_KEYWORDS = re.compile(
    r"\b(CBC|Complete Blood Count|LFT|Liver Function|KFT|Kidney Function|"
    r"Lipid Profile|Thyroid Profile|HbA1c|Biochemistry|Haematology|"
    r"Hematology|Pathology|Investigation|Test Name|Result|Reference)\b",
    re.IGNORECASE,
)

NARRATIVE_KEYWORDS = re.compile(
    r"\b(discharge summary|diagnosis|chief complaint|history|impression|"
    r"admitted|discharged|patient was|presented with|clinical findings)\b",
    re.IGNORECASE,
)


def detect_report_type(text: str) -> str:
    """Return 'structured_lab', 'narrative', or 'mixed'."""
    lab_score = len(LAB_SECTION_KEYWORDS.findall(text))
    narrative_score = len(NARRATIVE_KEYWORDS.findall(text))
    numeric_rows = len(re.findall(r"^[\w\s./()-]{2,40}\s+[\d.<>]+", text, re.MULTILINE))

    if narrative_score > 2 and lab_score <= 1:
        return "narrative"
    if lab_score >= 1 or numeric_rows >= 3:
        return "structured_lab" if narrative_score == 0 else "mixed"
    return "mixed"


def _extract_flag_from_text(flag_str: Optional[str], value_str: str) -> Optional[str]:
    if flag_str:
        f = flag_str.upper()
        if f in ("H", "HIGH"):
            return "high"
        if f in ("L", "LOW"):
            return "low"
    if value_str.startswith("<"):
        return "low"
    if value_str.startswith(">"):
        return "high"
    return None


def _build_finding(
    raw_name: str,
    value_str: str,
    unit: Optional[str],
    ref_str: Optional[str],
    explicit_flag: Optional[str],
    gender: str = "default",
) -> Optional[dict]:
    norm = normalize_test_name(raw_name)
    value = parse_numeric(value_str)
    if value is None and not value_str:
        return None

    ref_from_report = parse_reference_range(ref_str) if ref_str else None
    ref_from_loinc = get_loinc_range(norm.get("loinc"), gender, unit)

    ref_range = ref_from_report or ref_from_loinc
    if ref_from_report and ref_from_loinc:
        ref_range = {**ref_from_loinc, **ref_from_report, "source": "report+loinc"}

    flag = explicit_flag or determine_flag(value, ref_range, norm.get("loinc"))

    ref_display = ref_str
    if not ref_display and ref_range:
        low, high = ref_range.get("low"), ref_range.get("high")
        if low is not None and high is not None:
            ref_display = f"{low} - {high}"
        elif high is not None:
            ref_display = f"< {high}"
        elif low is not None:
            ref_display = f"> {low}"

    return {
        "test_name": norm["full_name"],
        "canonical_name": norm["canonical"],
        "raw_name": raw_name.strip(),
        "value": value,
        "value_raw": value_str.strip(),
        "unit": unit or (ref_range or {}).get("unit", ""),
        "reference_range": ref_display or "",
        "reference_range_parsed": ref_range,
        "flag": flag,
        "flag_color": flag_color(flag),
        "loinc": norm.get("loinc"),
        "category": norm.get("category"),
        "source": "rule_parser",
    }


def parse_structured_lab(text: str, gender: str = "default") -> list[dict]:
    """Extract lab findings from structured report text."""
    findings: list[dict] = []
    seen: set[str] = set()

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) < 5 or line.startswith("---"):
            continue

        for pattern in LAB_ROW_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue

            groups = match.groups()
            raw_name = groups[0]
            value_str = groups[1]
            unit = groups[2] if len(groups) > 2 else None
            ref_str = None
            flag_str = None

            if len(groups) >= 5:
                ref_str = groups[3]
                flag_str = groups[4]
            elif len(groups) >= 4:
                if groups[3] and groups[3].upper() in ("H", "L", "HIGH", "LOW"):
                    flag_str = groups[3]
                else:
                    ref_str = groups[3]

            explicit_flag = _extract_flag_from_text(flag_str, value_str)
            finding = _build_finding(
                raw_name, value_str, unit, ref_str, explicit_flag, gender
            )
            if finding:
                dedup_key = f"{finding['canonical_name']}|{finding['value_raw']}"
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    findings.append(finding)
            break

    return findings


def parse_report(text: str, gender: str = "default") -> dict:
    """
    Main entry: parse structured lab content from OCR text.
    Returns findings list + report type metadata.
    """
    report_type = detect_report_type(text)
    findings = []

    if report_type in ("structured_lab", "mixed"):
        findings = parse_structured_lab(text, gender)

    return {
        "report_type": report_type,
        "findings": findings,
        "finding_count": len(findings),
        "parser": "rule_based",
    }
