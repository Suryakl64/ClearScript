"""
Rule-based regex parser for structured / tabular lab report text.

Extracts: test name, numeric value, unit, reference range (low/high),
and flag (HIGH / LOW / NORMAL) from Indian pathology report formats.

Uses the NER abbreviation dictionary to normalise test names.
"""

import re
from typing import Optional

from backend.ner.abbreviations import normalize_test_name


# ── Regex patterns for common Indian lab report row formats ───────────────────

LAB_ROW_PATTERNS = [
    # ── Pattern 1: tabular with explicit flag ─────────────────────────────
    # TestName   Value   Unit   RefLow-RefHigh   H/L/HIGH/LOW
    re.compile(
        r"^(.{2,50}?)\s+([\d.<>]+)\s+(\S+)\s+"
        r"([\d.\s\-–—<>]+(?:\s*[-–—to]+\s*[\d.]+)?)\s+"
        r"(H|L|HIGH|LOW|NORMAL)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 2: colon-separated ────────────────────────────────────────
    # TestName : Value Unit  (Ref: X-Y)  H/L
    re.compile(
        r"^(.{2,50}?)\s*[:]\s*([\d.<>]+)\s*(\S+)?\s*"
        r"(?:\(?\s*(?:Ref\.?|Reference|Normal|Biological Ref)\s*[:\s]?\s*"
        r"([\d.\s\-–—<>]+)\)?)?\s*(H|L)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 3: value in parenthesised range ───────────────────────────
    # TestName     Value    (Ref Range)   Flag
    re.compile(
        r"^(.{2,50}?)\s{2,}([\d.<>]+)\s*(\S+)?\s*"
        r"(?:\(([\d.\s\-–—]+)\))?\s*(H|L|HIGH|LOW)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 4: dash-separated simple ──────────────────────────────────
    # TestName - Value Unit
    re.compile(
        r"^(.{2,40}?)\s*[-–]\s*([\d.<>]+)\s+(\S+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 5: tab/space-delimited columns (common in EasyOCR output) ─
    # TestName     Value     Unit     Low     High
    re.compile(
        r"^(.{2,45}?)\s{2,}([\d.<>]+)\s{1,}(\S+)\s{1,}"
        r"([\d.]+)\s*[-–—]\s*([\d.]+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_numeric(value_str: str) -> Optional[float]:
    """Parse a numeric lab value, handling '<', '>', commas."""
    if not value_str:
        return None
    cleaned = value_str.strip().replace(",", "")
    match = re.search(r"[\d.]+", cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _parse_reference_range(ref_str: Optional[str]) -> Optional[dict]:
    """
    Parse reference range strings.
    Handles: '13.0 - 17.0', '13-17', '< 200', '> 40', '13.0 to 17.0'.
    Returns {low, high} or a subset.
    """
    if not ref_str:
        return None
    ref = ref_str.strip().replace("–", "-").replace("—", "-")

    # Range:  low – high
    range_match = re.search(
        r"([\d.]+)\s*[-–—to]+\s*([\d.]+)", ref, re.IGNORECASE
    )
    if range_match:
        return {
            "low": float(range_match.group(1)),
            "high": float(range_match.group(2)),
        }

    # Less-than
    lt_match = re.search(r"<\s*([\d.]+)", ref)
    if lt_match:
        return {"low": None, "high": float(lt_match.group(1))}

    # Greater-than
    gt_match = re.search(r">\s*([\d.]+)", ref)
    if gt_match:
        return {"low": float(gt_match.group(1)), "high": None}

    return None


def _determine_flag(
    value: Optional[float],
    ref_range: Optional[dict],
    explicit_flag: Optional[str] = None,
) -> str:
    """
    Determine flag:  HIGH / LOW / NORMAL / UNKNOWN.
    Uses explicit flag from the report text if available; otherwise
    compares value against the reference range.
    """
    if explicit_flag:
        f = explicit_flag.upper()
        if f in ("H", "HIGH"):
            return "HIGH"
        if f in ("L", "LOW"):
            return "LOW"
        if f == "NORMAL":
            return "NORMAL"

    if value is None or not ref_range:
        return "UNKNOWN"

    low = ref_range.get("low")
    high = ref_range.get("high")

    if low is not None and value < low:
        return "LOW"
    if high is not None and value > high:
        return "HIGH"
    return "NORMAL"


def _extract_explicit_flag(flag_str: Optional[str], value_str: str) -> Optional[str]:
    """Extract an explicit flag from the matched regex group or the value prefix."""
    if flag_str:
        f = flag_str.upper()
        if f in ("H", "HIGH"):
            return "HIGH"
        if f in ("L", "LOW"):
            return "LOW"
        if f == "NORMAL":
            return "NORMAL"
    # Infer from value prefix
    if value_str.startswith("<"):
        return "LOW"
    if value_str.startswith(">"):
        return "HIGH"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_structured_report(text: str) -> list[dict]:
    """
    Parse structured / tabular lab report text and extract findings.

    Each finding is a dict:
        test       – canonical test name
        value      – numeric value (float or None)
        unit       – unit string
        range_low  – lower bound of reference range (float or None)
        range_high – upper bound of reference range (float or None)
        flag       – HIGH / LOW / NORMAL / UNKNOWN
        status     – same as flag, kept for schema compatibility
        full_name  – descriptive test name
        loinc      – LOINC code or None
        category   – clinical category
        source     – "rule_parser"
        raw_name   – original text from the report
        raw_line   – original line from the report
    """
    findings: list[dict] = []
    seen: set[str] = set()

    for line in text.split("\n"):
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
            ref_str: Optional[str] = None
            flag_str: Optional[str] = None

            if len(groups) >= 5:
                # Patterns 1/2/3 — check if groups[3] is a range or flag
                candidate_ref = groups[3]
                candidate_flag = groups[4]
                if candidate_ref and candidate_ref.upper() in ("H", "L", "HIGH", "LOW", "NORMAL"):
                    flag_str = candidate_ref
                else:
                    ref_str = candidate_ref
                    flag_str = candidate_flag
            elif len(groups) >= 4:
                candidate = groups[3]
                if candidate and candidate.upper() in ("H", "L", "HIGH", "LOW", "NORMAL"):
                    flag_str = candidate
                else:
                    ref_str = candidate

            # For pattern 5 (5 groups with low/high as separate captures)
            if len(groups) == 5 and groups[3] and groups[4]:
                try:
                    low_val = float(groups[3])
                    high_val = float(groups[4])
                    ref_str = f"{groups[3]}-{groups[4]}"
                    flag_str = None
                except ValueError:
                    pass

            # Normalise test name
            norm = normalize_test_name(raw_name)

            # Parse value and reference range
            value = _parse_numeric(value_str)
            ref_range = _parse_reference_range(ref_str)

            # Determine flag
            explicit_flag = _extract_explicit_flag(flag_str, value_str)
            flag = _determine_flag(value, ref_range, explicit_flag)

            # Deduplicate by canonical name + value
            dedup_key = f"{norm['canonical']}|{value_str.strip()}"
            if dedup_key in seen:
                break
            seen.add(dedup_key)

            findings.append({
                "test": norm["canonical"],
                "full_name": norm["full_name"],
                "value": value,
                "unit": (unit or "").strip(),
                "range_low": ref_range["low"] if ref_range else None,
                "range_high": ref_range["high"] if ref_range else None,
                "flag": flag,
                "status": flag,
                "loinc": norm.get("loinc"),
                "category": norm.get("category", "unknown"),
                "source": "rule_parser",
                "raw_name": raw_name.strip(),
                "raw_line": line,
            })
            break  # first matching pattern wins for this line

    return findings
