"""
Rule-based regex parser for structured / tabular lab report text.

Extracts: test name, numeric value, unit, reference range (low/high),
and flag (HIGH / LOW / NORMAL) from Indian pathology report formats.

Uses the NER abbreviation dictionary to normalise test names.
"""

import re
from typing import Optional
import logging

from backend.ner.abbreviations import normalize_test_name

logger = logging.getLogger(__name__)


# ── Regex patterns for common Indian lab report row formats ───────────────────

LAB_ROW_PATTERNS = [
    # ── Pattern 7: TestName Value Flag Range Unit (like Drlogy) ───────────
    re.compile(
        r"^(?P<name>(?:(?!\s{2,}).){2,45}?)\s{2,}(?P<val>(?!\d{8})[\d.<>]+)\s{1,}(?P<flag>H|L|HIGH|LOW|NORMAL|BORDERLINE)\s{1,}(?P<range>[\d.]+\s*(?:[-–—to]|\s+)\s*[\d.]+)\s{1,}(?P<unit>\S+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 8: TestName Value Range Unit (like Drlogy without flag) ───
    re.compile(
        r"^(?P<name>(?:(?!\s{2,}).){2,45}?)\s{2,}(?P<val>(?!\d{8})[\d.<>]+)\s{1,}(?P<range>[\d.]+\s*(?:[-–—to]|\s+)\s*[\d.]+)\s{1,}(?P<unit>\S+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 1: tabular with explicit flag ─────────────────────────────
    re.compile(
        r"^(?P<name>(?:(?!\s{2,}).){2,50}?)\s+(?P<val>(?!\d{8})[\d.<>]+)\s+(?P<unit>\S+)\s+"
        r"(?P<range>[\d.\s\-–—<>]+(?:\s*[-–—to]+\s*[\d.]+)?)\s+"
        r"(?P<flag>H|L|HIGH|LOW|NORMAL)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 2: colon-separated ────────────────────────────────────────
    re.compile(
        r"^(?P<name>(?:(?!\s{2,}).){2,50}?)\s*[:]\s*(?P<val>(?!\d{8})[\d.<>]+)\s*(?P<unit>\S+)?\s*"
        r"(?:\(?\s*(?:Ref\.?|Reference|Normal|Biological Ref)\s*[:\s]?\s*"
        r"(?P<range>[\d.\s\-–—<>]+)\)?)?\s*(?P<flag>H|L)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 3: value in parenthesised range ───────────────────────────
    # Requires parenthesis around range to prevent greedy match
    re.compile(
        r"^(?P<name>(?:(?!\s{2,}).){2,50}?)\s{2,}(?P<val>(?!\d{8})[\d.<>]+)\s*(?P<unit>\S+)?\s*"
        r"\((?P<range>[\d.\s\-–—]+)\)\s*(?P<flag>H|L|HIGH|LOW)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 5: tab/space-delimited columns ────────────────────────────
    # TestName     Value     Unit     Low     High -> map Low-High to range
    re.compile(
        r"^(?P<name>(?:(?!\s{2,}).){2,45}?)\s{2,}(?P<val>(?!\d{8})[\d.<>]+)\s{1,}(?P<unit>\S+)\s{1,}"
        r"(?P<range>[\d.]+\s*(?:[-–—]|\s+)\s*[\d.]+)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 6: single-space or tab-delimited row ──────────────────────
    re.compile(
        r"^(?P<name>(?:(?!\s{2,})[A-Za-z0-9\s()/\-]){2,45}?)\s+(?P<val>(?!\d{8})[\d.<>]+)\s+(?P<unit>[A-Za-z/%\s]{1,15}?)\s+(?P<range>[\d.]+\s*(?:[-–—to]|\s+)\s*[\d.]+)\s*(?P<flag>H|L|HIGH|LOW|NORMAL)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # ── Pattern 4: dash-separated simple (must be last!) ──────────────────
    re.compile(
        r"^(?P<name>(?:(?!\s{2,})[A-Za-z\s()]){2,40}?)\s*[-–]\s*(?P<val>(?!\d{8})[\d.<>]+)\s+(?P<unit>\S+)\s*$",
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
    Handles: '13.0 - 17.0', '13-17', '< 200', '> 40', '13.0 to 17.0', '13.0 17.0'.
    Returns {low, high} or a subset.
    """
    if not ref_str:
        return None
    ref = ref_str.strip().replace("–", "-").replace("—", "-")

    # Range:  low – high or low high
    range_match = re.search(
        r"([\d.]+)\s*(?:[-to]+\s*|\s+)([\d.]+)", ref, re.IGNORECASE
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
    Determine flag:  HIGH / LOW / NORMAL / BORDERLINE / UNKNOWN.
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
        if f in ("BORDERLINE", "BORDER LINE", "BORDER"):
            return "BORDERLINE"

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
        if f in ("BORDERLINE", "BORDER LINE", "BORDER"):
            return "BORDERLINE"
    # Infer from value prefix
    if value_str.startswith("<"):
        return "LOW"
    if value_str.startswith(">"):
        return "HIGH"
    return None


# ---------------------------------------------------------------------------
# Metadata Filters
# ---------------------------------------------------------------------------

_METADATA_IGNORE_PATTERNS = re.compile(
    r"\b(page\s*\d|generated|printed|report\s*date|print\s*date|sample\s*date|"
    r"collected|registered|approved|verified|barcode|mrn|ipd|opd|patient\s*id|"
    r"bill\s*no|invoice|hospital|clinic|doctor|dr\.|ph:\s*\d|tel:\s*\d|"
    r"lab|laboratory|pathology|diagnostics|scan|center|address|phone|email|"
    r"www|\.com|\.org|\.in|\.net|sample|collection|phlebotomist|technician)\b",
    re.IGNORECASE,
)

def _is_metadata_line(line: str, raw_name: str) -> bool:
    """Check if line or raw_name is document metadata (headers/footers/timestamps)."""
    if _METADATA_IGNORE_PATTERNS.search(line) or _METADATA_IGNORE_PATTERNS.search(raw_name):
        return True
    if re.search(r"\b\d{1,2}[-/\s][A-Za-z]{3,9}[-/\s]\d{2,4}\b", raw_name):
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Known differential test names that may lose their value in OCR
_KNOWN_TEST_NAMES = re.compile(
    r"^(?P<name>Eosinophils?|Basophils?|Neutrophils?|Lymphocytes?|Monocytes?|Bands?)\s*$",
    re.IGNORECASE,
)

# Pattern to detect a line that is ONLY a reference range + unit (no value)
_RANGE_ONLY_LINE = re.compile(
    r"^(?P<range>[\d.]+\s*[-–—to]+\s*[\d.]+)\s+(?P<unit>\S+)\s*$",
    re.IGNORECASE,
)


def parse_structured_report(text: str) -> list[dict]:
    """
    Parse structured / tabular lab report text and extract findings.
    """
    findings: list[dict] = []
    seen: set[str] = set()
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if len(line) < 5 or line.startswith("---"):
            continue

        if _METADATA_IGNORE_PATTERNS.search(line):
            continue

        # ── Special case: test name on its own line (OCR dropped its value) ──
        # Detects differential count names (e.g. "Eosinophils") with no numbers.
        name_only = _KNOWN_TEST_NAMES.match(line)
        if name_only and not re.search(r"\d", line):
            # Look ahead one or two lines for a range-only line
            ref_range = None
            unit = ""
            for j in range(i, min(i + 2, len(lines))):
                range_m = _RANGE_ONLY_LINE.match(lines[j].strip())
                if range_m:
                    ref_range = _parse_reference_range(range_m.group("range"))
                    unit = range_m.group("unit")
                    break
            raw_name = name_only.group("name")
            norm = normalize_test_name(raw_name)
            canon = norm["canonical"]
            if canon not in seen:
                seen.add(canon)
                findings.append({
                    "test": canon,
                    "full_name": norm["full_name"],
                    "value": None,
                    "unit": unit,
                    "range_low": ref_range["low"] if ref_range else None,
                    "range_high": ref_range["high"] if ref_range else None,
                    "flag": "UNKNOWN",
                    "status": "UNKNOWN",
                    "loinc": norm.get("loinc"),
                    "category": norm.get("category", "unknown"),
                    "source": "rule_parser",
                    "raw_name": raw_name,
                    "raw_line": line,
                })
            continue

        matched_any = False
        for pattern in LAB_ROW_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue

            group_dict = match.groupdict()

            raw_name = group_dict.get('name')
            if _is_metadata_line(line, raw_name):
                break

            value_str = group_dict.get('val')
            unit = group_dict.get('unit')
            ref_str = group_dict.get('range')
            flag_str = group_dict.get('flag')

            # Normalise test name
            norm = normalize_test_name(raw_name)

            # Parse value and reference range
            value = _parse_numeric(value_str)
            ref_range = _parse_reference_range(ref_str)

            # Determine flag
            explicit_flag = _extract_explicit_flag(flag_str, value_str or "")
            flag = _determine_flag(value, ref_range, explicit_flag)

            # Deduplicate by canonical name
            canon = norm['canonical']
            dedup_key = f"{canon}|{(value_str or '').strip()}"
            if dedup_key in seen:
                break
            seen.add(dedup_key)
            # Also mark test name seen so name-only fallback doesn't double-add
            seen.add(canon)

            findings.append({
                "test": canon,
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
            matched_any = True
            break  # first matching pattern wins for this line

        if not matched_any and len(line) > 10 and re.search(r'\d', line):
            logger.debug(f"Unmatched potential lab row: {line}")

    return findings
