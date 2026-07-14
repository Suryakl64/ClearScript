"""LOINC-based reference range lookup."""
import json
import re
from functools import lru_cache
from typing import Optional

from backend.config import DATA_DIR

LOINC_FILE = DATA_DIR / "loinc_ranges.json"


@lru_cache(maxsize=1)
def _load_loinc() -> dict:
    with open(LOINC_FILE, encoding="utf-8") as f:
        return json.load(f)


def parse_numeric(value_str: str) -> Optional[float]:
    """Parse a numeric lab value, handling '<', '>', and commas."""
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


def parse_reference_range(ref_str: str) -> Optional[dict]:
    """
    Parse reference range strings like '13.0 - 17.0', '13-17', '< 200', '> 40'.
    Returns {low, high} or {low/high only}.
    """
    if not ref_str:
        return None
    ref = ref_str.strip().replace("–", "-").replace("—", "-")

    range_match = re.search(
        r"([\d.]+)\s*[-–—to]+\s*([\d.]+)", ref, re.IGNORECASE
    )
    if range_match:
        return {
            "low": float(range_match.group(1)),
            "high": float(range_match.group(2)),
            "source": "report",
        }

    lt_match = re.search(r"<\s*([\d.]+)", ref)
    if lt_match:
        return {"low": None, "high": float(lt_match.group(1)), "source": "report"}

    gt_match = re.search(r">\s*([\d.]+)", ref)
    if gt_match:
        return {"low": float(gt_match.group(1)), "high": None, "source": "report"}

    return None


def get_loinc_range(
    loinc_code: str,
    gender: str = "default",
    unit: Optional[str] = None,
) -> Optional[dict]:
    """Look up reference range by LOINC code."""
    if not loinc_code:
        return None

    loinc_data = _load_loinc()
    entry = loinc_data.get(loinc_code)
    if not entry:
        return None

    gender_key = gender.lower() if gender.lower() in ("male", "female") else "default"
    range_info = entry.get(gender_key) or entry.get("default")
    if not range_info:
        return None

    result = {
        "low": range_info.get("low"),
        "high": range_info.get("high"),
        "unit": entry.get("unit"),
        "source": "loinc",
        "loinc": loinc_code,
        "name": entry.get("name"),
    }

    if unit and entry.get("unit_alternate"):
        alt_unit = entry["unit_alternate"].lower()
        if unit.lower().replace(" ", "") in alt_unit.replace(" ", "") or alt_unit in unit.lower():
            alt_range = entry.get("default_alternate", range_info)
            result["low"] = alt_range.get("low")
            result["high"] = alt_range.get("high")
            result["unit"] = entry["unit_alternate"]

    result["critical_low"] = entry.get("critical_low")
    result["critical_high"] = entry.get("critical_high")
    return result


def determine_flag(
    value: Optional[float],
    ref_range: Optional[dict],
    loinc_code: Optional[str] = None,
) -> str:
    """
    Determine flag: normal | low | high | critical_low | critical_high | unknown.
    """
    if value is None or not ref_range:
        return "unknown"

    low = ref_range.get("low")
    high = ref_range.get("high")

    critical_low = ref_range.get("critical_low")
    critical_high = ref_range.get("critical_high")

    if loinc_code:
        entry = _load_loinc().get(loinc_code, {})
        critical_low = critical_low or entry.get("critical_low")
        critical_high = critical_high or entry.get("critical_high")

    if critical_low is not None and value <= critical_low:
        return "critical_low"
    if critical_high is not None and value >= critical_high:
        return "critical_high"
    if low is not None and value < low:
        return "low"
    if high is not None and value > high:
        return "high"
    return "normal"


def flag_color(flag: str) -> str:
    """Map flag to dashboard color."""
    if flag in ("critical_low", "critical_high"):
        return "red"
    if flag in ("low", "high"):
        return "yellow"
    if flag == "normal":
        return "green"
    return "gray"
