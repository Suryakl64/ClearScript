"""Indian medical abbreviation normalisation."""
import json
import re
from functools import lru_cache
from pathlib import Path

from backend.config import DATA_DIR

ABBREV_FILE = DATA_DIR / "abbreviations.json"


@lru_cache(maxsize=1)
def _load_abbreviations() -> dict:
    with open(ABBREV_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_alias_map() -> dict[str, str]:
    """Map every alias (lowercase) → canonical key."""
    abbrevs = _load_abbreviations()
    alias_map: dict[str, str] = {}
    for canonical, info in abbrevs.items():
        alias_map[canonical.lower()] = canonical
        for alias in info.get("aliases", []):
            alias_map[alias.lower()] = canonical
        alias_map[info["full_name"].lower()] = canonical
    return alias_map


@lru_cache(maxsize=1)
def get_alias_map() -> dict[str, str]:
    return build_alias_map()


def normalize_test_name(raw_name: str) -> dict:
    """
    Normalise a raw test name to canonical form.
    Returns {canonical, full_name, loinc, category} or partial match.
    """
    abbrevs = _load_abbreviations()
    alias_map = get_alias_map()
    cleaned = re.sub(r"[^\w\s./()-]", "", raw_name).strip()
    key = cleaned.lower()

    canonical = alias_map.get(key)
    if not canonical:
        # Partial match — e.g. "Haemoglobin (Hb)" contains Hb
        for alias, canon in sorted(alias_map.items(), key=lambda x: -len(x[0])):
            if len(alias) >= 2 and alias in key:
                canonical = canon
                break

    if canonical and canonical in abbrevs:
        info = abbrevs[canonical]
        return {
            "canonical": canonical,
            "full_name": info["full_name"],
            "loinc": info.get("loinc"),
            "category": info.get("category"),
            "matched_from": raw_name,
        }

    return {
        "canonical": cleaned,
        "full_name": cleaned,
        "loinc": None,
        "category": "unknown",
        "matched_from": raw_name,
    }
