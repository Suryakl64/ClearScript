"""
NER Pipeline — main entry point.

Takes raw OCR text, detects report type, runs the appropriate parser(s),
merges and deduplicates findings, and returns a clean JSON-serialisable
list of findings.

Unified finding schema::

    {
        "test":       str,     # canonical short name  (e.g. "Hb")
        "value":      float?,  # numeric value or None
        "unit":       str,     # unit string
        "range_low":  float?,  # lower bound of reference range
        "range_high": float?,  # upper bound
        "flag":       str,     # HIGH / LOW / NORMAL / UNKNOWN
        "status":     str,     # alias of flag
        "full_name":  str,     # descriptive name
        "loinc":      str?,    # LOINC code
        "category":   str,     # clinical category
        "source":     str,     # "rule_parser" | "biobert_ner" | "narrative_regex"
    }
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.ner.report_type_detector import detect_report_type
from backend.ner.rule_based_parser import parse_structured_report

logger = logging.getLogger(__name__)

# BioBERT is expensive — import lazily
_biobert_available: Optional[bool] = None


def _try_import_biobert():
    """Attempt to import the BioBERT module; return True if available."""
    global _biobert_available
    if _biobert_available is not None:
        return _biobert_available
    try:
        from backend.ner.biobert_ner import extract_narrative_findings  # noqa: F401
        _biobert_available = True
    except Exception:
        logger.warning("BioBERT NER not available (transformers not installed?)")
        _biobert_available = False
    return _biobert_available


# ---------------------------------------------------------------------------
# Merge / deduplicate
# ---------------------------------------------------------------------------

def _merge_findings(
    rule_findings: list[dict],
    ner_findings: list[dict],
) -> list[dict]:
    """
    Merge findings from the rule-based parser and BioBERT NER.

    Rule-based findings take precedence (they have value/unit/range);
    NER findings are appended only if they are not already covered.
    """
    merged: list[dict] = list(rule_findings)
    seen: set[str] = set()

    for f in rule_findings:
        seen.add(f["test"].lower())
        if f.get("full_name"):
            seen.add(f["full_name"].lower())

    for f in ner_findings:
        key = f["test"].lower()
        if key not in seen:
            seen.add(key)
            merged.append(f)

    return merged


def _normalise_finding(f: dict) -> dict:
    """Ensure every finding has the full schema with defaults."""
    return {
        "test": f.get("test", ""),
        "value": f.get("value"),
        "unit": f.get("unit", ""),
        "range_low": f.get("range_low"),
        "range_high": f.get("range_high"),
        "flag": f.get("flag", "UNKNOWN"),
        "status": f.get("status", f.get("flag", "UNKNOWN")),
        "full_name": f.get("full_name", f.get("test", "")),
        "loinc": f.get("loinc"),
        "category": f.get("category", "unknown"),
        "source": f.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ner_pipeline(
    text: str,
    *,
    force_parser: Optional[str] = None,
    skip_biobert: bool = False,
) -> dict:
    """
    Main NER pipeline entry point.

    Parameters
    ----------
    text : str
        Raw OCR text from Phase 1.
    force_parser : str, optional
        Force a specific parser: ``"structured"``, ``"narrative"``,
        or ``"both"``.  If *None*, the report type is auto-detected.
    skip_biobert : bool
        If *True*, skip the BioBERT NER even for narrative text.
        Useful for fast / CPU-limited environments.

    Returns
    -------
    dict
        {
            "findings": [...],      # unified finding list
            "finding_count": int,
            "report_type": str,     # "structured" | "narrative" | "mixed"
            "parsers_used": [str],  # which parsers ran
            "detection": {...},     # raw detection diagnostics
        }
    """
    if not text or len(text.strip()) < 10:
        return {
            "findings": [],
            "finding_count": 0,
            "report_type": "empty",
            "parsers_used": [],
            "detection": {},
        }

    # ── Step 1: Detect report type ────────────────────────────────────────
    detection = detect_report_type(text)
    report_type = force_parser or detection.report_type
    parsers_used: list[str] = []

    # ── Step 2: Run appropriate parser(s) ─────────────────────────────────
    rule_findings: list[dict] = []
    ner_findings: list[dict] = []

    if report_type in ("structured", "mixed", "both"):
        rule_findings = parse_structured_report(text)
        parsers_used.append("rule_parser")

    if report_type in ("narrative", "mixed", "both") and not skip_biobert:
        if _try_import_biobert():
            from backend.ner.biobert_ner import extract_narrative_findings
            try:
                ner_findings = extract_narrative_findings(text)
                parsers_used.append("biobert_ner")
            except Exception as exc:
                logger.error("BioBERT NER failed: %s", exc)
                parsers_used.append("biobert_ner (failed)")
        else:
            logger.info("Skipping BioBERT NER (not available)")

    # ── Step 3: Merge and normalise ───────────────────────────────────────
    merged = _merge_findings(rule_findings, ner_findings)
    findings = [_normalise_finding(f) for f in merged]

    return {
        "findings": findings,
        "finding_count": len(findings),
        "report_type": detection.report_type,
        "parsers_used": parsers_used,
        "detection": {
            "report_type": detection.report_type,
            "lab_score": detection.lab_score,
            "narrative_score": detection.narrative_score,
            "numeric_row_count": detection.numeric_row_count,
            "confidence": detection.confidence,
        },
    }
