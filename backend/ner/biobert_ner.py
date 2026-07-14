"""
BioBERT-based NER for unstructured medical narrative text.

Uses the HuggingFace model ``d4data/biomedical-ner-all`` via the
``transformers`` token-classification pipeline to extract clinical
entities from discharge summaries, doctor notes, and similar narrative
report text.

Entities are post-processed and normalised through the abbreviation
dictionary, then returned in the unified finding schema used by the
NER pipeline.
"""

import re
from typing import Optional

from backend.config import BIOBERT_NER_MODEL
from backend.ner.abbreviations import normalize_test_name

# ── Lazy-loaded singleton ─────────────────────────────────────────────────────
_ner_pipeline = None


def get_ner_pipeline():
    """Load the BioBERT NER pipeline on first call (singleton)."""
    global _ner_pipeline
    if _ner_pipeline is None:
        print(f"Loading BioBERT NER model ({BIOBERT_NER_MODEL}) …")
        from transformers import pipeline as hf_pipeline

        _ner_pipeline = hf_pipeline(
            "token-classification",
            model=BIOBERT_NER_MODEL,
            aggregation_strategy="simple",
            device=-1,  # CPU
        )
        print("BioBERT NER ready.")
    return _ner_pipeline


# Entity labels we care about from biomedical-ner-all
CLINICAL_ENTITY_TYPES = {
    "Disease_disorder",
    "Sign_symptom",
    "Diagnostic_procedure",
    "Therapeutic_procedure",
    "Medication",
    "Biological_structure",
    "Lab_value",
    "Detailed_description",
}

# Max characters to feed into the model per chunk (avoids OOM on long texts)
_MAX_CHUNK = 4000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_len: int = _MAX_CHUNK) -> list[str]:
    """Split text into sentence-boundary-aligned chunks ≤ *max_len* chars."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_len:
            if current:
                chunks.append(current)
            current = sent[:max_len]  # safety truncation
        else:
            current = f"{current} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_clinical_entities(text: str) -> list[dict]:
    """
    Run BioBERT NER on narrative text.

    Returns a list of raw entity dicts::

        {entity, label, score, start, end}
    """
    if not text or len(text.strip()) < 20:
        return []

    try:
        ner = get_ner_pipeline()
        entities: list[dict] = []
        for chunk in _chunk_text(text):
            raw_entities = ner(chunk)
            for ent in raw_entities:
                label = ent.get("entity_group") or ent.get("entity", "")
                entities.append({
                    "entity": ent["word"],
                    "label": label,
                    "score": round(float(ent["score"]), 3),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                })
        return entities
    except Exception as exc:
        return [{"error": str(exc), "entity": "", "label": "ERROR", "score": 0}]


def extract_narrative_findings(text: str) -> list[dict]:
    """
    Convert NER output + keyword heuristics into structured findings
    for discharge summaries and narrative reports.

    Returns a list of dicts in the unified finding schema::

        {test, value, unit, range_low, range_high, flag, status, …}
    """
    entities = extract_clinical_entities(text)
    findings: list[dict] = []

    for ent in entities:
        if ent.get("label") == "ERROR":
            continue
        if ent["label"] not in CLINICAL_ENTITY_TYPES:
            continue
        if ent["score"] < 0.6:
            continue

        entity_text = ent["entity"].strip()
        if len(entity_text) < 3:
            continue

        norm = normalize_test_name(entity_text)

        findings.append({
            "test": norm.get("canonical", entity_text),
            "full_name": norm.get("full_name", entity_text),
            "value": None,
            "unit": "",
            "range_low": None,
            "range_high": None,
            "flag": "UNKNOWN",
            "status": "UNKNOWN",
            "loinc": norm.get("loinc"),
            "category": norm.get("category", "clinical"),
            "ner_label": ent["label"],
            "confidence": ent["score"],
            "source": "biobert_ner",
        })

    # ── Also extract vitals mentioned as "BP: 130/80 mmHg" etc. ───────────
    vitals_pattern = re.compile(
        r"(blood pressure|BP|pulse|heart rate|temperature|SpO2|"
        r"oxygen saturation|respiratory rate|RR)"
        r"\s*[:=]?\s*([\d./\s]+(?:mmHg|bpm|°F|°C|%)?)",
        re.IGNORECASE,
    )
    for match in vitals_pattern.finditer(text):
        findings.append({
            "test": match.group(1).strip().upper(),
            "full_name": match.group(1).strip().title(),
            "value": None,
            "unit": "",
            "range_low": None,
            "range_high": None,
            "flag": "UNKNOWN",
            "status": "UNKNOWN",
            "loinc": None,
            "category": "vitals",
            "source": "narrative_regex",
            "value_raw": match.group(2).strip(),
        })

    # Deduplicate by test name (case-insensitive)
    seen: set[str] = set()
    unique: list[dict] = []
    for f in findings:
        key = f["test"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def process_narrative(text: str) -> dict:
    """Full NER pipeline for unstructured reports (convenience wrapper)."""
    entities = extract_clinical_entities(text)
    findings = extract_narrative_findings(text)
    return {
        "entities": entities,
        "findings": findings,
        "entity_count": len(entities),
        "finding_count": len(findings),
        "parser": "biobert_ner",
    }
