"""
Google Gemini Vision API client for direct image-to-JSON extraction.

Uses the free tier of Google AI Studio (Gemini 2.0 Flash).
No billing account needed — just a free API key from:
    https://aistudio.google.com/apikey

Set the API key via environment variable: GEMINI_API_KEY
"""

import json
import logging
from typing import Optional

from backend.config import GEMINI_API_KEY, GEMINI_MODEL
from backend.llm.vision_prompts import MEDICAL_REPORT_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Raised when a Gemini API call fails."""
    pass


def check_gemini_available() -> dict:
    """Check if Gemini API is configured and reachable."""
    if not GEMINI_API_KEY:
        return {
            "available": False,
            "error": "GEMINI_API_KEY not set. Get a free key from https://aistudio.google.com/apikey",
            "model": GEMINI_MODEL,
        }

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Quick test — list models
        models = client.models.list()
        return {
            "available": True,
            "model": GEMINI_MODEL,
            "models_count": len(list(models)),
        }
    except ImportError:
        return {
            "available": False,
            "error": "google-genai package not installed. Run: pip install google-genai",
            "model": GEMINI_MODEL,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "model": GEMINI_MODEL,
        }


def extract_findings_from_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    prompt: Optional[str] = None,
) -> dict:
    """
    Send a medical report image to Gemini Vision and extract structured findings.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file content.
    mime_type : str
        MIME type of the image (e.g., "image/jpeg", "image/png").
    prompt : str, optional
        Custom extraction prompt. Uses the default medical report prompt if None.

    Returns
    -------
    dict
        Parsed JSON result with "patient", "findings", and "overall_status" keys.

    Raises
    ------
    GeminiError
        If the API key is missing, the call fails, or the response is not valid JSON.
    """
    if not GEMINI_API_KEY:
        raise GeminiError(
            "GEMINI_API_KEY not set. Get a free key from https://aistudio.google.com/apikey"
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise GeminiError(
            "google-genai package not installed. Run: pip install google-genai"
        )

    extraction_prompt = prompt or MEDICAL_REPORT_EXTRACTION_PROMPT

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                extraction_prompt,
            ],
        )

        raw_text = response.text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        # Parse JSON
        result = json.loads(raw_text)
        return result

    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s", raw_text[:500])
        raise GeminiError(f"Gemini response was not valid JSON: {exc}")
    except Exception as exc:
        raise GeminiError(f"Gemini API error: {exc}")


def normalize_gemini_findings(gemini_result: dict) -> list[dict]:
    """
    Convert Gemini's extracted findings into the unified ClearScript finding schema.

    Maps Gemini's output format to the standard schema used by the NER pipeline.
    """
    from backend.ner.abbreviations import normalize_test_name

    findings = []
    for f in gemini_result.get("findings", []):
        test_name = f.get("test_name", "")
        if not test_name:
            continue

        # Normalize through the abbreviation dictionary
        norm = normalize_test_name(test_name)

        # Parse numeric value if possible
        value = None
        value_raw = f.get("value", "")
        if value_raw:
            import re
            match = re.search(r"[\d.]+", str(value_raw))
            if match:
                try:
                    value = float(match.group())
                except ValueError:
                    pass

        # Parse reference range
        range_low = None
        range_high = None
        ref = f.get("reference_range", "")
        if ref:
            import re
            range_match = re.search(r"([\d.]+)\s*[-–—to]+\s*([\d.]+)", ref, re.IGNORECASE)
            if range_match:
                range_low = float(range_match.group(1))
                range_high = float(range_match.group(2))
            else:
                lt = re.search(r"<\s*([\d.]+)", ref)
                gt = re.search(r">\s*([\d.]+)", ref)
                if lt:
                    range_high = float(lt.group(1))
                if gt:
                    range_low = float(gt.group(1))

        # Determine flag
        flag = f.get("flag", "UNKNOWN").upper()
        if flag not in ("HIGH", "LOW", "NORMAL", "UNKNOWN"):
            flag = "UNKNOWN"

        findings.append({
            "test": norm.get("canonical", test_name),
            "full_name": norm.get("full_name", test_name),
            "value": value,
            "value_raw": str(value_raw),
            "unit": f.get("unit", ""),
            "range_low": range_low,
            "range_high": range_high,
            "flag": flag,
            "status": flag,
            "loinc": norm.get("loinc"),
            "category": f.get("category", norm.get("category", "unknown")),
            "source": "gemini_vision",
        })

    return findings
