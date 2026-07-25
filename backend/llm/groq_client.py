"""
Groq API client — Llama 3.2 Vision for direct image-to-JSON extraction.

Uses the free tier of Groq Cloud (very fast inference).
Get a free API key from: https://console.groq.com/keys

Set the API key via environment variable: GROQ_API_KEY
"""

import base64
import json
import logging
from typing import Optional

import httpx

from backend.config import GROQ_API_KEY, GROQ_VISION_MODEL
from backend.llm.vision_prompts import MEDICAL_REPORT_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class GroqError(Exception):
    """Raised when a Groq API call fails."""
    pass


def check_groq_available() -> dict:
    """Check if Groq API is configured and reachable."""
    if not GROQ_API_KEY:
        return {
            "available": False,
            "error": "GROQ_API_KEY not set. Get a free key from https://console.groq.com/keys",
            "model": GROQ_VISION_MODEL,
        }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            )
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            vision_ready = any(GROQ_VISION_MODEL in m for m in models)
            return {
                "available": True,
                "model": GROQ_VISION_MODEL,
                "vision_ready": vision_ready,
                "models_count": len(models),
            }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "model": GROQ_VISION_MODEL,
        }


def extract_findings_from_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    prompt: Optional[str] = None,
) -> dict:
    """
    Send a medical report image to Groq (Llama 3.2 Vision) and extract
    structured findings.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file content.
    mime_type : str
        MIME type of the image.
    prompt : str, optional
        Custom extraction prompt. Uses default medical report prompt if None.

    Returns
    -------
    dict
        Parsed JSON result with "patient", "findings", and "overall_status".

    Raises
    ------
    GroqError
        If the API key is missing, the call fails, or response is not valid JSON.
    """
    if not GROQ_API_KEY:
        raise GroqError(
            "GROQ_API_KEY not set. Get a free key from https://console.groq.com/keys"
        )

    extraction_prompt = prompt or MEDICAL_REPORT_EXTRACTION_PROMPT

    # Encode image as base64 data URL
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{img_b64}"

    # Build OpenAI-compatible chat completion request
    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": extraction_prompt,
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_text = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        result = json.loads(raw_text)
        return result

    except json.JSONDecodeError:
        logger.error("Groq returned invalid JSON: %s", raw_text[:500])
        raise GroqError(f"Groq response was not valid JSON")
    except httpx.HTTPStatusError as exc:
        error_body = exc.response.text[:300]
        logger.error("Groq HTTP error %s: %s", exc.response.status_code, error_body)
        raise GroqError(f"Groq API error ({exc.response.status_code}): {error_body}")
    except httpx.ConnectError:
        raise GroqError("Could not connect to Groq API (api.groq.com)")
    except Exception as exc:
        raise GroqError(f"Groq error: {exc}")


def normalize_groq_findings(groq_result: dict) -> list[dict]:
    """
    Convert Groq/Llama's extracted findings into the unified ClearScript
    finding schema.
    """
    import re
    from backend.ner.abbreviations import normalize_test_name

    findings = []
    for f in groq_result.get("findings", []):
        test_name = f.get("test_name", "")
        if not test_name:
            continue

        # Normalize through the abbreviation dictionary
        norm = normalize_test_name(test_name)

        # Parse numeric value if possible
        value = None
        value_raw = f.get("value", "")
        if value_raw:
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
            range_match = re.search(
                r"([\d.]+)\s*[-\u2013\u2014to]+\s*([\d.]+)", ref, re.IGNORECASE
            )
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
            "source": "groq_llama_vision",
        })

    return findings
