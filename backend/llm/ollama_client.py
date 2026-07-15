"""Local Ollama client — no external API keys needed.

Supports both text-only generation (mistral) and multimodal vision
generation (llava) for direct image understanding.
"""
import base64
import json
import logging
from typing import Optional

import httpx

from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    pass


def check_ollama_available() -> dict:
    """Check if Ollama is running and model is available."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            model_available = any(OLLAMA_MODEL in m for m in models)
            vision_available = any(OLLAMA_VISION_MODEL in m for m in models)
            return {
                "available": True,
                "models": models,
                "target_model": OLLAMA_MODEL,
                "model_ready": model_available,
                "vision_model": OLLAMA_VISION_MODEL,
                "vision_ready": vision_available,
            }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "target_model": OLLAMA_MODEL,
            "model_ready": False,
            "vision_model": OLLAMA_VISION_MODEL,
            "vision_ready": False,
        }


def generate(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """
    Generate text via Ollama /api/generate endpoint.
    Raises OllamaError on failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except httpx.ConnectError:
        raise OllamaError(
            "Ollama is not running. Start it with: ollama serve && ollama pull mistral"
        )
    except httpx.HTTPStatusError as exc:
        raise OllamaError(f"Ollama HTTP error: {exc.response.status_code}")
    except Exception as exc:
        raise OllamaError(str(exc))


def generate_with_image(
    prompt: str,
    image_bytes: bytes,
    model: Optional[str] = None,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """
    Generate text from a prompt + image via Ollama's multimodal API.

    Uses a vision-capable model (e.g., llava) that accepts base64-encoded
    images alongside the text prompt.

    Parameters
    ----------
    prompt : str
        The text instruction (e.g., "Extract medical findings from this image").
    image_bytes : bytes
        Raw image file bytes (JPEG, PNG, etc.).
    model : str, optional
        Ollama model name. Defaults to OLLAMA_VISION_MODEL (llava).
    system : str
        System prompt.
    temperature : float
        Sampling temperature (lower = more deterministic).
    max_tokens : int
        Maximum response length.

    Returns
    -------
    str
        Model's text response.

    Raises
    ------
    OllamaError
        If Ollama is not running or the model is unavailable.
    """
    vision_model = model or OLLAMA_VISION_MODEL

    # Encode image as base64 for the Ollama API
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": vision_model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except httpx.ConnectError:
        raise OllamaError(
            f"Ollama is not running. Start it with: ollama serve && ollama pull {vision_model}"
        )
    except httpx.HTTPStatusError as exc:
        raise OllamaError(f"Ollama HTTP error: {exc.response.status_code}")
    except Exception as exc:
        raise OllamaError(str(exc))


def extract_findings_from_image(image_bytes: bytes) -> dict:
    """
    Extract structured medical findings from an image using Ollama's
    vision model (llava).

    Returns parsed JSON with findings in the standard schema.

    Raises
    ------
    OllamaError
        If Ollama or the vision model is not available.
    """
    from backend.llm.vision_prompts import MEDICAL_REPORT_EXTRACTION_PROMPT

    raw_text = generate_with_image(
        prompt=MEDICAL_REPORT_EXTRACTION_PROMPT,
        image_bytes=image_bytes,
        temperature=0.1,
        max_tokens=4096,
    )

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    try:
        result = json.loads(raw_text)
        return result
    except json.JSONDecodeError:
        logger.error("Ollama vision returned invalid JSON: %s", raw_text[:500])
        raise OllamaError(f"Ollama vision model returned invalid JSON response")
