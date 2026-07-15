"""
Vision extraction API route.

POST /vision/extract — accepts an image or PDF and extracts structured
medical findings using vision AI models (Gemini, Ollama llava, or
fallback to standard OCR + NER pipeline).

Extraction priority:
    1. Gemini Vision (if GEMINI_API_KEY is configured)
    2. Ollama Vision / llava (if Ollama is running with llava model)
    3. Layout-aware OCR + NER pipeline (always available)
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["Vision"])

# MIME type mapping for Gemini
_MIME_MAP = {
    "application/pdf": "application/pdf",
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/webp": "image/webp",
    "image/tiff": "image/tiff",
    "image/bmp": "image/bmp",
}


@router.post("/extract")
async def extract_with_vision(
    file: UploadFile = File(...),
    prefer: Optional[str] = Form(None),
):
    """
    Extract structured medical findings from a report image using
    vision AI models.

    **Upload** a PDF or image (JPG, PNG, etc.).

    **Optional params:**

    - ``prefer`` — force a specific extraction backend:
        ``"gemini"``, ``"ollama"``, or ``"ocr"`` (standard pipeline).
        If not set, the endpoint tries them in priority order.

    **Returns** a JSON body with::

        {
            "success": true,
            "extractor": "gemini_vision" | "ollama_vision" | "ocr_ner_pipeline",
            "patient": {...} | null,
            "findings": [...],
            "finding_count": int,
            "overall_status": str | null
        }
    """
    # Validate file type
    allowed_types = {
        "application/pdf",
        "image/png", "image/jpeg",
        "image/jpg", "image/webp",
        "image/tiff", "image/bmp",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload a PDF or image.",
        )

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20 MB).")

    mime_type = _MIME_MAP.get(file.content_type, "image/jpeg")

    # Determine extraction order based on 'prefer' parameter
    if prefer == "gemini":
        extractors = [_try_gemini]
    elif prefer == "ollama":
        extractors = [_try_ollama_vision]
    elif prefer == "ocr":
        extractors = [_try_ocr_pipeline]
    else:
        extractors = [_try_gemini, _try_ollama_vision, _try_ocr_pipeline]

    # Try extractors in order
    last_error = None
    for extractor_fn in extractors:
        try:
            result = extractor_fn(contents, file.filename, mime_type)
            if result is not None:
                return {"success": True, **result}
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Extractor %s failed: %s", extractor_fn.__name__, exc)
            continue

    # All extractors failed
    raise HTTPException(
        status_code=503,
        detail=f"All extraction methods failed. Last error: {last_error}. "
               f"Ensure Ollama is running or set GEMINI_API_KEY.",
    )


# ---------------------------------------------------------------------------
# Extractor backends
# ---------------------------------------------------------------------------

def _try_gemini(image_bytes: bytes, filename: str, mime_type: str) -> Optional[dict]:
    """Try Gemini Vision extraction."""
    from backend.config import GEMINI_API_KEY
    if not GEMINI_API_KEY:
        logger.info("Gemini skipped — no API key configured")
        return None

    from backend.llm.gemini_client import (
        GeminiError,
        extract_findings_from_image,
        normalize_gemini_findings,
    )

    try:
        raw_result = extract_findings_from_image(image_bytes, mime_type=mime_type)
        findings = normalize_gemini_findings(raw_result)
        return {
            "extractor": "gemini_vision",
            "patient": raw_result.get("patient"),
            "findings": findings,
            "finding_count": len(findings),
            "overall_status": raw_result.get("overall_status"),
        }
    except GeminiError as exc:
        logger.warning("Gemini extraction failed: %s", exc)
        raise


def _try_ollama_vision(image_bytes: bytes, filename: str, mime_type: str) -> Optional[dict]:
    """Try local Ollama vision model (llava) extraction."""
    from backend.llm.ollama_client import check_ollama_available

    status = check_ollama_available()
    if not status.get("available"):
        logger.info("Ollama not running — skipping vision extraction")
        return None
    if not status.get("vision_ready"):
        logger.info("Ollama vision model (llava) not pulled — skipping. "
                     "Run: ollama pull llava")
        return None

    from backend.llm.ollama_client import OllamaError, extract_findings_from_image
    from backend.llm.gemini_client import normalize_gemini_findings

    try:
        raw_result = extract_findings_from_image(image_bytes)
        # Reuse the same normalizer — Ollama returns the same JSON schema
        findings = normalize_gemini_findings(raw_result)
        return {
            "extractor": "ollama_vision",
            "patient": raw_result.get("patient"),
            "findings": findings,
            "finding_count": len(findings),
            "overall_status": raw_result.get("overall_status"),
        }
    except OllamaError as exc:
        logger.warning("Ollama vision extraction failed: %s", exc)
        raise


def _try_ocr_pipeline(image_bytes: bytes, filename: str, mime_type: str) -> Optional[dict]:
    """Fallback: standard OCR + NER pipeline with layout-aware pre-processing."""
    from backend.ocr.extractor import extract_text
    from backend.ner.pipeline import run_ner_pipeline

    ocr_result = extract_text(image_bytes, filename)
    if "error" in ocr_result:
        raise RuntimeError(ocr_result["error"])

    raw_text = ocr_result.get("text", "")
    ner_result = run_ner_pipeline(raw_text, skip_biobert=False)

    return {
        "extractor": "ocr_ner_pipeline",
        "patient": None,
        "findings": ner_result.get("findings", []),
        "finding_count": ner_result.get("finding_count", 0),
        "overall_status": None,
        "report_type": ner_result.get("report_type"),
        "parsers_used": ner_result.get("parsers_used"),
        "ocr_method": ocr_result.get("method"),
    }


@router.get("/status")
def vision_status():
    """Check availability of all vision extraction backends."""
    from backend.llm.gemini_client import check_gemini_available
    from backend.llm.ollama_client import check_ollama_available

    return {
        "gemini": check_gemini_available(),
        "ollama": check_ollama_available(),
        "ocr_pipeline": {"available": True, "note": "Always available as fallback"},
    }
