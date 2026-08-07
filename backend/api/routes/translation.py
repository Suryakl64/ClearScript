"""
Translation API route.

POST /translate       -- translate plain text to an Indian language
POST /translate/findings -- translate explanation fields in findings
GET  /translate/status   -- check translator availability & supported languages
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translate", tags=["Translation"])


# ── Request / Response models ─────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str = Field(..., description="English text to translate")
    target_lang: str = Field(..., description="Target language code: hi, ta, kn, te")


class TranslateResponse(BaseModel):
    success: bool
    translated: str
    source_lang: str
    target_lang: str
    language_label: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None


class FindingInput(BaseModel):
    test: str = ""
    value: Optional[float] = None
    unit: str = ""
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    flag: str = "UNKNOWN"
    full_name: str = ""
    category: str = "unknown"
    explanation: str = ""
    explanation_available: bool = False


class TranslateFindingsRequest(BaseModel):
    findings: list[FindingInput]
    target_lang: str = Field(..., description="Target language code: hi, ta, kn, te")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=TranslateResponse)
def translate_text_route(req: TranslateRequest):
    """
    Translate English text to an Indian language.

    Supported languages: Hindi (hi), Tamil (ta), Kannada (kn), Telugu (te).
    """
    if req.target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{req.target_lang}'. "
                   f"Supported: {', '.join(k for k in SUPPORTED_LANGUAGES if k != 'en')}",
        )

    from backend.translation.translator import translate_text

    result = translate_text(req.text, req.target_lang)

    return {
        "success": "error" not in result,
        **result,
    }


@router.post("/findings")
def translate_findings_route(req: TranslateFindingsRequest):
    """
    Translate explanation fields in a findings array.

    Returns the same findings with an added `explanation_translated` field.
    """
    if req.target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{req.target_lang}'.",
        )

    from backend.translation.translator import translate_findings

    findings_dicts = [f.model_dump() for f in req.findings]
    translated = translate_findings(findings_dicts, req.target_lang)

    return {
        "success": True,
        "target_lang": req.target_lang,
        "language_label": SUPPORTED_LANGUAGES[req.target_lang]["label"],
        "findings": translated,
        "finding_count": len(translated),
    }


@router.get("/status")
def translation_status():
    """Check translator availability and supported languages."""
    from backend.translation.translator import check_translator_available
    return check_translator_available()
