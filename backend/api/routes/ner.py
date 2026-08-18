"""
NER extraction API route.

POST /ner/extract — accepts raw text (or a file upload to reuse Phase 1
OCR) and returns structured medical findings via the NER pipeline.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ner", tags=["NER"])


@router.post("/extract")
async def extract_findings(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    skip_biobert: bool = Form(False),
    force_parser: Optional[str] = Form(None, description="Force 'structured', 'narrative', or 'both'"),
):
    """
    Extract structured medical findings from raw text or an uploaded file.

    **Provide either:**

    - ``text`` — raw OCR text string (e.g. from Phase 1 output), OR
    - ``file`` — a PDF / image that will be OCR'd first (reuses Phase 1).

    **Optional params:**

    - ``skip_biobert`` — if true, skip the BioBERT NER (faster, CPU only).
    - ``force_parser`` — force ``"structured"``, ``"narrative"``, or ``"both"``.

    **Returns** a JSON body with::

        {
            "success": true,
            "findings": [...],
            "finding_count": int,
            "report_type": str,
            "parsers_used": [str],
            "detection": {...}
        }
    """
    raw_text: Optional[str] = None

    # ── Source 1: raw text submitted directly ──────────────────────────────
    if text and text.strip():
        raw_text = text.strip()

    # ── Source 2: file upload → run Phase 1 OCR first ─────────────────────
    elif file is not None:
        allowed_types = {
            "application/pdf",
            "image/png", "image/jpeg",
            "image/jpg", "image/webp",
            "image/tiff", "image/bmp",
        }
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. "
                       f"Upload a PDF or image.",
            )

        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 20 MB).")

        from backend.ocr.extractor import extract_text
        ocr_result = extract_text(contents, file.filename)
        if "error" in ocr_result:
            raise HTTPException(status_code=422, detail=ocr_result["error"])
        raw_text = ocr_result.get("text", "")

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' (form field) or 'file' (upload).",
        )

    # ── Run NER pipeline ──────────────────────────────────────────────────
    from backend.ner.pipeline import run_ner_pipeline

    logger.info(f"Starting NER extraction (skip_biobert={skip_biobert}, force_parser={force_parser})")
    start_time = time.time()
    
    try:
        result = run_ner_pipeline(
            raw_text,
            force_parser=force_parser,
            skip_biobert=skip_biobert,
        )
    except Exception as exc:
        logger.error(f"NER extraction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal NER error: {exc}")

    process_time = time.time() - start_time
    logger.info(f"NER extraction completed in {process_time:.3f}s. "
                f"Found {result.get('finding_count', 0)} findings. "
                f"Parsers used: {', '.join(result.get('parsers_used', []))}")

    return {
        "success": True,
        **result,
    }
