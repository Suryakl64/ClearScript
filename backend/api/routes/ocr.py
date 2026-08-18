from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
import time
from backend.ocr.extractor import extract_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/extract")
async def extract_report_text(file: UploadFile = File(...)):
    """
    Upload a medical report (PDF or image) and get back extracted text.
    """
    # Validate file type
    allowed_types = {
        "application/pdf",
        "image/png", "image/jpeg",
        "image/jpg", "image/webp",
        "image/tiff", "image/bmp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload a PDF or image."
        )

    # File size check — reject files over 20MB
    contents = await file.read()
    file_size = len(contents)
    if file_size > 20 * 1024 * 1024:
        logger.warning(f"File {file.filename} rejected: too large ({file_size} bytes)")
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB."
        )

    logger.info(f"Processing OCR for {file.filename} ({file_size} bytes, type: {file.content_type})")
    start_time = time.time()
    
    try:
        result = extract_text(contents, file.filename)
    except Exception as exc:
        logger.error(f"Unexpected error in extract_text for {file.filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal OCR error: {exc}")

    if "error" in result:
        logger.warning(f"OCR extraction failed for {file.filename}: {result['error']}")
        raise HTTPException(status_code=422, detail=result["error"])

    process_time = time.time() - start_time
    logger.info(f"OCR successful for {file.filename} in {process_time:.3f}s via {result['method']}")

    return {
        "success": True,
        "filename": result["filename"],
        "method": result["method"],
        "pages": result.get("pages", 1),
        "ocr_pages": result.get("ocr_pages", []),
        "char_count": result["char_count"],
        "text": result["text"],
        # Preview — first 500 chars so the response isn't huge in logs
        "preview": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
    }