from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.ocr.extractor import extract_text

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
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB."
        )

    result = extract_text(contents, file.filename)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

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