import fitz  # PyMuPDF
import cv2
import numpy as np
import re

# ── Engine singleton ──────────────────────────────────────────────────────────
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        print("Loading EasyOCR engine (first time only, ~30s + model download)...")
        import easyocr
        # gpu=False since we're on CPU; english only for medical reports
        _ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("EasyOCR ready.")
    return _ocr_engine


# ── Core OCR call ─────────────────────────────────────────────────────────────
def extract_from_image_array(img_array: np.ndarray) -> str:
    """
    Run EasyOCR on a numpy BGR image array.
    Returns all detected text as a single string, ordered top-to-bottom.
    """
    # EasyOCR expects RGB, OpenCV gives BGR — convert
    if len(img_array.shape) == 2:
        # Grayscale → RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        # RGBA → RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
    else:
        # BGR → RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    reader = get_ocr_engine()

    # EasyOCR returns: [ (box, text, confidence), ... ]
    # box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    raw = reader.readtext(img_rgb, detail=1, paragraph=False)

    if not raw:
        return ""

    # Sort top-to-bottom by y-coordinate of top-left corner of bounding box
    raw_sorted = sorted(raw, key=lambda x: x[0][0][1])

    out = []
    prev_y = None
    for (box, text, confidence) in raw_sorted:
        if confidence < 0.4:        # skip very uncertain reads
            continue

        y = box[0][1]               # top-left y coordinate

        # Large vertical gap between lines = table row separator
        if prev_y is not None and (y - prev_y) > 40:
            out.append("")

        out.append(text.strip())
        prev_y = y

    return "\n".join(out)


# ── File-type handlers ────────────────────────────────────────────────────────
def extract_from_image_file(image_bytes: bytes) -> dict:
    """Extract text from an uploaded image (JPG / PNG / etc.)."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "text"  : "",
            "method": "image_ocr",
            "error" : "Could not decode image"
        }

    text = extract_from_image_array(img)
    return {
        "text"     : text,
        "method"   : "image_ocr",
        "pages"    : 1,
        "ocr_pages": [1]
    }


def extract_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Extract text from a PDF.
    Uses PyMuPDF direct extraction for digital PDFs (fast, perfect accuracy).
    Falls back to EasyOCR for scanned / image-based pages.
    """
    doc            = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count     = len(doc)
    all_text       = ""
    pages_used_ocr = []

    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text().strip()

        if len(text) < 50:
            print(f"  Page {page_num + 1}: no text layer, running OCR...")
            pix       = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img_array = np.frombuffer(img_bytes, np.uint8)
            img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            text      = extract_from_image_array(img)
            pages_used_ocr.append(page_num + 1)

        all_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    return {
        "text"     : all_text.strip(),
        "pages"    : page_count,
        "ocr_pages": pages_used_ocr,
        "method"   : "pdf+ocr" if pages_used_ocr else "pdf_direct"
    }


# ── Main router ───────────────────────────────────────────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> dict:
    """
    Entry point for the API route and Phase 2 NER pipeline.
    Auto-detects file type and routes to the right extractor.
    """
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        result = extract_from_pdf(file_bytes)
    elif ext in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
        result = extract_from_image_file(file_bytes)
    else:
        return {
            "text"  : "",
            "method": "unsupported",
            "error" : f"Unsupported file type: .{ext}"
        }

    result["text"]       = clean_text(result["text"])
    result["filename"]   = filename
    result["char_count"] = len(result["text"])
    return result


# ── Text post-processing ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Fix common OCR errors found in Indian medical reports."""
    fixes = {
        r'\bl\b(?=\s*\d)': '1',    # lowercase l → 1 before numbers
        r'(?<=\d)O(?=\s)' : '0',   # letter O → 0 after numbers
        r'(?<=\d),(?=\d)' : '.',   # comma as decimal separator
        r'\s{2,}'         : ' ',   # collapse multiple spaces
        r'\n{3,}'         : '\n\n' # collapse multiple blank lines
    }
    for pattern, repl in fixes.items():
        text = re.sub(pattern, repl, text)
    return text.strip()