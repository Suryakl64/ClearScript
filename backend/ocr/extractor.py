import fitz  # PyMuPDF
import cv2
import numpy as np
import re

from backend.ocr.layout_segmenter import detect_layout, reassemble_column_texts

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


# ── Core OCR call (original — single column) ─────────────────────────────────
def _ocr_image_raw(img_rgb: np.ndarray) -> list[tuple]:
    """
    Run EasyOCR on an RGB image array.
    Returns the raw EasyOCR results: [(bounding_box, text, confidence), ...]
    """
    reader = get_ocr_engine()
    return reader.readtext(img_rgb, detail=1, paragraph=False)


def _to_rgb(img_array: np.ndarray) -> np.ndarray:
    """Convert any OpenCV image format to RGB for EasyOCR."""
    if len(img_array.shape) == 2:
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        return cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
    else:
        return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)


def _raw_to_text(raw: list[tuple]) -> str:
    """Convert raw EasyOCR results to a plain text string (top-to-bottom)."""
    if not raw:
        return ""

    raw_sorted = sorted(raw, key=lambda x: x[0][0][1])
    out = []
    prev_y = None
    for (box, text, confidence) in raw_sorted:
        if confidence < 0.4:
            continue
        y = box[0][1]
        if prev_y is not None and (y - prev_y) > 40:
            out.append("")
        out.append(text.strip())
        prev_y = y

    return "\n".join(out)


# ── Layout-aware OCR ──────────────────────────────────────────────────────────
def extract_from_image_array(img_array: np.ndarray) -> str:
    """
    Run EasyOCR on a numpy BGR image array with layout-aware pre-processing.

    If the image contains a multi-column layout (e.g., side-by-side tables),
    the columns are detected via OpenCV, OCR'd separately, and the results
    are reassembled row-by-row to preserve the tabular structure.

    Falls back to standard single-pass OCR for single-column documents.
    """
    # Step 1: Detect layout (columns)
    layout = detect_layout(img_array)

    if layout.is_multi_column and layout.column_count >= 2:
        print(f"  Layout: {layout.column_count} columns detected "
              f"(dividers at x={layout.divider_x_positions})")

        # Step 2: OCR each column separately
        column_results = []
        for i, col_img in enumerate(layout.columns):
            col_rgb = _to_rgb(col_img)
            raw = _ocr_image_raw(col_rgb)
            column_results.append(raw)
            print(f"    Column {i+1}: {len(raw)} text blocks detected")

        # Step 3: Reassemble columns row-by-row
        text = reassemble_column_texts(column_results)
        return text
    else:
        # Single column — original behaviour
        img_rgb = _to_rgb(img_array)
        raw = _ocr_image_raw(img_rgb)
        return _raw_to_text(raw)


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