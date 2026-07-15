"""
Layout-aware pre-processor for multi-column medical reports.

Uses OpenCV to detect vertical divider lines (column boundaries) and
table row separators in scanned medical report images, then segments
the image into logical columns so each column can be OCR'd separately.

This fixes the core problem where EasyOCR reads across columns,
interleaving "Medical Examination" results with "Laboratory Investigation"
results from side-by-side tables commonly found in Indian pathology reports.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LayoutResult:
    """Result of layout analysis."""
    columns: list[np.ndarray]       # list of column sub-images (left-to-right)
    column_count: int               # number of columns detected
    divider_x_positions: list[int]  # x-coordinates of vertical dividers
    is_multi_column: bool           # True if 2+ columns detected
    debug_info: dict                # diagnostic information


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess_for_line_detection(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale, blur, and apply adaptive threshold."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold — works well for scanned documents with uneven lighting
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10,
    )
    return binary


def _detect_vertical_lines(
    binary: np.ndarray,
    img_height: int,
    img_width: int,
    min_line_length_ratio: float = 0.3,
) -> list[int]:
    """
    Detect long vertical lines that act as column dividers.

    Uses morphological operations to isolate vertical structures,
    then HoughLinesP to find long vertical line segments.

    Returns a sorted list of x-coordinates where vertical dividers exist.
    """
    min_line_length = int(img_height * min_line_length_ratio)

    # Morphological kernel: tall and narrow to isolate vertical lines
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(30, img_height // 20))
    )
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Dilate slightly to connect broken line segments
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    vertical_mask = cv2.dilate(vertical_mask, dilate_kernel, iterations=1)

    # HoughLinesP to detect line segments
    lines = cv2.HoughLinesP(
        vertical_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=30,
    )

    if lines is None:
        return []

    # Filter: keep only near-vertical lines (angle < 5°) in the middle 80% of the image
    margin = int(img_width * 0.10)
    candidate_xs = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check near-vertical
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy < 50 or dx > dy * 0.15:
            continue
        # Check not at the extreme edges
        avg_x = (x1 + x2) // 2
        if avg_x < margin or avg_x > img_width - margin:
            continue
        # Check line is long enough
        length = np.sqrt(dx**2 + dy**2)
        if length < min_line_length * 0.7:
            continue
        candidate_xs.append(avg_x)

    if not candidate_xs:
        return []

    # Cluster nearby x-coordinates (lines within 20px are the same divider)
    candidate_xs.sort()
    clustered = []
    cluster = [candidate_xs[0]]

    for x in candidate_xs[1:]:
        if x - cluster[-1] < 20:
            cluster.append(x)
        else:
            clustered.append(int(np.mean(cluster)))
            cluster = [x]
    clustered.append(int(np.mean(cluster)))

    return clustered


def _detect_columns_by_whitespace(
    binary: np.ndarray,
    img_width: int,
    min_gap_width: int = 15,
) -> list[int]:
    """
    Fallback: detect columns by looking for tall vertical whitespace gaps.

    Projects pixel density horizontally and finds sustained gaps (columns of
    mostly white pixels) that span the middle portion of the image.
    """
    # Vertical projection: sum of white (ink) pixels per column
    projection = np.sum(binary, axis=0) / 255

    # Normalize to [0, 1] relative to image height
    img_height = binary.shape[0]
    density = projection / img_height

    # Find columns where ink density is very low (< 5%)
    margin = int(img_width * 0.15)
    low_density = density[margin:img_width - margin] < 0.02

    # Find contiguous runs of low-density columns
    gaps = []
    in_gap = False
    gap_start = 0

    for i, is_low in enumerate(low_density):
        if is_low and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_low and in_gap:
            gap_width = i - gap_start
            if gap_width >= min_gap_width:
                center = margin + gap_start + gap_width // 2
                gaps.append(center)
            in_gap = False

    return gaps


def _split_image_at_dividers(
    img: np.ndarray,
    divider_xs: list[int],
    padding: int = 5,
) -> list[np.ndarray]:
    """Split the image into column sub-images at the given x-coordinates."""
    h, w = img.shape[:2]
    boundaries = [0] + divider_xs + [w]
    columns = []

    for i in range(len(boundaries) - 1):
        left = max(0, boundaries[i] + padding)
        right = min(w, boundaries[i + 1] - padding)
        if right - left < 30:  # skip very narrow slices
            continue
        col_img = img[:, left:right]
        columns.append(col_img)

    return columns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_layout(img: np.ndarray) -> LayoutResult:
    """
    Analyse an image for multi-column layout.

    Parameters
    ----------
    img : np.ndarray
        BGR image array (as read by OpenCV).

    Returns
    -------
    LayoutResult
        Contains the list of column sub-images and diagnostic info.
    """
    h, w = img.shape[:2]

    binary = _preprocess_for_line_detection(img)

    # Strategy 1: detect explicit vertical divider lines
    divider_xs = _detect_vertical_lines(binary, h, w)

    # Strategy 2: fallback to whitespace gap detection
    if not divider_xs:
        divider_xs = _detect_columns_by_whitespace(binary, w)

    is_multi = len(divider_xs) > 0

    if is_multi:
        columns = _split_image_at_dividers(img, divider_xs)
    else:
        columns = [img]

    return LayoutResult(
        columns=columns,
        column_count=len(columns),
        divider_x_positions=divider_xs,
        is_multi_column=is_multi,
        debug_info={
            "image_size": (w, h),
            "dividers_found": len(divider_xs),
            "divider_positions": divider_xs,
            "strategy": "line_detection" if divider_xs else "single_column",
        },
    )


def reassemble_column_texts(
    column_texts: list[list[tuple[list, str, float]]],
    y_tolerance: int = 15,
) -> str:
    """
    Reassemble OCR results from multiple columns into row-aligned text.

    Each column_texts entry is a list of EasyOCR results:
        [(bounding_box, text, confidence), ...]

    Text fragments from different columns that share approximately the
    same Y-coordinate are joined into the same output line, separated
    by wide spacing (to preserve the tabular structure for the regex
    parser downstream).

    Parameters
    ----------
    column_texts : list of list of (box, text, confidence)
        One list per column, each containing EasyOCR raw results.
    y_tolerance : int
        Maximum Y-coordinate difference (in pixels) for two text blocks
        from different columns to be considered on the same row.

    Returns
    -------
    str
        Reassembled text with columns merged row-by-row.
    """
    if len(column_texts) <= 1:
        # Single column — flatten normally
        if not column_texts:
            return ""
        items = column_texts[0]
        items_sorted = sorted(items, key=lambda x: x[0][0][1])
        return "\n".join(t.strip() for (_, t, c) in items_sorted if c >= 0.4)

    # Collect all text items with their column index and Y-coordinate
    all_items: list[dict] = []
    for col_idx, col_results in enumerate(column_texts):
        for (box, text, confidence) in col_results:
            if confidence < 0.4:
                continue
            y_top = box[0][1]   # top-left Y
            x_left = box[0][0]  # top-left X
            all_items.append({
                "col": col_idx,
                "y": y_top,
                "x": x_left,
                "text": text.strip(),
            })

    if not all_items:
        return ""

    # Sort by Y-coordinate
    all_items.sort(key=lambda item: item["y"])

    # Group items into rows by Y-coordinate proximity
    rows: list[list[dict]] = []
    current_row: list[dict] = [all_items[0]]

    for item in all_items[1:]:
        # Check if this item belongs to the current row
        row_y = np.mean([r["y"] for r in current_row])
        if abs(item["y"] - row_y) <= y_tolerance:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    # Build output: for each row, sort items left-to-right and join
    lines = []
    for row in rows:
        # Sort items within the row by column index, then by x-position
        row.sort(key=lambda item: (item["col"], item["x"]))
        line_text = "   ".join(item["text"] for item in row)
        lines.append(line_text)

    return "\n".join(lines)
