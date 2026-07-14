"""
Quick test for the OCR pipeline.
Put any medical report PDF or image in reports_test/ and run this.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.ocr.extractor import extract_text

def test_file(filepath: str):
    print(f"\n{'='*60}")
    print(f"Testing: {filepath}")
    print('='*60)

    with open(filepath, "rb") as f:
        file_bytes = f.read()

    filename = os.path.basename(filepath)
    result = extract_text(file_bytes, filename)

    print(f"Method used    : {result['method']}")
    print(f"Pages          : {result.get('pages', 1)}")
    print(f"OCR pages      : {result.get('ocr_pages', [])}")
    print(f"Characters     : {result['char_count']}")
    print(f"\n--- Extracted Text (first 1000 chars) ---\n")
    print(result['text'][:1000])
    print("\n--- End Preview ---")

    return result

if __name__ == "__main__":
    # Test all files in reports_test/
    test_dir = "reports_test"
    files = [f for f in os.listdir(test_dir)
             if f.endswith(('.pdf', '.png', '.jpg', '.jpeg'))]

    if not files:
        print("No test files found. Put a medical report PDF or image in reports_test/")
    else:
        for f in files:
            test_file(os.path.join(test_dir, f))