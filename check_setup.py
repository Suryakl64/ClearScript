"""
ClearScript - Setup Verification Script
Checks that every required package imports correctly and prints version info.
Run:  python check_setup.py
"""

import sys
import os
import importlib

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PACKAGES = [
    # (display_name, import_name, optional_version_attr)
    ("NumPy",                "numpy",                  "__version__"),
    ("OpenCV (headless)",    "cv2",                    "__version__"),
    ("SciPy",                "scipy",                  "__version__"),
    ("Pillow",               "PIL",                    "__version__"),
    ("tifffile",             "tifffile",               "__version__"),
    ("scikit-image",         "skimage",                "__version__"),
    ("python-bidi",          "bidi",                   None),
    ("pyclipper",            "pyclipper",              None),
    ("Shapely",              "shapely",                "__version__"),
    ("PyYAML",               "yaml",                   "__version__"),
    ("ninja",                "ninja",                  "__version__"),
    ("EasyOCR",              "easyocr",                None),
    ("PyMuPDF",              "fitz",                   None),
    ("FastAPI",              "fastapi",                "__version__"),
    ("Uvicorn",              "uvicorn",                None),
    ("python-multipart",     "multipart",              None),
    ("python-dotenv",        "dotenv",                 None),
    ("Pydantic",             "pydantic",               "__version__"),
    ("Requests",             "requests",               "__version__"),
    ("Pandas",               "pandas",                 "__version__"),
    ("Transformers",         "transformers",           "__version__"),
    ("sentence-transformers","sentence_transformers",  "__version__"),
    ("ChromaDB",             "chromadb",               "__version__"),
    ("spaCy",                "spacy",                  "__version__"),
    ("PyTorch",              "torch",                  "__version__"),
    ("TorchVision",          "torchvision",            "__version__"),
]

OK = "[OK]"
FAIL = "[FAIL]"


def check_spacy_model():
    """Try loading the en_core_web_sm spaCy model."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("ClearScript is an AI medical report translator.")
        return True, f"loaded - {len(list(doc.ents))} entities in test sentence"
    except Exception as exc:
        return False, str(exc)


def main():
    print("=" * 60)
    print("  ClearScript - Setup Verification")
    print(f"  Python {sys.version}")
    print("=" * 60)
    print()

    passed, failed = 0, 0

    for display, module, ver_attr in PACKAGES:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, ver_attr, "ok") if ver_attr else "ok"
            print(f"  {OK} {display:<26s} {version}")
            passed += 1
        except ImportError as exc:
            print(f"  {FAIL} {display:<26s} FAILED - {exc}")
            failed += 1

    # spaCy model
    ok, info = check_spacy_model()
    if ok:
        print(f"  {OK} {'spaCy en_core_web_sm':<26s} {info}")
        passed += 1
    else:
        print(f"  {FAIL} {'spaCy en_core_web_sm':<26s} FAILED - {info}")
        failed += 1

    print()
    print("-" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("  ALL CHECKS PASSED - ClearScript is ready!")
    else:
        print("  WARNING: Some packages are missing. Fix them before proceeding.")
    print("-" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
