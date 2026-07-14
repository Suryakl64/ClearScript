print("Checking ClearScript setup...\n")

try:
    import fitz
    print("✅ PyMuPDF (fitz) — OK")
except ImportError:
    print("❌ PyMuPDF — FAILED")

try:
    import cv2
    print("✅ OpenCV — OK")
except ImportError:
    print("❌ OpenCV — FAILED")

try:
    import easyocr
    print("✅ EasyOCR — OK")
except ImportError:
    print("❌ EasyOCR — FAILED")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} — OK")
except ImportError:
    print("❌ PyTorch — FAILED")

try:
    import transformers
    print(f"✅ Transformers {transformers.__version__} — OK")
except ImportError:
    print("❌ Transformers — FAILED")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy + en_core_web_sm — OK")
except Exception:
    print("❌ spaCy — FAILED")

try:
    import chromadb
    print("✅ ChromaDB — OK")
except ImportError:
    print("❌ ChromaDB — FAILED")

try:
    import fastapi
    print("✅ FastAPI — OK")
except ImportError:
    print("❌ FastAPI — FAILED")

print("\n✅ Setup check complete.")