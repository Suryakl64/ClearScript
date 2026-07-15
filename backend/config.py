"""ClearScript configuration — all paths and model names in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_data"
REPORTS_DIR = BASE_DIR / "stored_reports"

# HuggingFace models (downloaded on first use)
BIOBERT_NER_MODEL = "d4data/biomedical-ner-all"
INDICTRANS_MODEL = "ai4bharat/indictrans2-en-indic-1B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama (local — no API keys)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"
OLLAMA_VISION_MODEL = "llava"  # local multimodal model for image understanding

# Google Gemini (free tier — optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# Supported translation languages (IndicTrans2 BCP-47 tags)
SUPPORTED_LANGUAGES = {
    "en": {"label": "English", "indic_tag": None},
    "hi": {"label": "Hindi", "indic_tag": "hin_Deva"},
    "ta": {"label": "Tamil", "indic_tag": "tam_Taml"},
    "kn": {"label": "Kannada", "indic_tag": "kan_Knda"},
}

# File upload limits
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png", "image/jpeg", "image/jpg",
    "image/webp", "image/tiff", "image/bmp",
}

# Ensure runtime directories exist
for _dir in (DATA_DIR, CHROMA_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
