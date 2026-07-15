"""
ClearScript API — AI Medical Report Translator
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.ocr import router as ocr_router
from backend.api.routes.ner import router as ner_router
from backend.api.routes.vision import router as vision_router

app = FastAPI(
    title="ClearScript API",
    description="AI-powered medical report translator — converts complex medical "
                "documents into plain-language summaries patients can understand.",
    version="0.2.0",
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(ocr_router)
app.include_router(ner_router)
app.include_router(vision_router)

# CORS — allow the Vite dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Root endpoint — confirms the API is live."""
    return {
        "project": "ClearScript",
        "description": "AI Medical Report Translator",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", tags=["General"])
def health():
    """Health-check endpoint for monitoring / load-balancers."""
    return {"status": "healthy"}