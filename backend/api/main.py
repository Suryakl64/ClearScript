"""
ClearScript API — AI Medical Report Translator
Main FastAPI application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import uuid

from backend.api.routes.ocr import router as ocr_router
from backend.api.routes.ner import router as ner_router
from backend.api.routes.vision import router as vision_router
from backend.api.routes.llm import router as llm_router
from backend.api.routes.translation import router as translation_router
from backend.api.routes.chat import router as chat_router

# ── Logging Configuration ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api.main")

app = FastAPI(
    title="ClearScript API",
    description="AI-powered medical report translator — converts complex medical "
                "documents into plain-language summaries patients can understand.",
    version="0.4.0",
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(ocr_router)
app.include_router(ner_router)
app.include_router(vision_router)
app.include_router(llm_router)
app.include_router(translation_router)
app.include_router(chat_router)

# CORS — allow the Vite dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Middleware to add request ID and log timing."""
    req_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{req_id}] Started {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"[{req_id}] Completed {response.status_code} in {process_time:.3f}s")
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(f"[{req_id}] Failed with unhandled exception in {process_time:.3f}s: {exc}", exc_info=True)
        raise

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to return structured JSON errors instead of crashing."""
    logger.error(f"Unhandled server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)},
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