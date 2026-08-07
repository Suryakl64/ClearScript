"""
Chat API route — RAG-powered Q&A over medical reports.

POST /chat/ask      -- ask a question about a specific report
POST /chat/store    -- store a report's findings for chat
GET  /chat/reports  -- list all stored reports
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.constants import DISCLAIMER_SHORT, MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request / Response models ─────────────────────────────────────────────────

class FindingInput(BaseModel):
    test: str = ""
    value: Optional[float] = None
    unit: str = ""
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    flag: str = "UNKNOWN"
    status: str = "UNKNOWN"
    full_name: str = ""
    loinc: Optional[str] = None
    category: str = "unknown"
    source: str = "unknown"
    explanation: str = ""


class StoreReportRequest(BaseModel):
    filename: str = Field("unknown_report", description="Original filename")
    report_type: str = Field("structured", description="Report type")
    findings: list[FindingInput] = Field(..., description="Structured findings")
    summary: str = Field("", description="Report summary from Phase 3")


class AskRequest(BaseModel):
    report_id: str = Field(..., description="Report ID to query")
    question: str = Field(..., description="User's question about the report")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/store")
def store_report(req: StoreReportRequest):
    """
    Store a processed report's findings for chat and RAG.

    This indexes the findings into ChromaDB for similarity search and
    saves the report to SQLite for persistence.

    **Call this after Phase 2+3 processing**, before using /chat/ask.
    Returns the `report_id` needed for subsequent chat queries.
    """
    from backend.data.db import generate_report_id, save_report
    from backend.rag.chroma_store import index_findings

    findings_dicts = [f.model_dump() for f in req.findings]

    # Generate unique report ID
    report_id = generate_report_id(req.filename)

    # Save to SQLite
    db_result = save_report(
        report_id=report_id,
        filename=req.filename,
        report_type=req.report_type,
        findings=findings_dicts,
        summary=req.summary,
    )

    if db_result.get("error"):
        raise HTTPException(status_code=500, detail=db_result["error"])

    # Index findings into ChromaDB for RAG
    index_result = index_findings(report_id, findings_dicts)

    return {
        "success": True,
        "report_id": report_id,
        "findings_stored": len(findings_dicts),
        "findings_indexed": index_result.get("findings_indexed", 0),
        "message": f"Report stored. Use report_id '{report_id}' for /chat/ask queries.",
    }


@router.post("/ask")
def ask_question(req: AskRequest):
    """
    Ask a question about a specific medical report.

    Uses RAG: retrieves relevant findings from ChromaDB and passes
    them to the local LLM (Ollama/Mistral) for a grounded answer.

    Falls back to showing retrieved findings directly when Ollama
    is offline.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Verify report exists
    from backend.data.db import get_report
    report = get_report(req.report_id)
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{req.report_id}' not found. "
                   f"Store it first via POST /chat/store.",
        )

    # Run RAG
    from backend.chat.rag_engine import ask_question as rag_ask
    result = rag_ask(req.report_id, req.question)

    return {
        "success": True,
        **result,
    }


@router.get("/reports")
def list_reports():
    """List all stored reports, newest first."""
    from backend.data.db import list_reports as db_list

    reports = db_list(limit=50)
    return {
        "success": True,
        "reports": reports,
        "count": len(reports),
    }
