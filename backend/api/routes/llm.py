"""
LLM explanation API route.

POST /explain/generate — takes structured findings from Phase 2 and returns
plain-English explanations + an overall report summary.

Uses Ollama (Mistral 7B) locally. Falls back to rule-based explanations
when Ollama is offline — never returns an error for missing LLM.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.constants import DISCLAIMER_SHORT, MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explanation"])


# ── Request / Response models ─────────────────────────────────────────────────

class Finding(BaseModel):
    """A single structured finding from the Phase 2 NER pipeline."""
    test: str = Field(..., description="Canonical short name (e.g. 'Hb')")
    value: Optional[float] = Field(None, description="Numeric value")
    unit: str = Field("", description="Unit string")
    range_low: Optional[float] = Field(None, description="Lower bound of reference range")
    range_high: Optional[float] = Field(None, description="Upper bound of reference range")
    flag: str = Field("UNKNOWN", description="HIGH / LOW / NORMAL / UNKNOWN")
    status: str = Field("UNKNOWN", description="Alias of flag")
    full_name: str = Field("", description="Descriptive name")
    loinc: Optional[str] = Field(None, description="LOINC code")
    category: str = Field("unknown", description="Clinical category")
    source: str = Field("unknown", description="Parser source")


class ExplainRequest(BaseModel):
    """Request body for POST /explain/generate."""
    findings: list[Finding] = Field(..., description="Structured findings from Phase 2")
    report_type: str = Field("structured", description="Report type: structured / narrative / mixed")


class ExplainedFinding(BaseModel):
    """A finding enriched with a plain-English explanation."""
    test: str
    value: Optional[float]
    unit: str
    range_low: Optional[float]
    range_high: Optional[float]
    flag: str
    full_name: str
    category: str
    explanation: str
    explanation_available: bool
    disclaimer: str


class ExplainResponse(BaseModel):
    """Response from POST /explain/generate."""
    success: bool
    findings: list[ExplainedFinding]
    finding_count: int
    summary: str
    summary_available: bool
    normal_count: int
    abnormal_count: int
    abnormal_tests: list[str]
    disclaimer: str
    disclaimer_full: str


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=ExplainResponse)
def generate_explanations(req: ExplainRequest):
    """
    Generate plain-English explanations for structured medical findings.

    **Send** a JSON body with the `findings` array from Phase 2
    (e.g. from ``POST /ner/extract`` or ``POST /vision/extract``).

    **Returns** each finding enriched with a human-readable explanation,
    plus an overall report summary.

    Works offline — when Ollama is not running, rule-based fallback
    explanations are returned automatically.
    """
    if not req.findings:
        raise HTTPException(
            status_code=400,
            detail="No findings provided. Send at least one finding.",
        )

    # Convert Pydantic models to dicts for the explainer
    findings_dicts = [f.model_dump() for f in req.findings]

    # ── Generate explanations (batch mode) ────────────────────────────────
    from backend.llm.explainer import explain_all_findings
    explained = explain_all_findings(findings_dicts)

    # ── Generate report summary ───────────────────────────────────────────
    from backend.llm.report_summary import generate_report_summary
    summary_result = generate_report_summary(findings_dicts, req.report_type)

    # ── Build response ────────────────────────────────────────────────────
    response_findings = []
    for f in explained:
        response_findings.append({
            "test": f.get("test", ""),
            "value": f.get("value"),
            "unit": f.get("unit", ""),
            "range_low": f.get("range_low"),
            "range_high": f.get("range_high"),
            "flag": f.get("flag", "UNKNOWN"),
            "full_name": f.get("full_name", f.get("test", "")),
            "category": f.get("category", "unknown"),
            "explanation": f.get("explanation", ""),
            "explanation_available": f.get("explanation_available", False),
            "disclaimer": DISCLAIMER_SHORT,
        })

    return {
        "success": True,
        "findings": response_findings,
        "finding_count": len(response_findings),
        "summary": summary_result["summary"],
        "summary_available": summary_result["summary_available"],
        "normal_count": summary_result["normal_count"],
        "abnormal_count": summary_result["abnormal_count"],
        "abnormal_tests": summary_result["abnormal_tests"],
        "disclaimer": DISCLAIMER_SHORT,
        "disclaimer_full": MEDICAL_DISCLAIMER,
    }


@router.get("/status")
def explain_status():
    """Check if the explanation engine (Ollama) is available."""
    from backend.llm.ollama_client import check_ollama_available
    ollama = check_ollama_available()

    return {
        "ollama": ollama,
        "fallback_available": True,
        "note": "Rule-based explanations are always available as fallback.",
        "disclaimer": DISCLAIMER_SHORT,
    }
