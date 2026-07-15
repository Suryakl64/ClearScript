"""
Shared vision extraction prompts for Gemini and Ollama vision models.

These prompts instruct multimodal AI models to extract structured
medical lab findings directly from an image of a medical report.
"""

MEDICAL_REPORT_EXTRACTION_PROMPT = """You are a medical report data extractor. Analyze this medical report image and extract ALL lab test results, medical examination findings, and investigation results.

For each finding, extract:
- test_name: The full name of the test or examination
- value: The result value (number, text like "ABSENT", "NORMAL", "POSITIVE", "NEGATIVE", "NAD", "NIL", etc.)
- unit: The measurement unit (if any)
- reference_range: The normal reference range (if shown)
- flag: "HIGH", "LOW", "NORMAL", or "UNKNOWN" based on whether the value is within the reference range

IMPORTANT RULES:
1. Handle multi-column layouts — this report may have side-by-side tables (e.g., "Medical Examination" on the left and "Laboratory Investigation" on the right). Extract from ALL columns.
2. Include qualitative results like ABSENT, PRESENT, NORMAL, NAD (No Abnormality Detected), NOT SEEN, POSITIVE, NEGATIVE, FIT, etc.
3. For Indian medical reports, common abbreviations include: CBC, LFT, KFT, RFT, ESR, TLC, DLC, SGPT, SGOT, HbA1c, FBS, PPBS, R.B.S., etc.
4. Also extract patient metadata if visible: name, age, gender, date.
5. Also extract the overall fitness status if present (e.g., "FIT FOR DUTY").

Return your response as valid JSON with this exact structure:
{
  "patient": {
    "name": "string or null",
    "age": "string or null",
    "gender": "string or null",
    "date": "string or null"
  },
  "findings": [
    {
      "test_name": "string",
      "value": "string",
      "unit": "string or empty",
      "reference_range": "string or empty",
      "flag": "HIGH | LOW | NORMAL | UNKNOWN",
      "category": "haematology | liver | kidney | lipid | thyroid | diabetes | urine | stool | serology | vitals | examination | other"
    }
  ],
  "overall_status": "string or null"
}

Return ONLY the JSON object, no additional text or markdown formatting."""


MEDICAL_REPORT_EXTRACTION_PROMPT_COMPACT = """Extract ALL medical test results from this report image as JSON.

For each test found, include: test_name, value, unit, reference_range, flag (HIGH/LOW/NORMAL/UNKNOWN), category.

Handle multi-column tables. Include qualitative results (ABSENT, NORMAL, NAD, POSITIVE, NEGATIVE, etc.).

Return JSON: {"patient": {"name", "age", "gender", "date"}, "findings": [{"test_name", "value", "unit", "reference_range", "flag", "category"}], "overall_status"}

Return ONLY valid JSON."""
