"""
Shared vision extraction prompts for Gemini and Ollama vision models.

These prompts instruct multimodal AI models to extract structured
medical lab findings directly from an image of a medical report.
"""

MEDICAL_REPORT_EXTRACTION_PROMPT = """You are a strict medical report data extractor. Your ONLY job is to extract test results that are VISUALLY PRESENT in this image.

CRITICAL RULES — READ CAREFULLY:
1. ONLY extract tests that have a VISIBLE test name AND a VISIBLE result value in this image.
2. DO NOT invent, infer, assume, or add any test that is not explicitly shown in the report.
3. DO NOT add tests just because they are common medical tests (e.g., do NOT add LFT, KFT, HbA1c, ESR, Blood Sugar unless they are actually present in this image).
4. DO NOT extract lab name, patient name, address, phone numbers, registration numbers, barcodes, or doctor names as test results.
5. If a test row has a name but no visible result value, SKIP IT entirely.
6. Handle multi-column table layouts — extract from ALL visible columns.
7. Preserve the source report's interpretation exactly — if the report says "Borderline", use "BORDERLINE" as the flag.

For each test found in the image:
- test_name: Exactly as written in the Investigation/Test column
- value: The numeric or text result from the Result column (NOT the reference range)
- unit: The unit from the Unit column
- reference_range: The range from the Reference Value column (e.g., "13.0-17.0")
- flag: Use the report's own flag if shown ("HIGH", "LOW", "NORMAL", "BORDERLINE"); otherwise compute from value vs range

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
      "flag": "HIGH | LOW | NORMAL | BORDERLINE | UNKNOWN",
      "category": "haematology | liver | kidney | lipid | thyroid | diabetes | urine | stool | serology | vitals | examination | other"
    }
  ],
  "overall_status": null
}

Return ONLY the JSON object, no additional text or markdown formatting."""




MEDICAL_REPORT_EXTRACTION_PROMPT_COMPACT = """Extract ALL medical test results from this report image as JSON.

For each test found, include: test_name, value, unit, reference_range, flag (HIGH/LOW/NORMAL/UNKNOWN), category.

Handle multi-column tables. Include qualitative results (ABSENT, NORMAL, NAD, POSITIVE, NEGATIVE, etc.).

Return JSON: {"patient": {"name", "age", "gender", "date"}, "findings": [{"test_name", "value", "unit", "reference_range", "flag", "category"}], "overall_status"}

Return ONLY valid JSON."""
