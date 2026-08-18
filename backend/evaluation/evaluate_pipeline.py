"""
ClearScript — Pipeline Evaluation Script

Evaluates the OCR + NER pipeline accuracy against manually verified
ground truth annotations. Calculates precision, recall, and F1 score
for entity extraction at three levels:
  1. Test name correctly identified
  2. Numeric value correctly extracted
  3. Flag (HIGH/LOW/NORMAL) correctly determined

Usage:
    python -m backend.evaluation.evaluate_pipeline
    # or from project root:
    python backend/evaluation/evaluate_pipeline.py
"""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ── Load ground truth and aliases ─────────────────────────────────────────────

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"


def _load_ground_truth() -> dict:
    """Load the ground truth JSON file."""
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_alias_map(aliases: dict[str, list[str]]) -> dict[str, str]:
    """
    Build a reverse lookup: alias → canonical name.
    All keys are lowercased for case-insensitive matching.
    """
    mapping: dict[str, str] = {}
    for canonical, alias_list in aliases.items():
        canonical_lower = canonical.lower()
        mapping[canonical_lower] = canonical_lower
        for alias in alias_list:
            mapping[alias.lower()] = canonical_lower
    return mapping


# ── Sample report texts (matching ground truth IDs) ───────────────────────────

SAMPLE_REPORTS = {
    "sample_structured_lab": """
--- Page 1 ---
PATHOLOGY LAB REPORT
Patient: Demo Patient    Age: 45 yrs   Sex: Male
Date: 01/07/2026          Ref Dr: Dr. Sharma

COMPLETE BLOOD COUNT (CBC)
Test Name                Result      Unit       Reference Range

Haemoglobin              9.2         g/dL       13.0 - 17.0           LOW
Total Leukocyte Count    12500       /cumm      4000 - 11000          HIGH
RBC Count                4.1         mill/cumm  4.5 - 5.5             LOW
PCV                      32          %          40 - 50               LOW
MCV                      78          fL         83 - 101
MCH                      26.5        pg         27 - 31
MCHC                     33.2        g/dL       31.5 - 34.5
Platelet Count           185000      /cumm      150000 - 410000
ESR                      35          mm/hr      0 - 10                HIGH

DIFFERENTIAL COUNT
Neutrophils              72          %          40 - 80
Lymphocytes              20          %          20 - 40
Monocytes                5           %          2 - 10
Eosinophils              2           %          1 - 6
Basophils                1           %          0 - 2

LIVER FUNCTION TESTS (LFT)
SGPT/ALT                 68          U/L        7 - 56                HIGH
SGOT/AST                 52          U/L        5 - 40                HIGH
ALP                      95          U/L        44 - 147
T.Bil                    1.8         mg/dL      0.1 - 1.2             HIGH
D.Bil                    0.5         mg/dL      0.0 - 0.3             HIGH
Albumin                  3.8         g/dL       3.5 - 5.0
Total Protein            7.2         g/dL       6.0 - 8.3

KIDNEY FUNCTION TESTS (KFT)
S.Creat                  1.5         mg/dL      0.7 - 1.3             HIGH
Urea                     48          mg/dL      17 - 43               HIGH
Uric Acid                7.8         mg/dL      3.4 - 7.0             HIGH

THYROID PROFILE
TSH                      6.8         mIU/L      0.27 - 4.2            HIGH

LIPID PROFILE
Total Cholesterol        245         mg/dL      <200                  HIGH
HDL Cholesterol          38          mg/dL      >40                   LOW
LDL Cholesterol          165         mg/dL      <100                  HIGH
Triglycerides            280         mg/dL      <150                  HIGH
VLDL                     42          mg/dL      <30                   HIGH

DIABETES
HbA1c                    8.2         %          <5.7                  HIGH
FBS                      158         mg/dL      70 - 100              HIGH
PPBS                     245         mg/dL      <140                  HIGH

OTHER
Vitamin D                12.5        ng/mL      30 - 100              LOW
Vitamin B12              180         pg/mL      211 - 946             LOW
CRP                      15.2        mg/L       <5.0                  HIGH
Ferritin                 18          ng/mL      20 - 250              LOW
""",
    "sample_e2e_structured": """
    COMPLETE BLOOD COUNT (CBC)
    Test                Result      Unit        Reference Range

    Hemoglobin          9.2         g/dL        13.0 - 17.0
    TLC                 12500       /cumm       4000 - 11000
    RBC                 4.1         mill/cumm   4.5 - 5.5
    Platelet Count      185000      /cumm       150000 - 410000
    ESR                 35          mm/hr       0 - 10

    LIVER FUNCTION TEST
    SGPT (ALT)          68          U/L         7 - 56
    SGOT (AST)          52          U/L         5 - 40

    THYROID PROFILE
    TSH                 6.8         mIU/L       0.27 - 4.2

    BLOOD SUGAR
    Fasting Blood Sugar 158         mg/dL       70 - 100

    VITAMINS
    Vitamin D           12.5        ng/mL       30.0 - 100.0
    Vitamin B12         180         pg/mL       211 - 946
    """,
}


# ── Matching logic ────────────────────────────────────────────────────────────


def _normalize_name(name: str, alias_map: dict[str, str]) -> str:
    """Normalize a test name using the alias map."""
    key = name.strip().lower()
    return alias_map.get(key, key)


def _values_match(expected: float, actual) -> bool:
    """Check if two numeric values match (within tolerance)."""
    if actual is None:
        return False
    try:
        return abs(float(expected) - float(actual)) < 0.01
    except (ValueError, TypeError):
        return False


def _flags_match(expected: str, actual: str) -> bool:
    """Check if two flags match (case-insensitive)."""
    e = expected.upper().strip()
    a = actual.upper().strip()
    # Map variations
    flag_map = {"H": "HIGH", "L": "LOW", "N": "NORMAL"}
    e = flag_map.get(e, e)
    a = flag_map.get(a, a)
    return e == a


# ── Evaluation metrics ────────────────────────────────────────────────────────


@dataclass
class MetricCounter:
    """Tracks TP, FP, FN for a single metric."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    details: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class EvaluationResult:
    """Holds evaluation results for one report."""
    report_id: str
    test_name_metric: MetricCounter = field(default_factory=MetricCounter)
    value_metric: MetricCounter = field(default_factory=MetricCounter)
    flag_metric: MetricCounter = field(default_factory=MetricCounter)
    expected_count: int = 0
    extracted_count: int = 0


def evaluate_report(
    report_id: str,
    expected_findings: list[dict],
    extracted_findings: list[dict],
    alias_map: dict[str, str],
) -> EvaluationResult:
    """
    Compare extracted findings against expected ground truth.

    Matching strategy:
    1. For each expected finding, find the best matching extracted finding
       using normalized test names (alias-aware fuzzy matching).
    2. Once matched, check value accuracy and flag accuracy.
    3. Unmatched expected → false negatives; unmatched extracted → false positives.
    """
    result = EvaluationResult(
        report_id=report_id,
        expected_count=len(expected_findings),
        extracted_count=len(extracted_findings),
    )

    # Build normalized lookup for extracted findings
    extracted_by_name: dict[str, list[dict]] = {}
    for f in extracted_findings:
        name = _normalize_name(f.get("test", ""), alias_map)
        extracted_by_name.setdefault(name, []).append(f)

    matched_extracted: set[int] = set()

    for exp in expected_findings:
        exp_name = _normalize_name(exp["test"], alias_map)
        candidates = extracted_by_name.get(exp_name, [])

        # Find best match (prefer value match)
        best_match = None
        best_idx = None
        for i, cand in enumerate(candidates):
            cand_id = id(cand)
            if cand_id in matched_extracted:
                continue
            if best_match is None:
                best_match = cand
                best_idx = cand_id
            elif _values_match(exp["value"], cand.get("value")):
                best_match = cand
                best_idx = cand_id
                break

        if best_match is not None and best_idx is not None:
            matched_extracted.add(best_idx)

            # Test name: TRUE POSITIVE
            result.test_name_metric.true_positives += 1
            result.test_name_metric.details.append(
                {"expected": exp["test"], "extracted": best_match.get("test"), "status": "TP"}
            )

            # Value check
            if _values_match(exp["value"], best_match.get("value")):
                result.value_metric.true_positives += 1
                result.value_metric.details.append(
                    {"test": exp["test"], "expected": exp["value"],
                     "extracted": best_match.get("value"), "status": "TP"}
                )
            else:
                result.value_metric.false_negatives += 1
                result.value_metric.details.append(
                    {"test": exp["test"], "expected": exp["value"],
                     "extracted": best_match.get("value"), "status": "FN_value_mismatch"}
                )

            # Flag check
            if _flags_match(exp["flag"], best_match.get("flag", "UNKNOWN")):
                result.flag_metric.true_positives += 1
                result.flag_metric.details.append(
                    {"test": exp["test"], "expected": exp["flag"],
                     "extracted": best_match.get("flag"), "status": "TP"}
                )
            else:
                result.flag_metric.false_negatives += 1
                result.flag_metric.details.append(
                    {"test": exp["test"], "expected": exp["flag"],
                     "extracted": best_match.get("flag"), "status": "FN_flag_mismatch"}
                )
        else:
            # No match found — FALSE NEGATIVE for all metrics
            result.test_name_metric.false_negatives += 1
            result.test_name_metric.details.append(
                {"expected": exp["test"], "extracted": None, "status": "FN_not_found"}
            )
            result.value_metric.false_negatives += 1
            result.flag_metric.false_negatives += 1

    # Count unmatched extracted findings as false positives
    total_matched = result.test_name_metric.true_positives
    unmatched_extracted = len(extracted_findings) - total_matched
    if unmatched_extracted > 0:
        result.test_name_metric.false_positives = unmatched_extracted
        result.value_metric.false_positives = unmatched_extracted
        result.flag_metric.false_positives = unmatched_extracted

    return result


# ── Pretty printing ───────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_report_results(result: EvaluationResult):
    """Print evaluation results for a single report."""
    print(f"\n{'=' * 72}")
    print(f"  Report: {result.report_id}")
    print(f"  Expected: {result.expected_count} findings  |  "
          f"Extracted: {result.extracted_count} findings")
    print(f"{'=' * 72}")

    print(f"\n  {'Metric':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}  "
          f"{'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}  "
          f"{'-' * 4} {'-' * 4} {'-' * 4}")

    for label, m in [
        ("Test Name Detection", result.test_name_metric),
        ("Value Extraction", result.value_metric),
        ("Flag Determination", result.flag_metric),
    ]:
        print(f"  {label:<25} {_pct(m.precision):>10} {_pct(m.recall):>10} "
              f"{_pct(m.f1):>10}  {m.true_positives:>4} {m.false_positives:>4} "
              f"{m.false_negatives:>4}")

    # Print mismatches
    mismatches = [
        d for d in result.test_name_metric.details if d["status"] != "TP"
    ]
    if mismatches:
        print(f"\n  Missed test names:")
        for d in mismatches:
            print(f"    - {d['expected']} (not found in extracted findings)")

    value_mismatches = [
        d for d in result.value_metric.details
        if d["status"] == "FN_value_mismatch"
    ]
    if value_mismatches:
        print(f"\n  Value mismatches:")
        for d in value_mismatches:
            print(f"    - {d['test']}: expected={d['expected']}, "
                  f"got={d['extracted']}")

    flag_mismatches = [
        d for d in result.flag_metric.details
        if d["status"] == "FN_flag_mismatch"
    ]
    if flag_mismatches:
        print(f"\n  Flag mismatches:")
        for d in flag_mismatches:
            print(f"    - {d['test']}: expected={d['expected']}, "
                  f"got={d['extracted']}")


def print_aggregate(results: list[EvaluationResult]):
    """Print aggregate metrics across all reports."""
    agg_name = MetricCounter()
    agg_value = MetricCounter()
    agg_flag = MetricCounter()

    for r in results:
        agg_name.true_positives += r.test_name_metric.true_positives
        agg_name.false_positives += r.test_name_metric.false_positives
        agg_name.false_negatives += r.test_name_metric.false_negatives

        agg_value.true_positives += r.value_metric.true_positives
        agg_value.false_positives += r.value_metric.false_positives
        agg_value.false_negatives += r.value_metric.false_negatives

        agg_flag.true_positives += r.flag_metric.true_positives
        agg_flag.false_positives += r.flag_metric.false_positives
        agg_flag.false_negatives += r.flag_metric.false_negatives

    print(f"\n{'=' * 72}")
    print(f"  AGGREGATE RESULTS ({len(results)} reports)")
    print(f"{'=' * 72}")

    print(f"\n  {'Metric':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}  "
          f"{'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}  "
          f"{'-' * 4} {'-' * 4} {'-' * 4}")

    for label, m in [
        ("Test Name Detection", agg_name),
        ("Value Extraction", agg_value),
        ("Flag Determination", agg_flag),
    ]:
        print(f"  {label:<25} {_pct(m.precision):>10} {_pct(m.recall):>10} "
              f"{_pct(m.f1):>10}  {m.true_positives:>4} {m.false_positives:>4} "
              f"{m.false_negatives:>4}")

    print(f"{'=' * 72}\n")

    return {
        "test_name": {"precision": agg_name.precision, "recall": agg_name.recall, "f1": agg_name.f1},
        "value": {"precision": agg_value.precision, "recall": agg_value.recall, "f1": agg_value.f1},
        "flag": {"precision": agg_flag.precision, "recall": agg_flag.recall, "f1": agg_flag.f1},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Run the full evaluation pipeline."""
    print("\n" + "=" * 72)
    print("  ClearScript — Pipeline Evaluation")
    print("=" * 72)

    from backend.ner.pipeline import run_ner_pipeline

    # Load ground truth
    gt = _load_ground_truth()
    alias_map = _build_alias_map(gt.get("test_name_aliases", {}))

    results: list[EvaluationResult] = []

    for report in gt["reports"]:
        report_id = report["id"]
        expected = report["expected_findings"]

        if report_id not in SAMPLE_REPORTS:
            print(f"\n  [SKIP] No sample text for report: {report_id}")
            continue

        sample_text = SAMPLE_REPORTS[report_id]
        print(f"\n  Running NER pipeline on: {report_id} ...")

        # Run pipeline (skip BioBERT for speed — rule-based is the target)
        pipeline_result = run_ner_pipeline(sample_text, skip_biobert=True)
        extracted = pipeline_result.get("findings", [])

        print(f"  Report type: {pipeline_result.get('report_type')}")
        print(f"  Parsers used: {pipeline_result.get('parsers_used')}")
        print(f"  Extracted {len(extracted)} findings")

        # Evaluate
        eval_result = evaluate_report(report_id, expected, extracted, alias_map)
        results.append(eval_result)
        print_report_results(eval_result)

    # Aggregate
    if results:
        aggregate = print_aggregate(results)

        # Save results to JSON
        output_path = Path(__file__).parent / "results.json"
        output = {
            "reports": [
                {
                    "id": r.report_id,
                    "expected_count": r.expected_count,
                    "extracted_count": r.extracted_count,
                    "test_name": {
                        "precision": r.test_name_metric.precision,
                        "recall": r.test_name_metric.recall,
                        "f1": r.test_name_metric.f1,
                    },
                    "value": {
                        "precision": r.value_metric.precision,
                        "recall": r.value_metric.recall,
                        "f1": r.value_metric.f1,
                    },
                    "flag": {
                        "precision": r.flag_metric.precision,
                        "recall": r.flag_metric.recall,
                        "f1": r.flag_metric.f1,
                    },
                }
                for r in results
            ],
            "aggregate": aggregate,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to: {output_path}")

    print("\n  [OK] Evaluation complete.\n")


if __name__ == "__main__":
    main()
