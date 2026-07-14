"""
Indian medical lab abbreviation normalisation for the NER pipeline.

Maps common abbreviations used in Indian pathology reports to canonical
names, full descriptive names, LOINC codes, and clinical categories.

This module wraps the existing JSON-based abbreviation database in
``backend/data/abbreviations.json`` and adds additional high-frequency
Indian-lab aliases so the NER pipeline can normalise OCR-extracted test
names reliably.
"""

import re
from functools import lru_cache
from typing import Optional

# ---------------------------------------------------------------------------
# Inline abbreviation table — 50+ entries covering Indian lab report aliases
# ---------------------------------------------------------------------------
# Structure:  canonical_key → {full_name, aliases[], loinc?, category}
# This is self-contained so the NER module works even without the JSON file.
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, dict] = {
    # ── Haematology ──────────────────────────────────────────────────────
    "Hb": {
        "full_name": "Haemoglobin",
        "aliases": ["HGB", "Haemoglobin", "Hemoglobin", "H.B.", "HB", "Hgb"],
        "loinc": "718-7",
        "category": "haematology",
    },
    "TLC": {
        "full_name": "Total Leukocyte Count",
        "aliases": ["WBC", "Total WBC", "Leucocyte Count", "Total Leucocyte Count",
                     "TLC/WBC", "White Blood Cell Count", "WBC Count"],
        "loinc": "6690-2",
        "category": "haematology",
    },
    "DLC": {
        "full_name": "Differential Leukocyte Count",
        "aliases": ["DC", "Differential Count", "Diff Count"],
        "loinc": "49024-7",
        "category": "haematology",
    },
    "RBC": {
        "full_name": "Red Blood Cell Count",
        "aliases": ["Red Cell Count", "Erythrocyte Count", "RBC Count"],
        "loinc": "789-8",
        "category": "haematology",
    },
    "PCV": {
        "full_name": "Packed Cell Volume",
        "aliases": ["HCT", "Hematocrit", "Haematocrit"],
        "loinc": "4544-3",
        "category": "haematology",
    },
    "MCV": {
        "full_name": "Mean Corpuscular Volume",
        "aliases": ["Mean Cell Volume"],
        "loinc": "787-2",
        "category": "haematology",
    },
    "MCH": {
        "full_name": "Mean Corpuscular Haemoglobin",
        "aliases": [],
        "loinc": "785-6",
        "category": "haematology",
    },
    "MCHC": {
        "full_name": "Mean Corpuscular Haemoglobin Concentration",
        "aliases": [],
        "loinc": "786-4",
        "category": "haematology",
    },
    "RDW": {
        "full_name": "Red Cell Distribution Width",
        "aliases": ["RDW-CV", "RDW-SD"],
        "loinc": "788-0",
        "category": "haematology",
    },
    "PLT": {
        "full_name": "Platelet Count",
        "aliases": ["Platelets", "PLT Count", "Thrombocyte Count", "Platelet"],
        "loinc": "777-3",
        "category": "haematology",
    },
    "MPV": {
        "full_name": "Mean Platelet Volume",
        "aliases": [],
        "loinc": "32623-1",
        "category": "haematology",
    },
    "ESR": {
        "full_name": "Erythrocyte Sedimentation Rate",
        "aliases": ["Sed Rate"],
        "loinc": "4537-7",
        "category": "haematology",
    },
    "Reticulocyte": {
        "full_name": "Reticulocyte Count",
        "aliases": ["Retic Count", "Retics", "Reticulocytes"],
        "loinc": "17849-1",
        "category": "haematology",
    },
    "Neutrophils": {
        "full_name": "Neutrophils",
        "aliases": ["Neut", "Polymorphs", "PMN", "Neutrophil %"],
        "loinc": "770-8",
        "category": "haematology",
    },
    "Lymphocytes": {
        "full_name": "Lymphocytes",
        "aliases": ["Lymph", "Lymphs", "Lymphocyte %"],
        "loinc": "736-9",
        "category": "haematology",
    },
    "Monocytes": {
        "full_name": "Monocytes",
        "aliases": ["Mono", "Monocyte %"],
        "loinc": "742-7",
        "category": "haematology",
    },
    "Eosinophils": {
        "full_name": "Eosinophils",
        "aliases": ["Eosin", "Eos", "Eosinophil %"],
        "loinc": "713-8",
        "category": "haematology",
    },
    "Basophils": {
        "full_name": "Basophils",
        "aliases": ["Baso", "Basophil %"],
        "loinc": "706-2",
        "category": "haematology",
    },
    "Ferritin": {
        "full_name": "Serum Ferritin",
        "aliases": ["S. Ferritin", "S.Ferritin"],
        "loinc": "2276-4",
        "category": "haematology",
    },
    "Iron": {
        "full_name": "Serum Iron",
        "aliases": ["S. Iron", "Serum Fe", "S.Iron"],
        "loinc": "2498-4",
        "category": "haematology",
    },
    "TIBC": {
        "full_name": "Total Iron Binding Capacity",
        "aliases": ["Iron Binding Capacity"],
        "loinc": "2500-7",
        "category": "haematology",
    },

    # ── Liver Function Tests (LFT) ──────────────────────────────────────
    "SGPT": {
        "full_name": "Alanine Aminotransferase",
        "aliases": ["ALT", "GPT", "S.G.P.T.", "SGPT/ALT", "S.GPT"],
        "loinc": "1742-6",
        "category": "liver",
    },
    "SGOT": {
        "full_name": "Aspartate Aminotransferase",
        "aliases": ["AST", "GOT", "S.G.O.T.", "SGOT/AST", "S.GOT"],
        "loinc": "1920-8",
        "category": "liver",
    },
    "ALP": {
        "full_name": "Alkaline Phosphatase",
        "aliases": ["Alk Phos", "SAP", "S.ALP"],
        "loinc": "6768-6",
        "category": "liver",
    },
    "Bilirubin Total": {
        "full_name": "Total Bilirubin",
        "aliases": ["T.Bil", "Total Bili", "Bilirubin (Total)", "Serum Bilirubin",
                     "T. Bilirubin", "S. Bilirubin", "T.Bilirubin"],
        "loinc": "1975-2",
        "category": "liver",
    },
    "Bilirubin Direct": {
        "full_name": "Direct Bilirubin",
        "aliases": ["D.Bil", "Conjugated Bilirubin", "Direct Bili", "D. Bilirubin",
                     "D.Bilirubin"],
        "loinc": "1968-7",
        "category": "liver",
    },
    "Bilirubin Indirect": {
        "full_name": "Indirect Bilirubin",
        "aliases": ["I.Bil", "Unconjugated Bilirubin", "I. Bilirubin",
                     "I.Bilirubin"],
        "loinc": "1971-1",
        "category": "liver",
    },
    "Albumin": {
        "full_name": "Serum Albumin",
        "aliases": ["S. Albumin", "Alb", "S.Albumin"],
        "loinc": "1751-7",
        "category": "liver",
    },
    "Total Protein": {
        "full_name": "Total Protein",
        "aliases": ["S. Protein", "Serum Protein", "TP", "S.Protein"],
        "loinc": "2885-2",
        "category": "liver",
    },
    "GGT": {
        "full_name": "Gamma Glutamyl Transferase",
        "aliases": ["Gamma GT", "GGTP", "Gamma-GT"],
        "loinc": "2324-2",
        "category": "liver",
    },
    "A/G Ratio": {
        "full_name": "Albumin/Globulin Ratio",
        "aliases": ["AG Ratio", "A:G Ratio", "Alb/Glob"],
        "loinc": "10834-0",
        "category": "liver",
    },
    "Globulin": {
        "full_name": "Serum Globulin",
        "aliases": ["S. Globulin", "Glob"],
        "loinc": "10835-7",
        "category": "liver",
    },
    "LDH": {
        "full_name": "Lactate Dehydrogenase",
        "aliases": ["Lactic Dehydrogenase"],
        "loinc": "2532-0",
        "category": "liver",
    },

    # ── Kidney Function Tests (KFT / RFT) ───────────────────────────────
    "Urea": {
        "full_name": "Blood Urea",
        "aliases": ["BUN", "Blood Urea Nitrogen", "S. Urea", "Serum Urea",
                     "S.Urea"],
        "loinc": "3094-0",
        "category": "kidney",
    },
    "Creatinine": {
        "full_name": "Serum Creatinine",
        "aliases": ["S. Creatinine", "Creat", "Serum Creat", "S.Creat",
                     "S.Creatinine", "Sr. Creatinine"],
        "loinc": "2160-0",
        "category": "kidney",
    },
    "Uric Acid": {
        "full_name": "Serum Uric Acid",
        "aliases": ["S. Uric Acid", "S.Uric", "S.Uric Acid", "Serum Uric Acid",
                     "Uric acid"],
        "loinc": "3084-1",
        "category": "kidney",
    },
    "eGFR": {
        "full_name": "Estimated Glomerular Filtration Rate",
        "aliases": ["EGFR", "GFR"],
        "loinc": "98979-8",
        "category": "kidney",
    },
    "BUN/Creatinine Ratio": {
        "full_name": "BUN to Creatinine Ratio",
        "aliases": ["BUN/Creat"],
        "loinc": "3097-3",
        "category": "kidney",
    },

    # ── Thyroid Profile ──────────────────────────────────────────────────
    "TSH": {
        "full_name": "Thyroid Stimulating Hormone",
        "aliases": ["S. TSH", "S.TSH", "Ultra-sensitive TSH"],
        "loinc": "3016-3",
        "category": "thyroid",
    },
    "T3": {
        "full_name": "Triiodothyronine",
        "aliases": ["Total T3", "TT3", "S. T3", "S.T3"],
        "loinc": "3053-6",
        "category": "thyroid",
    },
    "T4": {
        "full_name": "Thyroxine",
        "aliases": ["Total T4", "TT4", "S. T4", "S.T4"],
        "loinc": "3026-2",
        "category": "thyroid",
    },
    "FT3": {
        "full_name": "Free T3",
        "aliases": ["Free Triiodothyronine", "F T3", "FreeT3"],
        "loinc": "83112-3",
        "category": "thyroid",
    },
    "FT4": {
        "full_name": "Free T4",
        "aliases": ["Free Thyroxine", "F T4", "FreeT4"],
        "loinc": "83113-1",
        "category": "thyroid",
    },

    # ── Diabetes / Blood Sugar ───────────────────────────────────────────
    "FBS": {
        "full_name": "Fasting Blood Sugar",
        "aliases": ["FBG", "Fasting Glucose", "Fasting Blood Glucose",
                     "FPG", "Glucose Fasting", "F.B.S."],
        "loinc": "1558-6",
        "category": "diabetes",
    },
    "PPBS": {
        "full_name": "Post Prandial Blood Sugar",
        "aliases": ["PPBG", "Post Meal Glucose", "2hr PPBS", "PP Blood Sugar",
                     "P.P.B.S."],
        "loinc": "1521-4",
        "category": "diabetes",
    },
    "RBS": {
        "full_name": "Random Blood Sugar",
        "aliases": ["Random Glucose", "RBG", "R.B.S."],
        "loinc": "2345-7",
        "category": "diabetes",
    },
    "HbA1c": {
        "full_name": "Glycated Haemoglobin",
        "aliases": ["HBA1C", "A1C", "Glycosylated Hemoglobin",
                     "Glycated Hemoglobin", "Glycated Haemoglobin", "HbA1C"],
        "loinc": "4548-4",
        "category": "diabetes",
    },
    "GTT": {
        "full_name": "Glucose Tolerance Test",
        "aliases": ["OGTT", "Oral Glucose Tolerance Test"],
        "loinc": "20436-2",
        "category": "diabetes",
    },

    # ── Electrolytes ─────────────────────────────────────────────────────
    "Sodium": {
        "full_name": "Serum Sodium",
        "aliases": ["Na", "S. Na", "Na+", "S.Na"],
        "loinc": "2951-2",
        "category": "electrolytes",
    },
    "Potassium": {
        "full_name": "Serum Potassium",
        "aliases": ["K", "S. K", "K+", "S.K"],
        "loinc": "2823-3",
        "category": "electrolytes",
    },
    "Chloride": {
        "full_name": "Serum Chloride",
        "aliases": ["Cl", "S. Cl", "Cl-", "S.Cl"],
        "loinc": "2075-0",
        "category": "electrolytes",
    },
    "Calcium": {
        "full_name": "Serum Calcium",
        "aliases": ["S. Calcium", "Ca", "Serum Ca", "S.Calcium"],
        "loinc": "17861-6",
        "category": "electrolytes",
    },
    "Phosphorus": {
        "full_name": "Serum Phosphorus",
        "aliases": ["Phosphate", "S. Phosphorus", "Inorganic Phosphorus",
                     "S.Phosphorus"],
        "loinc": "2777-1",
        "category": "electrolytes",
    },
    "Magnesium": {
        "full_name": "Serum Magnesium",
        "aliases": ["Mg", "S. Magnesium", "S.Magnesium"],
        "loinc": "19123-9",
        "category": "electrolytes",
    },
    "Bicarbonate": {
        "full_name": "Serum Bicarbonate",
        "aliases": ["HCO3", "CO2", "Total CO2"],
        "loinc": "1963-8",
        "category": "electrolytes",
    },

    # ── Lipid Profile ────────────────────────────────────────────────────
    "Total Cholesterol": {
        "full_name": "Total Cholesterol",
        "aliases": ["Cholesterol", "S. Cholesterol", "TC", "Serum Cholesterol"],
        "loinc": "2093-3",
        "category": "lipid",
    },
    "HDL": {
        "full_name": "High Density Lipoprotein",
        "aliases": ["HDL Cholesterol", "HDL-C", "HDL Chol"],
        "loinc": "2085-9",
        "category": "lipid",
    },
    "LDL": {
        "full_name": "Low Density Lipoprotein",
        "aliases": ["LDL Cholesterol", "LDL-C", "LDL Chol"],
        "loinc": "13457-7",
        "category": "lipid",
    },
    "VLDL": {
        "full_name": "Very Low Density Lipoprotein",
        "aliases": ["VLDL Cholesterol", "VLDL-C"],
        "loinc": "13458-5",
        "category": "lipid",
    },
    "Triglycerides": {
        "full_name": "Triglycerides",
        "aliases": ["TG", "S. Triglycerides", "Trigs", "S.TG"],
        "loinc": "2571-8",
        "category": "lipid",
    },
    "LDL/HDL Ratio": {
        "full_name": "LDL to HDL Ratio",
        "aliases": ["LDL:HDL"],
        "loinc": None,
        "category": "lipid",
    },
    "TC/HDL Ratio": {
        "full_name": "Total Cholesterol to HDL Ratio",
        "aliases": ["Cholesterol/HDL", "TC:HDL"],
        "loinc": None,
        "category": "lipid",
    },

    # ── Coagulation ──────────────────────────────────────────────────────
    "PT": {
        "full_name": "Prothrombin Time",
        "aliases": ["Protime"],
        "loinc": "5902-2",
        "category": "coagulation",
    },
    "INR": {
        "full_name": "International Normalised Ratio",
        "aliases": [],
        "loinc": "6301-6",
        "category": "coagulation",
    },
    "APTT": {
        "full_name": "Activated Partial Thromboplastin Time",
        "aliases": ["PTT", "aPTT"],
        "loinc": "3173-2",
        "category": "coagulation",
    },
    "D-Dimer": {
        "full_name": "D-Dimer",
        "aliases": ["D Dimer", "DDimer"],
        "loinc": "48066-5",
        "category": "coagulation",
    },
    "Fibrinogen": {
        "full_name": "Fibrinogen",
        "aliases": ["Plasma Fibrinogen"],
        "loinc": "3255-7",
        "category": "coagulation",
    },

    # ── Inflammatory Markers ─────────────────────────────────────────────
    "CRP": {
        "full_name": "C-Reactive Protein",
        "aliases": ["C Reactive Protein", "hs-CRP", "hsCRP"],
        "loinc": "1988-5",
        "category": "inflammatory",
    },
    "Procalcitonin": {
        "full_name": "Procalcitonin",
        "aliases": ["PCT"],
        "loinc": "33959-8",
        "category": "inflammatory",
    },
    "IL-6": {
        "full_name": "Interleukin-6",
        "aliases": ["Interleukin 6"],
        "loinc": "26881-3",
        "category": "inflammatory",
    },

    # ── Vitamins / Minerals ──────────────────────────────────────────────
    "Vitamin D": {
        "full_name": "25-Hydroxy Vitamin D",
        "aliases": ["Vit D", "25(OH)D", "Vitamin D3", "25 OH Vitamin D",
                     "Vit D3"],
        "loinc": "62292-8",
        "category": "vitamins",
    },
    "Vitamin B12": {
        "full_name": "Vitamin B12",
        "aliases": ["Vit B12", "Cobalamin", "B12"],
        "loinc": "2132-9",
        "category": "vitamins",
    },
    "Folic Acid": {
        "full_name": "Folic Acid",
        "aliases": ["Folate", "Serum Folate", "S. Folate"],
        "loinc": "2284-8",
        "category": "vitamins",
    },

    # ── Tumour Markers ───────────────────────────────────────────────────
    "PSA": {
        "full_name": "Prostate Specific Antigen",
        "aliases": ["Total PSA"],
        "loinc": "2857-1",
        "category": "tumour_marker",
    },
    "CEA": {
        "full_name": "Carcinoembryonic Antigen",
        "aliases": [],
        "loinc": "2039-6",
        "category": "tumour_marker",
    },
    "AFP": {
        "full_name": "Alpha Fetoprotein",
        "aliases": ["Alpha-Fetoprotein"],
        "loinc": "1834-1",
        "category": "tumour_marker",
    },
    "CA-125": {
        "full_name": "Cancer Antigen 125",
        "aliases": ["CA 125"],
        "loinc": "10334-1",
        "category": "tumour_marker",
    },
    "CA 19-9": {
        "full_name": "Cancer Antigen 19-9",
        "aliases": ["CA19-9", "CA-19-9"],
        "loinc": "24108-3",
        "category": "tumour_marker",
    },

    # ── Cardiac Markers ──────────────────────────────────────────────────
    "Troponin I": {
        "full_name": "Troponin I",
        "aliases": ["Trop I", "cTnI"],
        "loinc": "10839-9",
        "category": "cardiac",
    },
    "Troponin T": {
        "full_name": "Troponin T",
        "aliases": ["Trop T", "cTnT"],
        "loinc": "6598-7",
        "category": "cardiac",
    },
    "CK-MB": {
        "full_name": "Creatine Kinase-MB",
        "aliases": ["CKMB", "CK MB"],
        "loinc": "13969-1",
        "category": "cardiac",
    },
    "BNP": {
        "full_name": "Brain Natriuretic Peptide",
        "aliases": ["NT-proBNP", "ProBNP"],
        "loinc": "30934-4",
        "category": "cardiac",
    },
    "CPK": {
        "full_name": "Creatine Phosphokinase",
        "aliases": ["CK", "Creatine Kinase", "Total CK"],
        "loinc": "2157-6",
        "category": "cardiac",
    },

    # ── Urine Tests ──────────────────────────────────────────────────────
    "Urine Routine": {
        "full_name": "Urine Routine Examination",
        "aliases": ["Urine R/M", "Urinalysis", "Urine Analysis"],
        "loinc": None,
        "category": "urine",
    },
    "Urine Protein": {
        "full_name": "Urine Protein",
        "aliases": ["Urine Albumin"],
        "loinc": "2888-6",
        "category": "urine",
    },
    "Urine Sugar": {
        "full_name": "Urine Sugar",
        "aliases": ["Urine Glucose"],
        "loinc": "2350-7",
        "category": "urine",
    },
    "ACR": {
        "full_name": "Albumin Creatinine Ratio",
        "aliases": ["Urine ACR", "Microalbumin/Creatinine"],
        "loinc": "9318-7",
        "category": "urine",
    },

    # ── Pancreatic ───────────────────────────────────────────────────────
    "Amylase": {
        "full_name": "Serum Amylase",
        "aliases": ["S. Amylase"],
        "loinc": "1798-8",
        "category": "pancreatic",
    },
    "Lipase": {
        "full_name": "Serum Lipase",
        "aliases": ["S. Lipase"],
        "loinc": "3040-3",
        "category": "pancreatic",
    },

    # ── Miscellaneous ────────────────────────────────────────────────────
    "RA Factor": {
        "full_name": "Rheumatoid Factor",
        "aliases": ["RF", "Rheumatoid Factor"],
        "loinc": "11572-5",
        "category": "autoimmune",
    },
    "ASO": {
        "full_name": "Anti-Streptolysin O",
        "aliases": ["ASO Titre", "ASOT"],
        "loinc": "5370-2",
        "category": "inflammatory",
    },
    "Widal": {
        "full_name": "Widal Test",
        "aliases": ["Widal Test"],
        "loinc": None,
        "category": "infectious",
    },
    "Dengue NS1": {
        "full_name": "Dengue NS1 Antigen",
        "aliases": ["NS1", "Dengue NS1 Ag"],
        "loinc": "75377-2",
        "category": "infectious",
    },
    "Malaria": {
        "full_name": "Malaria Parasite",
        "aliases": ["MP", "Malaria Smear", "Malaria Antigen"],
        "loinc": None,
        "category": "infectious",
    },
    "HIV": {
        "full_name": "HIV Screening",
        "aliases": ["HIV I & II", "HIV 1/2", "HIV Antibody"],
        "loinc": "56888-1",
        "category": "infectious",
    },
    "HBsAg": {
        "full_name": "Hepatitis B Surface Antigen",
        "aliases": ["HBs Ag", "Hepatitis B"],
        "loinc": "5196-1",
        "category": "infectious",
    },
    "HCV": {
        "full_name": "Hepatitis C Antibody",
        "aliases": ["Anti-HCV", "HCV Ab"],
        "loinc": "16128-1",
        "category": "infectious",
    },
}


# ---------------------------------------------------------------------------
# Alias lookup cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_alias_map() -> dict[str, str]:
    """
    Build a flat lookup:  lowercase alias → canonical key.
    Includes the canonical key itself, all aliases, and the full_name.
    """
    alias_map: dict[str, str] = {}
    for canonical, info in ABBREVIATIONS.items():
        alias_map[canonical.lower()] = canonical
        alias_map[info["full_name"].lower()] = canonical
        for alias in info.get("aliases", []):
            alias_map[alias.lower()] = canonical
    return alias_map


def get_alias_map() -> dict[str, str]:
    """Return the cached alias → canonical map."""
    return _build_alias_map()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_test_name(raw_name: str) -> dict:
    """
    Normalise an OCR-extracted test name to its canonical form.

    Returns a dict with keys:
        canonical   – short canonical key (e.g. "Hb")
        full_name   – descriptive name  (e.g. "Haemoglobin")
        loinc       – LOINC code or None
        category    – clinical category
        matched_from – the original input string

    Falls back to the cleaned input if no match is found.
    """
    cleaned = re.sub(r"[^\w\s./():-]", "", raw_name).strip()
    key = cleaned.lower()
    alias_map = get_alias_map()

    canonical: Optional[str] = alias_map.get(key)

    # Partial / substring match — longer aliases tried first
    if not canonical:
        for alias, canon in sorted(alias_map.items(), key=lambda x: -len(x[0])):
            if len(alias) >= 2 and alias in key:
                canonical = canon
                break

    if canonical and canonical in ABBREVIATIONS:
        info = ABBREVIATIONS[canonical]
        return {
            "canonical": canonical,
            "full_name": info["full_name"],
            "loinc": info.get("loinc"),
            "category": info.get("category", "unknown"),
            "matched_from": raw_name,
        }

    return {
        "canonical": cleaned,
        "full_name": cleaned,
        "loinc": None,
        "category": "unknown",
        "matched_from": raw_name,
    }


def list_all_abbreviations() -> list[dict]:
    """Return all abbreviation entries for display / debugging."""
    result = []
    for canonical, info in ABBREVIATIONS.items():
        result.append({
            "canonical": canonical,
            "full_name": info["full_name"],
            "aliases": info.get("aliases", []),
            "loinc": info.get("loinc"),
            "category": info.get("category"),
        })
    return result
