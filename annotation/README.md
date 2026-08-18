# Medical NER Annotation Guide — Label Studio Setup

This guide walks you through setting up Label Studio to annotate Indian
medical lab reports for fine-tuning the BioBERT NER model.

---

## 1. Install Label Studio

```bash
pip install label-studio
```

## 2. Start Label Studio

```bash
label-studio start --port 8080
```

Open **http://localhost:8080** in your browser and create an account.

## 3. Create a New Project

1. Click **Create Project**
2. Name it: `ClearScript Medical NER`
3. Under **Labeling Setup** → **Custom Template**, paste the contents of
   [`labeling_config.xml`](labeling_config.xml)
4. Click **Save**

## 4. Prepare Report Text Files

Before importing, run OCR on your lab report scans to extract raw text:

```python
# prepare_for_annotation.py (run from project root)
import json, os
from backend.ocr.extractor import extract_text

reports_dir = "reports_test"  # put your PDF/image files here
output = []

for fname in os.listdir(reports_dir):
    if fname.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
        with open(os.path.join(reports_dir, fname), "rb") as f:
            result = extract_text(f.read(), fname)
        output.append({
            "data": {"text": result["text"]},
            "meta": {"filename": fname, "ocr_method": result["method"]}
        })

with open("annotation/import_tasks.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Prepared {len(output)} tasks for Label Studio import.")
```

## 5. Import Tasks

1. In your Label Studio project, click **Import**
2. Upload `annotation/import_tasks.json`
3. All reports will appear as annotation tasks

## 6. Annotation Guidelines

### Entity Types

| Label | Description | Example |
|-------|-------------|---------|
| `Test_Name` | Name of the lab test | `Haemoglobin`, `SGPT/ALT`, `TSH` |
| `Test_Value` | Numeric result value | `9.2`, `12500`, `6.8` |
| `Unit` | Unit of measurement | `g/dL`, `/cumm`, `mIU/L` |
| `Reference_Range` | Normal range | `13.0 - 17.0`, `<200` |
| `Flag` | Abnormality indicator | `HIGH`, `LOW`, `H`, `L` |
| `Disease` | Disease/condition name | `diabetes mellitus`, `hypothyroidism` |
| `Medication` | Drug name | `metformin`, `levothyroxine` |
| `Symptom` | Symptom/complaint | `fatigue`, `shortness of breath` |
| `Procedure` | Medical procedure | `blood transfusion`, `biopsy` |
| `Body_Part` | Anatomical structure | `liver`, `kidney`, `thyroid` |
| `Vital_Sign` | Vital sign with value | `BP: 130/80 mmHg`, `SpO2: 96%` |

### Annotation Rules

1. **Be precise**: Select only the exact text span for each entity
2. **Include all tests**: Even if values are normal, annotate them
3. **Abbreviations**: Annotate abbreviations as `Test_Name` (e.g., `TLC`, `Hb`)
4. **Composite names**: Include the full test name including slashes
   (e.g., `SGPT/ALT` as one `Test_Name`)
5. **Reference ranges**: Include the full range string including dashes
   (e.g., `13.0 - 17.0`)
6. **Flags**: Only annotate explicit flags printed in the report
   (e.g., `HIGH`, `LOW`, `H`, `L`)
7. **Report type**: Classify each report using the radio buttons at the bottom

### Quality Targets

- Aim for **30–50 annotated reports** for meaningful fine-tuning
- Include a mix of:
  - 60% structured lab reports (CBC, LFT, KFT, Lipid, Thyroid)
  - 25% narrative reports (discharge summaries)
  - 15% mixed reports
- Prioritize diverse pathology lab formats from different Indian labs

## 7. Export Annotations

1. Click **Export** in your Label Studio project
2. Choose **CoNLL 2003** format (for token-level NER fine-tuning)
3. Save the exported file to `annotation/exported_annotations.conll`

Alternatively, export as **JSON** for maximum flexibility:
- Save to `annotation/exported_annotations.json`

## 8. Fine-tune BioBERT

Once you have 30+ annotated reports, run the fine-tuning script:

```bash
python -m backend.ner.finetune_biobert \
    --data annotation/exported_annotations.conll \
    --output backend/models/finetuned-biobert-ner \
    --epochs 10 \
    --batch-size 8
```

See [`backend/ner/finetune_biobert.py`](../backend/ner/finetune_biobert.py)
for details and hyperparameter configuration.
