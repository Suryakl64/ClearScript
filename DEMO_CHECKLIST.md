# ClearScript — Live Demo Script & Checklist

*A structured 5-minute presentation guide for your final year project viva.*

---

## 🛑 Pre-Demo Setup Checklist (Do this 10 minutes before)

- [ ] **Virtual Environment**: Ensure `venv` is activated.
- [ ] **Ollama Running**: Run `ollama serve` in a background terminal.
- [ ] **Model Pre-loaded**: Run `ollama run phi3` once to load the model into memory, then exit it (so the first API call is fast).
- [ ] **Backend Server**: Run `uvicorn backend.api.main:app --port 8000` (ensure no errors in the console).
- [ ] **Frontend Server**: Run `npm run dev` in the `frontend/` folder.
- [ ] **Sample Files Ready**: Have 2-3 sample reports (PDFs and Images) ready in a clearly visible folder on your desktop. Include at least one complex tabular lab report and one narrative discharge summary.
- [ ] **Browser Ready**: Open `http://localhost:5173` in a clean browser window (hide bookmarks/clutter).

---

## ⏱️ The 5-Minute Demo Script

### Minute 0–1: Introduction & Architecture
* **Say:** "Welcome to ClearScript, an offline AI system that translates complex medical reports into plain language for Indian patients. 60% of patients struggle to understand their lab values. Our solution fixes this using a 4-phase local pipeline without sending private health data to the cloud."
* **Show:** Briefly display the system architecture diagram from the README or presentation slide. Highlight the OCR → NER → Local LLM → Translation flow.

### Minute 1–2: Live Document Ingestion (Phase 1)
* **Say:** "Let's see it in action. I'm uploading a typical Indian pathology lab report. The system detects if it's a digital PDF or a scanned image."
* **Action:** Drag and drop a complex lab report (e.g., CBC or LFT panel) into the Upload interface.
* **Say:** "Our hybrid OCR engine kicks in. For multi-column reports, we use OpenCV layout segmentation to prevent data merging. The text is now extracted."
* **Show:** Point to the extracted text preview on the screen (if your UI shows it, or mention it happens instantly in the background).

### Minute 2–3: Structured Extraction (Phase 2)
* **Say:** "Raw OCR text isn't useful for analysis. Our NER pipeline kicks in. It uses regular expressions for tabular data and BioBERT for clinical narratives."
* **Action:** Click "Process Report" or wait for the results dashboard to load.
* **Show:** The Dashboard showing the structured table of findings.
* **Say:** "Notice how it correctly identified the test names, numeric values, units, and reference ranges. More importantly, it automatically flagged abnormal values (HIGH/LOW) based on the reference ranges, even if the lab didn't mark them explicitly."

### Minute 3–4: AI Explanations (Phase 3)
* **Say:** "A patient sees 'SGPT is 68 U/L (High)', but what does that mean? Instead of them googling it and panicking, our local LLM explains it."
* **Action:** Click on one of the findings (or show the explanation card).
* **Say:** "This plain-English explanation was generated locally on my machine using the phi3 model via Ollama. It's concise, reassuring, and reminds the patient to consult a doctor. We also generate an overall health summary counting normal vs. abnormal findings."
* **Show:** The Report Summary widget and the medical disclaimer.

### Minute 4–5: Multilingual Support & Q&A
* **Say:** "To bridge the language barrier, we integrated Facebook's NLLB-200 model."
* **Action:** Use the Language Switcher to change the UI to Hindi or Tamil.
* **Show:** The translated explanations.
* **Say:** "We also have an interactive chat interface where patients can ask follow-up questions about their specific report."
* **Action:** (Optional) Type a quick question in the chat like "Should I be worried about my sugar?"
* **Say:** "That concludes the demo. The entire system is running offline on this machine. Thank you, I'm open to questions."

---

## 🎯 Common Viva Questions & Suggested Answers

**Q: Why use Ollama/phi3 locally instead of OpenAI/Gemini APIs?**
> **A:** Medical data is highly sensitive (PHI - Protected Health Information). Sending patient reports to third-party cloud APIs poses significant privacy risks and violates HIPAA/HIPAA-equivalent compliance. Local inference guarantees 100% data privacy. Phi-3 is heavily optimized and runs well on consumer hardware.

**Q: How does your OCR handle messy or handwritten reports?**
> **A:** We use a hybrid approach. For digital PDFs, PyMuPDF extracts text perfectly. For scans, we use EasyOCR. While handwritten text remains a challenge for all OCRs, our layout segmenter helps separate columns, and our post-processing regex fixes common Indian OCR errors (like confusing `l` and `1`).

**Q: What is BioBERT doing exactly?**
> **A:** For tabular lab reports, regex is more accurate for numbers. But for unstructured text like discharge summaries, regex fails. BioBERT is a named entity recognition (NER) model fine-tuned on biomedical text. It identifies diseases, symptoms, and medications buried within paragraphs of doctor notes.

**Q: How do you evaluate the accuracy of your pipeline?**
> **A:** We wrote an evaluation script that calculates Precision, Recall, and F1 scores against manually annotated ground-truth reports. We evaluate at three levels: test name detection, value extraction, and flag determination.

---

## 🚨 Fallback Plan (If Live Demo Fails)

Technology can fail during a high-pressure viva (e.g., out of memory errors, port conflicts).
* **Always have a pre-recorded video** of the perfect 5-minute demo ready on your laptop. If the live server crashes, instantly switch to the video: *"I seem to be having a port conflict, but I have a recording of the system running locally from earlier today."*
* **Have screenshots** embedded in your presentation slides as a final backup.
