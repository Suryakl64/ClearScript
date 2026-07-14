"""
IndicTrans2 translation — English to Tamil/Hindi/Kannada.
Model: ai4bharat/indictrans2-en-indic-1B
"""
from backend.config import INDICTRANS_MODEL, SUPPORTED_LANGUAGES

_translator = None
_tokenizer = None


def get_translator():
    global _translator, _tokenizer
    if _translator is None:
        print(f"Loading IndicTrans2 ({INDICTRANS_MODEL})...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            INDICTRANS_MODEL, trust_remote_code=True
        )
        _translator = AutoModelForSeq2SeqLM.from_pretrained(
            INDICTRANS_MODEL, trust_remote_code=True
        )
        _translator.eval()
        print("IndicTrans2 ready.")
    return _translator, _tokenizer


def translate_text(text: str, target_lang: str) -> dict:
    """
    Translate English text to target language.
    target_lang: 'hi', 'ta', 'kn' (or 'en' returns as-is)
    """
    if not text or not text.strip():
        return {"translated": "", "source_lang": "en", "target_lang": target_lang}

    if target_lang == "en":
        return {"translated": text, "source_lang": "en", "target_lang": "en"}

    lang_info = SUPPORTED_LANGUAGES.get(target_lang)
    if not lang_info or not lang_info.get("indic_tag"):
        return {"translated": text, "error": f"Unsupported language: {target_lang}"}

    indic_tag = lang_info["indic_tag"]

    try:
        model, tokenizer = get_translator()

        # IndicTrans2 expects language tags in input
        input_text = f"eng_Latn {indic_tag} {text}"

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True,
        )
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "translated": translated.strip(),
            "source_lang": "en",
            "target_lang": target_lang,
            "language_label": lang_info["label"],
        }
    except Exception as exc:
        return {
            "translated": text,
            "source_lang": "en",
            "target_lang": target_lang,
            "error": str(exc),
        }


def translate_findings(findings: list[dict], target_lang: str) -> list[dict]:
    """Translate explanation fields in findings list."""
    if target_lang == "en":
        return findings

    translated = []
    for finding in findings:
        updated = dict(finding)
        explanation = finding.get("explanation", "")
        if explanation:
            result = translate_text(explanation, target_lang)
            updated["explanation_translated"] = result.get("translated", explanation)
            if result.get("error"):
                updated["translation_error"] = result["error"]
        translated.append(updated)
    return translated
