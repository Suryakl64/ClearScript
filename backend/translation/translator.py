"""
Multilingual translation — English to Indian languages.

Uses facebook/nllb-200-distilled-600M (~1.2 GB, CPU-friendly).
Supports: Hindi, Tamil, Kannada, Telugu.

Model is lazy-loaded on first translation call and cached for subsequent use.
"""

import logging
from typing import Optional

from backend.config import NLLB_MODEL, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    """Lazy-load the NLLB translation pipeline."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading NLLB-200 (%s) — first time only (~1.2 GB download)...", NLLB_MODEL)
        print(f"Loading NLLB-200 ({NLLB_MODEL}) — first time only...")

        from transformers import pipeline

        _pipeline = pipeline(
            "translation",
            model=NLLB_MODEL,
            max_length=512,
        )
        logger.info("NLLB-200 ready.")
        print("NLLB-200 ready.")
    return _pipeline


def translate_text(text: str, target_lang: str) -> dict:
    """
    Translate English text to a target Indian language.

    Parameters
    ----------
    text : str
        English text to translate.
    target_lang : str
        Language code: 'hi' (Hindi), 'ta' (Tamil), 'kn' (Kannada),
        'te' (Telugu), or 'en' (returns as-is).

    Returns
    -------
    dict
        {
            "translated": str,
            "source_lang": "en",
            "target_lang": str,
            "language_label": str,
            "model": str,
        }
    """
    if not text or not text.strip():
        return {"translated": "", "source_lang": "en", "target_lang": target_lang}

    if target_lang == "en":
        return {
            "translated": text,
            "source_lang": "en",
            "target_lang": "en",
            "language_label": "English",
        }

    lang_info = SUPPORTED_LANGUAGES.get(target_lang)
    if not lang_info or not lang_info.get("nllb_code"):
        return {
            "translated": text,
            "source_lang": "en",
            "target_lang": target_lang,
            "error": f"Unsupported language: {target_lang}. "
                     f"Supported: {', '.join(k for k in SUPPORTED_LANGUAGES if k != 'en')}",
        }

    nllb_code = lang_info["nllb_code"]

    try:
        pipe = _get_pipeline()
        result = pipe(
            text,
            src_lang="eng_Latn",
            tgt_lang=nllb_code,
        )
        translated = result[0]["translation_text"]

        return {
            "translated": translated.strip(),
            "source_lang": "en",
            "target_lang": target_lang,
            "language_label": lang_info["label"],
            "model": NLLB_MODEL,
        }
    except Exception as exc:
        logger.error("Translation failed: %s", exc)
        return {
            "translated": text,
            "source_lang": "en",
            "target_lang": target_lang,
            "error": str(exc),
        }


def translate_findings(findings: list[dict], target_lang: str) -> list[dict]:
    """
    Translate the explanation field in each finding to the target language.

    Adds an 'explanation_translated' field to each finding dict.
    """
    if target_lang == "en" or not findings:
        return findings

    translated = []
    for finding in findings:
        updated = dict(finding)
        explanation = finding.get("explanation", "")
        if explanation:
            result = translate_text(explanation, target_lang)
            updated["explanation_translated"] = result.get("translated", explanation)
            updated["translation_lang"] = target_lang
            updated["translation_label"] = SUPPORTED_LANGUAGES.get(
                target_lang, {}
            ).get("label", target_lang)
            if result.get("error"):
                updated["translation_error"] = result["error"]
        translated.append(updated)
    return translated


def check_translator_available() -> dict:
    """Check if the translation model is loaded or can be loaded."""
    return {
        "available": True,
        "model": NLLB_MODEL,
        "loaded": _pipeline is not None,
        "languages": {
            k: v["label"]
            for k, v in SUPPORTED_LANGUAGES.items()
            if k != "en"
        },
        "note": "Model downloads ~1.2 GB on first use. Subsequent calls are instant.",
    }
