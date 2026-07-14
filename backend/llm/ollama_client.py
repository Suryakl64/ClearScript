"""Local Ollama client — no external API keys."""
import httpx

from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaError(Exception):
    pass


def check_ollama_available() -> dict:
    """Check if Ollama is running and model is available."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            model_available = any(OLLAMA_MODEL in m for m in models)
            return {
                "available": True,
                "models": models,
                "target_model": OLLAMA_MODEL,
                "model_ready": model_available,
            }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "target_model": OLLAMA_MODEL,
            "model_ready": False,
        }


def generate(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """
    Generate text via Ollama /api/generate endpoint.
    Raises OllamaError on failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except httpx.ConnectError:
        raise OllamaError(
            "Ollama is not running. Start it with: ollama serve && ollama pull mistral"
        )
    except httpx.HTTPStatusError as exc:
        raise OllamaError(f"Ollama HTTP error: {exc.response.status_code}")
    except Exception as exc:
        raise OllamaError(str(exc))
