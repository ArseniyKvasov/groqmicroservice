import hmac
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request
from functools import wraps
from groq import Groq


def _env_int(name: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_float(
    name: str,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


API_KEY_STORAGE = os.getenv("API_KEY_STORAGE", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
MAX_PROMPT_CHARS = _env_int("MAX_PROMPT_CHARS", 20000, min_value=1)
MAX_AUDIO_FILE_SIZE_BYTES = _env_int("MAX_AUDIO_FILE_SIZE_BYTES", 25 * 1024 * 1024, min_value=1024)
GROQ_TIMEOUT_SECONDS = _env_float("GROQ_TIMEOUT_SECONDS", 45.0, min_value=5.0, max_value=300.0)
GROQ_RETRIES = _env_int("GROQ_RETRIES", 3, min_value=1, max_value=6)
GROQ_RETRY_BASE_DELAY_SECONDS = _env_float("GROQ_RETRY_BASE_DELAY_SECONDS", 0.75, min_value=0.1, max_value=10.0)

AVAILABLE_MODELS = {
    "allam-2-7b": "Allam 2 7B",
    "canopylabs/orpheus-arabic-saudi": "Canopy Labs Orpheus Arabic Saudi",
    "canopylabs/orpheus-v1-english": "Canopy Labs Orpheus V1 English",
    "groq/compound": "Groq Compound",
    "groq/compound-mini": "Groq Compound Mini",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant",
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Meta Llama 4 Scout 17B 16E Instruct",
    "meta-llama/llama-prompt-guard-2-22m": "Meta Llama Prompt Guard 2 22M",
    "meta-llama/llama-prompt-guard-2-86m": "Meta Llama Prompt Guard 2 86M",
    "moonshotai/kimi-k2-instruct": "Moonshot AI Kimi K2 Instruct",
    "moonshotai/kimi-k2-instruct-0905": "Moonshot AI Kimi K2 Instruct 0905",
    "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B",
    "openai/gpt-oss-20b": "OpenAI GPT-OSS 20B",
    "openai/gpt-oss-safeguard-20b": "OpenAI GPT-OSS Safeguard 20B",
    "qwen/qwen3-32b": "Qwen Qwen3 32B",
    "whisper-large-v3": "Whisper Large V3",
    "whisper-large-v3-turbo": "Whisper Large V3 Turbo",
}


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("groq-service")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = _env_int("MAX_REQUEST_BYTES", 1 * 1024 * 1024, min_value=1024)

_groq_client: Optional[Groq] = None


def json_error(message: str, status_code: int, error_type: str, **extra: Any):
    payload = {
        "error": message,
        "error_type": error_type,
        "status_code": status_code,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload), status_code


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        _groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    return _groq_client


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY_STORAGE:
            logger.error("API_KEY_STORAGE is empty; requests are blocked for safety")
            return json_error(
                "Server authentication is not configured",
                503,
                "service_misconfigured",
            )

        api_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(api_key, API_KEY_STORAGE):
            return json_error("Invalid or missing API key", 401, "unauthorized")
        return f(*args, **kwargs)

    return decorated


def _is_retryable(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    if status_code is not None:
        return status_code == 429 or status_code >= 500

    lowered = str(exc).lower()
    transient_markers = ["timeout", "temporarily", "connection reset", "econnreset", "unavailable"]
    return any(marker in lowered for marker in transient_markers)


def _validate_common_payload(data: Any):
    if not isinstance(data, dict):
        return None, None, None, json_error("JSON body is required", 400, "invalid_json")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, None, None, json_error("Prompt is required and must be a string", 400, "missing_field")

    if len(prompt) > MAX_PROMPT_CHARS:
        return None, None, None, json_error(
            f"Prompt is too long (max {MAX_PROMPT_CHARS} characters)",
            400,
            "invalid_parameter",
        )

    model = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        return None, None, None, json_error(
            f"Model '{model}' is not available",
            400,
            "invalid_model",
            available_models=list(AVAILABLE_MODELS.keys()),
        )

    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 5000)

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return None, None, None, json_error("Temperature must be a number", 400, "invalid_parameter")

    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return None, None, None, json_error("max_tokens must be an integer", 400, "invalid_parameter")

    if not (0 <= temperature <= 2):
        return None, None, None, json_error("Temperature must be between 0 and 2", 400, "invalid_parameter")

    if not (1 <= max_tokens <= 32768):
        return None, None, None, json_error("max_tokens must be between 1 and 32768", 400, "invalid_parameter")

    return prompt, model, {"temperature": temperature, "max_tokens": max_tokens}, None


def _safe_usage(completion: Any) -> Optional[dict[str, Any]]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _call_groq_with_retries(model: str, prompt: str, options: dict[str, Any]):
    last_exc: Optional[Exception] = None

    for attempt in range(1, GROQ_RETRIES + 1):
        try:
            client = get_groq_client()
            if model.startswith("whisper"):
                audio_path = Path(prompt)
                if not audio_path.exists() or not audio_path.is_file():
                    raise FileNotFoundError("For whisper models, prompt must be a valid local file path")
                if audio_path.stat().st_size > MAX_AUDIO_FILE_SIZE_BYTES:
                    raise ValueError(
                        f"Audio file is too large (max {MAX_AUDIO_FILE_SIZE_BYTES} bytes)"
                    )

                with audio_path.open("rb") as audio_file:
                    completion = client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                        response_format="json",
                    )
                text = getattr(completion, "text", None) or ""
                return text, completion

            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=options["temperature"],
                max_tokens=options["max_tokens"],
            )
            text = completion.choices[0].message.content
            return text, completion
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == GROQ_RETRIES or not _is_retryable(exc):
                break
            delay = GROQ_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Groq request failed (attempt %s/%s): %s. Retrying in %.2fs",
                attempt,
                GROQ_RETRIES,
                str(exc),
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


@app.errorhandler(404)
def not_found(_):
    return json_error("Route not found", 404, "not_found")


@app.errorhandler(405)
def method_not_allowed(_):
    return json_error("Method not allowed", 405, "method_not_allowed")


@app.errorhandler(413)
def payload_too_large(_):
    return json_error("Request payload is too large", 413, "payload_too_large")


@app.errorhandler(Exception)
def unexpected_error(exc):  # noqa: ANN001
    logger.exception("Unhandled exception: %s", str(exc))
    return json_error("Internal server error", 500, "internal_error")


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if GROQ_API_KEY and API_KEY_STORAGE else "degraded"
    return jsonify(
        {
            "status": status,
            "available_models": list(AVAILABLE_MODELS.keys()),
            "limits": {
                "max_request_bytes": app.config["MAX_CONTENT_LENGTH"],
                "max_prompt_chars": MAX_PROMPT_CHARS,
            },
        }
    )


@app.route("/ready", methods=["GET"])
def ready():
    if not GROQ_API_KEY:
        return json_error("GROQ_API_KEY is not configured", 503, "service_misconfigured")
    if not API_KEY_STORAGE:
        return json_error("API_KEY_STORAGE is not configured", 503, "service_misconfigured")
    return jsonify({"status": "ready"})


@app.route("/models", methods=["GET"])
@require_api_key
def list_models():
    return jsonify({
        "models": AVAILABLE_MODELS,
        "model_names": list(AVAILABLE_MODELS.keys()),
    })


@app.route("/generate", methods=["POST"])
@require_api_key
def generate():
    data = request.get_json(silent=True)
    prompt, model, options, validation_error = _validate_common_payload(data)
    if validation_error is not None:
        return validation_error

    try:
        response_text, completion = _call_groq_with_retries(model, prompt, options)
    except FileNotFoundError as exc:
        return json_error(str(exc), 400, "invalid_parameter")
    except ValueError as exc:
        return json_error(str(exc), 400, "invalid_parameter")
    except RuntimeError as exc:
        return json_error(str(exc), 503, "service_misconfigured")
    except Exception as groq_error:  # noqa: BLE001
        logger.error("Groq API error for model %s: %s", model, str(groq_error))
        return json_error("Groq API request failed", 502, "groq_api_error", model=model)

    return jsonify(
        {
            "success": True,
            "response": response_text,
            "prompt": prompt,
            "model": model,
            "usage": _safe_usage(completion),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
