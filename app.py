import hmac
import logging
import os
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import groq
from flask import Flask, g, jsonify, request
from groq import Groq
from werkzeug.exceptions import HTTPException


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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant")
MAX_PROMPT_CHARS = _env_int("MAX_PROMPT_CHARS", 20000, min_value=1)
MAX_AUDIO_FILE_SIZE_BYTES = _env_int("MAX_AUDIO_FILE_SIZE_BYTES", 25 * 1024 * 1024, min_value=1024)
MAX_REQUEST_BYTES = _env_int("MAX_REQUEST_BYTES", 1 * 1024 * 1024, min_value=1024)

GROQ_TIMEOUT_SECONDS = _env_float("GROQ_TIMEOUT_SECONDS", 45.0, min_value=3.0, max_value=300.0)
GROQ_RETRY_ATTEMPTS = _env_int("GROQ_RETRY_ATTEMPTS", 3, min_value=2, max_value=4)
GROQ_RETRY_BASE_DELAY_SECONDS = _env_float("GROQ_RETRY_BASE_DELAY_SECONDS", 0.6, min_value=0.1, max_value=10.0)

HEALTH_CACHE_SECONDS = _env_int("HEALTH_CACHE_SECONDS", 25, min_value=5, max_value=60)
HEALTH_TIMEOUT_SECONDS = _env_float("HEALTH_TIMEOUT_SECONDS", 5.0, min_value=1.0, max_value=20.0)

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


class EmptyUpstreamResponseError(Exception):
    """Raised when upstream returns an empty response payload."""


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("groq-service")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

_groq_client: Optional[Groq] = None
_health_cache: Dict[str, Any] = {
    "checked_at": 0.0,
    "upstream_ok": False,
    "upstream_status": None,
    "error_type": None,
}
_health_cache_lock = threading.Lock()


def _json_response(payload: Dict[str, Any], status_code: int):
    response = jsonify(payload)
    response.status_code = status_code
    response.headers["Content-Type"] = "application/json"
    return response


def _generate_error(error: str, error_type: str, status_code: int):
    return _json_response(
        {
            "success": False,
            "error": error,
            "error_type": error_type,
            "status_code": status_code,
        },
        status_code,
    )


def _api_error(error: str, error_type: str, status_code: int):
    return _json_response(
        {
            "error": error,
            "error_type": error_type,
            "status_code": status_code,
        },
        status_code,
    )


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured")
        _groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=GROQ_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _groq_client


def _upstream_status_from_error(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return status_code


def _map_upstream_error(exc: Exception) -> Tuple[int, str, str, Optional[int]]:
    upstream_status = _upstream_status_from_error(exc)

    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)) or upstream_status in (401, 403):
        return 401, "upstream_auth_error", "Groq upstream authentication failed", upstream_status

    if isinstance(exc, groq.RateLimitError) or upstream_status == 429:
        return 429, "upstream_rate_limit", "Groq upstream rate limit exceeded", 429

    if isinstance(exc, (groq.APITimeoutError, groq.APIConnectionError)):
        return 504, "upstream_timeout", "Groq upstream timeout/connection error", upstream_status

    if isinstance(exc, groq.APIStatusError) and upstream_status is not None and upstream_status >= 500:
        return 502, "upstream_server_error", "Groq upstream server error", upstream_status

    if isinstance(exc, EmptyUpstreamResponseError):
        return 502, "upstream_empty_response", str(exc), upstream_status

    if isinstance(exc, RuntimeError):
        return 500, "service_misconfigured", str(exc), upstream_status

    return 502, "upstream_error", "Groq upstream request failed", upstream_status


def _should_retry_upstream(exc: Exception) -> bool:
    if isinstance(exc, (groq.APITimeoutError, groq.APIConnectionError)):
        return True

    status_code = _upstream_status_from_error(exc)
    if status_code == 429:
        return True
    if status_code is not None and status_code >= 500:
        return True
    return False


def _safe_usage(completion: Any) -> Dict[str, Any]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _extract_response_text(model: str, prompt: str, completion: Any) -> str:
    if model.startswith("whisper"):
        response_text = getattr(completion, "text", None)
    else:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise EmptyUpstreamResponseError("Groq returned no choices")
        message = getattr(choices[0], "message", None)
        response_text = getattr(message, "content", None)

    if not isinstance(response_text, str) or not response_text.strip():
        raise EmptyUpstreamResponseError("Groq returned empty response content")

    return response_text


def _validate_generate_payload(data: Any):
    if not isinstance(data, dict):
        return None, _generate_error("JSON body is required", "invalid_json", 400)

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, _generate_error("Field 'prompt' is required and must be a string", "missing_field", 400)

    model = data.get("model", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        return None, _generate_error("Unsupported model", "invalid_model", 400)

    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1024)

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return None, _generate_error("Field 'temperature' must be a number", "invalid_parameter", 400)

    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return None, _generate_error("Field 'max_tokens' must be an integer", "invalid_parameter", 400)

    if not 0 <= temperature <= 2:
        return None, _generate_error("Field 'temperature' must be between 0 and 2", "invalid_parameter", 400)

    if not 1 <= max_tokens <= 32768:
        return None, _generate_error("Field 'max_tokens' must be between 1 and 32768", "invalid_parameter", 400)

    if model.startswith("whisper"):
        audio_path = Path(prompt)
        if not audio_path.exists() or not audio_path.is_file():
            return None, _generate_error(
                "For whisper models, prompt must be a valid local file path",
                "invalid_parameter",
                400,
            )
        if audio_path.stat().st_size > MAX_AUDIO_FILE_SIZE_BYTES:
            return None, _generate_error(
                "Audio file is too large",
                "invalid_parameter",
                400,
            )
    elif len(prompt) > MAX_PROMPT_CHARS:
        return None, _generate_error(
            "Prompt exceeds max allowed length",
            "invalid_parameter",
            400,
        )

    payload = {
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return payload, None


def _check_upstream_cached() -> Dict[str, Any]:
    now = time.time()

    with _health_cache_lock:
        if _health_cache["checked_at"] and now - _health_cache["checked_at"] <= HEALTH_CACHE_SECONDS:
            return dict(_health_cache)

    result = {
        "checked_at": now,
        "upstream_ok": False,
        "upstream_status": None,
        "error_type": None,
    }

    try:
        client = get_groq_client()
        # Lightweight real upstream check.
        client.with_options(timeout=HEALTH_TIMEOUT_SECONDS, max_retries=0).models.list()
        result["upstream_ok"] = True
        result["upstream_status"] = 200
    except Exception as exc:  # noqa: BLE001
        _, error_type, _, upstream_status = _map_upstream_error(exc)
        result["upstream_ok"] = False
        result["upstream_status"] = upstream_status
        result["error_type"] = error_type

    with _health_cache_lock:
        _health_cache.update(result)
        return dict(_health_cache)


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY_STORAGE:
            logger.error("request_id=%s service auth key is not configured", getattr(g, "request_id", "n/a"))
            if request.path == "/generate":
                return _generate_error("Server authentication is not configured", "service_misconfigured", 500)
            return _api_error("Server authentication is not configured", "service_misconfigured", 500)

        api_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(api_key, API_KEY_STORAGE):
            if request.path == "/generate":
                return _generate_error("Invalid or missing API key", "unauthorized", 401)
            return _api_error("Invalid or missing API key", "unauthorized", 401)
        return f(*args, **kwargs)

    return decorated


@app.before_request
def attach_request_id():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())


@app.after_request
def ensure_json_and_request_id(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    response.headers["Content-Type"] = "application/json"
    return response


@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    status_code = exc.code or 500
    if request.path == "/generate":
        return _generate_error(exc.description or "HTTP error", "http_error", status_code)
    return _api_error(exc.description or "HTTP error", "http_error", status_code)


@app.errorhandler(Exception)
def handle_unexpected_exception(exc):  # noqa: ANN001
    logger.exception("request_id=%s unhandled_error=%s", getattr(g, "request_id", "n/a"), str(exc))
    if request.path == "/generate":
        return _generate_error("Internal server error", "internal_error", 500)
    return _api_error("Internal server error", "internal_error", 500)


@app.route("/health", methods=["GET"])
def health():
    upstream = _check_upstream_cached()
    return _json_response(
        {
            "status": "ok" if upstream["upstream_ok"] else "degraded",
            "upstream_ok": bool(upstream["upstream_ok"]),
            "model_default": DEFAULT_MODEL,
        },
        200 if upstream["upstream_ok"] else 503,
    )


@app.route("/models", methods=["GET"])
@require_api_key
def list_models():
    return _json_response(
        {
            "models": AVAILABLE_MODELS,
            "model_names": list(AVAILABLE_MODELS.keys()),
        },
        200,
    )


@app.route("/generate", methods=["POST"])
@require_api_key
def generate():
    data = request.get_json(silent=True)
    payload, validation_error = _validate_generate_payload(data)
    if validation_error is not None:
        return validation_error

    prompt = payload["prompt"]
    model = payload["model"]
    temperature = payload["temperature"]
    max_tokens = payload["max_tokens"]

    request_id = getattr(g, "request_id", "n/a")

    last_exception: Optional[Exception] = None
    for attempt in range(1, GROQ_RETRY_ATTEMPTS + 1):
        try:
            client = get_groq_client()

            if model.startswith("whisper"):
                completion = client.audio.transcriptions.create(
                    model=model,
                    file=Path(prompt),
                    response_format="json",
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            response_text = _extract_response_text(model, prompt, completion)
            return _json_response(
                {
                    "success": True,
                    "response": response_text,
                    "model": model,
                    "usage": _safe_usage(completion),
                },
                200,
            )
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            should_retry = attempt < GROQ_RETRY_ATTEMPTS and _should_retry_upstream(exc)
            status_code, error_type, error_message, upstream_status = _map_upstream_error(exc)

            logger.warning(
                (
                    "request_id=%s model=%s upstream_status=%s error_type=%s "
                    "attempt=%s/%s retry=%s timeout=%s error=%s"
                ),
                request_id,
                model,
                upstream_status,
                error_type,
                attempt,
                GROQ_RETRY_ATTEMPTS,
                should_retry,
                isinstance(exc, (groq.APITimeoutError, groq.APIConnectionError)),
                str(exc),
            )

            if should_retry:
                delay = GROQ_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                time.sleep(delay)
                continue

            return _generate_error(error_message, error_type, status_code)

    status_code, error_type, error_message, upstream_status = _map_upstream_error(last_exception or Exception())
    logger.error(
        "request_id=%s model=%s upstream_status=%s timeout=%s error_type=%s final_error=%s",
        request_id,
        model,
        upstream_status,
        isinstance(last_exception, (groq.APITimeoutError, groq.APIConnectionError)),
        error_type,
        str(last_exception),
    )
    return _generate_error(error_message, error_type, status_code)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
