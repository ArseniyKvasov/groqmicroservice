import os
from flask import Flask, request, jsonify
from groq import Groq
from functools import wraps

app = Flask(__name__)

API_KEY_STORAGE = os.getenv("API_KEY_STORAGE", "my-secret-key-123")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

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
    "whisper-large-v3-turbo": "Whisper Large V3 Turbo"
}


def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != API_KEY_STORAGE:
            return jsonify({
                "error": "Invalid or missing API key",
                "status_code": 401
            }), 401
        return f(*args, **kwargs)

    return decorated


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "available_models": list(AVAILABLE_MODELS.keys())})


@app.route('/models', methods=['GET'])
@require_api_key
def list_models():
    return jsonify({
        "models": AVAILABLE_MODELS,
        "model_names": list(AVAILABLE_MODELS.keys())
    })


@app.route('/generate', methods=['POST'])
@require_api_key
def generate():
    try:
        data = request.json

        prompt = data.get('prompt')
        if not prompt:
            return jsonify({
                "error": "Prompt is required",
                "error_type": "missing_field",
                "status_code": 400
            }), 400

        model = data.get('model', 'llama-3.3-70b-versatile')

        if model not in AVAILABLE_MODELS:
            return jsonify({
                "error": f"Model '{model}' is not available",
                "error_type": "invalid_model",
                "available_models": list(AVAILABLE_MODELS.keys()),
                "models_info": AVAILABLE_MODELS,
                "status_code": 400
            }), 400

        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 5000)

        if not (0 <= temperature <= 2):
            return jsonify({
                "error": "Temperature must be between 0 and 2",
                "error_type": "invalid_parameter",
                "status_code": 400
            }), 400

        if not (1 <= max_tokens <= 32768):
            return jsonify({
                "error": "max_tokens must be between 1 and 32768",
                "error_type": "invalid_parameter",
                "status_code": 400
            }), 400

        try:
            client = get_groq_client()

            if model.startswith('whisper'):
                completion = client.audio.transcriptions.create(
                    model=model,
                    file=open(prompt, 'rb'),
                    response_format="json"
                )
                response_text = completion.text
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response_text = completion.choices[0].message.content

            return jsonify({
                "success": True,
                "response": response_text,
                "prompt": prompt,
                "model": model,
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens if hasattr(completion, 'usage') else None,
                    "completion_tokens": completion.usage.completion_tokens if hasattr(completion, 'usage') else None,
                    "total_tokens": completion.usage.total_tokens if hasattr(completion, 'usage') else None
                } if hasattr(completion, 'usage') else None
            })

        except Exception as groq_error:
            return jsonify({
                "error": f"Groq API error: {str(groq_error)}",
                "error_type": "groq_api_error",
                "model": model,
                "status_code": 500
            }), 500

    except Exception as e:
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "error_type": "internal_error",
            "status_code": 500
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)