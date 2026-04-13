# groqmicroservice

## Local run (1 CPU, concurrent requests)

```bash
cd /Users/arseniy/PycharmProjects/GroqService/microservice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill GROQ_API_KEY and API_KEY_STORAGE
export $(grep -v '^#' .env | xargs)
gunicorn --bind 0.0.0.0:8080 --worker-class gthread --workers 1 --threads 8 --timeout 120 --graceful-timeout 30 --keep-alive 5 app:app
```

## Docker run

```bash
cd /Users/arseniy/PycharmProjects/GroqService/microservice
docker build -t groq-service:stable .
docker run --rm -p 8080:8080 --env-file .env groq-service:stable
```

## API checks

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY_STORAGE}" \
  -d '{"prompt":"Hello","model":"llama-3.1-8b-instant","temperature":0.7,"max_tokens":128}'
```


## Notes

- `/live` is for container liveness checks (no upstream call).
- `/health` checks Groq upstream and uses cache (`HEALTH_CACHE_SECONDS`).
- If you get many `429`, reduce `max_tokens`, lower request rate, or upgrade Groq plan.
