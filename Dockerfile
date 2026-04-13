FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys;urllib.request.urlopen('http://127.0.0.1:8080/live', timeout=3);sys.exit(0)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--worker-class", "gthread", "--workers", "1", "--threads", "8", "--timeout", "120", "--graceful-timeout", "30", "--keep-alive", "5", "app:app"]
