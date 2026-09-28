FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

# Recorded-judge only over HTTP (service.py) — no anthropic/litellm needed
# in this image at all, since no live model call is reachable from it.
RUN pip install --no-cache-dir ".[service]"

EXPOSE 8080

CMD ["uvicorn", "eval_pipeline.service:app", "--host", "0.0.0.0", "--port", "8080"]
