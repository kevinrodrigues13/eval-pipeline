FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

# The service is live-judge only, gated behind LIVE_JUDGE_API_KEY /
# ANTHROPIC_API_KEY at deploy time — see DEPLOYMENT.md.
RUN pip install --no-cache-dir ".[service]"

EXPOSE 8080

CMD ["uvicorn", "eval_pipeline.service:app", "--host", "0.0.0.0", "--port", "8080"]
