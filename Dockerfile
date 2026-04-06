# ── Email Triage OpenEnv — Dockerfile ─────────────────────────────────────
FROM python:3.11-slim

# HF Spaces expects port 7860
EXPOSE 7860

WORKDIR /app/env

# Install dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

ENV PYTHONPATH=/app/env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
