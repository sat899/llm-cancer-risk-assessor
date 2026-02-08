# Use Python 3.11 slim image as base
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    GOOGLE_CLOUD_PROJECT="" \
    GOOGLE_CLOUD_LOCATION="us-central1" \
    VERTEX_AI_MODEL_NAME="gemini-2.5-flash-lite" \
    VERTEX_AI_EMBEDDING_MODEL="text-embedding-004" \
    API_PORT=8000

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with uv (uses lockfile, no dev deps)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --system

# Data dir: mount host data at runtime, e.g. -v "$(pwd)/data:/app/data"
RUN mkdir -p data

# Create credentials directory (mount service account key here)
RUN mkdir -p /app/credentials

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${API_PORT:-8000}"]
