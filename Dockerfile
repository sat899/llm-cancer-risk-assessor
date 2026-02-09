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
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY streamlit_app.py ./
RUN uv sync --frozen --no-dev

# Data dir: mount host data at runtime, e.g. -v "$(pwd)/data:/app/data"
RUN mkdir -p data

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start both API and Streamlit
CMD sh -c "uv run uvicorn src.main:app --host 0.0.0.0 --port ${API_PORT:-8000} & uv run streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"
