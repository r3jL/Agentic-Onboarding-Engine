# -----------------------------------------------------------------------
# Dockerfile for AI Onboarding Engine
# Runs FastAPI, plus Ollama (qwen2.5:7b) when built with WITH_OLLAMA=true.
# Default build is lean: it expects a hosted LLM via LLM_BASE_URL/LLM_API_KEY.
# docker-compose.yml builds with WITH_OLLAMA=true for fully local use; then the
# first run downloads the model (~4.7 GB) and later runs use the cache.
# -----------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama only for local/offline builds
ARG WITH_OLLAMA=false
RUN if [ "$WITH_OLLAMA" = "true" ]; then curl -fsSL https://ollama.com/install.sh | sh; fi

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY public/ ./public/

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Directory for SQLite database
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SKIP_VENV=1
ENV DB_PATH=/app/data/onboarding.db

EXPOSE 8000

# Health check (waits up to 5 min for model download on first run)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=10 \
    CMD curl -sf http://localhost:${PORT:-8000}/ || exit 1

CMD ["./start.sh"]
