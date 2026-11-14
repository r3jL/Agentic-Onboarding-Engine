#!/bin/bash
set -e

# ── .env check ────────────────────────────────────────────────────────────────
# Hosting platforms inject env vars directly, so a .env file is only required
# when ADMIN_PASSWORD isn't already set.
if [ ! -f ".env" ] && [ -z "$ADMIN_PASSWORD" ]; then
  echo "ERROR: .env file not found."
  echo "Copy .env.example to .env and fill in your credentials:"
  echo "  cp .env.example .env"
  exit 1
fi

# Read LLM settings from .env when they aren't already in the environment
env_from_file() {
  [ -f ".env" ] && grep -E "^$1=" .env | tail -n 1 | cut -d= -f2- | tr -d "\"'" || true
}
LLM_BASE_URL="${LLM_BASE_URL:-$(env_from_file LLM_BASE_URL)}"
LLM_MODEL="${LLM_MODEL:-$(env_from_file LLM_MODEL)}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434/v1}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
case "$LLM_BASE_URL" in
  *localhost*|*127.0.0.1*) USE_OLLAMA=1 ;;
  *) USE_OLLAMA=0 ;;
esac

# ── Python virtualenv + dependencies ──────────────────────────────────────────
# The Docker image sets SKIP_VENV=1 because dependencies are installed system-wide.
if [ -n "$SKIP_VENV" ]; then
  UVICORN="uvicorn"
else
  if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
  fi

  echo "Installing Python dependencies..."
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
  UVICORN=".venv/bin/uvicorn"
fi

# ── Ollama (only when the LLM endpoint is local) ─────────────────────────────
if [ "$USE_OLLAMA" = "1" ]; then
  if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed."
  fi

  # Start Ollama in the background (ignore error if already running)
  ollama serve 2>/dev/null &
  OLLAMA_PID=$!

  echo "Waiting for Ollama to start..."
  until curl -s http://localhost:11434 > /dev/null 2>&1; do
    sleep 1
  done
  echo "Ollama is ready."

  if ! ollama list | grep -q "$LLM_MODEL"; then
    echo "Pulling $LLM_MODEL model..."
    ollama pull $LLM_MODEL
  fi
else
  echo "Using hosted LLM at $LLM_BASE_URL (model: $LLM_MODEL) — skipping Ollama."
fi

# ── FastAPI ───────────────────────────────────────────────────────────────────
echo "Starting FastAPI..."
exec $UVICORN app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
