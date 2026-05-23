#!/bin/sh
set -e

OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"

echo "Ожидание Ollama на ${OLLAMA_URL}..."
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; do
  sleep 3
done
echo "Ollama готов."

exec python app.py
