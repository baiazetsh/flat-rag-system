#!/bin/sh
set -e

echo "=== Ollama init script starting ==="
echo "LLM_MODEL: ${LLM_MODEL}"
echo "EMBEDDING_MODEL: ${EMBEDDING_MODEL}"

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Stop server on container shutdown
trap "kill $OLLAMA_PID 2>/dev/null || true; exit 0" TERM INT

# Wait for API to become ready
echo "Waiting for Ollama API..."
MAX_RETRIES=120
i=0

while [ $i -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:11434/api/tags >/dev/null; then
        echo "Ollama API is ready."
        break
    fi
    i=$((i+1))
    sleep 1
done

if [ $i -eq $MAX_RETRIES ]; then
    echo "❌ ERROR: Ollama API failed to start"
    kill $OLLAMA_PID || true
    exit 1
fi

model_exists() {
    curl -s http://localhost:11434/api/show \
      -d "{\"name\":\"$1\"}" | grep -q '"model"'
}

# Pull embedding model
if [ -n "$EMBEDDING_MODEL" ]; then
    echo "Checking embedding model: $EMBEDDING_MODEL"
    if ! model_exists "$EMBEDDING_MODEL"; then
        echo "Pulling embedding model..."
        ollama pull "$EMBEDDING_MODEL"
        sleep 2
    else
        echo "Embedding model already installed."
    fi
fi

# Pull main model
if [ -n "$LLM_MODEL" ]; then
    echo "Checking LLM model: $LLM_MODEL"
    if ! model_exists "$LLM_MODEL"; then
        echo "Pulling LLM model..."
        ollama pull "$LLM_MODEL"
        sleep 2
    else
        echo "LLM model already installed."
    fi
fi

echo "=== All models ready ==="

wait $OLLAMA_PID
