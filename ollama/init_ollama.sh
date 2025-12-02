#!/bin/sh
set -e

echo "=== Ollama init script starting ==="
echo "LLM_MODEL: ${LLM_MODEL}"
echo "EMBEDDING_MODEL: ${EMBEDDING_MODEL}"
echo "KEEP_ALIVE: ${OLLAMA_KEEP_ALIVE}"

# ---------- Start server ----------
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

trap "kill $OLLAMA_PID 2>/dev/null || true; exit 0" TERM INT

# ---------- Wait for server ----------
echo "Waiting for Ollama API ready..."

for i in $(seq 1 120); do
    if curl -sf http://localhost:11434/api/tags >/dev/null; then
        echo "✔ Ollama API is ready"
        break
    fi
    sleep 1
done

# ---------- Model exists ----------
model_exists() {
    curl -s http://localhost:11434/api/show \
      -d "{\"name\":\"$1\"}" | grep -q '"name"'
}

# ---------- Pull / Load embedding model ----------
if [ -n "$EMBEDDING_MODEL" ]; then
    echo "Checking embedding model: $EMBEDDING_MODEL"

    if ! model_exists "$EMBEDDING_MODEL"; then
        echo "⬇ Pulling embedding model..."
        ollama pull "$EMBEDDING_MODEL"
    else
        echo "✔ Embedding model already installed"
    fi

    echo "🔥 Warming up embeddings..."
    curl -s http://localhost:11434/api/embeddings \
      -d "{\"model\": \"$EMBEDDING_MODEL\", \"prompt\": \"warmup\"}" >/dev/null
fi

# ---------- Pull / Load LLM ----------
if [ -n "$LLM_MODEL" ]; then
    echo "Checking LLM model: $LLM_MODEL"

    if ! model_exists "$LLM_MODEL"; then
        echo "⬇ Pulling LLM model..."
        ollama pull "$LLM_MODEL"
    else
        echo "✔ LLM model already installed"
    fi

    echo "🔥 Warming up LLM..."
    curl -s http://localhost:11434/api/generate \
      -d "{\"model\": \"$LLM_MODEL\", \"prompt\": \"warmup\", \"stream\": false}" >/dev/null
fi

echo "=== All models loaded & warmed ==="

# ---------- Do NOT block the entrypoint ----------
wait $OLLAMA_PID
