#scripts/init-ollama.sh

#!/bin/sh
set -e

echo "🚀 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama to be ready..."
sleep 15

# Проверка, что сервер действительно жив
if ! kill -0 $OLLAMA_PID 2>/dev/null; then
    echo "❌ Ollama failed to start"
    exit 1
fi

# Pull embedding model
if [ -n "$OLLAMA_EMBED_MODEL" ]; then
    echo "📦 Pulling embedding model: $OLLAMA_EMBED_MODEL"
    ollama pull "$OLLAMA_EMBED_MODEL" || echo "⚠️  Failed to pull $OLLAMA_EMBED_MODEL"
fi

# Pull main model
if [ -n "$OLLAMA_MODEL" ]; then
    echo "🤖 Pulling LLM model: $OLLAMA_MODEL"
    ollama pull "$OLLAMA_MODEL" || echo "⚠️  Failed to pull $OLLAMA_MODEL"
fi

echo "✅ All models ready! Ollama is running."

# Держим Ollama живым
wait $OLLAMA_PID
