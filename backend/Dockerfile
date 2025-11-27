#backend/Dockerfile
# === Base image ===
FROM python:3.11-slim

# === Environment optimization ===
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC 
   
# === Working directory ===
WORKDIR /app

# === System deps (нужны sentence-transformers, tokenizers, pydantic-core) ===
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        git \
        curl \
        cargo \
    && rm -rf /var/lib/apt/lists/*

# === Python dependencies ===
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# === Copy app code ===
COPY ./app ./app

# === Expose port ===
EXPOSE 8000

# === Health check (опционально) ===
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

# === Run FastAPI ===
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

