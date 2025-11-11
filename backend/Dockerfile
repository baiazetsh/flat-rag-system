# rag_local/backend/Dockerfile

# === Base image ===
FROM python:3.11-slim

# === Environment optimization ===
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC

# === Working directory ===
WORKDIR /app

# === System deps ===
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# === Python deps ===
COPY requirements.txt .
COPY backend/requirements-test.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-test.txt --break-system-packages

# === Copy app code ===
COPY ./app ./app

# === Security: non-root user ===
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# === Expose port ===
EXPOSE 8000

# === Health check ===
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# === Run FastAPI ===
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
