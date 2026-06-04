# ──────────────────────────────────────────────
# RiesgoVial API — Dockerfile multi-stage
# Etapa 1: builder (instala deps pesadas)
# Etapa 2: runtime mínimo
# ──────────────────────────────────────────────

# ── STAGE 1: builder ──────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Dependencias del sistema para compilar paquetes nativos
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (cache layer)
COPY requirements.txt .

# Instalar en directorio temporal para copiar al runtime
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── STAGE 2: runtime ──────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Solo libgomp para XGBoost/LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias instaladas
COPY --from=builder /install /usr/local

# Copiar código de la aplicación
COPY main.py model_loader.py explainer.py ./
COPY rag/ ./rag/
COPY models/ ./models/
COPY frontend/ ./frontend/

RUN mkdir -p /app/logs

# Variables de entorno
ENV MODELS_DIR=/app/models \
    RAG_DIR=/app/rag \
    PORT=8000 \
    WORKERS=2 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Usuario no-root por seguridad
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Arrancar con Gunicorn + Uvicorn workers
CMD ["sh", "-c", \
     "gunicorn main:app \
      --worker-class uvicorn.workers.UvicornWorker \
      --workers ${WORKERS} \
      --bind 0.0.0.0:${PORT} \
      --timeout 120 \
      --log-level ${LOG_LEVEL} \
      --access-logfile - \
      --error-logfile -"]
