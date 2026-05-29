# RiesgoVial API

Plataforma de predicción de riesgo vial urbano. Multi-ciudad, multi-modelo.

## Arquitectura

```
notebook (.ipynb)
    ↓ export_models.py
models/*.pkl  ←──────────────────────────────────┐
    ↓                                             │
FastAPI (main.py)                                 │
  ├── model_loader.py  ← carga y cachea pkl       │
  ├── explainer.py     ← RAG + SHAP              │
  └── /api/v1/*                                  │
         ↓                                        │
     Nginx                                        │
         ↓                                        │
  Frontend HTML ──── llama a la API ─────────────┘
```

## Estructura de archivos

```
riesgo_api/
├── main.py             ← FastAPI app, endpoints
├── model_loader.py     ← ModelRegistry, CityModel, pipeline de entrenamiento
├── explainer.py        ← RAG + SHAP + LLM opcional
├── export_models.py    ← Script para exportar modelos desde el notebook
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── models/             ← Aquí van los .pkl (crear antes de arrancar)
│   ├── medellin_xgboost.pkl
│   ├── medellin_rf.pkl
│   └── bogota_xgboost.pkl   ← (cuando esté disponible)
└── frontend/
    └── index.html      ← riesgo_urbano_platform.html renombrado
```

## Paso 1 — Exportar modelos desde el notebook

Abre `export_models.py`, descomenta los `export_model(...)` que correspondan
y ejecútalo en el mismo kernel del notebook donde están entrenados los modelos:

```bash
python export_models.py
# Crea: models/medellin_xgboost.pkl
```

## Paso 2 — Levantar con Docker

```bash
# Crear directorio de modelos si no existe
mkdir -p models frontend logs

# Copiar el frontend
cp riesgo_urbano_platform.html frontend/index.html

# Construir y arrancar
docker compose up --build

# Verificar que funciona
curl http://localhost:8000/health
```

La API estará en `http://localhost:8000`
El frontend en `http://localhost:80`
Swagger docs en `http://localhost:8000/docs`

## Paso 3 — Desarrollo local (sin Docker)

```bash
pip install -r requirements.txt

# Correr en modo desarrollo con recarga automática
uvicorn main:app --reload --port 8000
```

## Endpoints principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado de la API y modelos cargados |
| POST | `/api/v1/predict` | Predicción individual |
| GET | `/api/v1/heatmap/{city}/{date}` | Matriz de riesgo completa |
| POST | `/api/v1/explain` | SHAP + RAG explicación |
| GET | `/api/v1/cities` | Modelos disponibles |
| POST | `/api/v1/train/{city}` | Entrenar nuevo modelo (async) |
| GET | `/api/v1/jobs/{job_id}` | Estado de entrenamiento |

## Ejemplo de uso

```bash
# Predecir riesgo
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "medellin",
    "neighborhood": "La Candelaria",
    "day_of_week": 4,
    "hour": 23,
    "date": "2025-12-31"
  }'

# Respuesta:
# {
#   "city": "medellin",
#   "neighborhood": "La Candelaria",
#   "probability": 0.9421,
#   "risk_level": "alto",
#   "model_used": "medellin_xgboost_v1.0"
# }
```

```bash
# Entrenar nuevo modelo (ej: Bogotá)
curl -X POST http://localhost:8000/api/v1/train/bogota \
  -H "Content-Type: application/json" \
  -d '{
    "city": "bogota",
    "data_path": "/app/data/bogota_incidentes.csv",
    "model_type": "xgboost",
    "year_from": 2018
  }'
# Devuelve job_id para seguimiento
```

## Agregar una ciudad nueva

1. Prepara el CSV con columnas:
   `LLAVE, AÑO, FECHA_INCIDENTE, HORA_INCIDENTE, CLASE_INCIDENTE,
    GRAVEDAD_INCIDENTE, Codigo Comuna Planeacion, Comuna Planeacion,
    Codigo Barrio Planeacion, Barrio Planeacion`

2. Coloca el CSV en `/app/data/` dentro del contenedor (o ajusta el volumen en `docker-compose.yml`)

3. Llama a `POST /api/v1/train/{ciudad}` — el pipeline completo corre en background

4. Consulta el estado con `GET /api/v1/jobs/{job_id}` hasta ver `"status": "done"`

5. El nuevo modelo queda disponible inmediatamente en `GET /api/v1/cities`

## Activar RAG con LLM real

Agrega tu API key en `docker-compose.yml`:

```yaml
environment:
  - ANTHROPIC_API_KEY=sk-ant-...
```

El `explainer.py` detectará automáticamente la key y usará Claude Sonnet
para generar explicaciones enriquecidas en lugar de las basadas en reglas.

## Despliegue en producción (Railway / Render)

```bash
# Railway
railway login
railway init
railway up

# Variables de entorno en el dashboard:
# MODELS_DIR=/app/models
# ANTHROPIC_API_KEY=sk-ant-...
```
