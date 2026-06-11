# RiesgoVial — Manual de inicio

## Requisitos previos

### Linux
- **Docker Engine** y **Docker Compose** instalados
- Usuario en el grupo `docker`:
  ```bash
  sudo usermod -aG docker $USER
  # Cerrar sesión y volver a entrar (o ejecutar: newgrp docker)
  ```

### Windows
- **Docker Desktop** instalado y corriendo (icono en la barra del sistema)
- Usar **PowerShell** o **Git Bash** como terminal

---

## Orden de preparación de artefactos

El sistema depende de **4 artefactos** que deben generarse en este orden antes de levantar Docker. Si alguno falta o está desactualizado, la funcionalidad afectada falla **silenciosamente** (sin error visible en el frontend).

```
Paso 0 ── run_tabnet.py / notebook
              │  genera
              ▼
Paso 1 ── models/medellin_xgboost.pkl        ← predicciones + SHAP
              │  requiere (mismo bundle)
              ▼
Paso 2 ── python3 export_model_knowledge.py
              │  genera
              ├──▶ rag/documents/medellin/model_knowledge.json  ← Chat IA
              └──▶ rag/documents/medellin/todos_los_barrios.txt ← endpoint /explain

Paso 3 ── python3 riesgo_api/rag/geocode_barrios.py
              │  genera
              └──▶ rag/documents/medellin/coordinates.json      ← mapa Leaflet

Paso 4 ── docker compose up --build -d
```

| Si omites | Se rompe |
|---|---|
| Paso 1 — `.pkl` | La API no arranca (`ModelRegistry` lanza error al inicio) |
| Paso 2 — `model_knowledge.json` | Chat responde "base de conocimiento no disponible" (Error 404) |
| Paso 2 — `todos_los_barrios.txt` | `/explain` devuelve explicación sin contexto del barrio (degradada, sin error visible) |
| Paso 3 — `coordinates.json` | Mapa Leaflet aparece vacío sin círculos (sin error visible) |

> **Después de reentrenar el modelo** siempre ejecuta los pasos 2 y 3 antes de `docker compose up --build`.

---

## Levantar el proyecto

**Paso 1 — Verificar que Docker esté corriendo:**

```bash
docker info
```

Si aparece `dial unix /var/run/docker.sock: no such file or directory`:

```bash
# Linux
sudo systemctl start docker

# Windows
# Abrir Docker Desktop desde el menú de inicio
```

**Paso 2 — Levantar los contenedores:**

```bash
cd ~/Proyectos/Notebooks/riesgo_api
docker compose up -d
```

Primera vez o después de cambios en el código:

```bash
docker compose up --build -d
```

**Accesos:**

| Servicio | URL |
|---|---|
| Frontend | http://localhost |
| API (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

---

## Comandos útiles

```bash
# Ver estado de los contenedores
docker compose ps

# Ver logs en tiempo real
docker compose logs -f api

# Parar los contenedores
docker compose down

# Reiniciar solo la API (sin reconstruir imagen)
docker compose restart api

# Reiniciar solo nginx (cambios en HTML)
docker compose restart nginx
```

---

## Variables de configuración

Todas se definen en `docker-compose.yml` bajo `environment`. Cambiar y ejecutar `docker compose up -d` (sin `--build`):

| Variable | Default | Descripción |
|---|---|---|
| `GROQ_API_KEY` | — | API key de Groq (requerida para el Chat IA; para `/explain` es una de las dos opciones) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Modelo LLM usado para explicaciones y chat |
| `ANTHROPIC_API_KEY` | — | API key de Anthropic Claude (opcional). En `/explain` se usa como proveedor si no hay `GROQ_API_KEY`, o como respaldo automático si Groq falla. El Chat IA no usa Anthropic. |
| `RAG_MAX_TOKENS` | `400` | Tokens máximos en explicaciones de barrios |
| `CHAT_MAX_TOKENS` | `1000` | Tokens máximos en respuestas del chatbot |
| `CHAT_HISTORY_TURNS` | `10` | Turnos de historial que se envían al chat |
| `CHAT_RAG_TOP_K` | `5` | Chunks relevantes que se recuperan por pregunta |

---

## Funcionalidades

### Consulta de riesgo
Predice la probabilidad de accidente para un barrio, fecha y hora. Deriva el día automáticamente de la fecha seleccionada. Muestra porcentaje de riesgo, gráfico gauge, gráfico de barras SHAP y explicación generada por IA (Groq y/o Anthropic Claude, según las API keys configuradas).

### Dashboard
- **KPIs**: total incidentes, hora pico, barrio más crítico, días desde último incidente
- **Gráfico por hora**: distribución histórica 0–23h con rangos 7d / 30d / 90d
- **Tendencia diaria**: evolución día a día del período seleccionado
- **Mapa de riesgo**: hasta 315 barrios con círculos coloreados por nivel de riesgo del día actual. Hover muestra nombre y %. Clic abre consulta directa.
- **Heatmap**: matriz barrio × franja horaria con riesgo predicho

### Chat IA
Chatbot sobre el modelo, los datos, los barrios y el proceso de entrenamiento. Usa RAG con TF-IDF sobre el knowledge JSON para enviar solo el contexto relevante (~500–700 tokens por pregunta). Muestra tokens consumidos por iteración. Filtra preguntas no relacionadas con RiesgoVial.

### Modelos
Gestión de modelos por ciudad. Métricas, distribución de riesgo, top 10 barrios más/menos peligrosos y tabla comparativa cargados dinámicamente desde la API.

### Documentación
Guía completa de uso, arquitectura, visualizaciones y referencia técnica.

---

## Archivos de conocimiento (RAG)

Ubicados en `rag/documents/medellin/`:

| Archivo | Usado por | Generado por |
|---|---|---|
| `todos_los_barrios.txt` | `explainer.py` — endpoint `/explain` | `export_model_knowledge.py` |
| `model_knowledge.json` | `main.py` — Chat IA (TF-IDF) | `export_model_knowledge.py` |
| `coordinates.json` | `main.py` — mapa Leaflet | `rag/geocode_barrios.py` |

Para regenerar después de reentrenar el modelo (genera los tres artefactos en orden):

```bash
cd ~/Proyectos/Notebooks

# 1. Regenera model_knowledge.json Y todos_los_barrios.txt
python3 export_model_knowledge.py

# 2. Regenera coordinates.json (solo si cambian los barrios; tarda ~8 min)
python3 riesgo_api/rag/geocode_barrios.py

# 3. Reconstruir y levantar
cd riesgo_api
docker compose up --build -d
```

---

## Estructura de archivos clave

```
riesgo_api/
├── main.py                     ← API FastAPI (endpoints)
├── model_loader.py             ← Carga del modelo y predicciones
├── explainer.py                ← Explicaciones con Groq + RAG de barrios
├── requirements.txt
├── Dockerfile
├── docker-compose.yml          ← Variables de configuración
├── nginx.conf
├── models/
│   └── medellin_xgboost.pkl   ← Modelo XGBoost entrenado (~2.4 MB)
├── rag/
│   ├── geocode_barrios.py      ← Geocodifica barrios con Nominatim
│   └── documents/medellin/
│       ├── todos_los_barrios.txt
│       ├── coordinates.json
│       └── model_knowledge.json
├── frontend/
│   └── index.html              ← Aplicación web (5 páginas)
└── tests/
    └── test_api.py             ← pytest (17 tests)
```

---

## Endpoints principales

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Estado de la API |
| `POST` | `/api/v1/predict` | Predicción de riesgo para barrio/fecha/hora |
| `POST` | `/api/v1/explain` | Explicación SHAP + RAG con Groq |
| `GET` | `/api/v1/heatmap/{city}/{date}` | Matriz de riesgo por barrio y franja |
| `GET` | `/api/v1/neighborhoods/{city}` | Lista de barrios válidos |
| `GET` | `/api/v1/model-info/{city}` | Métricas, distribución y top barrios |
| `GET` | `/api/v1/coordinates/{city}` | Coordenadas geográficas por ciudad |
| `POST` | `/api/v1/chat` | Chat IA con RAG TF-IDF |
| `GET` | `/api/v1/chat/config` | Configuración del chat (turns, top_k) |

---

## Instalar en otro computador

### Opción A — Copiar el proyecto completo (recomendado)

```bash
cd ~/Proyectos/Notebooks
tar -czf riesgovial_completo.tar.gz \
  riesgo_api/ \
  Fatal_Road_Traffic_Normalizado.xlsx \
  modelo_riesgo_enriquecido_covid.ipynb \
  export_model_knowledge.py

scp riesgovial_completo.tar.gz usuario@ip-destino:~/
```

En el destino:

```bash
tar -xzf riesgovial_completo.tar.gz
cd riesgo_api
# Editar GROQ_API_KEY en docker-compose.yml
docker compose up -d
```

### Opción B — Solo el código (requiere reentrenar)

```bash
# 1. Instalar Python y dependencias
pip install xgboost scikit-learn pandas numpy openpyxl groq

# 2. Colocar Fatal_Road_Traffic_Normalizado.xlsx en ~/Proyectos/Notebooks/

# 3. Ejecutar el notebook para entrenar
jupyter nbconvert --to script modelo_riesgo_enriquecido_covid.ipynb
python3 modelo_riesgo_enriquecido_covid.py

# 4. Copiar el modelo generado
cp models/medellin_xgboost.pkl riesgo_api/models/

# 5. Generar knowledge base del chat Y descripciones de barrios para /explain
python3 export_model_knowledge.py
# → genera model_knowledge.json y todos_los_barrios.txt

# 6. Geocodificar barrios para el mapa (tarda ~8 min)
python3 riesgo_api/rag/geocode_barrios.py
# → genera coordinates.json

# 7. Levantar Docker
cd riesgo_api
docker compose up --build -d
```

---

## Solución a problemas comunes

| Problema | Solución |
|---|---|
| Docker daemon apagado (Linux) | `sudo systemctl start docker` |
| Docker daemon apagado (Windows) | Abrir Docker Desktop |
| `permission denied docker.sock` | `newgrp docker` o abrir terminal nueva |
| Puerto 80 ocupado | Cambiar `"80:80"` → `"8080:80"` en `docker-compose.yml` |
| Página no actualiza | `Ctrl+Shift+R` en el browser |
| Heatmap tarda (primera vez) | Normal — ~4s, luego caché instantáneo |
| Chat responde "Error 500" | Verificar `GROQ_API_KEY` en `docker-compose.yml` |
| Chat sin respuesta de IA | Sin `GROQ_API_KEY` usa RAG local sin LLM |
| Mapa sin barrios | Verificar que `coordinates.json` esté en `rag/documents/medellin/` |
| `/explain` sin contexto del barrio | Ejecutar `python3 export_model_knowledge.py` para regenerar `todos_los_barrios.txt` |
| Chat responde "base de conocimiento no disponible" | Ejecutar `python3 export_model_knowledge.py` para regenerar `model_knowledge.json` |
| Explicaciones hablan de features que ya no existen | El modelo fue reentrenado pero no se corrió `export_model_knowledge.py` después |
| Contenedor no inicia | `docker compose logs api` para ver el error |

---

## Ejecutar tests

```bash
cd riesgo_api
pip install pytest httpx fastapi
pytest tests/ -v
```

Resultado esperado: **17 passed**.
