"""
RiesgoVial API — main.py
Pydantic v2 compatible: sin @validator, sin model_config, sin prefijo model_
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from contextlib import asynccontextmanager
from datetime import date, datetime
from enum import Enum
import logging
import uuid
import os
import json as _json
from pathlib import Path as _Path

from model_loader import ModelRegistry
from explainer import RiskExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("riesgovial")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precalienta el índice RAG del Chat IA para cada ciudad con modelo cargado."""
    for c in registry.city_summary():
        _get_rag(c["city"])
    yield

app = FastAPI(
    title="RiesgoVial API",
    description="Predicción de riesgo vial urbano. Multi-ciudad, multi-modelo.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()
explainer = RiskExplainer()


# ── Enums simples (evitan cualquier conflicto de namespace) ─────────

class CityEnum(str, Enum):
    medellin = "medellin"
    bogota   = "bogota"
    cali     = "cali"

class AlgoEnum(str, Enum):
    xgboost = "xgboost"
    rf      = "rf"
    lr      = "lr"


# ── Schemas ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    city:        CityEnum         = CityEnum.medellin
    neighborhood: str
    day_of_week: Annotated[int, Field(ge=0, le=6)]
    hour:        Annotated[int, Field(ge=0, le=23)]
    fecha:       Optional[str]    = None   # "YYYY-MM-DD"
    algo:        Optional[AlgoEnum] = None


class PredictResponse(BaseModel):
    city:          str
    neighborhood:  str
    day_of_week:   int
    hour:          int
    fecha:         Optional[str]
    probability:   float
    risk_level:    str
    used_model:    str
    prediction_id: str


class ExplainRequest(BaseModel):
    city:         CityEnum = CityEnum.medellin
    neighborhood: str
    day_of_week:  Annotated[int, Field(ge=0, le=6)]
    hour:         Annotated[int, Field(ge=0, le=23)]
    fecha:        Optional[str] = None


class ExplainResponse(BaseModel):
    shap_values:     dict
    rag_explanation: str
    rag_context:     str
    top_features:    List[dict]


class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message:  str
    city:     str = "medellin"
    history:  List[ChatMessage] = []

class ChatResponse(BaseModel):
    response:     str
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0


class HeatmapResponse(BaseModel):
    city:     str
    fecha:    str
    matrix:   List[dict]
    metadata: dict


class TrainRequest(BaseModel):
    city:      str
    data_path: str
    algo:      str         = "xgboost"
    year_from: Optional[int] = None


# ── Helpers ──────────────────────────────────────────────────────────

def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Fecha inválida '{s}'. Formato: YYYY-MM-DD")


# ── Frontend ─────────────────────────────────────────────────────────

_FRONTEND = _Path(__file__).parent / "frontend" / "index.html"

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(_FRONTEND)


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health", tags=["sistema"])
def health():
    return {
        "status": "ok",
        "models_loaded": registry.available_models(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── POST /api/v1/predict ─────────────────────────────────────────────

@app.post("/api/v1/predict", response_model=PredictResponse, tags=["predicción"])
def predict(req: PredictRequest):
    model = registry.get(req.city.value, model_type=req.algo.value if req.algo else None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {req.city}")

    try:
        prob = model.predict(
            neighborhood=req.neighborhood,
            day_of_week=req.day_of_week,
            hour=req.hour,
            date=parse_date(req.fecha),
        )
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Barrio no reconocido: {e}")

    risk_level = "alto" if prob >= 0.70 else "medio" if prob >= 0.40 else "bajo"

    return PredictResponse(
        city=req.city.value,
        neighborhood=req.neighborhood,
        day_of_week=req.day_of_week,
        hour=req.hour,
        fecha=req.fecha,
        probability=round(prob, 4),
        risk_level=risk_level,
        used_model=model.model_id,
        prediction_id=str(uuid.uuid4())[:8],
    )


# ── GET /api/v1/heatmap/{city}/{date} ────────────────────────────────

_heatmap_cache: dict = {}   # {(city, fecha): HeatmapResponse}

@app.get("/api/v1/heatmap/{city}/{query_date}", response_model=HeatmapResponse, tags=["dashboard"])
def heatmap(city: str, query_date: str):
    model = registry.get(city)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Modelo no disponible: {city}")

    cache_key = (city, query_date)
    if cache_key in _heatmap_cache:
        return _heatmap_cache[cache_key]

    target_date = parse_date(query_date)
    day_of_week = target_date.weekday()

    nb_probs = model.predict_heatmap(day_of_week, target_date)

    matrix = [
        {"neighborhood": nb, "hour_slot": franja, "day": day_of_week,
         "risk_score": round(prob, 4)}
        for nb, slots in nb_probs.items()
        for franja, prob in slots.items()
    ]

    result = HeatmapResponse(
        city=city, fecha=query_date, matrix=matrix,
        metadata={"total_slots": len(matrix),
                  "neighborhoods": len(nb_probs),
                  "model": model.model_id},
    )
    _heatmap_cache[cache_key] = result
    return result


# ── POST /api/v1/explain ─────────────────────────────────────────────

@app.post("/api/v1/explain", response_model=ExplainResponse, tags=["explicabilidad"])
def explain(req: ExplainRequest):
    model = registry.get(req.city.value)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Modelo no disponible: {req.city}")

    shap_vals = model.shap_values(neighborhood=req.neighborhood,
                                   day_of_week=req.day_of_week,
                                   hour=req.hour, date=parse_date(req.fecha))
    raw_ctx  = explainer._rag_context(city=req.city.value, neighborhood=req.neighborhood)
    rag_text = explainer.explain(city=req.city.value, neighborhood=req.neighborhood,
                                  day_of_week=req.day_of_week, hour=req.hour,
                                  shap_values=shap_vals)
    top = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return ExplainResponse(
        shap_values=shap_vals,
        rag_explanation=rag_text,
        rag_context=raw_ctx,
        top_features=[{"feature": k, "shap": round(v, 4)} for k, v in top],
    )


# ── GET /api/v1/cities ───────────────────────────────────────────────

@app.get("/api/v1/cities", tags=["modelos"])
def list_cities():
    return {"cities": registry.city_summary()}


# ── GET /api/v1/model-info/{city} ────────────────────────────────────

@app.get("/api/v1/model-info/{city}", tags=["modelos"])
def model_info(city: str):
    model = registry.get(city)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Modelo no disponible: {city}")

    meta = model.metadata
    enc  = model.encoders

    UNDEFINED = {"0", "sin informacion", "sin información", "no definido", "nd", "n/a", ""}

    def risk_level(rate: float) -> str:
        if rate >= 0.70: return "muy alto"
        if rate >= 0.55: return "alto"
        if rate >= 0.40: return "moderado"
        if rate >= 0.20: return "bajo"
        return "muy bajo"

    all_nb = [
        {"name": nb, "rate": round(rate, 4), "level": risk_level(rate)}
        for nb, rate in sorted(enc["target_enc"].items(), key=lambda x: x[1], reverse=True)
        if nb and nb.strip().lower() not in UNDEFINED
    ]

    top10_risk = all_nb[:10]
    top10_safe = sorted(all_nb, key=lambda x: x["rate"])[:10]

    dist: dict = {"muy alto": 0, "alto": 0, "moderado": 0, "bajo": 0, "muy bajo": 0}
    for nb in all_nb:
        dist[nb["level"]] += 1

    return {
        "city":          city,
        "display_name":  {"medellin": "Medellín", "bogota": "Bogotá", "cali": "Cali"}.get(city, city.capitalize()),
        "model_type":    meta.get("model_type", "xgboost"),
        "version":       meta.get("version", "1.0"),
        "period":        "2008 – 2025",
        "records":       meta.get("records", 0),
        "slots":         meta.get("slots", 0),
        "neighborhoods": len(all_nb),
        "metrics": {
            "roc_auc":   round(meta.get("ROC AUC",   0), 4),
            "f1":        round(meta.get("F1",         0), 4),
            "precision": round(meta.get("Precision",  0), 4),
            "recall":    round(meta.get("Recall",     0), 4),
            "accuracy":  round(meta.get("Accuracy",   0), 4),
        },
        "global_risk_rate": round(enc["global_rate"], 4),
        "risk_distribution": dist,
        "top_risk": top10_risk,
        "top_safe": top10_safe,
        "all_neighborhoods": all_nb,
        "features": {
            "total": 26,
            "list": [
                "DIA_SEMANA", "MES", "DIA_ANIO_SIN", "DIA_ANIO_COS",
                "ES_FIN_SEMANA", "ES_LABORAL", "HORA_PUNTO_MEDIO",
                "DIA_SEMANA_SIN", "DIA_SEMANA_COS", "MES_SIN", "MES_COS",
                "UBICACION_TARGET_ENC", "UBICACION_LOG_ODDS",
                "FRANJA_00-02h", "FRANJA_02-04h", "FRANJA_04-06h",
                "FRANJA_06-08h", "FRANJA_08-10h", "FRANJA_10-12h",
                "FRANJA_12-14h", "FRANJA_14-16h", "FRANJA_16-18h",
                "FRANJA_18-20h", "FRANJA_20-22h", "FRANJA_22-24h",
                "INTERACCION_FINDE_NOCHE",
            ],
        },
        "tech_stack": ["Python 3.11", "XGBoost 2.1", "FastAPI", "SHAP", "Groq", "Docker", "pandas", "nginx"],
        "comparison": [
            {"model": "Logistic Regression", "scenario": "Histórico",   "auc": 0.9129, "f1": 0.8362, "precision": 0.8201, "recall": 0.8483},
            {"model": "Logistic Regression", "scenario": "Post-COVID",  "auc": 0.8215, "f1": 0.6143, "precision": 0.5128, "recall": 0.7397},
            {"model": "Random Forest",        "scenario": "Histórico",   "auc": 0.9177, "f1": 0.8386, "precision": 0.8401, "recall": 0.8375},
            {"model": "Random Forest",        "scenario": "Post-COVID",  "auc": 0.8485, "f1": 0.6389, "precision": 0.5001, "recall": 0.8111},
            {"model": "XGBoost",              "scenario": "Histórico",   "auc": round(meta.get("ROC AUC", 0.9182), 4), "f1": round(meta.get("F1", 0.8384), 4), "precision": round(meta.get("Precision", 0.8401), 4), "recall": round(meta.get("Recall", 0.8367), 4), "winner": True},
            {"model": "XGBoost",              "scenario": "Post-COVID",  "auc": 0.8502, "f1": 0.6417, "precision": 0.5023, "recall": 0.8059},
        ],
    }


# ── GET /api/v1/coordinates/{city} ───────────────────────────────────

@app.get("/api/v1/coordinates/{city}", tags=["modelos"])
def get_coordinates(city: str):
    coords_path = _Path(os.getenv("RAG_DIR", _Path(__file__).parent / "rag")) / "documents" / city / "coordinates.json"
    if not coords_path.exists():
        raise HTTPException(status_code=404, detail=f"Coordenadas no disponibles para: {city}")
    coords = _json.loads(coords_path.read_text(encoding="utf-8"))
    return {"city": city, "count": len(coords), "coordinates": coords}


# ── GET /api/v1/neighborhoods/{city} ─────────────────────────────────

@app.get("/api/v1/neighborhoods/{city}", tags=["modelos"])
def list_neighborhoods(city: str):
    model = registry.get(city)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Modelo no disponible: {city}")
    UNDEFINED = {"0", "sin informacion", "sin información", "no definido", "nd", "n/a", ""}
    neighborhoods = sorted(
        nb for nb in model.neighborhoods
        if nb and nb.strip().lower() not in UNDEFINED
    )
    return {"city": city, "count": len(neighborhoods), "neighborhoods": neighborhoods}


# ── POST /api/v1/train/{city} ────────────────────────────────────────

@app.post("/api/v1/train/{city}", tags=["modelos"])
async def train(city: str, req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(
        registry.train_async,
        city=city, data_path=req.data_path,
        model_type=req.algo, year_from=req.year_from, job_id=job_id,
    )
    logger.info(f"Training job {job_id} started for city={city}")
    return {"job_id": job_id, "status": "started",
            "message": f"Entrenamiento iniciado para {city}. Consulta /api/v1/jobs/{job_id}"}


@app.get("/api/v1/jobs/{job_id}", tags=["modelos"])
def job_status(job_id: str):
    status = registry.job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return status


# ── POST /api/v1/chat ─────────────────────────────────────────────

# ── RAG con TF-IDF para el chat ───────────────────────────────────
# Usa scikit-learn (ya en el proyecto) — sin dependencias nuevas.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CHAT_RAG_TOP_K = int(os.getenv("CHAT_RAG_TOP_K", "5"))

_rag_cache: dict = {}   # {city: {"chunks": [...], "vectorizer": ..., "matrix": ...}}

def _build_chunks(data: dict) -> list[str]:
    """Convierte el knowledge JSON en chunks de texto independientes."""
    m  = data.get("metricas", {})
    b  = data.get("barrios", {})
    ds = data.get("dataset", {})
    r  = m.get("resultados", {})
    chunks = []

    # Proyecto y objetivo
    chunks.append(
        f"Proyecto: {data.get('proyecto','')}. Objetivo: {data.get('objetivo','')}. Ciudad: {data.get('ciudad','')}."
    )

    # Dataset
    bc = ds.get("balance_clases", {}).get("nota", "")
    chunks.append(
        f"Dataset: {ds.get('fuente','')}. Período {ds.get('periodo','')}. "
        f"{ds.get('registros_originales',0):,} registros originales, {ds.get('registros_post_limpieza',0):,} post-limpieza. "
        f"Slots de modelado: {ds.get('slots_modelado',0):,}. {bc}. "
        f"Eliminados en limpieza: {', '.join(ds.get('motivos_eliminacion', []))}."
    )

    # Métricas y modelo
    comp = "; ".join(
        f"{c['modelo']} {c['escenario']}: AUC={c['auc']} F1={c['f1']}"
        for c in m.get("comparativa", [])
    )
    chunks.append(
        f"Modelo: {m.get('modelo_final','')}. Hiperparámetros: {m.get('hiperparametros',{})}. "
        f"Split: {m.get('split','')}. "
        f"Métricas: ROC AUC={r.get('ROC_AUC','')} F1={r.get('F1','')} "
        f"Precisión={r.get('Precision','')} Recall={r.get('Recall','')} Accuracy={r.get('Accuracy','')}. "
        f"Comparativa de modelos: {comp}."
    )

    # Features
    imp = "; ".join(
        f"{f['nombre']} ({f['importancia']}): {f['descripcion']}"
        for f in data.get("features", {}).get("mas_importantes", [])
    )
    chunks.append(
        f"Features del modelo ({data.get('features',{}).get('total',26)} variables): "
        f"{', '.join(data.get('features',{}).get('list',[]))}. "
        f"Las más importantes: {imp}."
    )

    # Barrios
    top_r = ", ".join(f"{x['nombre']} {x['tasa']:.0%}" for x in b.get("top10_mayor_riesgo", []))
    top_s = ", ".join(f"{x['nombre']} {x['tasa']:.0%}" for x in b.get("top10_menor_riesgo", []))
    chunks.append(
        f"Barrios: {b.get('total',0)} barrios y comunas en Medellín. "
        f"Tasa promedio de accidentabilidad: {b.get('tasa_promedio_ciudad',0):.1%}. "
        f"Distribución por nivel: {b.get('distribucion',{})}. "
        f"Barrios con mayor riesgo: {top_r}. "
        f"Barrios con menor riesgo: {top_s}."
    )

    # Una chunk por decisión de diseño
    for d in data.get("decisiones_diseno", []):
        chunks.append(
            f"Decisión de diseño — {d['decision']}: {d['razon']} "
            f"Impacto: {d['impacto']}."
        )

    # Tech stack
    chunks.append(f"Stack tecnológico: {data.get('tech_stack',{})}.")

    # Arquitectura web de la aplicación
    chunks.append(
        "Arquitectura web de RiesgoVial: el frontend es una aplicación de una sola página (SPA) en HTML, CSS y "
        "JavaScript puro, sin frameworks, con cinco secciones (Consulta, Dashboard, Modelos, Chat IA y Documentación). "
        "Usa Leaflet para el mapa interactivo de barrios y Chart.js para los gráficos (gauge, barras SHAP, heatmap, tendencias). "
        "Un servidor Nginx entrega esa página estática y redirige las peticiones que empiezan por /api/ hacia el backend. "
        "El backend es una API construida con FastAPI (Python 3.11) y servida con Gunicorn, que carga el modelo de "
        "machine learning (XGBoost) en memoria al iniciar. "
        "Toda la aplicación corre empaquetada con Docker y Docker Compose, en dos contenedores: la API y Nginx. "
        "Las explicaciones de riesgo (/explain) y el Chat IA usan modelos de lenguaje configurables: Groq y/o "
        "Anthropic Claude, según las llaves de API (GROQ_API_KEY, ANTHROPIC_API_KEY) que estén configuradas. "
        "El backend no tiene base de datos: es sin estado (stateless). El historial del Chat IA se mantiene en "
        "memoria del navegador y se reenvía en cada pregunta como ventana de contexto; el servidor no guarda "
        "conversaciones, por lo que el historial se pierde al recargar la página."
    )

    # Una chunk por FAQ
    for fq in data.get("preguntas_frecuentes", []):
        chunks.append(f"Pregunta: {fq['pregunta']} Respuesta: {fq['respuesta']}")

    # Una chunk por sección del notebook (solo título + descripción)
    for s in data.get("secciones_notebook", []):
        desc = s.get("descripcion", "")[:400]
        if desc:
            chunks.append(f"Notebook — {s['titulo']}: {desc}")

    return chunks

def _load_text_chunks(path: _Path) -> list[str]:
    """Divide un .txt en chunks por bloques separados por línea en blanco."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip().replace("\n", " ") for b in text.strip().split("\n\n") if b.strip()]
    return [b for b in blocks if len(b) >= 30]

def _get_rag(city: str) -> dict | None:
    if city in _rag_cache:
        return _rag_cache[city]
    docs_dir = _Path(os.getenv("RAG_DIR", _Path(__file__).parent / "rag")) / "documents" / city
    kb_path = docs_dir / "model_knowledge.json"
    if not kb_path.exists():
        return None
    data = _json.loads(kb_path.read_text(encoding="utf-8"))
    chunks = _build_chunks(data)
    chunks += _load_text_chunks(docs_dir / "estadisticas_viales.txt")
    chunks += _load_text_chunks(docs_dir / "movilidad_medellin.txt")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(chunks)
    _rag_cache[city] = {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}
    logger.info(f"RAG chat: {len(chunks)} chunks indexados para {city}")
    return _rag_cache[city]

def _retrieve_context(question: str, city: str, top_k: int = CHAT_RAG_TOP_K) -> str:
    rag = _get_rag(city)
    if not rag:
        return ""
    q_vec = rag["vectorizer"].transform([question])
    scores = cosine_similarity(q_vec, rag["matrix"]).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    retrieved = [rag["chunks"][i] for i in top_idx if scores[i] > 0]
    return "\n\n".join(retrieved) if retrieved else "\n\n".join(rag["chunks"][:top_k])

CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "10"))
GROQ_MODEL         = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
CHAT_MAX_TOKENS    = int(os.getenv("CHAT_MAX_TOKENS", "1000"))

CHAT_SYSTEM = """Eres el asistente de RiesgoVial, una plataforma de predicción de riesgo vial en Medellín, Colombia.

Usa este contexto para responder:

{context}

Reglas estrictas:
- Responde siempre en español, de forma clara y en lenguaje natural.
- Máximo 3 oraciones por respuesta. Ve al grano.
- NUNCA incluyas código fuente, comandos, fragmentos de programación ni nombres de funciones.
- NUNCA uses términos técnicos sin explicarlos en palabras simples.
- Si preguntan por la arquitectura, el despliegue o las tecnologías de la plataforma (frontend, backend, infraestructura, IA), SÍ puedes nombrar las tecnologías del contexto (FastAPI, Nginx, Docker, Groq, Anthropic Claude, Leaflet, Chart.js, XGBoost, etc.) explicando brevemente para qué sirve cada una.
- Para preguntas sobre el modelo, los datos o los barrios, evita nombrar archivos internos o librerías de programación y usa lenguaje simple.
- Si la pregunta es sobre un barrio, menciona su nivel de riesgo y tasa histórica.
- Si no tienes la información, dilo en una oración.
- No inventes datos que no estén en el contexto.
"""

@app.get("/api/v1/chat/config", tags=["chat"])
def chat_config():
    return {"history_turns": CHAT_HISTORY_TURNS, "rag_top_k": CHAT_RAG_TOP_K}

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=503, detail="LLM no disponible — configura GROQ_API_KEY")

    context = _retrieve_context(req.message, req.city)
    if not context:
        raise HTTPException(status_code=404, detail=f"Base de conocimiento no disponible para: {req.city}")

    try:
        from groq import Groq
        client = Groq(api_key=groq_key)

        messages = [{"role": "system", "content": CHAT_SYSTEM.format(context=context)}]
        for msg in req.history[-(CHAT_HISTORY_TURNS * 2):]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": req.message})

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=CHAT_MAX_TOKENS,
            temperature=0.3,
        )
        usage = completion.usage
        return ChatResponse(
            response=completion.choices[0].message.content.strip(),
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            tokens_total=usage.total_tokens,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")
