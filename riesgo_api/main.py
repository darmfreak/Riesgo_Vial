"""
RiesgoVial API — main.py
Pydantic v2 compatible: sin @validator, sin model_config, sin prefijo model_
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from datetime import date, datetime
from enum import Enum
import logging
import uuid

from model_loader import ModelRegistry
from explainer import RiskExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("riesgovial")

app = FastAPI(
    title="RiesgoVial API",
    description="Predicción de riesgo vial urbano. Multi-ciudad, multi-modelo.",
    version="1.0.0",
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
    shap_values:    dict
    rag_explanation: str
    top_features:   List[dict]


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
    rag_text = explainer.explain(city=req.city.value, neighborhood=req.neighborhood,
                                  day_of_week=req.day_of_week, hour=req.hour,
                                  shap_values=shap_vals)
    top = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return ExplainResponse(
        shap_values=shap_vals,
        rag_explanation=rag_text,
        top_features=[{"feature": k, "shap": round(v, 4)} for k, v in top],
    )


# ── GET /api/v1/cities ───────────────────────────────────────────────

@app.get("/api/v1/cities", tags=["modelos"])
def list_cities():
    return {"cities": registry.city_summary()}


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
