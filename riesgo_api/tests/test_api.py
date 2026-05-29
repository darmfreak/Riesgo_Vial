"""
Tests de integración para RiesgoVial API.
Ejecutar: cd riesgo_api && pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient

# Importar app (requiere estar en riesgo_api/ o tener el path correcto)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models_loaded" in body
    assert isinstance(body["models_loaded"], list)
    assert len(body["models_loaded"]) >= 1


# ── Cities ────────────────────────────────────────────────────────

def test_list_cities():
    r = client.get("/api/v1/cities")
    assert r.status_code == 200
    body = r.json()
    assert "cities" in body
    assert len(body["cities"]) >= 1
    city = body["cities"][0]
    assert "city" in city
    assert "model_id" in city
    assert "neighborhoods" in city


# ── Predict ───────────────────────────────────────────────────────

def test_predict_medellin_alto():
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": "La Candelaria",
        "day_of_week": 5,
        "hour": 23,
    })
    assert r.status_code == 200
    d = r.json()
    assert 0.0 <= d["probability"] <= 1.0
    assert d["risk_level"] in ("bajo", "medio", "alto")
    assert d["used_model"].startswith("medellin")
    assert len(d["prediction_id"]) > 0


def test_predict_medellin_bajo():
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": "Corregimiento de San Cristóbal",
        "day_of_week": 1,
        "hour": 10,
    })
    assert r.status_code == 200
    d = r.json()
    assert 0.0 <= d["probability"] <= 1.0


def test_predict_with_fecha():
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": "El Poblado",
        "day_of_week": 4,
        "hour": 18,
        "fecha": "2025-12-31",
    })
    assert r.status_code == 200
    d = r.json()
    assert d["fecha"] == "2025-12-31"


def test_predict_invalid_city():
    r = client.post("/api/v1/predict", json={
        "city": "xyz",
        "neighborhood": "Algún barrio",
        "day_of_week": 0,
        "hour": 12,
    })
    assert r.status_code == 422


def test_predict_invalid_hour():
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": "La Candelaria",
        "day_of_week": 0,
        "hour": 25,
    })
    assert r.status_code == 422


def test_predict_invalid_day():
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": "La Candelaria",
        "day_of_week": 8,
        "hour": 10,
    })
    assert r.status_code == 422


def test_predict_bad_fecha():
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": "La Candelaria",
        "day_of_week": 0,
        "hour": 10,
        "fecha": "not-a-date",
    })
    assert r.status_code == 422


# ── Heatmap ───────────────────────────────────────────────────────

def test_heatmap_valid():
    r = client.get("/api/v1/heatmap/medellin/2025-06-15")
    assert r.status_code == 200
    d = r.json()
    assert d["city"] == "medellin"
    assert d["fecha"] == "2025-06-15"
    assert len(d["matrix"]) > 0
    row = d["matrix"][0]
    assert "neighborhood" in row
    assert "hour_slot" in row
    assert "risk_score" in row
    assert 0.0 <= row["risk_score"] <= 1.0


def test_heatmap_invalid_date():
    r = client.get("/api/v1/heatmap/medellin/bad-date")
    assert r.status_code == 422


def test_heatmap_unknown_city():
    r = client.get("/api/v1/heatmap/atlantida/2025-01-01")
    assert r.status_code == 404


# ── Explain ───────────────────────────────────────────────────────

def test_explain():
    r = client.post("/api/v1/explain", json={
        "city": "medellin",
        "neighborhood": "La Candelaria",
        "day_of_week": 5,
        "hour": 23,
    })
    assert r.status_code == 200
    d = r.json()
    assert "shap_values" in d
    assert "rag_explanation" in d
    assert "top_features" in d
    assert isinstance(d["shap_values"], dict)
    assert len(d["rag_explanation"]) > 10
    assert len(d["top_features"]) > 0


# ── Probability range sanity ──────────────────────────────────────

@pytest.mark.parametrize("barrio,dow,hour", [
    ("La Candelaria", 5, 23),
    ("El Poblado", 2, 10),
    ("Guayabal", 0, 7),
    ("Robledo", 6, 2),
])
def test_probability_in_range(barrio, dow, hour):
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": barrio,
        "day_of_week": dow,
        "hour": hour,
    })
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0
