# RiesgoVial — Plan de trabajo para Claude Code

> Contexto: API FastAPI + frontend HTML ya construidos.
> Docker corriendo localmente. Modelos ML en XGBoost (ROC AUC 0.9182).
> Este documento guía el trabajo restante sesión por sesión.

---

## Estado actual

| Componente | Estado |
|---|---|
| Notebook de entrenamiento | ✅ Completo (674K registros, 2008-2025) |
| Experimento COVID (histórico vs 2020+) | ✅ Completo |
| Frontend HTML (4 páginas) | ✅ Completo |
| API FastAPI estructura | ✅ Completo |
| Docker / docker-compose | ✅ Corriendo |
| Modelos .pkl exportados | ⬜ Pendiente |
| API conectada a modelos reales | ⬜ Pendiente |
| Deep Learning (TabNet) | ⬜ Pendiente |
| RAG con FAISS | ⬜ Pendiente |
| Frontend conectado a la API | ⬜ Pendiente |
| Tests | ⬜ Pendiente |

---

## Sesión 1 — Exportar modelos y conectar la API

**Objetivo:** que `POST /api/v1/predict` devuelva predicciones reales del XGBoost entrenado.

### Paso 1: Exportar modelos desde el notebook

Abrir `modelo_riesgo_enriquecido.ipynb` y ejecutar:

```python
# Al final del notebook, agregar y ejecutar esta celda:
import pickle
from pathlib import Path

Path("riesgo_api/models").mkdir(exist_ok=True)

def exportar_modelo(model_obj, model_df_, city, algo, metrics=None):
    M = 50.0
    gr = model_df_["HUBO_INCIDENTE"].mean()
    ub = model_df_.groupby("UBICACION_KEY").agg(
        m=("HUBO_INCIDENTE","mean"), n=("HUBO_INCIDENTE","count")
    ).reset_index()
    ub["tenc"] = (ub["n"]*ub["m"] + M*gr) / (ub["n"] + M)
    ub["logo"] = np.log((ub["tenc"]+0.001)/(1-ub["tenc"]+0.001))
    encoders = {
        "target_enc":  ub.set_index("UBICACION_KEY")["tenc"].to_dict(),
        "log_odds":    ub.set_index("UBICACION_KEY")["logo"].to_dict(),
        "global_rate": float(gr),
    }
    bundle = {
        "model":    model_obj,
        "encoders": encoders,
        "metadata": {"version":"1.0","city":city,"algo":algo,**(metrics or {})},
    }
    path = f"riesgo_api/models/{city}_{algo}.pkl"
    with open(path,"wb") as f:
        pickle.dump(bundle, f, protocol=5)
    print(f"✅ Exportado: {path}")

# Exportar el mejor modelo (XGBoost histórico completo)
exportar_modelo(
    res_completo["XGBoost"]["modelo"],
    model_df_completo,
    city="medellin",
    algo="xgboost",
    metrics=met_completo["XGBoost"]
)

# Opcional: exportar también Random Forest y post-COVID
# exportar_modelo(res_completo["Random Forest"]["modelo"], model_df_completo, "medellin", "rf")
# exportar_modelo(res_post["XGBoost"]["modelo"], model_df_post, "medellin_postcovid", "xgboost")
```

### Paso 2: Reconstruir el contenedor con los modelos

```bash
cd ~/Proyectos/Notebooks/riesgo_api
docker compose down
docker compose up --build
```

### Paso 3: Verificar que la API usa el modelo real

```bash
# El log ya NO debe decir "Cargando modelo DEMO"
# Debe decir: "Modelo cargado: medellin_xgboost (370 barrios)"

curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "medellin",
    "neighborhood": "La Candelaria",
    "day_of_week": 5,
    "hour": 23,
    "fecha": "2025-12-31"
  }'
```

**Resultado esperado:**
```json
{
  "probability": 0.94,
  "risk_level": "alto",
  "used_model": "medellin_xgboost_v1.0"
}
```

---

## Sesión 2 — Conectar el frontend a la API real

**Objetivo:** que los botones del HTML llamen a `localhost:8000` en lugar de usar datos simulados.

### Archivos a modificar

`riesgo_api/frontend/index.html` — reemplazar las funciones JS simuladas:

```javascript
// ANTES (simulado):
const base = RISK_BASE[barrio] || 0.5;
const raw  = base * HOUR_FACTOR(hora) * ...;

// DESPUÉS (API real):
async function runPrediction() {
  const body = {
    city:         document.getElementById('cityGlobal').value,
    neighborhood: document.getElementById('barrio').value,
    day_of_week:  ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
                    .indexOf(document.getElementById('diaSemana').value),
    hour:         parseInt(document.getElementById('horaSlider').value),
    fecha:        document.getElementById('fecha').value || null,
  };

  const res  = await fetch('/api/v1/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await res.json();

  // Actualizar UI con data.probability, data.risk_level, data.used_model
  const pct = Math.round(data.probability * 100);
  document.getElementById('riskNumber').textContent = pct + '%';
  drawGauge(pct);
  // ...
}
```

```javascript
// Dashboard: heatmap desde API real
async function buildHeatmap() {
  const city = document.getElementById('cityGlobal').value;
  const today = new Date().toISOString().split('T')[0];
  const res  = await fetch(`/api/v1/heatmap/${city}/${today}`);
  const data = await res.json();
  // data.matrix es array de {neighborhood, hour_slot, risk_score}
  // renderizar igual que antes pero con datos reales
}
```

### Manejo de errores en el frontend

```javascript
async function apiCall(url, options = {}) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      const err = await res.json();
      showError(err.detail || 'Error en la API');
      return null;
    }
    return await res.json();
  } catch (e) {
    showError('No se pudo conectar con la API');
    return null;
  }
}
```

---

## Sesión 3 — RAG con FAISS

**Objetivo:** que `POST /api/v1/explain` devuelva explicaciones en lenguaje natural con contexto real de los barrios.

### Estructura de archivos a crear

```
riesgo_api/
└── rag/
    ├── build_index.py      ← indexa documentos en FAISS
    ├── documents/
    │   ├── medellin_barrios.txt    ← contexto por barrio
    │   ├── movilidad_2024.txt      ← informe SDM Medellín
    │   └── estadisticas_viales.txt ← datos históricos
    └── index/
        └── medellin.faiss   ← índice vectorial (generado)
```

### `build_index.py`

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def build_rag_index(city: str = "medellin"):
    loader = DirectoryLoader(f"rag/documents/{city}/", glob="*.txt",
                             loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings locales (sin API key necesaria)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    Path(f"rag/index").mkdir(exist_ok=True)
    db.save_local(f"rag/index/{city}")
    print(f"✅ Índice FAISS creado: rag/index/{city} ({len(chunks)} chunks)")

if __name__ == "__main__":
    build_rag_index("medellin")
```

### Actualizar `explainer.py` para usar FAISS real

```python
# En RiskExplainer.__init__:
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

self._embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
self._indexes = {}  # cache por ciudad

def _get_index(self, city: str):
    if city not in self._indexes:
        path = f"rag/index/{city}"
        if Path(path).exists():
            self._indexes[city] = FAISS.load_local(
                path, self._embeddings, allow_dangerous_deserialization=True
            )
    return self._indexes.get(city)

def explain(self, city, neighborhood, day_of_week, hour, shap_values=None):
    db = self._get_index(city)
    query = f"riesgo vial {neighborhood} hora {hour} día {DAYS_ES[day_of_week]}"

    if db:
        docs = db.similarity_search(query, k=3)
        context = "\n".join(d.page_content for d in docs)
    else:
        context = self._get_neighborhood_context(city, neighborhood)

    # Con LLM:
    if self._llm_available:
        return self._llm_enrich(context, neighborhood, DAYS_ES[day_of_week], hour)

    return context[:500]  # fallback sin LLM
```

### Agregar dependencias al requirements.txt

```
sentence-transformers==3.3.1
```

---

## Sesión 4 — Deep Learning con TabNet

**Objetivo:** entrenar un modelo TabNet sobre el mismo dataset y comparar con XGBoost.

### Instalar dependencia (en el notebook)

```bash
pip install pytorch-tabnet
```

### Celda a agregar en el notebook (después del Paso 8)

```python
# ============================================================
# DEEP LEARNING: TabNet con embeddings de ubicación
# ============================================================
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

print("🧠 ENTRENAMIENTO TABNET")
print("=" * 70)

# Preparar datos (mismas features que XGBoost)
X_tr_tn = X_tr.values.astype('float32')
X_te_tn = X_te.values.astype('float32')
y_tr_tn = y_tr.values.astype('int64')
y_te_tn = y_te.values.astype('int64')

# Pesos de clase para desbalance
pos_weight = (y_tr_tn == 0).sum() / (y_tr_tn == 1).sum()

tabnet = TabNetClassifier(
    n_d=32, n_a=32,              # dimensión de las representaciones
    n_steps=5,                    # pasos de atención
    gamma=1.3,
    n_independent=2,
    n_shared=2,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 2e-3},
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax",
    verbose=10,
    seed=42,
)

tabnet.fit(
    X_train=X_tr_tn, y_train=y_tr_tn,
    eval_set=[(X_te_tn, y_te_tn)],
    eval_metric=["auc"],
    max_epochs=100,
    patience=20,
    batch_size=4096,
    weights={0: 1, 1: int(pos_weight)},
)

# Evaluar
from sklearn.metrics import roc_auc_score
probs_tn = tabnet.predict_proba(X_te_tn)[:, 1]
auc_tn   = roc_auc_score(y_te_tn, probs_tn)
print(f"\n🎯 TabNet ROC AUC: {auc_tn:.4f}")
print(f"   XGBoost ROC AUC: {met_completo['XGBoost']['ROC AUC']:.4f}")
print(f"   Diferencia: {auc_tn - met_completo['XGBoost']['ROC AUC']:+.4f}")

# Importancia de features (máscaras de atención)
feat_importances = dict(zip(feature_cols, tabnet.feature_importances_))
top_tn = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n🔍 Top 10 features (TabNet atención):")
for feat, imp in top_tn:
    print(f"   {feat:<35} {imp:.4f}")
```

### Exportar TabNet para la API

```python
# Agregar en export_models.py — TabNet usa save_model en lugar de pickle
tabnet.save_model("riesgo_api/models/medellin_tabnet")
# Guarda: medellin_tabnet.zip
```

### Actualizar `model_loader.py` para soportar TabNet

```python
# En _load_from_disk, detectar si es tabnet:
if "tabnet" in model_type:
    from pytorch_tabnet.tab_model import TabNetClassifier
    model_obj = TabNetClassifier()
    model_obj.load_model(str(path).replace(".pkl", ".zip"))
```

---

## Sesión 5 — Tests y documentación final

### Tests básicos con pytest

Crear `riesgo_api/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_medellin():
    r = client.post("/api/v1/predict", json={
        "city": "medellin",
        "neighborhood": "La Candelaria",
        "day_of_week": 5,
        "hour": 23,
    })
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["probability"] <= 1
    assert data["risk_level"] in ["bajo", "medio", "alto"]

def test_predict_invalid_city():
    r = client.post("/api/v1/predict", json={
        "city": "xyz",
        "neighborhood": "Algún barrio",
        "day_of_week": 0,
        "hour": 12,
    })
    assert r.status_code == 422

def test_heatmap():
    r = client.get("/api/v1/heatmap/medellin/2025-06-15")
    assert r.status_code == 200
    data = r.json()
    assert len(data["matrix"]) > 0

def test_cities():
    r = client.get("/api/v1/cities")
    assert r.status_code == 200
    assert len(r.json()["cities"]) >= 1
```

Ejecutar:
```bash
cd riesgo_api
pip install pytest httpx
pytest tests/ -v
```

---

## Roadmap completo

```
Semana 1
├── Sesión 1: Exportar PKL + API con modelos reales     ← SIGUIENTE
├── Sesión 2: Frontend conectado a API real
└── Sesión 3: RAG con FAISS + documentos de barrios

Semana 2
├── Sesión 4: TabNet en el notebook + exportar
├── Sesión 5: Tests + CI básico
└── Deploy: Railway o Render (gratis)
```

---

## Comandos útiles de referencia

```bash
# Ver logs en tiempo real
docker compose logs -f api

# Entrar al contenedor
docker exec -it riesgovial_api bash

# Reconstruir solo la API (sin nginx)
docker compose up --build api

# Ver modelos cargados
curl http://localhost:8000/api/v1/cities | python3 -m json.tool

# Hacer predicción de prueba
curl -s -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"medellin","neighborhood":"Laureles Estadio","day_of_week":6,"hour":22}' \
  | python3 -m json.tool

# Ver documentación interactiva
open http://localhost:8000/docs
```

---

## Estructura final del proyecto

```
riesgo_api/
├── main.py                  ← API FastAPI
├── model_loader.py          ← carga PKL + pipeline de entrenamiento
├── explainer.py             ← RAG + SHAP
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── export_models.py         ← ejecutar en el notebook
├── models/
│   ├── medellin_xgboost.pkl
│   ├── medellin_rf.pkl      (opcional)
│   └── medellin_tabnet.zip  (sesión 4)
├── rag/
│   ├── build_index.py
│   ├── documents/medellin/
│   └── index/medellin/
├── frontend/
│   └── index.html           ← riesgo_urbano_platform.html
└── tests/
    └── test_api.py
```
