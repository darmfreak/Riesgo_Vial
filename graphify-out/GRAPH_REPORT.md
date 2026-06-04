# Graph Report - .  (2026-06-04)

## Corpus Check
- 25 files · ~69,180 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 593 nodes · 738 edges · 23 communities (16 shown, 7 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 50 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Medellín Neighborhood Coordinates|Medellín Neighborhood Coordinates]]
- [[_COMMUNITY_FastAPI App & Core Modules|FastAPI App & Core Modules]]
- [[_COMMUNITY_Model Loading & RAG Build Index|Model Loading & RAG Build Index]]
- [[_COMMUNITY_RAG Knowledge Base & Infrastructure|RAG Knowledge Base & Infrastructure]]
- [[_COMMUNITY_Chat RAG Pipeline|Chat RAG Pipeline]]
- [[_COMMUNITY_Feature Engineering Concepts|Feature Engineering Concepts]]
- [[_COMMUNITY_API Integration Layer|API Integration Layer]]
- [[_COMMUNITY_Model Metrics & Hyperparameters|Model Metrics & Hyperparameters]]
- [[_COMMUNITY_Dataset Statistics|Dataset Statistics]]
- [[_COMMUNITY_Neighborhood Risk Data|Neighborhood Risk Data]]
- [[_COMMUNITY_LLM Explain Pipeline|LLM Explain Pipeline]]
- [[_COMMUNITY_Model Export & Encoding|Model Export & Encoding]]
- [[_COMMUNITY_Geocoding Pipeline|Geocoding Pipeline]]
- [[_COMMUNITY_Training Notebook Scripts|Training Notebook Scripts]]
- [[_COMMUNITY_Claude Code Settings|Claude Code Settings]]
- [[_COMMUNITY_Demo Model Fallback|Demo Model Fallback]]
- [[_COMMUNITY_Chat Request Schema|Chat Request Schema]]
- [[_COMMUNITY_Explain Request Schema|Explain Request Schema]]
- [[_COMMUNITY_Predict Request Schema|Predict Request Schema]]
- [[_COMMUNITY_CityModel Core|CityModel Core]]
- [[_COMMUNITY_Local Settings|Local Settings]]

## God Nodes (most connected - your core abstractions)
1. `ModelRegistry` - 25 edges
2. `RiskExplainer` - 24 edges
3. `CityModel` - 11 edges
4. `dataset` - 11 edges
5. `run_tabnet.py — Training & Prediction Script` - 10 edges
6. `str` - 9 edges
7. `str` - 9 edges
8. `main()` - 8 edges
9. `tech_stack` - 8 edges
10. `barrios` - 7 edges

## Surprising Connections (you probably didn't know these)
- `TabNetClassifier — Deep Learning Model` --semantically_similar_to--> `medellin_xgboost.pkl — Serialized Production Model`  [INFERRED] [semantically similar]
  run_tabnet.py → riesgo_api/MANUAL.md
- `COVID-19 Structural Break Experiment` --semantically_similar_to--> `estadisticas_viales.txt — Medellín Vial Statistics 2008-2025`  [INFERRED] [semantically similar]
  run_tabnet.py → riesgo_api/rag/documents/medellin/estadisticas_viales.txt
- `model_df — Spatiotemporal Modeling Dataset` --shares_data_with--> `medellin_xgboost.pkl — Serialized Production Model`  [INFERRED]
  run_tabnet.py → riesgo_api/MANUAL.md
- `export_model() — Model Serialization` --implements--> `medellin_xgboost.pkl — Serialized Production Model`  [EXTRACTED]
  run_tabnet.py → riesgo_api/MANUAL.md
- `RiesgoVial Plan — Claude Code Session Roadmap` --references--> `TabNetClassifier — Deep Learning Model`  [EXTRACTED]
  riesgo_api/PLAN_CLAUDE_CODE.md → run_tabnet.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **RiesgoVial Prediction Pipeline: Request → Features → Model → Explain** — main_predict, model_loader_citymodel__build_features, model_loader_citymodel_predict, main_explain, model_loader_citymodel_shap_values, explainer_riskexplainer_explain [EXTRACTED 1.00]
- **Model Training & Export Pipeline: Notebook → export_models → model_knowledge** — run_notebook_script, export_models_export_model, export_models_build_encoders, concept_xgboost_model, export_model_knowledge_main, medellin_model_knowledge [INFERRED 0.95]
- **Feature Engineering Core Concepts: Grid + Cyclic Encoding + Target Encoding** — concept_spatiotemporal_grid, concept_2h_franjas, concept_cyclic_encoding, concept_target_encoding [INFERRED 0.95]
- **RAG Pipeline: TF-IDF over Medellín Documents for Chat** — riesgo_api_rag_tfidf, medellin_barrios_doc, medellin_estadisticas_doc, medellin_movilidad_doc, medellin_todos_barrios_doc [INFERRED 0.95]
- **Training Pipeline: Feature Engineering → XGBoost → PKL Export → API** — run_tabnet_feature_engineering, run_tabnet_xgb_model, run_tabnet_export_model, medellin_xgboost_pkl [EXTRACTED 1.00]
- **Deployment Stack: API + Nginx + Frontend served via Docker Compose** — riesgo_api_dockercompose, riesgo_api_nginx, riesgo_api_frontend_index [EXTRACTED 1.00]

## Communities (23 total, 7 thin omitted)

### Community 0 - "Medellín Neighborhood Coordinates"
Cohesion: 0.01
Nodes (311): Aguas Frías, Aldea Pablo VI, Alejandro Echavarría, Alejandría, Alfonso López, Altamira, Altavista, Altos del Poblado (+303 more)

### Community 1 - "FastAPI App & Core Modules"
Cohesion: 0.09
Nodes (37): BackgroundTasks, BaseModel, Enum, hour_label(), int, str, explainer.py Sistema RAG para generar explicaciones en lenguaje natural. Usa dic, RiskExplainer (+29 more)

### Community 2 - "Model Loading & RAG Build Index"
Cohesion: 0.10
Nodes (23): float, Path, build_index(), build_index.py — Construye el índice FAISS para el RAG de RiesgoVial. Ejecutar d, CityModel, hour_to_franja(), ModelRegistry, DataFrame (+15 more)

### Community 3 - "RAG Knowledge Base & Infrastructure"
Cohesion: 0.10
Nodes (32): barrios.txt — Medellín Neighborhood Risk Profiles, estadisticas_viales.txt — Medellín Vial Statistics 2008-2025, movilidad_medellin.txt — Medellín Urban Mobility Context, todos_los_barrios.txt — 369 Barrios with Historical Risk Rates, medellin_xgboost.pkl — Serialized Production Model, docker-compose.yml — Container Orchestration, index.html — RiesgoVial Frontend SPA, Groq LLM Integration — AI Explanations (+24 more)

### Community 4 - "Chat RAG Pipeline"
Cohesion: 0.06
Nodes (30): build_index() — FAISS Index Builder, RAG Pipeline for Explanations, POST /api/v1/chat, _get_rag(), _retrieve_context(), TF-IDF RAG for Chat, ciclicas, franjas_horarias (+22 more)

### Community 5 - "Feature Engineering Concepts"
Cohesion: 0.12
Nodes (22): 2-Hour Time Slot Granularity, Cyclic Encoding (sin/cos) for Temporal Variables, Spatio-temporal Grid Modeling (370 barrios × 7 days × 12 franjas × 12 months), Smoothed Target Encoding for UBICACION (m=50), XGBoost Road Risk Model (medellin_xgboost.pkl), clean(), extract_outputs(), main() (+14 more)

### Community 6 - "API Integration Layer"
Cohesion: 0.11
Nodes (4): RiskExplainer, FastAPI App (main.py), ModelRegistry, Tests de integración para RiesgoVial API. Ejecutar: cd riesgo_api && pytest test

### Community 7 - "Model Metrics & Hyperparameters"
Cohesion: 0.13
Nodes (15): max_depth, n_estimators, random_state, scale_pos_weight, metricas, comparativa, hiperparametros, modelo_final (+7 more)

### Community 8 - "Dataset Statistics"
Cohesion: 0.14
Nodes (14): negativos, nota, positivos, dataset, balance_clases, columnas_originales, eliminados_limpieza, fuente (+6 more)

### Community 9 - "Neighborhood Risk Data"
Cohesion: 0.17
Nodes (12): barrios, distribucion, tasa_promedio_ciudad, todos, top10_mayor_riesgo, top10_menor_riesgo, total, alto_55_70 (+4 more)

### Community 10 - "LLM Explain Pipeline"
Cohesion: 0.36
Nodes (7): Groq → Anthropic Fallback Pattern, RiskExplainer._build_prompt(), RiskExplainer._llm_enrich_anthropic(), RiskExplainer._llm_enrich_groq(), RiskExplainer._rag_context(), RiskExplainer.explain(), POST /api/v1/explain

### Community 11 - "Model Export & Encoding"
Cohesion: 0.32
Nodes (7): build_encoders(), export_model(), DataFrame, str, export_models.py Ejecutar en el mismo entorno del notebook para exportar los mod, Construye los encoders de ubicación desde el dataframe de modelado., Serializa un modelo con sus encoders y métricas.

### Community 12 - "Geocoding Pipeline"
Cohesion: 0.48
Nodes (6): geocode_barrio(), in_medellin(), main(), nominatim_search(), geocode_barrios.py Geocodifica los barrios del modelo usando Nominatim (OpenStre, str

### Community 13 - "Training Notebook Scripts"
Cohesion: 0.40
Nodes (4): entrenar_escenario(), predecir_riesgo(), Predice la probabilidad de accidente.      Parámetros:     - ubicacion: str, nom, Entrena los tres modelos sobre un escenario dado y devuelve métricas.     Mantie

## Knowledge Gaps
- **380 isolated node(s):** `allow`, `str`, `float`, `str`, `La Candelaria` (+375 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `geocode_barrios.py — main()` connect `Feature Engineering Concepts` to `Medellín Neighborhood Coordinates`?**
  _High betweenness centrality (0.221) - this node is a cross-community bridge._
- **Why does `main()` connect `Feature Engineering Concepts` to `Chat RAG Pipeline`?**
  _High betweenness centrality (0.205) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `ModelRegistry` (e.g. with `BackgroundTasks` and `AlgoEnum`) actually correct?**
  _`ModelRegistry` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `RiskExplainer` (e.g. with `BackgroundTasks` and `AlgoEnum`) actually correct?**
  _`RiskExplainer` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `allow`, `export_model_knowledge.py Exporta todo el conocimiento del notebook a un JSON es`, `Extrae texto de los outputs de una celda de código.` to the rest of the system?**
  _408 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Medellín Neighborhood Coordinates` be split into smaller, more focused modules?**
  _Cohesion score 0.00641025641025641 - nodes in this community are weakly interconnected._
- **Should `FastAPI App & Core Modules` be split into smaller, more focused modules?**
  _Cohesion score 0.09125188536953242 - nodes in this community are weakly interconnected._