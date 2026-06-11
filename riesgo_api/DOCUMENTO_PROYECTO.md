# RiesgoVial — Documento del proyecto

> Documento de estudio y apoyo para la sustentación del proyecto **RiesgoVial**, plataforma de predicción de riesgo vial urbano para Medellín (Programa Talento Tech 2026).
> Reúne en un solo lugar el contexto del problema, los datos, el modelo de machine learning, la explicabilidad, el sistema de IA conversacional (RAG) y la arquitectura web completa.

---

## 1. Resumen ejecutivo

**RiesgoVial** predice la probabilidad de que ocurra un accidente de tráfico en un barrio de Medellín, para un día de la semana y una franja horaria determinados. A partir de esa probabilidad:

- Clasifica el riesgo en **bajo / medio / alto**.
- Explica **por qué** el modelo predijo ese riesgo (SHAP).
- Genera una **explicación en lenguaje natural** combinando esos datos con una IA (Groq y/o Anthropic Claude).
- Ofrece un **dashboard** con mapa de calor, mapa interactivo y KPIs históricos.
- Incluye un **Chat IA** que responde preguntas sobre el modelo, los datos y la propia plataforma.

| Dato | Valor |
|---|---|
| Ciudad | Medellín, Antioquia, Colombia |
| Período de datos | 2008 – 2025 |
| Registros originales | 702,540 |
| Registros tras limpieza | 674,394 |
| Barrios/comunas modelados | 369–370 |
| Algoritmo final | XGBoost (Gradient Boosted Trees) |
| ROC AUC | 0.9182 |
| F1 | 0.8384 |

---

## 2. Contexto y objetivo

**Objetivo del modelo:** predecir la probabilidad de que ocurra **al menos un accidente** de tráfico para una combinación de **(barrio, día de la semana, franja horaria, mes)**.

**Por qué es un problema interesante:**
- Un modelo que solo use `DIA_SEMANA` trataría todos los sábados como iguales. Pero un sábado de San Valentín no es igual a un sábado de Navidad: la movilidad cambia según la **época del año**.
- Para capturarlo, el proyecto incorpora **estacionalidad anual** mediante codificación cíclica (seno/coseno), sin explotar la memoria del computador.

**Fuente de los datos:** Sistema de Información de Accidentalidad Vial de Medellín (Secretaría de Movilidad), entregado como `Fatal_Road_Traffic_Normalizado.xlsx`.

---

## 3. Datos

### 3.1 Carga y limpieza

- Carga con `pandas.read_excel()` y tipos optimizados (`int16`, `category`) → ~156 MB en memoria.
- **702,540** registros originales → **674,394** tras limpieza (**28,146 eliminados**, ~4%).
- Motivos de eliminación:
  - Nulos en `BARRIO` o `UBICACION` (11,683 registros).
  - Fechas inválidas o mal formateadas.
  - Horas fuera del rango 0–23.

### 3.2 La grilla espacio-temporal ("el truco del target")

El dataset original **solo contiene accidentes que sí ocurrieron** (ejemplos positivos). Para entrenar un clasificador se necesitan también ejemplos de "no accidente":

1. Se construye una **grilla completa** con todas las combinaciones posibles de:
   ```
   370 ubicaciones × 7 días de la semana × 12 franjas horarias × 12 meses = 372,960 slots
   ```
2. Si una combinación **aparece** en el histórico → `HUBO_INCIDENTE = 1`.
3. Si **no aparece** → `HUBO_INCIDENTE = 0`.
4. El modelo aprende a distinguir condiciones de riesgo, generalizando incluso a combinaciones nunca observadas.

### 3.3 Balance de clases

- ~51% de los slots tienen al menos un incidente, ~49% no.
- El balance es **natural** (gracias a la grilla) — no hizo falta SMOTE ni undersampling, solo `scale_pos_weight` en XGBoost.

### 3.4 Franjas horarias de 2 horas

Se agruparon las 24 horas en **12 franjas de 2 horas** (`00-02h`, `02-04h`, … `22-24h`):
- Con franjas de 1h habría demasiado ruido estadístico (pocos eventos por slot).
- Con franjas de 4h se perdería detalle de patrones como la hora pico.
- 2h es el punto óptimo encontrado.

---

## 4. Feature engineering — 26 variables

| Categoría | Variables | Para qué sirven |
|---|---|---|
| **Temporales básicas** (5) | `DIA_SEMANA`, `MES`, `ES_FIN_SEMANA`, `ES_LABORAL`, `HORA_PUNTO_MEDIO` | Contexto directo de cuándo ocurre la consulta |
| **Cíclicas** (6) | `DIA_SEMANA_SIN/COS`, `MES_SIN/COS`, `DIA_ANIO_SIN/COS` | Capturan que el tiempo es circular: domingo y lunes son "cercanos", diciembre y enero también |
| **Ubicación** (2) | `UBICACION_TARGET_ENC`, `UBICACION_LOG_ODDS` | Tasa histórica de accidentes del barrio (suavizada) y su versión en log-odds |
| **Franjas horarias** (12) | `FRANJA_00-02h` … `FRANJA_22-24h` | Dummies one-hot: 1 en la franja activa, 0 en el resto |
| **Interacciones** (1) | `INTERACCION_FINDE_NOCHE` | 1 si es fin de semana **y** hora ≥ 20:00 — captura el patrón crítico nocturno-festivo |

### 4.1 Codificación cíclica (sin/cos)

`DIA_SEMANA` y `MES` son variables cíclicas: si se codifican como números lineales (0,1,2…6), el modelo "ve" que el domingo (6) está lejos del lunes (0), cuando en realidad son consecutivos. La solución:

```
sin = sin(2π · valor / período)
cos = cos(2π · valor / período)
```

Esto mapea el tiempo a un círculo, donde el final y el inicio del ciclo quedan cerca.

### 4.2 Estacionalidad anual sin explotar la memoria

Para no multiplicar las filas por 365 días, el dataset **no se agrupa por día exacto**:
1. Se agrupa por `MES` (12 categorías).
2. A cada mes se le asigna el **día 15** como representante (enero → día 15, febrero → día 46 del año, etc.).
3. Se calculan `DIA_ANIO_SIN` / `DIA_ANIO_COS` sobre ese día representativo.
4. El modelo aprende estacionalidad suave (ej. diciembre y enero son "parecidos") sin explotar la RAM.

### 4.3 Target encoding suavizado (m = 50)

`UBICACION_TARGET_ENC` reemplaza cada barrio por su **tasa histórica de accidentes**, pero suavizada:

```
tasa_suavizada = (n_barrio · tasa_barrio + m · tasa_global) / (n_barrio + m)
```

Con `m = 50`: si un barrio tiene pocos registros históricos, su tasa se "acerca" a la tasa global de la ciudad, evitando sobreajuste. `UBICACION_LOG_ODDS` es la versión `log(p / (1-p))` de esa tasa, que captura mejor el efecto no lineal.

### 4.4 Variables más importantes (según SHAP)

| Variable | Importancia | Por qué |
|---|---|---|
| `UBICACION_TARGET_ENC` | Muy alta | El historial de accidentes del barrio es el factor más predictivo |
| `HORA_PUNTO_MEDIO` | Alta | La hora del día como valor numérico continuo |
| `UBICACION_LOG_ODDS` | Alta | Complementa al target encoding, capta efectos no lineales |
| `FRANJA_18-20h` | Media-alta | Hora pico vespertina — mayor volumen de tráfico |
| `INTERACCION_FINDE_NOCHE` | Media | Fin de semana + noche = riesgo amplificado |

---

## 5. Modelado

### 5.1 Algoritmo final: XGBoost

| Hiperparámetro | Valor |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 8 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `scale_pos_weight` | negativos / positivos (≈1.0, automático) |
| `random_state` | 42 |
| Split | 80% entrenamiento / 20% prueba, estratificado por target |

### 5.2 Métricas finales (test, histórico 2008–2025)

| Métrica | Valor |
|---|---|
| ROC AUC | **0.9182** |
| F1 | 0.8384 |
| Precisión | 0.8401 |
| Recall | 0.8367 |
| Accuracy | 0.8345 |

### 5.3 Comparativa de modelos

| Modelo | Escenario | AUC | F1 |
|---|---|---|---|
| Regresión Logística | Histórico | 0.9129 | 0.8362 |
| Regresión Logística | Post-COVID | 0.8215 | 0.6143 |
| Random Forest | Histórico | 0.9177 | 0.8386 |
| Random Forest | Post-COVID | 0.8485 | 0.6389 |
| **XGBoost** | **Histórico** | **0.9182** ✅ | **0.8384** |
| XGBoost | Post-COVID | 0.8502 | 0.6417 |

**¿Por qué XGBoost?** Tuvo el mejor ROC AUC, maneja el desbalance de clases de forma nativa (`scale_pos_weight`) y es más rápido en inferencia que Random Forest (~12 ms por predicción en producción).

### 5.4 Experimento COVID — histórico vs post-COVID

- Modelo entrenado **solo con datos 2020+** (post-COVID): AUC = 0.85.
- Modelo entrenado con **histórico completo 2008–2025**: AUC = 0.9182.
- Conclusión: el COVID redujo la movilidad temporalmente, pero los patrones de accidentalidad **volvieron a la normalidad** después. Usar el histórico completo da mejor generalización → es el modelo elegido para producción.

---

## 6. Explicabilidad — SHAP

`TreeExplainer` (librería SHAP) calcula, para cada predicción, **cuánto contribuye cada una de las 26 variables** al resultado final (positivo = aumenta el riesgo, negativo = lo reduce).

- El endpoint `/api/v1/explain` devuelve los **top 5 features** ordenados por valor absoluto de su contribución SHAP.
- Esos valores se traducen a una narrativa simple, ej.: *"el historial del barrio aumenta el riesgo (+3.68), la hora del día lo reduce (-0.42)..."*
- Esa narrativa es la base del contexto que recibe el LLM para redactar la explicación final en lenguaje natural.

---

## 7. Sistema RAG (Recuperación de contexto)

El proyecto usa **dos mecanismos de recuperación de texto independientes**, ninguno basado en embeddings/vectores — ambos sobre archivos de texto plano.

### 7.1 RAG de `/explain` — explicación de riesgo por barrio

Archivo fuente: `rag/documents/medellin/todos_los_barrios.txt` (369 entradas, generado por `export_model_knowledge.py`).

Flujo (`explainer.py`):
1. Buscar el barrio por **coincidencia exacta de nombre** → obtener su descripción textual (factores de riesgo, comuna, puntos críticos).
2. Combinar esa descripción con la **narrativa SHAP** (top features de la predicción) y los datos de la consulta (día, hora, franja).
3. Enviar ese contexto a un LLM configurable (ver sección 8) para redactar **una explicación de máximo 3 oraciones**.
4. Si no hay ninguna API key de LLM configurada: se devuelve el contexto crudo (descripción + narrativa SHAP), sin pasar por IA.

### 7.2 RAG del Chat IA — TF-IDF sobre `model_knowledge.json`

Archivo fuente: `rag/documents/medellin/model_knowledge.json` (generado por `export_model_knowledge.py`), más `estadisticas_viales.txt` y `movilidad_medellin.txt`.

Flujo (`main.py`, función `_build_chunks` / `_get_rag` / `_retrieve_context`):
1. El JSON y los `.txt` se dividen en **chunks de texto** (uno por sección: dataset, métricas, features, barrios, decisiones de diseño, FAQs, secciones del notebook, estadísticas viales, movilidad, **arquitectura web**).
2. `TfidfVectorizer` (scikit-learn, bigramas, `sublinear_tf`) indexa esos chunks **en memoria**, la primera vez que se consulta cada ciudad.
3. Cuando llega una pregunta, se vectoriza con TF-IDF y se calcula **similitud coseno** contra todos los chunks.
4. Se recuperan los **top-K chunks** (`CHAT_RAG_TOP_K`, default 5) y se inyectan en el `{context}` del system prompt.
5. Groq genera la respuesta final, manteniendo hasta `CHAT_HISTORY_TURNS` (default 10) turnos de historial de la conversación.

> El Chat IA **solo funciona con Groq** (no tiene fallback a Anthropic). Si `GROQ_API_KEY` no está configurada, el endpoint `/api/v1/chat` responde error 503.

---

## 8. LLM configurable: Groq y/o Anthropic Claude

| Endpoint | Proveedor primario | Fallback / alternativa | Sin ninguna API key |
|---|---|---|---|
| `/api/v1/explain` | Groq (`GROQ_API_KEY`, modelo `GROQ_MODEL`, default `openai/gpt-oss-120b`) | Anthropic Claude (`ANTHROPIC_API_KEY`, `claude-sonnet-4-6`) — automático si Groq falla o no está configurado | Devuelve el contexto crudo (sin LLM) |
| `/api/v1/chat` | Groq (única opción) | — | Error 503 |

Lógica de detección (`explainer.py::_detect_llm`):
```python
if os.getenv("GROQ_API_KEY"):       return "groq"
if os.getenv("ANTHROPIC_API_KEY"):  return "anthropic"
return None  # RAG local sin LLM
```

---

## 9. Arquitectura web

```
┌─────────────────────────────────────────────────────────────┐
│  Cliente Web (Frontend) — SPA HTML/CSS/JS puro, sin frameworks│
│  5 módulos: Consulta · Dashboard · Modelos · Chat IA · Docs   │
│  Leaflet (mapa) · Chart.js (gráficos) · marked.js (markdown)  │
└───────────────────────────┬───────────────────────────────────┘
                             │ HTTP/JSON · puerto 80
┌───────────────────────────▼───────────────────────────────────┐
│  Nginx 1.27-alpine — sirve el HTML estático y hace             │
│  proxy reverso de /api/* hacia el backend (puerto 8000)        │
└───────────────────────────┬───────────────────────────────────┘
                             │
┌───────────────────────────▼───────────────────────────────────┐
│  API REST — FastAPI 0.115 + Gunicorn (2 workers Uvicorn)        │
│  Pydantic v2 valida requests. 9 endpoints.                      │
└──────┬───────────────────┬───────────────────┬─────────────────┘
       │                    │                   │
┌──────▼──────────┐ ┌───────▼────────────┐ ┌────▼─────────────────┐
│ Motor predicción │ │ RAG /explain        │ │ RAG Chat              │
│ XGBoost + SHAP   │ │ todos_los_barrios   │ │ TF-IDF sobre          │
│ (model_loader.py)│ │ + LLM (§8)          │ │ model_knowledge.json  │
└──────────────────┘ └─────────────────────┘ └───────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────────┐
│  Persistencia: models/medellin_xgboost.pkl (pickle p5),         │
│  rag/documents/medellin/*.json y *.txt — montados como          │
│  volúmenes Docker de solo lectura                                │
└──────────────────────────────────────────────────────────────────┘
```

### 9.1 Frontend
SPA en HTML/CSS/JS sin dependencias de build. Cinco secciones:
- **Consulta**: formulario (barrio, fecha, hora) → gauge de riesgo + gráfico SHAP + explicación IA.
- **Dashboard**: KPIs, gráfico por hora, tendencia diaria, mapa de calor (heatmap), mapa Leaflet con círculos de riesgo por barrio.
- **Modelos**: métricas del modelo, distribución de riesgo, top 10 barrios más/menos peligrosos, comparativa de algoritmos.
- **Chat IA**: conversación con historial, contador de tokens.
- **Documentación / Arquitectura**: este mismo contenido, en formato visual con tooltips.

### 9.2 Nginx
Sirve `./frontend` en el puerto 80 con headers `no-cache` (para que los cambios de HTML se vean al instante) y redirige `/api/*` al contenedor de la API.

### 9.3 API (FastAPI + Gunicorn)
- 2 workers Uvicorn asíncronos.
- Carga el modelo `.pkl` y construye los índices RAG **en memoria** al iniciar / en la primera consulta.
- Cachea resultados de `/api/v1/heatmap` por `(ciudad, fecha)`: primera carga ~4s (370 barrios × 12 franjas), siguientes ~30ms.

### 9.4 Docker
Dos contenedores orquestados con Docker Compose:
- `riesgovial_api` — puerto 8000 interno, monta `./models` y `./logs`.
- `riesgovial_nginx` — puerto 80, monta `./frontend` y `nginx.conf`. Espera a que la API pase el healthcheck antes de arrancar.

---

## 10. API — Endpoints principales

| Método | Endpoint | Descripción | Tiempo aprox. |
|---|---|---|---|
| `GET` | `/health` | Estado de la API y modelos cargados | — |
| `POST` | `/api/v1/predict` | Predicción de riesgo (city, neighborhood, day_of_week, hour, fecha) | ~12 ms |
| `POST` | `/api/v1/explain` | SHAP + RAG + LLM → explicación en lenguaje natural | ~2–4 s |
| `GET` | `/api/v1/heatmap/{city}/{date}` | Matriz de riesgo barrio × franja horaria | ~4s (1ª vez) / ~30ms (caché) |
| `GET` | `/api/v1/neighborhoods/{city}` | Lista de barrios válidos | — |
| `GET` | `/api/v1/cities` | Modelos cargados con métricas | — |
| `GET` | `/api/v1/model-info/{city}` | Métricas, distribución de riesgo, top barrios, comparativa | — |
| `GET` | `/api/v1/coordinates/{city}` | Coordenadas geográficas para el mapa Leaflet | — |
| `POST` | `/api/v1/chat` | Chat IA con RAG TF-IDF (requiere `GROQ_API_KEY`) | — |
| `GET` | `/api/v1/chat/config` | Configuración expuesta al frontend (`history_turns`, `rag_top_k`) | — |
| `POST` | `/api/v1/train/{city}` | Lanza entrenamiento asíncrono en background | — |

---

## 11. Decisiones de diseño clave (para defender en la sustentación)

| Decisión | Razón | Impacto |
|---|---|---|
| Grilla espacio-temporal en lugar de registros crudos | Los accidentes son eventos esporádicos; usar solo registros crudos genera desbalance extremo | Permite generalizar a combinaciones no vistas |
| Franjas de 2h en lugar de horas individuales | 1h genera demasiado ruido; 4h pierde detalle de patrones de movilidad | 12 franjas diarias — mejor balance señal/ruido |
| Target encoding suavizado (m=50) | Barrios con pocos registros tendrían tasas inestables | Evita sobreajuste en barrios con <50 registros históricos |
| Codificación cíclica sin/cos | DIA_SEMANA y MES son cíclicos; codificación lineal rompe esa continuidad | El modelo aprende que diciembre y enero son similares |
| XGBoost sobre RF y Regresión Logística | Mejor AUC (0.9182) + manejo nativo de desbalance + menor latencia | AUC superior y predicciones más rápidas en producción |
| Histórico completo (2008-2025) sobre post-COVID | Post-COVID tiene AUC 0.85 vs 0.92 del histórico — los patrones volvieron tras la pandemia | Mejor generalización del modelo en producción |
| RAG basado en texto plano (sin FAISS/embeddings) | Los datasets de contexto son pequeños (cientos de chunks); TF-IDF y lookup exacto son suficientes y evitan dependencias pesadas | Arranque más rápido, sin modelos de embeddings que descargar |
| LLM configurable (Groq y/o Anthropic) | Evita depender de un único proveedor; Groq es más rápido/económico, Anthropic como respaldo de calidad | Resiliencia ante caídas o ausencia de una de las dos API keys |

---

## 12. Estructura del notebook (proceso de ciencia de datos)

El notebook `modelo_riesgo_enriquecido_covid.ipynb` documenta todo el proceso, paso a paso:

1. Configuración del entorno
2. Carga del dataset
3. *Profundización:* estructura del dato crudo
4. Diagnóstico de valores nulos
5. Limpieza de datos
6. *Profundización:* impacto visual de la limpieza
7. Feature engineering temporal
8. *Profundización:* ¿por qué pivotear en franjas de 2 horas?
9. Análisis Exploratorio de Datos (EDA)
10. Construcción del dataset de modelado (la grilla)
11. *Profundización:* ejemplos reales y balance de clases
12. *Profundización:* tabla de coordenadas circulares por mes
13. Añadir estacionalidad (día del año)
14. Feature engineering para el modelo
15. *Profundización:* efecto de cada variable sobre la tasa de accidentes
16. Preparación para entrenamiento
17. Entrenamiento de modelos (LR, RF, XGBoost)
18. Evaluación de modelos
19. Importancia de features (XGBoost)
20. Función de predicción
21. Mapa de riesgo
22. Resumen y conclusiones
23. Experimento: histórico completo vs post-COVID

---

## 13. Stack tecnológico completo

| Categoría | Tecnologías |
|---|---|
| Lenguaje | Python 3.11 |
| Machine Learning | XGBoost 2.1.3, scikit-learn 1.5.2, SHAP 0.46.0 |
| Datos | pandas 2.2.3, numpy 1.26.4, openpyxl |
| API | FastAPI 0.115, Gunicorn, Uvicorn, Pydantic v2 |
| LLM | Groq API (`openai/gpt-oss-120b`), Anthropic Claude (fallback/alternativa) |
| Infraestructura | Docker, Docker Compose, Nginx 1.27-alpine |
| Frontend | HTML5, CSS, JavaScript puro, Leaflet 1.9, Chart.js 4, marked.js |
| Notebook | Jupyter (`modelo_riesgo_enriquecido_covid.ipynb`) |

---

## 14. Preguntas frecuentes (preparación para preguntas del jurado)

**¿Qué tan preciso es el modelo?**
ROC AUC de 0.9182, F1 de 0.8384 y precisión de 0.8401 sobre el histórico 2008–2025.

**¿Cuántos barrios cubre el modelo?**
369 barrios y comunas de Medellín, con una tasa promedio de accidentabilidad del 51.3%.

**¿Qué variable influye más en la predicción?**
La tasa histórica del barrio (`UBICACION_TARGET_ENC`), seguida de la hora del día (`HORA_PUNTO_MEDIO`). El historial de accidentes del barrio pesa más que el momento del día.

**¿Por qué se usaron franjas de 2 horas?**
Reducen el ruido estadístico manteniendo granularidad suficiente para capturar hora pico matutina (6-8h), vespertina (17-20h), noche y madrugada.

**¿Qué impacto tuvo el COVID en el modelo?**
El modelo entrenado solo con datos post-COVID (2020+) tiene AUC 0.85 vs 0.92 del histórico completo — los patrones de movilidad volvieron a la normalidad tras la pandemia.

**¿Cuál es el barrio más peligroso / más seguro?**
Más peligroso: **La Candelaria** (96.1% de tasa histórica). Más seguro: **El Jardín** (2.5%).

**¿Qué pasa si no hay API key de IA configurada?**
`/explain` devuelve el contexto crudo (descripción del barrio + narrativa SHAP) sin pasar por LLM. El Chat IA queda deshabilitado (error 503).

**¿Por qué no se usó un índice vectorial (FAISS/embeddings) para el RAG?**
Los documentos de contexto son pequeños (decenas de chunks por ciudad); TF-IDF (para el chat) y coincidencia exacta de nombre (para `/explain`) son suficientes, más livianos y no requieren descargar modelos de embeddings.

---

## 15. Glosario de términos técnicos

| Término | Explicación simple |
|---|---|
| **XGBoost** | Algoritmo de machine learning que combina muchos "árboles de decisión" pequeños para hacer una predicción más precisa |
| **SHAP** | Técnica que explica cuánto aporta cada variable a una predicción concreta (positivo o negativo) |
| **ROC AUC** | Métrica entre 0 y 1 que mide qué tan bien el modelo distingue entre "hubo accidente" y "no hubo accidente"; 1 = perfecto |
| **F1 / Precisión / Recall** | Métricas de calidad de un clasificador: qué tan pocos errores comete y qué tan bien detecta los casos positivos |
| **Target encoding** | Reemplazar una categoría (ej. un barrio) por un número que resume su comportamiento histórico (su tasa de riesgo) |
| **Codificación cíclica (sin/cos)** | Forma de representar números que "dan la vuelta" (como los días de la semana o los meses) para que el inicio y el fin queden cerca |
| **RAG** (Retrieval-Augmented Generation) | Técnica de IA que busca información relevante en documentos propios antes de generar una respuesta, para que la IA no "invente" datos |
| **TF-IDF** | Técnica que mide qué tan importante es una palabra/frase para un texto, sin usar IA — útil para buscar el fragmento más relevante a una pregunta |
| **LLM** (Large Language Model) | Modelo de inteligencia artificial que genera texto en lenguaje natural (ej. Groq, Anthropic Claude) |
| **API REST** | Forma estándar en que el frontend le pide datos al backend a través de internet |
| **Docker / contenedor** | Forma de empaquetar una aplicación con todo lo que necesita para correr igual en cualquier computador |
| **Nginx / proxy reverso** | Servidor que recibe las peticiones del navegador y las reparte entre el sitio web estático y la API |
| **SPA** (Single Page Application) | Aplicación web que vive en una sola página HTML y cambia su contenido dinámicamente con JavaScript |

---

## 16. Equipo y créditos

- **Desarrollador:** Juan Camilo Ramirez
- **Propietario / proveedor de datos:** Cristian David Correa Álvarez — Universidad Nacional de Colombia
- **Programa:** Talento Tech 2026 — Gobernación
- **Licencia de los datos:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Fuente de datos:** Sistema de Información de Accidentalidad Vial de Medellín, Secretaría de Movilidad

---

## 17. Dónde está cada cosa en el código (mapa rápido)

| Componente | Archivo |
|---|---|
| Endpoints de la API | `riesgo_api/main.py` |
| Carga del modelo y predicciones | `riesgo_api/model_loader.py` |
| RAG `/explain` + LLM | `riesgo_api/explainer.py` |
| Frontend (SPA) | `riesgo_api/frontend/index.html` |
| Configuración Docker | `riesgo_api/docker-compose.yml`, `riesgo_api/Dockerfile`, `riesgo_api/nginx.conf` |
| Conocimiento del Chat IA | `riesgo_api/rag/documents/medellin/model_knowledge.json` |
| Descripciones de barrios (`/explain`) | `riesgo_api/rag/documents/medellin/todos_los_barrios.txt` |
| Coordenadas del mapa | `riesgo_api/rag/documents/medellin/coordinates.json` |
| Notebook de entrenamiento | `modelo_riesgo_enriquecido_covid.ipynb` |
| Generador de `model_knowledge.json` | `export_model_knowledge.py` |
| Manual de despliegue | `riesgo_api/MANUAL.md` |
| Tests | `riesgo_api/tests/test_api.py` (17 tests) |
