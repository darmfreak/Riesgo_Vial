"""
export_model_knowledge.py
Exporta todo el conocimiento del notebook a un JSON estructurado
para uso en chatbot/RAG sobre el modelo de riesgo vial.
"""
import json
import pickle
import re
from pathlib import Path

NB_PATH    = Path(__file__).parent / "modelo_riesgo_enriquecido_covid.ipynb"
MODEL_PATH = Path(__file__).parent / "riesgo_api/models/medellin_xgboost.pkl"
OUT_PATH   = Path(__file__).parent / "riesgo_api/rag/documents/medellin/model_knowledge.json"

def clean(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_outputs(cell):
    """Extrae texto de los outputs de una celda de código."""
    results = []
    for out in cell.get("outputs", []):
        if out.get("output_type") in ("stream", "execute_result", "display_data"):
            text = out.get("text", out.get("data", {}).get("text/plain", ""))
            if isinstance(text, list): text = "".join(text)
            text = text.strip()
            # Solo outputs que tengan métricas o texto relevante
            if text and len(text) > 20 and not text.startswith("<Figure"):
                results.append(text[:2000])
    return "\n".join(results)

def main():
    with open(NB_PATH) as f:
        nb = json.load(f)

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    meta = bundle["metadata"]
    enc  = bundle["encoders"]

    UNDEFINED = {"0","sin informacion","sin información","no definido","nd","n/a",""}

    def risk_level(r):
        if r >= 0.70: return "muy alto"
        if r >= 0.55: return "alto"
        if r >= 0.40: return "moderado"
        if r >= 0.20: return "bajo"
        return "muy bajo"

    all_nb = sorted(
        [{"nombre": nb_, "tasa": round(r, 4), "nivel": risk_level(r)}
         for nb_, r in enc["target_enc"].items()
         if nb_ and nb_.strip().lower() not in UNDEFINED],
        key=lambda x: x["tasa"], reverse=True
    )

    # ── Secciones del notebook ───────────────────────────────────────
    sections = []
    current_section = None

    for cell in nb["cells"]:
        src = "".join(cell["source"]).strip()
        if not src:
            continue

        if cell["cell_type"] == "markdown":
            # Detectar encabezados como nuevas secciones
            lines = src.split("\n")
            title_line = next((l for l in lines if l.startswith("#")), None)
            if title_line:
                title = re.sub(r'^#+\s*', '', title_line).strip()
                title = re.sub(r'[🚗🔬📋🧹📐📊🌐🎯⚙️🤖🔍📈🗺️🧪✅🌙]', '', title).strip()
                body = clean(" ".join(
                    l for l in lines if not l.startswith("#") and l.strip()
                ))
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "titulo": title,
                    "descripcion": body,
                    "codigo_explicacion": [],
                    "outputs_clave": []
                }
            elif current_section:
                current_section["descripcion"] += " " + clean(src)

        elif cell["cell_type"] == "code" and current_section:
            code_preview = clean(src[:300])
            output = extract_outputs(cell)
            if output:
                current_section["outputs_clave"].append(output[:1500])
            if code_preview:
                current_section["codigo_explicacion"].append(code_preview)

    if current_section:
        sections.append(current_section)

    # ── Decisiones de diseño clave (extraídas de los markdowns) ─────
    design_decisions = [
        {
            "decision": "Grilla espacio-temporal en lugar de registros crudos",
            "razon": "Los accidentes son eventos esporádicos. Usar registros crudos genera desbalance extremo. La grilla completa (370 ubicaciones × 7 días × 12 franjas × 12 meses = 372,960 slots) permite al modelo aprender tanto de cuándo SÍ ocurrió un accidente como de cuándo NO ocurrió.",
            "impacto": "Permite generalizar a combinaciones no vistas en el histórico"
        },
        {
            "decision": "Franjas horarias de 2 horas en lugar de horas individuales",
            "razon": "Las franjas de 2h reducen el ruido estadístico en slots con pocos eventos y capturan mejor los patrones de movilidad (hora pico, noche, madrugada). Granularidad de 1h genera demasiado ruido; granularidad de 4h pierde detalle.",
            "impacto": "12 franjas diarias vs 24 horas — mejor balance señal/ruido"
        },
        {
            "decision": "Target encoding suavizado (m=50) para UBICACION",
            "razon": "Barrios con pocos registros tendrían estimaciones de tasa inestables. El suavizado con m=50 mezcla la tasa del barrio con la tasa global, dando más peso a la global cuando hay pocos datos.",
            "impacto": "Evita sobreajuste en barrios con menos de 50 registros históricos"
        },
        {
            "decision": "Codificación cíclica (sin/cos) para variables temporales",
            "razon": "Variables como DIA_SEMANA y MES son cíclicas: el domingo (6) es 'cercano' al lunes (0). Usar valores numéricos lineales rompe esta continuidad. Sin/cos preserva la distancia circular.",
            "impacto": "El modelo aprende correctamente que diciembre y enero son similares estacionalmente"
        },
        {
            "decision": "XGBoost sobre Random Forest y Regresión Logística",
            "razon": "XGBoost tuvo el mejor ROC AUC (0.9182 vs 0.9177 RF vs 0.9129 LR) con manejo nativo de desbalance (scale_pos_weight). Además es más rápido en inferencia que RF para producción.",
            "impacto": "AUC superior y menor latencia de predicción en API"
        },
        {
            "decision": "Historial completo 2008-2025 sobre dataset post-COVID",
            "razon": "El experimento COVID mostró que el modelo post-COVID (2020+) tiene AUC 0.85 vs 0.92 del histórico. El COVID redujo la movilidad temporalmente pero los patrones volvieron. El dataset completo tiene más generalización.",
            "impacto": "AUC histórico 0.9182 vs post-COVID 0.8502"
        }
    ]

    # ── Features del modelo ──────────────────────────────────────────
    features = {
        "total": 26,
        "categorias": {
            "temporales_basicas": ["DIA_SEMANA", "MES", "ES_FIN_SEMANA", "ES_LABORAL", "HORA_PUNTO_MEDIO"],
            "ciclicas": ["DIA_SEMANA_SIN", "DIA_SEMANA_COS", "MES_SIN", "MES_COS", "DIA_ANIO_SIN", "DIA_ANIO_COS"],
            "ubicacion": ["UBICACION_TARGET_ENC", "UBICACION_LOG_ODDS"],
            "franjas_horarias": [
                "FRANJA_00-02h","FRANJA_02-04h","FRANJA_04-06h","FRANJA_06-08h",
                "FRANJA_08-10h","FRANJA_10-12h","FRANJA_12-14h","FRANJA_14-16h",
                "FRANJA_16-18h","FRANJA_18-20h","FRANJA_20-22h","FRANJA_22-24h"
            ],
            "interacciones": ["INTERACCION_FINDE_NOCHE"]
        },
        "mas_importantes": [
            {"nombre": "UBICACION_TARGET_ENC", "importancia": "muy alta", "descripcion": "Tasa histórica del barrio — el factor más predictivo"},
            {"nombre": "HORA_PUNTO_MEDIO", "importancia": "alta", "descripcion": "Hora del día como valor numérico continuo"},
            {"nombre": "UBICACION_LOG_ODDS", "importancia": "alta", "descripcion": "Log-odds de la tasa del barrio — complementa target_enc"},
            {"nombre": "FRANJA_18-20h", "importancia": "media-alta", "descripcion": "Hora pico vespertina — mayor volumen de tráfico"},
            {"nombre": "INTERACCION_FINDE_NOCHE", "importancia": "media", "descripcion": "Combinación fin de semana + noche — riesgo amplificado"},
        ]
    }

    # ── Dataset stats ────────────────────────────────────────────────
    dataset = {
        "fuente": "Fatal_Road_Traffic_Normalizado.xlsx — registros de la Secretaría de Movilidad de Medellín",
        "periodo": "2008 – 2025",
        "registros_originales": 702540,
        "registros_post_limpieza": 674394,
        "eliminados_limpieza": 28146,
        "motivos_eliminacion": [
            "Nulos en BARRIO o UBICACION (11,683)",
            "Fechas inválidas o mal formateadas",
            "Horas fuera del rango 0–23"
        ],
        "columnas_originales": ["LLAVE","AÑO","CLASE_INCIDENTE","MES","DIA","HORA_INCIDENTE","DIA_NOMBRE","BARRIO","COMUNA"],
        "slots_modelado": meta.get("slots", 372960),
        "ubicaciones": meta.get("neighborhoods", 370),
        "balance_clases": {
            "positivos": "~51% de slots con al menos un incidente",
            "negativos": "~49% sin incidentes",
            "nota": "Balance natural — no requirió SMOTE ni undersampling, solo scale_pos_weight en XGBoost"
        }
    }

    # ── Métricas de entrenamiento ────────────────────────────────────
    metricas = {
        "modelo_final": "XGBoost — histórico completo 2008-2025",
        "hiperparametros": {
            "n_estimators": 200,
            "max_depth": 8,
            "scale_pos_weight": "neg/pos (auto)",
            "random_state": 42
        },
        "split": "80% entrenamiento / 20% prueba — estratificado por target",
        "resultados": {
            "ROC_AUC": meta.get("ROC AUC", 0.9182),
            "F1": meta.get("F1", 0.8384),
            "Precision": meta.get("Precision", 0.8401),
            "Recall": meta.get("Recall", 0.8367),
            "Accuracy": meta.get("Accuracy", 0.8345)
        },
        "comparativa": [
            {"modelo": "Regresión Logística", "escenario": "Histórico",  "auc": 0.9129, "f1": 0.8362},
            {"modelo": "Regresión Logística", "escenario": "Post-COVID", "auc": 0.8215, "f1": 0.6143},
            {"modelo": "Random Forest",       "escenario": "Histórico",  "auc": 0.9177, "f1": 0.8386},
            {"modelo": "Random Forest",       "escenario": "Post-COVID", "auc": 0.8485, "f1": 0.6389},
            {"modelo": "XGBoost",             "escenario": "Histórico",  "auc": 0.9182, "f1": 0.8384, "ganador": True},
            {"modelo": "XGBoost",             "escenario": "Post-COVID", "auc": 0.8502, "f1": 0.6417}
        ]
    }

    # ── Barrios ──────────────────────────────────────────────────────
    barrios = {
        "total": len(all_nb),
        "tasa_promedio_ciudad": round(enc["global_rate"], 4),
        "distribucion": {
            "muy_alto_gte70": sum(1 for x in all_nb if x["tasa"] >= 0.70),
            "alto_55_70":     sum(1 for x in all_nb if 0.55 <= x["tasa"] < 0.70),
            "moderado_40_55": sum(1 for x in all_nb if 0.40 <= x["tasa"] < 0.55),
            "bajo_20_40":     sum(1 for x in all_nb if 0.20 <= x["tasa"] < 0.40),
            "muy_bajo_lt20":  sum(1 for x in all_nb if x["tasa"] < 0.20)
        },
        "top10_mayor_riesgo": all_nb[:10],
        "top10_menor_riesgo": sorted(all_nb, key=lambda x: x["tasa"])[:10],
        "todos": all_nb
    }

    # ── Stack técnico ────────────────────────────────────────────────
    tech_stack = {
        "lenguaje": "Python 3.11",
        "ml": ["XGBoost 2.1.3", "scikit-learn 1.5.2", "SHAP 0.46.0"],
        "datos": ["pandas 2.2.3", "numpy 1.26.4", "openpyxl"],
        "api": ["FastAPI 0.115", "Gunicorn", "Uvicorn", "Pydantic v2"],
        "llm": ["Groq API (openai/gpt-oss-120b)", "Anthropic Claude (fallback)"],
        "infraestructura": ["Docker", "Docker Compose", "Nginx"],
        "notebook": "Jupyter — modelo_riesgo_enriquecido_covid.ipynb"
    }

    # ── JSON final ───────────────────────────────────────────────────
    knowledge = {
        "proyecto": "RiesgoVial — Sistema de predicción de riesgo de accidentes de tráfico en Medellín, Colombia",
        "version": meta.get("version", "1.0"),
        "ciudad": "Medellín, Antioquia, Colombia",
        "objetivo": "Predecir la probabilidad de que ocurra un accidente de tráfico en un barrio específico, en un día de la semana y una franja horaria determinados.",
        "dataset": dataset,
        "features": features,
        "metricas": metricas,
        "barrios": barrios,
        "decisiones_diseno": design_decisions,
        "tech_stack": tech_stack,
        "secciones_notebook": sections,
        "preguntas_frecuentes": [
            {"pregunta": "¿Qué tan preciso es el modelo?",
             "respuesta": f"El modelo XGBoost logra un ROC AUC de {meta.get('ROC AUC', 0.9182):.4f}, F1 de {meta.get('F1', 0.8384):.4f} y precisión de {meta.get('Precision', 0.8401):.4f} sobre el dataset histórico 2008-2025 de Medellín."},
            {"pregunta": "¿Cuántos barrios cubre el modelo?",
             "respuesta": f"El modelo cubre {len(all_nb)} barrios y comunas de Medellín, con una tasa promedio de accidentabilidad del {enc['global_rate']*100:.1f}%."},
            {"pregunta": "¿Qué variable influye más en la predicción?",
             "respuesta": "La tasa histórica del barrio (UBICACION_TARGET_ENC) es el factor más predictivo, seguido de la hora del día (HORA_PUNTO_MEDIO). Esto significa que el historial de accidentes del barrio es más determinante que el momento del día."},
            {"pregunta": "¿Por qué se usaron franjas de 2 horas?",
             "respuesta": "Las franjas de 2h reducen el ruido estadístico manteniendo suficiente granularidad para capturar los patrones de movilidad (hora pico matutina 6-8h, vespertina 17-20h, noche, madrugada)."},
            {"pregunta": "¿Qué impacto tuvo el COVID en el modelo?",
             "respuesta": "El experimento mostró que el modelo entrenado solo con datos post-COVID (2020+) tiene AUC 0.85 vs 0.92 del histórico completo. El COVID redujo la movilidad temporalmente pero los patrones volvieron a la normalidad."},
            {"pregunta": "¿Cuál es el barrio más peligroso de Medellín?",
             "respuesta": f"Según el modelo, {all_nb[0]['nombre']} tiene la mayor tasa histórica de accidentabilidad con {all_nb[0]['tasa']*100:.1f}%."},
            {"pregunta": "¿Cuál es el barrio más seguro?",
             "respuesta": f"Según el modelo, {sorted(all_nb, key=lambda x: x['tasa'])[0]['nombre']} tiene la menor tasa histórica de accidentabilidad con {sorted(all_nb, key=lambda x: x['tasa'])[0]['tasa']*100:.1f}%."},
            {"pregunta": "¿Cómo está construida la plataforma?",
             "respuesta": "RiesgoVial tiene un frontend (SPA en HTML, CSS y JavaScript con Leaflet y Chart.js), un servidor Nginx que sirve esa página y redirige las peticiones /api/ hacia el backend, y un backend en FastAPI (Python) que carga el modelo XGBoost y los índices RAG. Todo corre empaquetado con Docker y Docker Compose."},
            {"pregunta": "¿Qué tecnologías usa el backend?",
             "respuesta": "El backend está construido con FastAPI (Python 3.11) y servido con Gunicorn. Carga el modelo de predicción XGBoost y usa SHAP para las explicaciones, TF-IDF (scikit-learn) para el RAG del Chat IA, y Groq y/o Anthropic Claude como modelos de lenguaje."},
            {"pregunta": "¿Se guarda el historial del chat?",
             "respuesta": "No, el backend no tiene base de datos: es sin estado. El historial de la conversación se mantiene en memoria del navegador y se reenvía en cada pregunta como contexto; el servidor no lo guarda y se pierde al recargar la página."},
        ]
    }

    OUT_PATH.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"✓ Exportado: {OUT_PATH}")
    print(f"  Tamaño: {size_kb} KB")
    print(f"  Secciones del notebook: {len(sections)}")
    print(f"  Barrios: {len(all_nb)}")
    print(f"  Decisiones de diseño: {len(design_decisions)}")
    print(f"  Preguntas frecuentes: {len(knowledge['preguntas_frecuentes'])}")

    # ── todos_los_barrios.txt  (RAG de /explain — leído por explainer.py) ──
    BARRIOS_PATH = Path(__file__).parent / "riesgo_api/rag/documents/medellin/todos_los_barrios.txt"

    avg = enc["global_rate"]

    def _desc(nombre, tasa, nivel):
        t, a = f"{tasa*100:.0f}%", f"{avg*100:.0f}%"
        if nivel == "muy alto":
            return (f"{nombre} (Medellín) tiene una tasa histórica de accidentabilidad del {t}. "
                    f"Zona de alta concentración de incidentes viales, muy por encima del promedio "
                    f"de la ciudad ({a}). Especialmente crítica en horas pico y fines de semana nocturnos.")
        if nivel == "alto":
            return (f"{nombre} (Medellín) tiene una tasa histórica de accidentabilidad del {t}. "
                    f"Accidentabilidad elevada respecto al promedio de la ciudad ({a}). "
                    f"Se recomiendan medidas de precaución adicionales en horas de alta circulación.")
        if nivel == "moderado":
            return (f"{nombre} (Medellín) tiene una tasa histórica de accidentabilidad del {t}. "
                    f"Accidentabilidad cercana al promedio de la ciudad ({a}). "
                    f"Los incidentes se distribuyen principalmente en hora pico matutina y vespertina. "
                    f"Flujo vehicular mixto con presencia de motocicletas.")
        if nivel == "bajo":
            return (f"{nombre} (Medellín) tiene una tasa histórica de accidentabilidad del {t}. "
                    f"Accidentabilidad por debajo del promedio de la ciudad ({a}). "
                    f"Flujo vehicular moderado con incidentes concentrados en puntos específicos.")
        return (f"{nombre} (Medellín) tiene una tasa histórica de accidentabilidad del {t}. "
                f"Zona periurbana o de baja densidad vehicular con muy pocos incidentes registrados. "
                f"Principalmente vías rurales o corredores de baja circulación.")

    blocks = []
    for b in sorted(all_nb, key=lambda x: x["nombre"]):
        blocks.append(
            f"BARRIO: {b['nombre']}\n"
            f"ZONA: Medellín\n"
            f"TASA_HISTORICA: {b['tasa']*100:.0f}% (promedio ciudad: {avg*100:.0f}%)\n"
            f"NIVEL_RIESGO: {b['nivel']}\n"
            f"DESCRIPCIÓN: {_desc(b['nombre'], b['tasa'], b['nivel'])}"
        )

    BARRIOS_PATH.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"✓ Exportado: {BARRIOS_PATH}")
    print(f"  Barrios: {len(blocks)}  |  Tamaño: {BARRIOS_PATH.stat().st_size // 1024} KB")

if __name__ == "__main__":
    main()
