"""
explainer.py
Sistema RAG para generar explicaciones en lenguaje natural.
Usa FAISS + sentence-transformers + (opcional) LLM Claude.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("riesgovial.explainer")

DAYS_ES = ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"]
HOUR_CONTEXT = {
    (0,5):   "madrugada (menor visibilidad, velocidad vehicular elevada)",
    (6,8):   "hora pico matutina (alto flujo hacia centros de trabajo y estudio)",
    (9,11):  "mañana laboral (flujo moderado)",
    (12,14): "mediodía (pausa almuerzo, cruces peatonales frecuentes)",
    (15,16): "tarde temprana",
    (17,20): "hora pico vespertina (mayor volumen de tráfico del día)",
    (21,23): "noche (menor visibilidad, mayor presencia de alcohol en vía)",
}

RAG_DIR = Path(os.getenv("RAG_DIR", Path(__file__).parent / "rag"))


def hour_label(h: int) -> str:
    for (a, b), label in HOUR_CONTEXT.items():
        if a <= h <= b:
            return label
    return f"{h}:00h"


class RiskExplainer:
    def __init__(self):
        self._indexes: dict = {}
        self._embeddings = None
        self._llm_available = self._check_llm()
        self._load_embeddings()

    def _check_llm(self) -> bool:
        key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if key:
            logger.info("LLM disponible para explicaciones enriquecidas")
            return True
        logger.info("Sin API key LLM — usando RAG local")
        return False

    def _load_embeddings(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
            )
            logger.info("Embeddings sentence-transformers cargados")
        except Exception as e:
            logger.warning(f"No se pudo cargar embeddings: {e}. Usando fallback basado en reglas.")

    def _get_index(self, city: str):
        if city in self._indexes:
            return self._indexes[city]
        if self._embeddings is None:
            return None
        idx_path = RAG_DIR / "index" / city
        if not idx_path.exists():
            logger.warning(f"Índice FAISS no encontrado: {idx_path}")
            return None
        try:
            from langchain_community.vectorstores import FAISS
            db = FAISS.load_local(str(idx_path), self._embeddings,
                                  allow_dangerous_deserialization=True)
            self._indexes[city] = db
            logger.info(f"Índice FAISS cargado: {city}")
            return db
        except Exception as e:
            logger.warning(f"Error cargando índice FAISS {city}: {e}")
            return None

    def _rag_context(self, city: str, neighborhood: str, day_of_week: int, hour: int) -> str:
        db = self._get_index(city)
        if db is None:
            return ""
        query = (f"riesgo vial accidente {neighborhood} "
                 f"hora {hour} {hour_label(hour)} "
                 f"día {DAYS_ES[day_of_week] if 0 <= day_of_week <= 6 else ''}")
        try:
            docs = db.similarity_search(query, k=3)
            return "\n".join(d.page_content for d in docs)
        except Exception as e:
            logger.warning(f"FAISS similarity_search falló: {e}")
            return ""

    def _shap_narrative(self, shap_values: dict) -> str:
        if not shap_values:
            return ""
        top = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        parts = []
        for feat, val in top:
            direction = "aumenta" if val > 0 else "reduce"
            if "UBICACION" in feat:
                parts.append(f"el historial del barrio {direction} el riesgo ({val:+.3f})")
            elif "HORA" in feat or "FRANJA" in feat:
                parts.append(f"la hora del día {direction} el riesgo ({val:+.3f})")
            elif "FIN_SEMANA" in feat:
                parts.append(f"ser fin de semana {direction} el riesgo ({val:+.3f})")
            elif "NOCHE" in feat:
                parts.append(f"la combinación noche+fin de semana {direction} el riesgo ({val:+.3f})")
        return ("Según el modelo, " + ", ".join(parts) + ".") if parts else ""

    def explain(self, city: str, neighborhood: str, day_of_week: int, hour: int,
                shap_values: Optional[dict] = None) -> str:
        day_label = DAYS_ES[day_of_week] if 0 <= day_of_week <= 6 else "día"
        h_label   = hour_label(hour)
        rag_ctx   = self._rag_context(city, neighborhood, day_of_week, hour)
        shap_txt  = self._shap_narrative(shap_values or {})
        finde_note = (
            " Los fines de semana amplían el riesgo nocturno hasta un 25% respecto a días laborales."
            if day_of_week >= 5 else ""
        )

        if rag_ctx:
            base = (
                f"Contexto de {neighborhood} ({city.capitalize()}):\n{rag_ctx[:600]}\n\n"
                f"Consulta: {day_label} a las {hour}:00h ({h_label}).{finde_note} {shap_txt}"
            ).strip()
        else:
            base = (
                f"La consulta corresponde a {neighborhood} un {day_label} "
                f"a las {hour}:00h ({h_label}).{finde_note} {shap_txt}"
            ).strip()

        if self._llm_available:
            return self._llm_enrich(base, neighborhood, day_label, hour)

        # Fallback: resumen limpio sin el bloque de contexto crudo
        if rag_ctx:
            lines = [l.strip() for l in rag_ctx.split('\n') if l.strip() and not l.startswith('BARRIO:') and not l.startswith('COMUNA:')]
            summary = ' '.join(lines[:4])[:400]
            return (f"{summary} "
                    f"Consulta: {day_label} a las {hour}:00h ({h_label}).{finde_note} {shap_txt}").strip()
        return base

    def _llm_enrich(self, context: str, neighborhood: str, day: str, hour: int) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic()
            prompt = (
                f"Eres un analista de seguridad vial urbana. "
                f"Basándote en este contexto, escribe UNA explicación clara y útil "
                f"(máximo 3 oraciones, en español) para un ciudadano sobre el riesgo vial "
                f"en {neighborhood} un {day} a las {hour}:00h:\n\n{context}"
            )
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=220,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            logger.warning(f"LLM enrich falló: {e}")
            return context
