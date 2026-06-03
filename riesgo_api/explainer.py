"""
explainer.py
Sistema RAG para generar explicaciones en lenguaje natural.
Usa diccionario directo de barrios + LLM (Groq / Anthropic).
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

RAG_DIR    = Path(os.getenv("RAG_DIR", Path(__file__).parent / "rag"))
GROQ_MODEL = "openai/gpt-oss-120b"


def hour_label(h: int) -> str:
    for (a, b), label in HOUR_CONTEXT.items():
        if a <= h <= b:
            return label
    return f"{h}:00h"


class RiskExplainer:
    def __init__(self):
        self._llm_provider = self._detect_llm()
        self._barrio_cache: dict = {}

    def _detect_llm(self) -> Optional[str]:
        if os.getenv("GROQ_API_KEY"):
            logger.info("LLM: Groq activo")
            return "groq"
        if os.getenv("ANTHROPIC_API_KEY"):
            logger.info("LLM: Anthropic activo")
            return "anthropic"
        logger.info("Sin API key LLM — usando RAG local")
        return None

    def _load_barrio_docs(self, city: str) -> dict:
        if city in self._barrio_cache:
            return self._barrio_cache[city]

        lookup = {}
        docs_path = RAG_DIR / "documents" / city / "todos_los_barrios.txt"
        if not docs_path.exists():
            self._barrio_cache[city] = lookup
            return lookup

        try:
            text = docs_path.read_text(encoding="utf-8")
            for block in text.strip().split("\n\n"):
                lines = block.strip().split("\n")
                name, desc = None, []
                for line in lines:
                    if line.startswith("BARRIO:"):
                        name = line[len("BARRIO:"):].strip()
                    elif line.startswith("DESCRIPCIÓN:"):
                        desc.append(line[len("DESCRIPCIÓN:"):].strip())
                if name and desc:
                    lookup[name] = " ".join(desc)
        except Exception as e:
            logger.warning(f"Error cargando lookup de barrios: {e}")

        self._barrio_cache[city] = lookup
        return lookup

    def _rag_context(self, city: str, neighborhood: str) -> str:
        lookup = self._load_barrio_docs(city)
        if neighborhood in lookup:
            return lookup[neighborhood]
        nb_lower = neighborhood.lower()
        for name, text in lookup.items():
            if name.lower() == nb_lower:
                return text
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

    def _build_prompt(self, context: str, neighborhood: str, day: str, hour: int) -> str:
        return (
            f"Eres un analista de seguridad vial urbana de Medellín, Colombia. "
            f"Basándote en el siguiente contexto, escribe UNA explicación clara y útil "
            f"(máximo 3 oraciones, en español) para un ciudadano sobre el riesgo vial "
            f"en {neighborhood} un {day} a las {hour}:00h:\n\n{context}"
        )

    def explain(self, city: str, neighborhood: str, day_of_week: int, hour: int,
                shap_values: Optional[dict] = None) -> str:
        day_label  = DAYS_ES[day_of_week] if 0 <= day_of_week <= 6 else "día"
        h_label    = hour_label(hour)
        rag_ctx    = self._rag_context(city, neighborhood)
        shap_txt   = self._shap_narrative(shap_values or {})
        finde_note = (
            " Los fines de semana amplían el riesgo nocturno hasta un 25% respecto a días laborales."
            if day_of_week >= 5 else ""
        )

        context = (
            f"{rag_ctx}\n\nConsulta: {day_label} a las {hour}:00h ({h_label}).{finde_note} {shap_txt}"
            if rag_ctx else
            f"Barrio: {neighborhood}. Consulta: {day_label} a las {hour}:00h ({h_label}).{finde_note} {shap_txt}"
        ).strip()

        if self._llm_provider == "groq":
            return self._llm_enrich_groq(context, neighborhood, day_label, hour)
        if self._llm_provider == "anthropic":
            return self._llm_enrich_anthropic(context, neighborhood, day_label, hour)

        return f"{rag_ctx or f'Barrio {neighborhood}'}. Consulta: {day_label} a las {hour}:00h ({h_label}).{finde_note} {shap_txt}".strip()

    def _llm_enrich_groq(self, context: str, neighborhood: str, day: str, hour: int) -> str:
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": self._build_prompt(context, neighborhood, day, hour)}],
                max_tokens=400,
                temperature=0.4,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq LLM falló: {e}. Usando fallback Anthropic.")
            return self._llm_enrich_anthropic(context, neighborhood, day, hour)

    def _llm_enrich_anthropic(self, context: str, neighborhood: str, day: str, hour: int) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": self._build_prompt(context, neighborhood, day, hour)}],
            )
            return msg.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic LLM falló: {e}")
            return context
