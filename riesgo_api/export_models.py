"""
export_models.py
Ejecutar en el mismo entorno del notebook para exportar
los modelos entrenados al formato que consume la API.

Uso:
    python export_models.py

Requiere que en el namespace existan:
    - xgb_model (o rf_model, lr_model)
    - model_df con UBICACION_KEY y HUBO_INCIDENTE
    - (Opcional) res_post, met_post para el modelo post-COVID
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

M_SMOOTH = 50.0

def build_encoders(model_df_: pd.DataFrame) -> dict:
    """Construye los encoders de ubicación desde el dataframe de modelado."""
    gr = model_df_["HUBO_INCIDENTE"].mean()
    ub = model_df_.groupby("UBICACION_KEY").agg(
        m=("HUBO_INCIDENTE", "mean"),
        n=("HUBO_INCIDENTE", "count"),
    ).reset_index()
    ub["tenc"] = (ub["n"] * ub["m"] + M_SMOOTH * gr) / (ub["n"] + M_SMOOTH)
    ub["log_odds"] = np.log((ub["tenc"] + 0.001) / (1 - ub["tenc"] + 0.001))
    return {
        "target_enc":  ub.set_index("UBICACION_KEY")["tenc"].to_dict(),
        "log_odds":    ub.set_index("UBICACION_KEY")["log_odds"].to_dict(),
        "global_rate": float(gr),
    }


def export_model(model_obj, model_df_: pd.DataFrame, city: str,
                 model_type: str, metrics: dict = None):
    """Serializa un modelo con sus encoders y métricas."""
    encoders = build_encoders(model_df_)
    metadata = {
        "version": "1.0",
        "city": city,
        "model_type": model_type,
        "records": len(model_df_),
        "slots": len(model_df_),
        "neighborhoods": len(encoders["target_enc"]),
        **(metrics or {}),
    }
    bundle = {
        "model":    model_obj,
        "encoders": encoders,
        "metadata": metadata,
    }
    out = MODELS_DIR / f"{city}_{model_type}.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=5)
    print(f"✅ Exportado: {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return out


# ── Ejecutar desde el notebook ─────────────────────────────────────
if __name__ == "__main__":
    # Estas variables deben existir en el entorno del notebook
    # Descomenta y ajusta según corresponda:

    # Exportar XGBoost histórico completo (Medellín)
    # export_model(
    #     model_obj  = res_completo["XGBoost"]["modelo"],
    #     model_df_  = model_df_completo,
    #     city       = "medellin",
    #     model_type = "xgboost",
    #     metrics    = met_completo["XGBoost"],
    # )

    # Exportar XGBoost post-COVID (Medellín)
    # export_model(
    #     model_obj  = res_post["XGBoost"]["modelo"],
    #     model_df_  = model_df_post,
    #     city       = "medellin_postcovid",
    #     model_type = "xgboost",
    #     metrics    = met_post["XGBoost"],
    # )

    # Exportar Random Forest
    # export_model(
    #     model_obj  = res_completo["Random Forest"]["modelo"],
    #     model_df_  = model_df_completo,
    #     city       = "medellin",
    #     model_type = "rf",
    #     metrics    = met_completo["Random Forest"],
    # )

    print("\nPara usar en la API:")
    print("  cp -r models/ riesgo_api/models/")
    print("  cd riesgo_api && docker compose up --build")
