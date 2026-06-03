"""
model_loader.py
Gestiona la carga, caché y entrenamiento de modelos por ciudad.
Soporta múltiples tipos de modelo por ciudad (xgboost, rf, lr).
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Optional
from threading import Lock

logger = logging.getLogger("riesgovial.loader")

# ──────────────────────────────────────────────
# Rutas de modelos serializados
# (joblib.dump en el notebook → modelos/*.pkl)
# ──────────────────────────────────────────────
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))

# Día-del-año representativo por mes (idéntico al notebook)
DIA_REP = {1:15,2:46,3:74,4:105,5:135,6:166,7:196,8:227,9:258,10:288,11:319,12:349}

# Features exactas en el mismo orden que el entrenamiento
FEATURE_COLS = [
    "DIA_SEMANA","MES","DIA_ANIO_SIN","DIA_ANIO_COS",
    "ES_FIN_SEMANA","ES_LABORAL","HORA_PUNTO_MEDIO",
    "DIA_SEMANA_SIN","DIA_SEMANA_COS","MES_SIN","MES_COS",
    "UBICACION_TARGET_ENC","UBICACION_LOG_ODDS",
    "FRANJA_00-02h","FRANJA_02-04h","FRANJA_04-06h","FRANJA_06-08h",
    "FRANJA_08-10h","FRANJA_10-12h","FRANJA_12-14h","FRANJA_14-16h",
    "FRANJA_16-18h","FRANJA_18-20h","FRANJA_20-22h","FRANJA_22-24h",
    "INTERACCION_FINDE_NOCHE",
]

FRANJA_SLOTS = [
    "00-02h","02-04h","04-06h","06-08h","08-10h","10-12h",
    "12-14h","14-16h","16-18h","18-20h","20-22h","22-24h",
]


def hour_to_franja(hour: int) -> str:
    """Convierte hora entera a etiqueta de franja de 2h."""
    slot = (hour // 2) * 2
    return f"{slot:02d}-{slot+2:02d}h"


# ──────────────────────────────────────────────
# CityModel — wrapper de un modelo entrenado
# ──────────────────────────────────────────────
class CityModel:
    """
    Encapsula un modelo sklearn/xgboost ya entrenado junto con
    los encoders de ubicación necesarios para feature engineering.
    """

    def __init__(self, city: str, model_type: str, model_obj, encoders: dict, metadata: dict):
        self.city = city
        self.model_type = model_type
        self.model_id = f"{city}_{model_type}_v{metadata.get('version','1.0')}"
        self._model = model_obj
        # encoders contiene: {"target_enc": dict, "log_odds": dict, "global_rate": float}
        self._encoders = encoders
        self.encoders = encoders
        self.metadata = metadata
        self.neighborhoods = list(encoders.get("target_enc", {}).keys())

    def _build_features(self, neighborhood: str, day_of_week: int, hour: int, query_date: Optional[date]) -> pd.DataFrame:
        """Construye el vector de features igual que en el notebook."""
        mes = query_date.month if query_date else 6
        dia_rep = DIA_REP[mes]
        franja = hour_to_franja(hour)
        es_finde = int(day_of_week >= 5)

        target_enc = self._encoders["target_enc"].get(
            neighborhood, self._encoders.get("global_rate", 0.5)
        )
        log_odds_val = self._encoders["log_odds"].get(
            neighborhood, np.log((target_enc + 0.001) / (1 - target_enc + 0.001))
        )

        row = {
            "DIA_SEMANA": day_of_week,
            "MES": mes,
            "DIA_ANIO_SIN": np.sin(2 * np.pi * dia_rep / 365),
            "DIA_ANIO_COS": np.cos(2 * np.pi * dia_rep / 365),
            "ES_FIN_SEMANA": es_finde,
            "ES_LABORAL": 1 - es_finde,
            "HORA_PUNTO_MEDIO": (hour // 2) * 2 + 1,
            "DIA_SEMANA_SIN": np.sin(2 * np.pi * day_of_week / 7),
            "DIA_SEMANA_COS": np.cos(2 * np.pi * day_of_week / 7),
            "MES_SIN": np.sin(2 * np.pi * mes / 12),
            "MES_COS": np.cos(2 * np.pi * mes / 12),
            "UBICACION_TARGET_ENC": target_enc,
            "UBICACION_LOG_ODDS": log_odds_val,
            "INTERACCION_FINDE_NOCHE": int(es_finde and hour >= 20),
        }
        # One-hot de franja
        for slot in FRANJA_SLOTS:
            row[f"FRANJA_{slot}"] = int(franja == slot)

        return pd.DataFrame([row])[FEATURE_COLS]

    def predict(self, neighborhood: str, day_of_week: int, hour: int, date: Optional[date] = None) -> float:
        X = self._build_features(neighborhood, day_of_week, hour, date)
        prob = float(self._model.predict_proba(X)[0][1])
        return min(max(prob, 0.0), 1.0)

    def predict_heatmap(self, day_of_week: int, query_date: Optional[date] = None) -> dict:
        """Vectorizado: predice todos los barrios × 12 franjas en una sola llamada."""
        rows = []
        keys = []
        for hour_slot in range(0, 24, 2):
            for nb in self.neighborhoods:
                row = self._build_features(nb, day_of_week, hour_slot, query_date).iloc[0]
                rows.append(row)
                keys.append((nb, hour_slot))

        X = pd.DataFrame(rows)[FEATURE_COLS]
        probs = self._model.predict_proba(X)[:, 1]
        result = {}
        for (nb, hour_slot), prob in zip(keys, probs):
            franja = hour_to_franja(hour_slot)
            result.setdefault(nb, {})[franja] = float(min(max(prob, 0.0), 1.0))
        return result

    def shap_values(self, neighborhood: str, day_of_week: int, hour: int, date: Optional[date] = None) -> dict:
        """Devuelve SHAP values como dict feature→valor."""
        try:
            import shap
            X = self._build_features(neighborhood, day_of_week, hour, date)
            explainer = shap.TreeExplainer(self._model)
            sv = explainer.shap_values(X)
            # sv puede ser array 2D (multiclass) o 1D (binario)
            vals = sv[1][0] if isinstance(sv, list) else sv[0]
            return dict(zip(FEATURE_COLS, [float(v) for v in vals]))
        except Exception as e:
            logger.warning(f"SHAP falló ({e}), usando importancias del modelo")
            importances = getattr(self._model, "feature_importances_", np.zeros(len(FEATURE_COLS)))
            return dict(zip(FEATURE_COLS, [float(v) for v in importances]))


# ──────────────────────────────────────────────
# ModelRegistry — gestiona todos los modelos
# ──────────────────────────────────────────────
class ModelRegistry:
    """
    Carga y cachea modelos desde disco.
    Estructura esperada en MODELS_DIR:
      models/
        medellin_xgboost.pkl      ← {"model": <obj>, "encoders": {...}, "metadata": {...}}
        medellin_rf.pkl
        bogota_xgboost.pkl
        ...
    """

    def __init__(self):
        self._cache: dict[str, CityModel] = {}
        self._lock = Lock()
        self._jobs: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        """Carga todos los .pkl disponibles al arrancar."""
        MODELS_DIR.mkdir(exist_ok=True)
        for pkl_path in MODELS_DIR.glob("*.pkl"):
            parts = pkl_path.stem.split("_", 1)   # ej: "medellin_xgboost"
            if len(parts) != 2:
                continue
            city, model_type = parts
            self._load_from_disk(city, model_type, pkl_path)

        if not self._cache:
            logger.warning("No se encontraron modelos en disco. Cargando modelo DEMO.")
            self._load_demo()

    def _load_from_disk(self, city: str, model_type: str, path: Path):
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            cm = CityModel(
                city=city,
                model_type=model_type,
                model_obj=bundle["model"],
                encoders=bundle["encoders"],
                metadata=bundle.get("metadata", {}),
            )
            key = f"{city}_{model_type}"
            self._cache[key] = cm
            logger.info(f"Modelo cargado: {key} ({len(cm.neighborhoods)} barrios)")
        except Exception as e:
            logger.error(f"Error cargando {path}: {e}")

    def _load_demo(self):
        """
        Modelo DEMO sin archivo pkl — usa reglas heurísticas para
        que la API funcione aunque no haya modelos entrenados en disco.
        """
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        demo_neighborhoods = [
            "La Candelaria","Laureles Estadio","El Poblado","Belén","Robledo",
            "Aranjuez","Manrique","Guayabal","Castilla","San Javier",
            "Doce de Octubre","Villa Hermosa","Buenos Aires","La América",
            "Corregimiento de San Cristóbal",
        ]
        base_rates = [0.92,0.67,0.45,0.58,0.53,0.61,0.64,0.71,0.68,0.62,0.59,0.55,0.50,0.56,0.39]
        target_enc = dict(zip(demo_neighborhoods, base_rates))
        log_odds = {k: float(np.log((v+0.001)/(1-v+0.001))) for k,v in target_enc.items()}

        # Entrenar LR minimalista con datos sintéticos para que predict_proba funcione
        np.random.seed(42)
        n = 1000
        X_demo = np.random.rand(n, len(FEATURE_COLS))
        y_demo = (X_demo[:, 11] > 0.5).astype(int)   # col 11 = UBICACION_TARGET_ENC
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_demo, y_demo)

        cm = CityModel(
            city="medellin",
            model_type="demo",
            model_obj=lr,
            encoders={"target_enc": target_enc, "log_odds": log_odds, "global_rate": 0.58},
            metadata={"version": "demo", "note": "Modelo heurístico — reemplazar con pkl real"},
        )
        self._cache["medellin_demo"] = cm
        self._cache["medellin_xgboost"] = cm   # alias para que /predict funcione
        logger.info("Modelo DEMO cargado para Medellín")

    def get(self, city: str, model_type: Optional[str] = None) -> Optional[CityModel]:
        """Devuelve el mejor modelo disponible para la ciudad."""
        with self._lock:
            if model_type:
                return self._cache.get(f"{city}_{model_type}")
            # Preferencia: xgboost > rf > lr > demo
            for mt in ("xgboost", "rf", "lr", "demo"):
                m = self._cache.get(f"{city}_{mt}")
                if m:
                    return m
        return None

    def available_models(self) -> list:
        return list(self._cache.keys())

    def city_summary(self) -> list:
        seen = set()
        result = []
        for key, cm in self._cache.items():
            if cm.city not in seen:
                seen.add(cm.city)
                result.append({
                    "city": cm.city,
                    "model_id": cm.model_id,
                    "neighborhoods": len(cm.neighborhoods),
                    "metadata": cm.metadata,
                })
        return result

    def job_status(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def train_async(self, city: str, data_path: str, model_type: str,
                    year_from: Optional[int], job_id: str):
        """
        Pipeline completo de entrenamiento ejecutado en background thread.
        Replica el flujo del notebook: limpieza → features → grilla → modelo.
        """
        self._jobs[job_id] = {"status": "running", "step": "carga", "progress": 0}
        try:
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score
            import xgboost as xgb

            # 1. Cargar datos
            self._jobs[job_id].update({"step": "carga", "progress": 5})
            df = pd.read_csv(data_path, low_memory=False)
            logger.info(f"[{job_id}] Datos cargados: {len(df):,} registros")

            # 2. Filtro por año
            if year_from and "AÑO" in df.columns:
                df = df[df["AÑO"] >= year_from]
                logger.info(f"[{job_id}] Filtrado desde {year_from}: {len(df):,} registros")

            # 3. Limpieza básica
            self._jobs[job_id].update({"step": "limpieza", "progress": 15})
            df = df.dropna(subset=["Barrio Planeacion","HORA_INCIDENTE","AÑO"])
            df = df[df["Barrio Planeacion"] != "0"]
            df["HORA_NUM"] = df["HORA_INCIDENTE"].str[:2].astype(int)
            df["FRANJA_2H"] = pd.cut(
                df["HORA_NUM"], bins=list(range(0,25,2)),
                labels=FRANJA_SLOTS, right=False, include_lowest=True
            )
            df["DIA_SEMANA"] = pd.to_datetime(df["FECHA_INCIDENTE"]).dt.dayofweek
            df["MES"] = pd.to_datetime(df["FECHA_INCIDENTE"]).dt.month
            df["UBICACION_KEY"] = df["Comuna Planeacion"].str.strip() + " – " + df["Barrio Planeacion"].str.strip()

            # 4. Grilla + target
            self._jobs[job_id].update({"step": "grilla", "progress": 35})
            agg = df.groupby(["UBICACION_KEY","DIA_SEMANA","FRANJA_2H","MES"], observed=False).agg(
                NUM_INCIDENTES=("LLAVE","count"),
                INCIDENTES_FATALES=("GRAVEDAD_INCIDENTE", lambda x: (x=="MUERTO").sum()),
            ).reset_index()
            ubs = df["UBICACION_KEY"].unique()
            idx = pd.MultiIndex.from_product(
                [ubs, list(range(7)), FRANJA_SLOTS, list(range(1,13))],
                names=["UBICACION_KEY","DIA_SEMANA","FRANJA_2H","MES"]
            )
            grid = pd.DataFrame(index=idx).reset_index()
            mdf = grid.merge(agg, on=["UBICACION_KEY","DIA_SEMANA","FRANJA_2H","MES"], how="left")
            mdf["NUM_INCIDENTES"] = mdf["NUM_INCIDENTES"].fillna(0).astype("int32")
            mdf["HUBO_INCIDENTE"] = (mdf["NUM_INCIDENTES"] > 0).astype("int8")

            # 5. Feature engineering
            self._jobs[job_id].update({"step": "features", "progress": 55})
            m_smooth = 50.0
            gr = mdf["HUBO_INCIDENTE"].mean()
            ub_stats = mdf.groupby("UBICACION_KEY").agg(
                m=("HUBO_INCIDENTE","mean"), n=("HUBO_INCIDENTE","count")
            ).reset_index()
            ub_stats["tenc"] = (ub_stats["n"]*ub_stats["m"] + m_smooth*gr) / (ub_stats["n"] + m_smooth)
            ub_stats["log_odds"] = np.log((ub_stats["tenc"]+0.001)/(1-ub_stats["tenc"]+0.001))
            mdf = mdf.merge(ub_stats[["UBICACION_KEY","tenc","log_odds"]], on="UBICACION_KEY", how="left")
            mdf.rename(columns={"tenc":"UBICACION_TARGET_ENC","log_odds":"UBICACION_LOG_ODDS"}, inplace=True)

            mdf["DIA_ANIO_REP"] = mdf["MES"].map(DIA_REP).astype("int16")
            mdf["DIA_ANIO_SIN"] = np.sin(2*np.pi*mdf["DIA_ANIO_REP"]/365).astype("float32")
            mdf["DIA_ANIO_COS"] = np.cos(2*np.pi*mdf["DIA_ANIO_REP"]/365).astype("float32")
            mdf["ES_FIN_SEMANA"] = mdf["DIA_SEMANA"].isin([5,6]).astype("int8")
            mdf["ES_LABORAL"] = (~mdf["DIA_SEMANA"].isin([5,6])).astype("int8")
            franja_mid = {f: int(f[:2])+1 for f in FRANJA_SLOTS}
            mdf["HORA_PUNTO_MEDIO"] = mdf["FRANJA_2H"].map(franja_mid).astype("int8")
            mdf["DIA_SEMANA_SIN"] = np.sin(2*np.pi*mdf["DIA_SEMANA"]/7).astype("float32")
            mdf["DIA_SEMANA_COS"] = np.cos(2*np.pi*mdf["DIA_SEMANA"]/7).astype("float32")
            mdf["MES_SIN"] = np.sin(2*np.pi*mdf["MES"]/12).astype("float32")
            mdf["MES_COS"] = np.cos(2*np.pi*mdf["MES"]/12).astype("float32")
            mdf["INTERACCION_FINDE_NOCHE"] = ((mdf["ES_FIN_SEMANA"]==1)&(mdf["HORA_PUNTO_MEDIO"]>=20)).astype("int8")
            dummies = pd.get_dummies(mdf["FRANJA_2H"], prefix="FRANJA", dtype="int8")
            for slot in FRANJA_SLOTS:
                col = f"FRANJA_{slot}"
                if col not in dummies.columns:
                    dummies[col] = 0
            mdf = pd.concat([mdf, dummies[[f"FRANJA_{s}" for s in FRANJA_SLOTS]]], axis=1)

            # 6. Entrenamiento
            self._jobs[job_id].update({"step": "entrenamiento", "progress": 70})
            X = mdf[FEATURE_COLS]
            y = mdf["HUBO_INCIDENTE"]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            spw = float((y_tr==0).sum()/(y_tr==1).sum())

            if model_type == "xgboost":
                import xgboost as xgb
                model_obj = xgb.XGBClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.05,
                    scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, tree_method="hist", n_jobs=-1
                )
            elif model_type == "rf":
                from sklearn.ensemble import RandomForestClassifier
                model_obj = RandomForestClassifier(
                    n_estimators=100, max_depth=12, class_weight="balanced",
                    random_state=42, n_jobs=-1
                )
            else:
                from sklearn.linear_model import LogisticRegression
                model_obj = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)

            model_obj.fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, model_obj.predict_proba(X_te)[:,1])
            logger.info(f"[{job_id}] Entrenamiento completado. AUC={auc:.4f}")

            # 7. Guardar
            self._jobs[job_id].update({"step": "guardando", "progress": 90})
            encoders = {
                "target_enc": ub_stats.set_index("UBICACION_KEY")["tenc"].to_dict(),
                "log_odds":   ub_stats.set_index("UBICACION_KEY")["log_odds"].to_dict(),
                "global_rate": float(gr),
            }
            metadata = {"version": "1.0", "auc": round(auc, 4), "city": city,
                        "records": len(df), "slots": len(mdf), "model_type": model_type}
            bundle = {"model": model_obj, "encoders": encoders, "metadata": metadata}
            out_path = MODELS_DIR / f"{city}_{model_type}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(bundle, f)

            # Registrar en caché
            cm = CityModel(city=city, model_type=model_type, model_obj=model_obj,
                           encoders=encoders, metadata=metadata)
            with self._lock:
                self._cache[f"{city}_{model_type}"] = cm

            self._jobs[job_id].update({"status": "done", "step": "completado",
                                        "progress": 100, "auc": round(auc, 4),
                                        "model_id": cm.model_id})
        except Exception as e:
            logger.error(f"[{job_id}] Error en entrenamiento: {e}")
            self._jobs[job_id].update({"status": "error", "error": str(e)})
