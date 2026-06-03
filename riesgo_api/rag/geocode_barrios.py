"""
geocode_barrios.py
Geocodifica los barrios del modelo usando Nominatim (OpenStreetMap).
Respeta el límite de 1 req/s. Guarda progreso incremental.
"""

import json
import time
import pickle
import urllib.request
import urllib.parse
from pathlib import Path

MODEL_PATH  = Path(__file__).parent.parent / "models" / "medellin_xgboost.pkl"
OUTPUT_PATH = Path(__file__).parent / "documents" / "medellin" / "coordinates.json"

# Bounding box de Medellín (lat_min, lat_max, lon_min, lon_max)
MED_BBOX = (6.10, 6.45, -75.75, -75.40)

UNDEFINED = {"0", "sin informacion", "sin información", "no definido", "nd", "n/a", ""}

def in_medellin(lat, lon):
    return MED_BBOX[0] <= lat <= MED_BBOX[1] and MED_BBOX[2] <= lon <= MED_BBOX[3]

def nominatim_search(query: str):
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "co",
    })
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "RiesgoVial/1.0 geocoder"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            results = json.loads(r.read())
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                return lat, lon
    except Exception as e:
        print(f"  Error: {e}")
    return None, None

def geocode_barrio(name: str):
    queries = [
        f"barrio {name}, Medellín, Antioquia, Colombia",
        f"{name}, Medellín, Colombia",
        f"{name}, Antioquia, Colombia",
    ]
    for q in queries:
        lat, lon = nominatim_search(q)
        time.sleep(1.1)  # respetar rate limit Nominatim
        if lat and lon and in_medellin(lat, lon):
            return round(lat, 4), round(lon, 4)
    return None, None

def main():
    # Cargar barrios del modelo
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    all_barrios = [
        nb for nb in bundle["encoders"]["target_enc"].keys()
        if nb and nb.strip().lower() not in UNDEFINED
    ]
    print(f"Total barrios en modelo: {len(all_barrios)}")

    # Cargar coordenadas existentes
    existing = {}
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    print(f"Ya geocodificados: {len(existing)}")

    pending = [nb for nb in all_barrios if nb not in existing]
    print(f"Por geocodificar: {len(pending)}\n")

    found = 0
    not_found = []

    for i, barrio in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {barrio} ... ", end="", flush=True)
        lat, lon = geocode_barrio(barrio)
        if lat and lon:
            existing[barrio] = [lat, lon]
            found += 1
            print(f"✓ ({lat}, {lon})")
        else:
            not_found.append(barrio)
            print("✗ no encontrado")

        # Guardar progreso cada 10 barrios
        if i % 10 == 0:
            OUTPUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → Progreso guardado ({len(existing)} coordenadas)")

    # Guardar final
    OUTPUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"Encontrados: {found}/{len(pending)}")
    print(f"Total en archivo: {len(existing)}")
    if not_found:
        print(f"No encontrados ({len(not_found)}): {', '.join(not_found[:20])}")

if __name__ == "__main__":
    main()
