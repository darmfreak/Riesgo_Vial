# RiesgoVial — Manual de inicio

## Requisitos previos

- **Docker** y **Docker Compose** instalados
- Usuario en el grupo `docker`:
  ```bash
  sudo usermod -aG docker $USER
  # Cerrar sesión y volver a entrar (o ejecutar: newgrp docker)
  ```

---

## Levantar el proyecto

**Paso 1 — Verificar que Docker esté corriendo:**

```bash
docker info
```

Si aparece `dial unix /var/run/docker.sock: no such file or directory`, el daemon está apagado. Iniciarlo:

```bash
sudo systemctl start docker
# o si lo anterior no funciona:
sudo service docker start
```

Verificar que quedó activo:

```bash
sudo systemctl status docker
```

**Paso 2 — Levantar los contenedores:**

```bash
cd ~/Proyectos/Notebooks/riesgo_api
docker compose up -d
```

Primera vez o después de cambios en el código:

```bash
docker compose up --build -d
```

**Accesos:**

| Servicio | URL |
|---|---|
| Frontend | http://localhost |
| API (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

---

## Comandos útiles

```bash
# Ver estado de los contenedores
docker compose ps

# Ver logs en tiempo real
docker compose logs -f api

# Parar los contenedores
docker compose down

# Reiniciar solo nginx (cambios en HTML)
docker compose restart nginx
```

---

## Instalar en otro computador

### Opción A — Copiar el proyecto completo (recomendado)

Desde este computador:

```bash
cd ~/Proyectos/Notebooks
tar -czf riesgovial_completo.tar.gz \
  riesgo_api/ \
  Fatal_Road_Traffic_Normalizado.xlsx \
  modelo_riesgo_enriquecido_covid.ipynb

scp riesgovial_completo.tar.gz usuario@ip-destino:~/
```

En el otro computador:

```bash
tar -xzf riesgovial_completo.tar.gz
cd riesgo_api
docker compose up -d
```

### Opción B — Solo el código (requiere reentrenar)

Si solo se transfiere el código sin el modelo `.pkl`:

```bash
# 1. Instalar dependencias Python
pip install xgboost scikit-learn pandas numpy openpyxl \
            langchain langchain-community faiss-cpu \
            sentence-transformers

# 2. Colocar Fatal_Road_Traffic_Normalizado.xlsx en ~/Proyectos/Notebooks/

# 3. Regenerar el script del notebook
cd ~/Proyectos/Notebooks
python3 -c "
import json
with open('modelo_riesgo_enriquecido_covid.ipynb') as f:
    nb = json.load(f)
cells = [c for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
script = '\n'.join('# Celda ' + str(i) + '\n' + ''.join(c['source']) for i, c in enumerate(cells))
open('run_notebook.py','w').write(script)
"

# 4. Entrenar el modelo (tarda ~10 min)
python3 run_notebook.py

# 5. Copiar el modelo generado
cp models/medellin_xgboost.pkl riesgo_api/models/

# 6. Construir el índice RAG
cd riesgo_api
python rag/build_index.py

# 7. Levantar Docker
docker compose up --build -d
```

---

## Estructura de archivos clave

```
riesgo_api/
├── main.py                  ← API FastAPI (endpoints)
├── model_loader.py          ← Carga del modelo y predicciones
├── explainer.py             ← RAG + SHAP
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── models/
│   └── medellin_xgboost.pkl ← Modelo entrenado (2.4 MB)
├── rag/
│   ├── build_index.py       ← Reconstruir índice FAISS
│   ├── documents/medellin/  ← Documentos de conocimiento
│   └── index/medellin/      ← Índice vectorial FAISS
├── frontend/
│   └── index.html           ← Aplicación web
└── tests/
    └── test_api.py          ← pytest (17 tests)
```

---

## Solución a problemas comunes

| Problema | Solución |
|---|---|
| `permission denied docker.sock` | `newgrp docker` o abrir terminal nueva |
| `docker: command not found` | Instalar Docker Engine |
| Puerto 80 ocupado | Cambiar `"80:80"` → `"8080:80"` en `docker-compose.yml` |
| Página no actualiza | `Ctrl+Shift+R` en el browser |
| Heatmap tarda (primera vez) | Normal — ~4s, luego caché instantáneo |
| Docker daemon apagado | `sudo systemctl start docker` |

---

## Ejecutar tests

```bash
cd riesgo_api
pip install pytest httpx fastapi
pytest tests/ -v
```

Resultado esperado: **17 passed**.