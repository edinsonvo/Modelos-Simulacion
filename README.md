# Fase 3 — API REST

**Proyecto:** Predicción de la variación mensual del IPC en Colombia por ciudad y grupo de gasto
**Curso:** Modelos y Simulación de Sistemas I — Universidad de Antioquia

Esta fase expone el modelo de la Fase 2 mediante una API REST construida
con **FastAPI**, reutilizando `features.py`, `train.py` (ya con
`run_training()` como función) y `predict.py` sin duplicar lógica.

## Contenido de la carpeta

```
fase-3/
├── data/                          # heredado
├── model/                         # modelo versionado (model_v1.joblib, latest.json)
├── tests/
│   └── test_api.py                # pruebas con pytest + TestClient
├── features.py                    # + validación de ciudad/grupo_gasto válidos
├── train.py                       # run_training() reutilizado por /train
├── predict.py                     # predict_from_payload() reutilizado por /predict
├── app.py                         # API FastAPI: /health, /predict, /train
├── cliente_prueba.py              # cliente de prueba manual (requests)
├── ejemplo_input.json
├── Dockerfile
├── requirements.txt
└── README.md
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y si hay un modelo cargado |
| POST | `/predict` | Predicción puntual (recibe features ya calculadas) |
| POST | `/train` | Reentrena el modelo y lo versiona |

Documentación interactiva autogenerada (Swagger UI): `http://localhost:8000/docs`

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "model_version": 1}
```

### `POST /predict`

Cuerpo esperado (todos los campos son obligatorios):

```json
{
  "ciudad": "Medellín",
  "grupo_gasto": "Transporte",
  "ipc_lag1": 0.42,
  "ipc_lag3": 0.31,
  "ipc_lag6": 0.55,
  "mes_sin": 0.5,
  "mes_cos": 0.8660254,
  "tasa_banrep": 9.5,
  "var_trm": 1.2,
  "var_petroleo": -0.5
}
```

Respuesta:

```json
{
  "ciudad": "Medellín",
  "grupo_gasto": "Transporte",
  "ipc_variacion_mensual_predicha": 0.1943,
  "model_version": 1
}
```

Manejo de errores:
- **422** — campo faltante o de tipo incorrecto (validado automáticamente por Pydantic).
- **400** — ciudad o grupo de gasto no reconocidos (validado explícitamente contra las 24 ciudades y 12 grupos de gasto válidos).
- **503** — no hay ningún modelo entrenado todavía (llamar primero a `/train`).

### `POST /train`

Cuerpo opcional (todos los campos tienen valores por defecto):

```json
{
  "data_path": "data/ipc_panel_sintetico.csv",
  "model_type": "random_forest",
  "test_start": "2023-01-01"
}
```

Respuesta: metadatos y métricas del modelo recién entrenado (mismo formato
que `model_v{N}_metadata.json`). Errores: **404** si `data_path` no existe,
**400** si el dataset no cumple el mínimo de observaciones o el corte
train/test deja algún conjunto vacío.

## Por qué `/predict` no recibe "ciudad + fecha" y calcula los rezagos internamente

Calcular `ipc_lag1`, `ipc_lag3`, `ipc_lag6` o la variación de TRM/petróleo
requiere el histórico completo de esa serie ciudad–grupo de gasto. En una
API de predicción puntual sin base de datos propia, esa responsabilidad le
corresponde a quien arma la solicitud (un sistema externo que sí tiene
acceso al histórico), no a este servicio. Esto mantiene la API **sin
estado** entre solicitudes.

## Uso local sin Docker

```bash
pip install -r requirements.txt

# Levantar la API
uvicorn app:app --reload --port 8000

# En otra terminal: entrenar
curl -X POST http://localhost:8000/train

# Predecir
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ciudad":"Medellín","grupo_gasto":"Transporte","ipc_lag1":0.42,"ipc_lag3":0.31,"ipc_lag6":0.55,"mes_sin":0.5,"mes_cos":0.8660254,"tasa_banrep":9.5,"var_trm":1.2,"var_petroleo":-0.5}'

# O usar el cliente de prueba, que ejecuta health -> train -> predict (válido e inválido)
python cliente_prueba.py --base-url http://localhost:8000
```

## Uso con Docker

```bash
docker build -t ipc-api .
docker run --rm -p 8000:8000 -v "$(pwd)/model:/app/model" ipc-api
```

La API queda disponible en `http://localhost:8000`. El volumen sobre
`model/` permite que el modelo entrenado (o reentrenado vía `/train`)
persista fuera del contenedor.

## Pruebas (pytest)

`tests/test_api.py` usa `TestClient` de FastAPI (no requiere levantar el
servidor) y cubre:
- `/health` sin modelo cargado.
- `/predict` sin modelo entrenado → 503.
- `/predict` con campo faltante → 422 (validación de Pydantic).
- `/predict` con ciudad inválida → 400 (validación de negocio).
- `/predict` con entrada válida tras entrenar → 200 y predicción numérica.
- `/train` crea el modelo, actualiza métricas y `/health` refleja el cambio.
- `/train` incrementa versión en ejecuciones sucesivas.
- `/train` con `data_path` inexistente → 404.
- `/train` con cuerpo vacío usa los valores por defecto (random_forest).

```bash
pytest -v
# 9 passed
```

**Verificación adicional (no solo TestClient):** se levantó el servidor
real con `uvicorn` y se corrió `cliente_prueba.py` contra `localhost:8000`,
confirmando el ciclo completo `health → train → predict (válido) → predict
(ciudad inválida, 400) → predict (campo faltante, 422)`.

## Próximos pasos (Fase 4 — Monitoreo)

- Registrar cada predicción (entrada, salida, fecha) en un log o base de datos ligera.
- Endpoint o reporte de error rodante (MAE/RMSE) cuando se conozca el valor real.
- Política simple de reentrenamiento automático ante señales de drift.
