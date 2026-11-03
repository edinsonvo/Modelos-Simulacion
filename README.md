# Fase 2 — Scripts y Docker

**Proyecto:** Predicción de la variación mensual del IPC en Colombia por ciudad y grupo de gasto
**Curso:** Modelos y Simulación de Sistemas I — Universidad de Antioquia

Esta fase convierte el modelo construido en la Fase 1 en una aplicación
reutilizable desde línea de comandos y empaquetada en un contenedor Docker.
**No se entrena un modelo nuevo**: se reutiliza exactamente el mismo pipeline
(features + Random Forest) de la Fase 1, ahora envuelto en `train.py` y
`predict.py`.

## Contenido de la carpeta

```
fase-2/
├── data/
│   ├── generar_datos.py          # generador del dataset (heredado de Fase 1)
│   └── ipc_panel_sintetico.csv
├── model/                        # se crea/actualiza al ejecutar train.py
│   ├── model_v1.joblib
│   ├── model_v1_metadata.json
│   └── latest.json               # apunta a la versión más reciente
├── tests/
│   └── test_pipeline.py          # pruebas con pytest
├── features.py                   # ingeniería de variables (compartida)
├── train.py                      # entrena y versiona el modelo
├── predict.py                    # carga el modelo y predice
├── ejemplo_input.json            # ejemplo de entrada para predict.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## ⚠️ Nota sobre los datos (heredada de la Fase 1)

`data/ipc_panel_sintetico.csv` sigue siendo **sintético** (ver
`data/generar_datos.py` para el detalle y las fuentes reales de reemplazo:
DANE vía datos.gov.co, Banco de la República, `yfinance` para petróleo).
El pipeline de `features.py`/`train.py`/`predict.py` no cambia al usar
datos reales: solo se reemplaza el CSV de entrada.

## Uso sin Docker

```bash
pip install -r requirements.txt

# Entrenar (crea model/model_v1.joblib, o v2, v3... en ejecuciones sucesivas)
python train.py --data data/ipc_panel_sintetico.csv

# Predecir con un archivo JSON
python predict.py --input-file ejemplo_input.json

# Predecir pasando el JSON directamente
python predict.py --json '{"ciudad": "Medellín", "grupo_gasto": "Transporte", "ipc_lag1": 0.4, "ipc_lag3": 0.3, "ipc_lag6": 0.5, "mes_sin": 0.5, "mes_cos": 0.87, "tasa_banrep": 9.5, "var_trm": 1.2, "var_petroleo": -0.5}'

# Usar una versión específica del modelo en vez de la más reciente
python predict.py --input-file ejemplo_input.json --model-version 1

# Correr las pruebas
pytest -v
```

### ¿Por qué `predict.py` recibe features ya calculadas y no fecha/ciudad crudos?

Calcular un rezago (`ipc_lag1`, `ipc_lag3`, `ipc_lag6`) o una variación
porcentual (`var_trm`, `var_petroleo`) requiere el histórico completo de esa
serie ciudad–grupo de gasto, no solo el dato del día. Esa responsabilidad
es de quien construye la solicitud (en la Fase 3, la propia API), no de
`predict.py`. Esto mantiene `predict.py` simple, rápido y sin estado.

## Versionado del modelo

Cada ejecución de `train.py` crea `model_v{N}.joblib` con su
`model_v{N}_metadata.json` (métricas, período de entrenamiento, features
usadas) y actualiza `model/latest.json`, que `predict.py` lee por defecto.
Los modelos anteriores nunca se sobrescriben, lo que permite comparar
versiones o hacer rollback simplemente apuntando `--model-version` a una
versión anterior.

## Uso con Docker

```bash
# Construir la imagen
docker build -t ipc-model .

# Entrenar dentro del contenedor, persistiendo el modelo en el host
docker run --rm \
  -v "$(pwd)/model:/app/model" \
  ipc-model train.py --data data/ipc_panel_sintetico.csv

# Predecir dentro del contenedor, usando el modelo ya entrenado
docker run --rm \
  -v "$(pwd)/model:/app/model" \
  ipc-model predict.py --json '{"ciudad": "Medellín", "grupo_gasto": "Transporte", "ipc_lag1": 0.4, "ipc_lag3": 0.3, "ipc_lag6": 0.5, "mes_sin": 0.5, "mes_cos": 0.87, "tasa_banrep": 9.5, "var_trm": 1.2, "var_petroleo": -0.5}'
```

> El montaje de `model/` como volumen (`-v`) es lo que permite que el
> modelo entrenado sobreviva fuera del contenedor y que `predict.py` (en
> otra ejecución del contenedor) pueda encontrarlo.

## Pruebas (pytest)

`tests/test_pipeline.py` cubre:
- `features.py` produce las columnas esperadas y falla ante columnas faltantes.
- `train.py` (vía CLI) crea un modelo versionado con metadatos y métricas válidas.
- Ejecuciones sucesivas de `train.py` incrementan la versión (v1, v2, ...) sin sobrescribir.
- `predict.py` devuelve una predicción numérica válida ante una entrada correcta.
- `predict.py` lanza un error controlado ante campos faltantes.
- El flujo CLI completo de `predict.py` (end-to-end) funciona con un archivo JSON real.
- Solicitar una versión de modelo inexistente falla de forma controlada.

```bash
pytest -v
# 8 passed
```

## Resultados obtenidos (modelo versión 1, Random Forest)

| Métrica | Valor |
|---|---|
| MAE | 0.284 |
| RMSE | 0.378 |
| RMSE baseline ingenuo (persistencia) | 0.663 |

## Próximos pasos (Fase 3)

- Exponer `train.py`/`predict.py` mediante una API REST (`/health`, `/predict`, `/train`) usando FastAPI.
- `predict_from_payload()` en `predict.py` ya está aislada para ser reutilizada directamente por el endpoint `/predict`, sin duplicar lógica.
