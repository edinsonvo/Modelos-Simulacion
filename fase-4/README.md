# Fase 4 — Monitoreo

**Proyecto:** Predicción de la variación mensual del IPC en Colombia por ciudad y grupo de gasto
**Curso:** Modelos y Simulación de Sistemas I — Universidad de Antioquia

Esta fase amplía la API de la Fase 3 con mecanismos básicos de monitoreo:
registro de **todas** las solicitudes a `/predict` (incluidas las rechazadas),
indicadores de operación, retroalimentación con valores reales, error rodante,
historial de versiones y una política de reentrenamiento.

## Contenido de la carpeta

```
fase-4/
├── data/
├── model/                         # model_v1 (random_forest) y model_v2 (ridge)
│   ├── latest.json                # versión que usa la API al arrancar
│   ├── model_v1.joblib
│   ├── model_v1_metadata.json
│   ├── model_v2.joblib
│   └── model_v2_metadata.json
├── monitoring/
│   ├── predictions_log.jsonl      # una fila por intento de /predict
│   ├── feedback_log.jsonl         # valores reales reportados
│   └── reporte_monitoreo.md       # reporte generado por monitor.py
├── tests/
│   ├── test_api.py                # pruebas heredadas de la Fase 3
│   └── test_monitoring.py         # pruebas de monitoreo y registro de rechazos
├── features.py
├── train.py                       # run_training()
├── predict.py                     # predict_from_payload()
├── monitoring_utils.py            # registro y métricas (Fase 4)
├── retraining_policy.py           # regla de error de la política de reentrenamiento
├── monitor.py                     # script de monitoreo (genera el reporte)
├── app.py                         # API: /health, /predict, /train, /feedback, /metrics
├── cliente_prueba.py
├── POLITICA_REENTRENAMIENTO.md    # documento de la política completa
├── Dockerfile
├── requirements.txt
└── README.md
```

## Qué se agregó respecto a la Fase 3

| Requisito de la guía | Cómo se cumple |
|---|---|
| Registrar cada solicitud de predicción | Cada llamada a `/predict` deja una sola fila en `monitoring/predictions_log.jsonl` con `fecha_hora`, `estado_http`, `version` y `prediccion`. Las rechazadas por validación (422), ciudad inválida (400) o sin modelo (503) también quedan registradas, con `prediccion` y `version` vacíos. |
| Calcular indicadores de operación | `GET /metrics` y `monitor.py` muestran el total de solicitudes y el porcentaje con código 4xx/5xx. |
| Mantener y seleccionar versiones | El repositorio conserva v1 y v2; `model/latest.json` indica cuál carga la API al iniciar. |
| Definir política de reentrenamiento | `POLITICA_REENTRENAMIENTO.md` + `retraining_policy.py`, evaluados en `/metrics` y en `monitor.py`. |

## Reporte

Sobre un registro vacío se enviaron **dos solicitudes válidas** y **una solicitud incompleta** (le faltaba `ipc_lag1`). La predicción válida usó la versión 2.

**Ejemplo de solicitud**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ciudad":"Medellín","grupo_gasto":"Transporte","ipc_lag1":0.42,"ipc_lag3":0.31,"ipc_lag6":0.55,"mes_sin":0.5,"mes_cos":0.8660254,"tasa_banrep":9.5,"var_trm":1.2,"var_petroleo":-0.5}'
```

**Respuesta obtenida**

```json
{
  "prediction_id": "d50f6a71-9be3-43bb-98a9-140844a5e97c",
  "ciudad": "Medellín",
  "grupo_gasto": "Transporte",
  "ipc_variacion_mensual_predicha": 0.4931,
  "model_version": 2
}
```

**Indicadores obtenidos** (mismo resultado que imprimen `GET /metrics` y `python monitor.py`):

| Indicador | Valor |
|---|---|
| Solicitudes totales | 3 |
| Solicitudes fallidas (4xx/5xx) | 1 |
| Porcentaje de fallos | 33,3 % |

La API procesó dos registros con la versión 2 y rechazó el tercero porque
faltaba `ipc_lag1`; ese rechazo era el resultado esperado de la prueba de
validación, no un fallo del modelo.

### Versiones disponibles y selección

| Versión | Tipo | RMSE (test) | Estado |
|---|---|---|---|
| v1 (`model_v1.joblib`) | random_forest | 0.3781 | versión anterior |
| v2 (`model_v2.joblib`) | ridge | 0.4543 | versión activa en `model/latest.json` |

La API carga la versión indicada en `model/latest.json` al arrancar. Para
volver a la versión 1, edita ese archivo dejando `"version": 1`,
`"model_file": "model_v1.joblib"` y `"metadata_file": "model_v1_metadata.json"`,
y reinicia el servidor. Desde la línea de comandos también se puede forzar una
versión con `python predict.py --input-file ejemplo_input.json --model-version 1`.
El historial completo se consulta en `GET /metrics`.

### Política de reentrenamiento

Cuando se reúnan al menos 15 predicciones con valor real, o cuando pasen 30 días desde el último entrenamiento, el equipo revisa el reporte y la persona responsable del modelo ejecuta `POST /train` para generar una versión candidata. La candidata se activa únicamente si su RMSE sobre el mismo conjunto reservado es menor que el de la versión vigente; de lo contrario se mantiene la versión actual ajustando `model/latest.json`. El detalle de los umbrales está en `POLITICA_REENTRENAMIENTO.md`.

## Endpoints nuevos

### `POST /feedback`

Reporta el valor real de IPC una vez se conoce (por ejemplo, cuando el DANE publica el dato oficial del mes), asociándolo a una predicción previa mediante su `prediction_id`.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": "<id devuelto por /predict>", "actual_value": 0.35}'
```

Respuesta:
```json
{"prediction_id": "...", "predicted_value": 0.4931, "actual_value": 0.35, "error_abs": 0.1431}
```

Error **404** si el `prediction_id` no existe en el log.

### `GET /metrics`

```json
{
  "n_predicciones_totales": 2,
  "n_predicciones_con_feedback": 2,
  "total_solicitudes": 3,
  "solicitudes_fallidas": 1,
  "porcentaje_fallos": 33.3,
  "mensaje_porcentaje": "33,3 %",
  "model_version_actual": 2,
  "versiones_disponibles": ["... historial completo con métricas de entrenamiento ..."],
  "politica_reentrenamiento": {
    "recomendacion": "esperar",
    "razon": "Solo hay 2 predicciones con feedback; se requieren al menos 15 para evaluar la regla de error."
  }
}
```

## Script de monitoreo (`monitor.py`)

Lee los mismos logs que usa la API y genera un reporte en Markdown:

```bash
python monitor.py
# genera monitoring/reporte_monitoreo.md e imprime el reporte en pantalla

python monitor.py --rolling-window 30 --output monitoring/reporte_monitoreo.md
```

El reporte incluye los indicadores de operación, el volumen de predicciones,
el error rodante, el historial de versiones y la recomendación final que
combina la **regla de error** y la **regla de calendario**. El archivo
`monitoring/reporte_monitoreo.md` contiene el resultado de la ejecución
descrita en la sección «Reporte».

## Uso local

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# En otra terminal:
python cliente_prueba.py --base-url http://localhost:8000   # health -> predict -> feedback -> metrics
python cliente_prueba.py --train                             # opcional: genera una versión candidata
python monitor.py                                            # genera el reporte a partir de los logs acumulados
```

## Uso con Docker

```bash
docker build -t ipc-api .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/monitoring:/app/monitoring" \
  ipc-api
```

El volumen sobre `monitoring/` es importante: sin él, los registros de
solicitudes y feedback se pierden al detener el contenedor.

## Pruebas (pytest)

`tests/test_api.py` (heredadas de la Fase 3) y `tests/test_monitoring.py`
(nuevas) cubren, entre otros casos:
- `/health` responde 200.
- Una entrada válida devuelve predicción y versión.
- Una entrada con un campo obligatorio ausente se rechaza con 422 y queda registrada.
- Los rechazos 503 y 400 también dejan una sola fila en el log.
- El total de solicitudes y el porcentaje de fallos salen correctamente, incluido el caso «sin solicitudes».
- `/feedback` calcula el error absoluto y avisa cuando el id no existe.
- La política responde `"esperar"`, `"reentrenar"` o `"sin_accion"` según el feedback acumulado.

```bash
pytest -v
```

**Verificación adicional:** se generaron las dos versiones guardadas (v1 y v2), se levantó la API con `model/latest.json` apuntando a v2, se enviaron dos predicciones válidas y una incompleta, y se ejecutó `monitor.py` sobre ese registro. Los indicadores y el reporte incluidos arriba corresponden a esa ejecución.
