"""
monitoring_utils.py — Registro de solicitudes, retroalimentación (valores reales)
y cálculo de métricas básicas de monitoreo.

Usado tanto por app.py (endpoints /predict, /feedback, /metrics) como por
monitor.py (script standalone que genera el reporte).

El archivo `monitoring/predictions_log.jsonl` guarda una fila por cada intento
de /predict, incluidas las respuestas rechazadas. Campos mínimos de cada fila:

    fecha_hora, estado_http, version, prediccion

En un error, `prediccion` y `version` quedan vacíos cuando no se usó modelo.
También se conserva `prediction_id` e `input` para las predicciones válidas,
necesarios para asociar un valor real mediante /feedback.
"""

import datetime
import json
import math
import uuid
from pathlib import Path
from typing import Optional

MONITORING_DIR = Path("monitoring")
PREDICTIONS_LOG = MONITORING_DIR / "predictions_log.jsonl"
FEEDBACK_LOG = MONITORING_DIR / "feedback_log.jsonl"


def _ensure_dir():
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _append_prediction_record(
    status_code: int,
    model_version: Optional[int],
    prediction: Optional[float],
    payload: Optional[dict],
    prediction_id: Optional[str],
) -> dict:
    """Escribe una sola fila del registro de solicitudes y la devuelve."""
    _ensure_dir()
    record = {
        "fecha_hora": _now_iso(),
        "estado_http": status_code,
        "version": model_version,
        "prediccion": prediction,
        "prediction_id": prediction_id,
        "input": payload if status_code < 400 else None,
    }
    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def log_prediction(payload: dict, prediction: float, model_version: Optional[int]) -> str:
    """Registra una predicción servida (HTTP 200) y devuelve su id único."""
    prediction_id = str(uuid.uuid4())
    _append_prediction_record(200, model_version, prediction, payload, prediction_id)
    return prediction_id


def log_rejected(status_code: int, payload: Optional[dict] = None) -> dict:
    """Registra una solicitud de /predict rechazada (4xx o 5xx).

    No se usó ninguna versión de modelo para producir la respuesta, por lo
    que `version` y `prediccion` se guardan vacíos.
    """
    return _append_prediction_record(status_code, None, None, payload, None)


def log_feedback(prediction_id: str, actual_value: float) -> dict:
    """Registra el valor real de IPC observado para una predicción previa.

    Lanza ValueError si el prediction_id no existe en el log de predicciones.
    """
    _ensure_dir()
    predicciones = _read_jsonl(PREDICTIONS_LOG)
    match = next(
        (p for p in predicciones if p.get("prediction_id") == prediction_id),
        None,
    )
    if match is None:
        raise ValueError(f"No existe una predicción registrada con id: {prediction_id}")

    prediccion = match["prediccion"]
    error_abs = abs(prediccion - actual_value)
    record = {
        "prediction_id": prediction_id,
        "fecha_hora": _now_iso(),
        "actual_value": actual_value,
        "predicted_value": prediccion,
        "error_abs": error_abs,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_operation_metrics(log_path: Optional[Path] = None) -> dict:
    """Indicadores de operación exigidos en la fase 4.

    Cuenta todas las solicitudes registradas y cuántas respondieron 4xx/5xx.
    Con el registro vacío devuelve total 0 y el mensaje «sin solicitudes».
    """
    registros = _read_jsonl(log_path or PREDICTIONS_LOG)
    total = len(registros)
    fallidas = sum(
        1 for r in registros
        if isinstance(r.get("estado_http"), int) and 400 <= r["estado_http"] <= 599
    )

    if total == 0:
        porcentaje = None
        mensaje = "sin solicitudes"
    else:
        porcentaje = round(fallidas / total * 100, 1)
        mensaje = f"{str(porcentaje).replace('.', ',')} %"

    return {
        "total_solicitudes": total,
        "solicitudes_fallidas": fallidas,
        "porcentaje_fallos": porcentaje,
        "mensaje_porcentaje": mensaje,
    }


def compute_metrics(rolling_window: int = 30) -> dict:
    """Calcula métricas de monitoreo a partir de los logs.

    Incluye los indicadores de operación (total y porcentaje de fallos), el
    volumen de predicciones válidas y el error rodante sobre las predicciones
    que ya tienen valor real reportado.
    """
    registros = _read_jsonl(PREDICTIONS_LOG)
    predicciones = [r for r in registros if r.get("prediccion") is not None]
    feedback = _read_jsonl(FEEDBACK_LOG)
    operacion = compute_operation_metrics()

    por_ciudad, por_grupo = {}, {}
    for p in predicciones:
        entrada = p.get("input") or {}
        ciudad = entrada.get("ciudad", "desconocida")
        grupo = entrada.get("grupo_gasto", "desconocido")
        por_ciudad[ciudad] = por_ciudad.get(ciudad, 0) + 1
        por_grupo[grupo] = por_grupo.get(grupo, 0) + 1

    ultimos = feedback[-rolling_window:]
    if ultimos:
        errores = [f["error_abs"] for f in ultimos]
        mae_rodante = sum(errores) / len(errores)
        rmse_rodante = math.sqrt(sum(e ** 2 for e in errores) / len(errores))
    else:
        mae_rodante = None
        rmse_rodante = None

    return {
        "n_predicciones_totales": len(predicciones),
        "n_predicciones_con_feedback": len(feedback),
        "predicciones_por_ciudad": por_ciudad,
        "predicciones_por_grupo_gasto": por_grupo,
        "rolling_window": rolling_window,
        "mae_rodante": mae_rodante,
        "rmse_rodante": rmse_rodante,
        **operacion,
    }


def list_model_versions(model_dir: str = "model") -> list[dict]:
    """Lista todas las versiones de modelo disponibles con sus métricas."""
    versiones = []
    for meta_path in sorted(Path(model_dir).glob("model_v*_metadata.json")):
        with open(meta_path, encoding="utf-8") as f:
            versiones.append(json.load(f))
    return versiones
