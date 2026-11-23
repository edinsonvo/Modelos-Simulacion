"""
Pruebas de la capa de monitoreo (Fase 4): registro de predicciones,
feedback, métricas y política de reentrenamiento.

Ejecutar con:
    pytest -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402
import monitoring_utils as mon  # noqa: E402
from retraining_policy import MIN_FEEDBACK_PARA_EVALUAR, RMSE_UMBRAL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ipc_panel_sintetico.csv"

VALID_PAYLOAD = {
    "ciudad": "Medellín",
    "grupo_gasto": "Transporte",
    "ipc_lag1": 0.42,
    "ipc_lag3": 0.31,
    "ipc_lag6": 0.55,
    "mes_sin": 0.5,
    "mes_cos": 0.8660254,
    "tasa_banrep": 9.5,
    "var_trm": 1.2,
    "var_petroleo": -0.5,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Cliente de pruebas con model_dir y monitoring_dir aislados por test."""
    model_dir = tmp_path / "model"
    monitoring_dir = tmp_path / "monitoring"

    monkeypatch.setattr(app_module, "MODEL_DIR", str(model_dir))
    monkeypatch.setattr(mon, "MONITORING_DIR", monitoring_dir)
    monkeypatch.setattr(mon, "PREDICTIONS_LOG", monitoring_dir / "predictions_log.jsonl")
    monkeypatch.setattr(mon, "FEEDBACK_LOG", monitoring_dir / "feedback_log.jsonl")

    app_module._model_cache["pipe"] = None
    app_module._model_cache["version"] = None

    with TestClient(app_module.app) as c:
        yield c


def _train(client):
    r = client.post("/train", json={"data_path": str(DATA_PATH), "model_type": "ridge"})
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------------
# Registro de predicciones
# ---------------------------------------------------------------------

def test_predict_returns_prediction_id_and_logs_it(client):
    _train(client)
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "prediction_id" in body and len(body["prediction_id"]) > 0

    metrics = client.get("/metrics").json()
    assert metrics["n_predicciones_totales"] == 1
    assert metrics["predicciones_por_ciudad"] == {"Medellín": 1}


def test_multiple_predictions_are_all_logged(client):
    _train(client)
    for _ in range(5):
        client.post("/predict", json=VALID_PAYLOAD)

    metrics = client.get("/metrics").json()
    assert metrics["n_predicciones_totales"] == 5
    assert metrics["predicciones_por_ciudad"]["Medellín"] == 5


# ---------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------

def test_feedback_valid_prediction_id(client):
    _train(client)
    pred = client.post("/predict", json=VALID_PAYLOAD).json()

    r = client.post("/feedback", json={
        "prediction_id": pred["prediction_id"],
        "actual_value": pred["ipc_variacion_mensual_predicha"] + 0.1,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["error_abs"] == pytest.approx(0.1, abs=1e-6)


def test_feedback_unknown_prediction_id_returns_404(client):
    _train(client)
    r = client.post("/feedback", json={"prediction_id": "id-inexistente", "actual_value": 0.5})
    assert r.status_code == 404


# ---------------------------------------------------------------------
# /metrics — error rodante
# ---------------------------------------------------------------------

def test_metrics_rolling_error_none_before_min_feedback(client):
    _train(client)
    for _ in range(3):  # menos que MIN_FEEDBACK_PARA_EVALUAR
        pred = client.post("/predict", json=VALID_PAYLOAD).json()
        client.post("/feedback", json={
            "prediction_id": pred["prediction_id"],
            "actual_value": pred["ipc_variacion_mensual_predicha"],
        })

    metrics = client.get("/metrics").json()
    assert metrics["politica_reentrenamiento"]["recomendacion"] == "esperar"


def test_metrics_recommends_retrain_on_high_rolling_error(client):
    _train(client)
    # Generar suficiente feedback con error grande y consistente
    n = MIN_FEEDBACK_PARA_EVALUAR + 5
    for _ in range(n):
        pred = client.post("/predict", json=VALID_PAYLOAD).json()
        error_grande = pred["ipc_variacion_mensual_predicha"] + (RMSE_UMBRAL * 3)
        client.post("/feedback", json={
            "prediction_id": pred["prediction_id"],
            "actual_value": error_grande,
        })

    metrics = client.get("/metrics").json()
    assert metrics["politica_reentrenamiento"]["recomendacion"] == "reentrenar"
    assert metrics["rmse_rodante"] > RMSE_UMBRAL


def test_metrics_no_action_on_low_rolling_error(client):
    _train(client)
    n = MIN_FEEDBACK_PARA_EVALUAR + 5
    for _ in range(n):
        pred = client.post("/predict", json=VALID_PAYLOAD).json()
        client.post("/feedback", json={
            "prediction_id": pred["prediction_id"],
            "actual_value": pred["ipc_variacion_mensual_predicha"] + 0.01,  # error mínimo
        })

    metrics = client.get("/metrics").json()
    assert metrics["politica_reentrenamiento"]["recomendacion"] == "sin_accion"


def test_metrics_includes_model_version_history(client):
    _train(client)
    metrics = client.get("/metrics").json()
    assert len(metrics["versiones_disponibles"]) == 1
    assert metrics["versiones_disponibles"][0]["version"] == 1


# ---------------------------------------------------------------------
# Indicadores de operación y registro de rechazos
# ---------------------------------------------------------------------

def test_predict_without_model_logs_503(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503

    registros = mon._read_jsonl(mon.PREDICTIONS_LOG)
    assert len(registros) == 1
    assert registros[0]["estado_http"] == 503
    assert registros[0]["prediccion"] is None
    assert registros[0]["version"] is None


def test_rejected_predictions_are_logged_and_counted_as_failures(client):
    _train(client)
    for _ in range(2):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    incompleto = dict(VALID_PAYLOAD)
    del incompleto["ipc_lag1"]
    resp = client.post("/predict", json=incompleto)
    assert resp.status_code == 422

    metricas = client.get("/metrics").json()
    assert metricas["total_solicitudes"] == 3
    assert metricas["solicitudes_fallidas"] == 1
    assert metricas["porcentaje_fallos"] == pytest.approx(33.3, abs=0.05)
    assert metricas["mensaje_porcentaje"] == "33,3 %"

    registros = mon._read_jsonl(mon.PREDICTIONS_LOG)
    assert len(registros) == 3
    assert registros[-1]["estado_http"] == 422
    assert registros[-1]["prediccion"] is None
    assert registros[-1]["version"] is None


def test_operation_metrics_with_empty_log(client):
    metricas = client.get("/metrics").json()
    assert metricas["total_solicitudes"] == 0
    assert metricas["solicitudes_fallidas"] == 0
    assert metricas["porcentaje_fallos"] is None
    assert metricas["mensaje_porcentaje"] == "sin solicitudes"


def test_report_includes_operation_indicators(client, tmp_path):
    from monitor import generar_reporte

    _train(client)
    for _ in range(2):
        client.post("/predict", json=VALID_PAYLOAD)
    incompleto = dict(VALID_PAYLOAD)
    del incompleto["ipc_lag1"]
    client.post("/predict", json=incompleto)

    reporte = generar_reporte(model_dir=str(tmp_path / "model"))
    assert "Solicitudes totales: **3**" in reporte
    assert "Solicitudes fallidas (4xx/5xx): **1**" in reporte
    assert "Porcentaje de fallos: **33,3 %**" in reporte
