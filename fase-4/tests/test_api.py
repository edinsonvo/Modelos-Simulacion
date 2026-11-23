"""
Pruebas de la API REST (Fase 3).

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
    """Cliente de pruebas con model_dir y monitoring_dir aislados por test
    (no comparte estado con el model/ ni el monitoring/ reales del repo)."""
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


# ---------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------

def test_health_without_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["model_version"] is None


# ---------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------

def test_predict_without_trained_model_returns_503(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 503


def test_predict_with_missing_field_returns_422(client):
    incompleto = dict(VALID_PAYLOAD)
    del incompleto["ipc_lag1"]
    resp = client.post("/predict", json=incompleto)
    assert resp.status_code == 422  # error de validación de Pydantic


def test_predict_with_invalid_city_returns_400(client):
    # primero entrenamos para tener un modelo cargado
    client.post("/train", json={"data_path": str(DATA_PATH), "model_type": "ridge"})
    payload_invalido = dict(VALID_PAYLOAD, ciudad="Ciudad Inexistente")
    resp = client.post("/predict", json=payload_invalido)
    assert resp.status_code == 400
    assert "Ciudad desconocida" in resp.json()["detail"]


def test_predict_after_training_returns_valid_prediction(client):
    client.post("/train", json={"data_path": str(DATA_PATH), "model_type": "ridge"})
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["ipc_variacion_mensual_predicha"], float)
    assert body["model_version"] == 1


# ---------------------------------------------------------------------
# /train
# ---------------------------------------------------------------------

def test_train_creates_model_and_updates_health(client):
    resp = client.post("/train", json={"data_path": str(DATA_PATH), "model_type": "ridge"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 1
    assert body["n_observations"] >= 1000
    assert "RMSE" in body["metrics"]

    health = client.get("/health").json()
    assert health["model_loaded"] is True
    assert health["model_version"] == 1


def test_train_increments_version(client):
    client.post("/train", json={"data_path": str(DATA_PATH), "model_type": "ridge"})
    resp2 = client.post("/train", json={"data_path": str(DATA_PATH), "model_type": "ridge"})
    assert resp2.json()["version"] == 2


def test_train_with_missing_data_file_returns_404(client):
    resp = client.post("/train", json={"data_path": "data/no_existe.csv"})
    assert resp.status_code == 404


def test_train_with_default_body_uses_random_forest(client):
    # Body vacío -> debe usar los defaults (data por defecto, random_forest)
    resp = client.post("/train", json={})
    assert resp.status_code == 200
    assert resp.json()["model_type"] == "random_forest"
