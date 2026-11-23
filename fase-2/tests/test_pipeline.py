"""
Pruebas básicas del pipeline de la Fase 2.

Ejecutar con:
    pytest -v
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import FEATURE_COLS_CAT, FEATURE_COLS_NUM, TARGET_COL, build_features
from predict import load_model, predict_from_payload

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ipc_panel_sintetico.csv"


# ---------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------

def test_build_features_produces_expected_columns():
    df = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
    panel = build_features(df.head(5000))  # subconjunto para que la prueba sea rápida

    esperadas = {"fecha", TARGET_COL, *FEATURE_COLS_CAT, *FEATURE_COLS_NUM}
    assert esperadas.issubset(set(panel.columns))
    assert len(panel) > 0
    # No deben quedar nulos en las columnas de features tras el dropna interno
    assert panel[FEATURE_COLS_NUM].isna().sum().sum() == 0


def test_build_features_raises_on_missing_columns():
    df_incompleto = pd.DataFrame({"fecha": pd.to_datetime(["2020-01-01"]), "ciudad": ["Cali"]})
    with pytest.raises(ValueError):
        build_features(df_incompleto)


# ---------------------------------------------------------------------
# train.py (vía subprocess, para probar la CLI real)
# ---------------------------------------------------------------------

def test_train_cli_creates_versioned_model(tmp_path):
    model_dir = tmp_path / "model"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "train.py"),
            "--data", str(DATA_PATH),
            "--model-dir", str(model_dir),
            "--model-type", "ridge",  # más rápido que random_forest para la prueba
            "--test-start", "2023-01-01",
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    assert (model_dir / "model_v1.joblib").exists()
    assert (model_dir / "model_v1_metadata.json").exists()
    assert (model_dir / "latest.json").exists()

    with open(model_dir / "latest.json", encoding="utf-8") as f:
        latest = json.load(f)
    assert latest["version"] == 1

    with open(model_dir / "model_v1_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["n_observations"] >= 1000
    assert "RMSE" in metadata["metrics"]


def test_train_cli_versions_increment(tmp_path):
    model_dir = tmp_path / "model"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--data", str(DATA_PATH),
        "--model-dir", str(model_dir),
        "--model-type", "ridge",
    ]
    subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)

    assert (model_dir / "model_v1.joblib").exists()
    assert (model_dir / "model_v2.joblib").exists()
    with open(model_dir / "latest.json", encoding="utf-8") as f:
        latest = json.load(f)
    assert latest["version"] == 2


# ---------------------------------------------------------------------
# predict.py
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    subprocess.run(
        [
            sys.executable, str(ROOT / "train.py"),
            "--data", str(DATA_PATH),
            "--model-dir", str(model_dir),
            "--model-type", "ridge",
        ],
        capture_output=True, text=True, cwd=ROOT, check=True,
    )
    return model_dir


def test_predict_valid_input_returns_float(trained_model_dir):
    pipe = load_model(model_dir=str(trained_model_dir))
    payload = {
        "ciudad": "Medellín", "grupo_gasto": "Transporte",
        "ipc_lag1": 0.4, "ipc_lag3": 0.3, "ipc_lag6": 0.5,
        "mes_sin": 0.5, "mes_cos": 0.87,
        "tasa_banrep": 9.5, "var_trm": 1.2, "var_petroleo": -0.5,
    }
    pred = predict_from_payload(pipe, payload)
    assert isinstance(pred, float)


def test_predict_missing_field_raises(trained_model_dir):
    pipe = load_model(model_dir=str(trained_model_dir))
    payload_incompleto = {"ciudad": "Cali", "grupo_gasto": "Salud"}  # faltan campos numéricos
    with pytest.raises(ValueError):
        predict_from_payload(pipe, payload_incompleto)


def test_predict_cli_end_to_end(trained_model_dir):
    input_file = ROOT / "ejemplo_input.json"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "predict.py"),
            "--model-dir", str(trained_model_dir),
            "--input-file", str(input_file),
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    salida = json.loads(result.stdout)
    assert "ipc_variacion_mensual_predicha" in salida
    assert isinstance(salida["ipc_variacion_mensual_predicha"], float)


def test_predict_cli_unknown_model_version_fails(trained_model_dir):
    input_file = ROOT / "ejemplo_input.json"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "predict.py"),
            "--model-dir", str(trained_model_dir),
            "--model-version", "999",
            "--input-file", str(input_file),
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode != 0
