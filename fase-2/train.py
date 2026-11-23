#!/usr/bin/env python3
"""
train.py — Entrena el modelo de predicción de la variación mensual del IPC
y lo guarda de forma versionada en model/.

Uso:
    python train.py --data data/ipc_panel_sintetico.csv
    python train.py --data data/ipc_panel_sintetico.csv --model-type ridge
    python train.py --data data/ipc_panel_sintetico.csv --test-start 2023-01-01

Cada ejecución crea una nueva versión (model/model_v{N}.joblib +
model/model_v{N}_metadata.json) y actualiza model/latest.json para que
predict.py siempre use, por defecto, el modelo más reciente.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import FEATURE_COLS_CAT, FEATURE_COLS_NUM, TARGET_COL, build_features

RANDOM_STATE = 42


def build_pipeline(model_type: str) -> Pipeline:
    preprocesador = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS_CAT)],
        remainder="passthrough",
    )

    if model_type == "ridge":
        regresor = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    elif model_type == "random_forest":
        regresor = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"model_type desconocido: {model_type}")

    return Pipeline([("prep", preprocesador), ("reg", regresor)])


def next_version(model_dir: Path) -> int:
    versiones = [
        int(m.group(1))
        for f in model_dir.glob("model_v*.joblib")
        if (m := re.match(r"model_v(\d+)\.joblib", f.name))
    ]
    return max(versiones, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo de predicción del IPC.")
    parser.add_argument("--data", required=True, help="CSV con el panel de datos crudos.")
    parser.add_argument("--model-dir", default="model", help="Carpeta donde guardar el modelo.")
    parser.add_argument(
        "--model-type", default="random_forest", choices=["ridge", "random_forest"],
        help="Algoritmo a entrenar.",
    )
    parser.add_argument(
        "--test-start", default="2023-01-01",
        help="Fecha (YYYY-MM-DD) a partir de la cual los datos se usan como test temporal.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    print(f"Cargando datos desde {data_path} ...")
    df = pd.read_csv(data_path, parse_dates=["fecha"])

    print("Construyendo variables (features)...")
    panel = build_features(df)

    if len(panel) < 1000:
        raise ValueError(
            f"El dataset resultante tiene solo {len(panel)} observaciones; "
            "se requiere un mínimo de 1.000."
        )

    train = panel[panel["fecha"] < args.test_start]
    test = panel[panel["fecha"] >= args.test_start]

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            "El corte --test-start deja el conjunto de train o de test vacío. "
            "Verifica el rango de fechas de tus datos."
        )

    X_train, y_train = train[FEATURE_COLS_CAT + FEATURE_COLS_NUM], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS_CAT + FEATURE_COLS_NUM], test[TARGET_COL]

    print(f"Train: {len(train):,} filas | Test: {len(test):,} filas")
    print(f"Entrenando modelo: {args.model_type} ...")

    pipe = build_pipeline(args.model_type)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    y_naive = test["ipc_lag1"].values
    rmse_naive = mean_squared_error(y_test, y_naive) ** 0.5

    print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  (baseline ingenuo RMSE={rmse_naive:.4f})")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    version = next_version(model_dir)
    model_filename = f"model_v{version}.joblib"
    metadata_filename = f"model_v{version}_metadata.json"

    joblib.dump(pipe, model_dir / model_filename, compress=3)

    metadata = {
        "version": version,
        "model_type": args.model_type,
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": str(data_path),
        "n_observations": int(len(panel)),
        "train_period": [str(train["fecha"].min().date()), str(train["fecha"].max().date())],
        "test_period": [str(test["fecha"].min().date()), str(test["fecha"].max().date())],
        "features_num": FEATURE_COLS_NUM,
        "features_cat": FEATURE_COLS_CAT,
        "target": TARGET_COL,
        "metrics": {"MAE": mae, "RMSE": rmse, "baseline_naive_RMSE": rmse_naive},
    }
    with open(model_dir / metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    latest = {"version": version, "model_file": model_filename, "metadata_file": metadata_filename}
    with open(model_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado: {model_dir / model_filename}")
    print(f"Marcado como versión más reciente en {model_dir / 'latest.json'}")


if __name__ == "__main__":
    main()
