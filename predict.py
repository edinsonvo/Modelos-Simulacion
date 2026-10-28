#!/usr/bin/env python3
"""
predict.py — Carga el modelo entrenado (por defecto, la versión más
reciente) y genera una predicción de la variación mensual del IPC.

Uso:
    # usando un archivo JSON de entrada
    python predict.py --input-file ejemplo_input.json

    # pasando el JSON directamente
    python predict.py --json '{"ciudad": "Medellín", "grupo_gasto": "Transporte", \
"ipc_lag1": 0.4, "ipc_lag3": 0.3, "ipc_lag6": 0.5, "mes_sin": 0.5, "mes_cos": 0.87, \
"tasa_banrep": 9.5, "var_trm": 1.2, "var_petroleo": -0.5}'

    # usando una versión específica del modelo en vez de la más reciente
    python predict.py --input-file ejemplo_input.json --model-version 1

Este script está pensado para reutilizarse tal cual desde la API REST de la
Fase 3 (el endpoint /predict simplemente llamará a `predict_from_payload`).
"""

import argparse
import json
from pathlib import Path

import joblib

from features import build_single_prediction_row


def resolve_model_path(model_dir: Path, model_version: int | None) -> Path:
    if model_version is not None:
        candidate = model_dir / f"model_v{model_version}.joblib"
        if not candidate.exists():
            raise FileNotFoundError(f"No existe la versión solicitada: {candidate}")
        return candidate

    latest_path = model_dir / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No se encontró {latest_path}. Entrena un modelo primero con train.py."
        )
    with open(latest_path, encoding="utf-8") as f:
        latest = json.load(f)
    return model_dir / latest["model_file"]


def load_model(model_dir: str = "model", model_version: int | None = None):
    path = resolve_model_path(Path(model_dir), model_version)
    return joblib.load(path)


def predict_from_payload(pipe, payload: dict) -> float:
    """Valida el payload y devuelve la predicción como float de Python."""
    fila = build_single_prediction_row(payload)
    pred = pipe.predict(fila)[0]
    return float(pred)


def main():
    parser = argparse.ArgumentParser(description="Predice la variación mensual del IPC.")
    parser.add_argument("--model-dir", default="model", help="Carpeta donde está el modelo.")
    parser.add_argument("--model-version", type=int, default=None, help="Versión específica a usar (por defecto: la más reciente).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", help="Ruta a un archivo JSON con los campos de entrada.")
    group.add_argument("--json", dest="json_str", help="JSON de entrada como cadena de texto.")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(args.json_str)

    try:
        pipe = load_model(args.model_dir, args.model_version)
        prediccion = predict_from_payload(pipe, payload)
    except (FileNotFoundError, ValueError) as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        raise SystemExit(1)

    resultado = {
        "ciudad": payload["ciudad"],
        "grupo_gasto": payload["grupo_gasto"],
        "ipc_variacion_mensual_predicha": round(prediccion, 4),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
