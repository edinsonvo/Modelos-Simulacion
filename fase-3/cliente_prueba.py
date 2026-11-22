#!/usr/bin/env python3
"""
cliente_prueba.py — Cliente de prueba manual para la API REST del proyecto.

Requiere que la API ya esté corriendo, por ejemplo:
    uvicorn app:app --port 8000

Uso:
    python cliente_prueba.py
    python cliente_prueba.py --base-url http://localhost:8000
"""

import argparse
import json

import requests


def main():
    parser = argparse.ArgumentParser(description="Prueba manual de la API de predicción del IPC.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print("1) GET /health")
    r = requests.get(f"{base}/health")
    print(f"   status={r.status_code}  body={r.json()}\n")

    print("2) POST /train  (entrena/reentrena el modelo)")
    r = requests.post(f"{base}/train", json={"model_type": "random_forest"})
    print(f"   status={r.status_code}")
    print(f"   {json.dumps(r.json(), indent=2, ensure_ascii=False)}\n")

    print("3) GET /health  (debe mostrar model_loaded=true)")
    r = requests.get(f"{base}/health")
    print(f"   status={r.status_code}  body={r.json()}\n")

    print("4) POST /predict  (entrada válida)")
    payload = {
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
    r = requests.post(f"{base}/predict", json=payload)
    print(f"   status={r.status_code}")
    print(f"   {json.dumps(r.json(), indent=2, ensure_ascii=False)}\n")

    print("5) POST /predict  (ciudad inválida -> se espera error 400)")
    payload_invalido = dict(payload, ciudad="Ciudad Inexistente")
    r = requests.post(f"{base}/predict", json=payload_invalido)
    print(f"   status={r.status_code}  body={r.json()}\n")

    print("6) POST /predict  (campo faltante -> se espera error 422)")
    payload_incompleto = dict(payload)
    del payload_incompleto["tasa_banrep"]
    r = requests.post(f"{base}/predict", json=payload_incompleto)
    print(f"   status={r.status_code}  body={r.json()}\n")


if __name__ == "__main__":
    main()
