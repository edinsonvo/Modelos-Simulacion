"""
features.py — Ingeniería de variables compartida entre train.py y predict.py

Mantener esta lógica en un único módulo evita que el entrenamiento y la
predicción diverjan (un error clásico: "training/serving skew").
"""

import numpy as np
import pandas as pd

FEATURE_COLS_NUM = [
    "ipc_lag1", "ipc_lag3", "ipc_lag6",
    "mes_sin", "mes_cos",
    "tasa_banrep", "var_trm", "var_petroleo",
]
FEATURE_COLS_CAT = ["ciudad", "grupo_gasto"]
TARGET_COL = "ipc_variacion_mensual"

REQUIRED_RAW_COLUMNS = [
    "fecha", "ciudad", "grupo_gasto", "ipc_variacion_mensual",
    "trm_promedio_mensual", "tasa_banrep", "petroleo_brent_usd",
]

# Campos que debe enviar un cliente (p. ej. la API de la Fase 3) para pedir
# una predicción puntual: ya vienen "pre-calculados" (rezagos, variaciones),
# porque calcular un rezago requiere el histórico completo de esa serie,
# algo que no es responsabilidad del cliente de la API.
PREDICTION_INPUT_FIELDS = FEATURE_COLS_CAT + FEATURE_COLS_NUM


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables predictoras a partir del panel crudo.

    Espera un DataFrame con las columnas de REQUIRED_RAW_COLUMNS y una
    columna `fecha` de tipo datetime. Devuelve un DataFrame ordenado con
    las columnas de FEATURE_COLS_NUM + FEATURE_COLS_CAT + TARGET_COL + fecha,
    habiendo eliminado las filas iniciales sin rezago suficiente.
    """
    faltantes = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en los datos: {sorted(faltantes)}")

    panel = df.sort_values(["ciudad", "grupo_gasto", "fecha"]).copy()

    grp = panel.groupby(["ciudad", "grupo_gasto"])["ipc_variacion_mensual"]
    panel["ipc_lag1"] = grp.shift(1)
    panel["ipc_lag3"] = grp.shift(3)
    panel["ipc_lag6"] = grp.shift(6)

    panel["mes"] = panel["fecha"].dt.month
    panel["mes_sin"] = np.sin(2 * np.pi * panel["mes"] / 12)
    panel["mes_cos"] = np.cos(2 * np.pi * panel["mes"] / 12)

    panel["var_trm"] = panel.groupby(["ciudad", "grupo_gasto"])["trm_promedio_mensual"].pct_change() * 100
    panel["var_petroleo"] = panel.groupby(["ciudad", "grupo_gasto"])["petroleo_brent_usd"].pct_change() * 100

    panel = panel.dropna(
        subset=["ipc_lag1", "ipc_lag3", "ipc_lag6", "var_trm", "var_petroleo"]
    ).reset_index(drop=True)

    columnas_finales = ["fecha"] + FEATURE_COLS_CAT + FEATURE_COLS_NUM + [TARGET_COL]
    return panel[columnas_finales]


def build_single_prediction_row(payload: dict) -> pd.DataFrame:
    """Valida y convierte un payload de predicción (dict) en un DataFrame
    de una fila con exactamente las columnas que el modelo espera.
    """
    faltantes = [c for c in PREDICTION_INPUT_FIELDS if c not in payload]
    if faltantes:
        raise ValueError(f"Faltan campos requeridos en la solicitud: {faltantes}")

    fila = {col: payload[col] for col in PREDICTION_INPUT_FIELDS}
    return pd.DataFrame([fila])
