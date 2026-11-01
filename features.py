
"""
features.py — Ingeniería de variables compartida entre train.py y predict.py

Mantener esta lógica en un único módulo evita que el entrenamiento y la
predicción diverjan (un error clásico: "training/serving skew").
"""

from typing import List
import numpy as np
import pandas as pd

FEATURE_COLS_NUM: List[str] = [
    "ipc_lag1", "ipc_lag3", "ipc_lag6",
    "mes_sin", "mes_cos",
    "tasa_banrep", "var_trm", "var_petroleo",
]
FEATURE_COLS_CAT: List[str] = ["ciudad", "grupo_gasto"]
TARGET_COL: str = "ipc_variacion_mensual"

REQUIRED_RAW_COLUMNS: List[str] = [
    "fecha", "ciudad", "grupo_gasto", "ipc_variacion_mensual",
    "trm_promedio_mensual", "tasa_banrep", "petroleo_brent_usd",
]

PREDICTION_INPUT_FIELDS: List[str] = FEATURE_COLS_CAT + FEATURE_COLS_NUM


def _compute_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula los rezagos históricos del IPC agrupados por serie."""
    grp = df.groupby(["ciudad", "grupo_gasto"])[TARGET_COL]
    df["ipc_lag1"] = grp.shift(1)
    df["ipc_lag3"] = grp.shift(3)
    df["ipc_lag6"] = grp.shift(6)
    return df


def _compute_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica la estacionalidad del mes mediante funciones seno y coseno."""
    mes = df["fecha"].dt.month
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    return df


def _compute_pct_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula variaciones porcentuales mensuales de variables macroeconómicas."""
    grp = df.groupby(["ciudad", "grupo_gasto"])
    df["var_trm"] = grp["trm_promedio_mensual"].pct_change() * 100
    df["var_petroleo"] = grp["petroleo_brent_usd"].pct_change() * 100
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las variables predictoras a partir del panel crudo."""
    faltantes = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en los datos: {sorted(faltantes)}")

    panel = df.sort_values(["ciudad", "grupo_gasto", "fecha"]).copy()

    # Transformaciones modulares
    panel = _compute_lags(panel)
    panel = _compute_cyclical_features(panel)
    panel = _compute_pct_changes(panel)

    cols_rezagos = ["ipc_lag1", "ipc_lag3", "ipc_lag6", "var_trm", "var_petroleo"]
    panel = panel.dropna(subset=cols_rezagos).reset_index(drop=True)

    columnas_finales = ["fecha"] + FEATURE_COLS_CAT + FEATURE_COLS_NUM + [TARGET_COL]
    return panel[columnas_finales]


def build_single_prediction_row(payload: dict) -> pd.DataFrame:
    """Valida y convierte un payload de predicción (dict) en un DataFrame."""
    faltantes = [c for c in PREDICTION_INPUT_FIELDS if c not in payload]
    if faltantes:
        raise ValueError(f"Faltan campos requeridos en la solicitud: {faltantes}")

    fila = {col: payload[col] for col in PREDICTION_INPUT_FIELDS}
    return pd.DataFrame([fila])