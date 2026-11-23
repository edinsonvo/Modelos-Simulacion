"""
retraining_policy.py — Política simple de reentrenamiento del modelo.

Se documenta en detalle (con la justificación de cada umbral) en
POLITICA_REENTRENAMIENTO.md. Este módulo implementa la parte evaluable
mecánicamente, para que tanto la API (/metrics) como monitor.py puedan
señalar si toca reentrenar.

Reglas (ver POLITICA_REENTRENAMIENTO.md para el detalle completo):
1. Umbral de error: si el RMSE rodante (últimas N predicciones con
   feedback) supera RMSE_UMBRAL, se recomienda reentrenar.
2. Datos mínimos: la regla de error solo se evalúa si hay al menos
   MIN_FEEDBACK_PARA_EVALUAR observaciones con valor real conocido;
   antes de eso no hay evidencia suficiente.
3. Calendario: independientemente del error, se recomienda un
   reentrenamiento periódico (mensual) para incorporar los datos más
   recientes del panel.
"""

# Referencia: el modelo original (Fase 1) tuvo RMSE=0.378 en test,
# frente a un baseline ingenuo de RMSE=0.663. Fijamos el umbral de alerta
# a medio camino entre ambos, es decir, un deterioro claro pero no tan
# extremo como para ya ser inútil frente al baseline.
RMSE_BASELINE_INGENUO = 0.663
RMSE_MODELO_ORIGINAL = 0.378
RMSE_UMBRAL = round((RMSE_BASELINE_INGENUO + RMSE_MODELO_ORIGINAL) / 2, 3)  # ~0.520

MIN_FEEDBACK_PARA_EVALUAR = 15
DIAS_ENTRE_REENTRENAMIENTOS_CALENDARIO = 30


def evaluate_retraining_policy(rmse_rodante: float | None, n_feedback: int) -> dict:
    """Evalúa la regla de error de la política y devuelve una recomendación.

    No evalúa aquí la regla de calendario (requeriría conocer la fecha del
    último entrenamiento en producción); esa parte se ilustra en
    monitor.py usando la fecha de model_metadata.json.
    """
    if n_feedback < MIN_FEEDBACK_PARA_EVALUAR:
        return {
            "recomendacion": "esperar",
            "razon": (
                f"Solo hay {n_feedback} predicciones con feedback; se requieren "
                f"al menos {MIN_FEEDBACK_PARA_EVALUAR} para evaluar la regla de error "
                "con suficiente evidencia."
            ),
            "rmse_rodante": rmse_rodante,
            "rmse_umbral": RMSE_UMBRAL,
        }

    if rmse_rodante is not None and rmse_rodante > RMSE_UMBRAL:
        return {
            "recomendacion": "reentrenar",
            "razon": (
                f"El RMSE rodante ({rmse_rodante:.3f}) supera el umbral de alerta "
                f"({RMSE_UMBRAL:.3f}), lo que sugiere una posible pérdida de "
                "desempeño (drift) del modelo en producción."
            ),
            "rmse_rodante": rmse_rodante,
            "rmse_umbral": RMSE_UMBRAL,
        }

    return {
        "recomendacion": "sin_accion",
        "razon": f"El RMSE rodante ({rmse_rodante:.3f}) está dentro del rango esperado.",
        "rmse_rodante": rmse_rodante,
        "rmse_umbral": RMSE_UMBRAL,
    }
