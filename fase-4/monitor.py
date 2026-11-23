#!/usr/bin/env python3
"""
monitor.py — Script de monitoreo (Fase 4).

Lee los logs de predicciones/feedback (monitoring/predictions_log.jsonl y
monitoring/feedback_log.jsonl) y las versiones de modelo disponibles en
model/, y genera un reporte en Markdown con:
- Indicadores de operación: total de solicitudes y porcentaje de fallos
- Volumen de predicciones servidas (total y por ciudad/grupo de gasto)
- Error rodante (MAE/RMSE) sobre las predicciones con feedback
- Historial de versiones del modelo y sus métricas de entrenamiento
- Recomendación de la política de reentrenamiento (regla de error + calendario)

Uso:
    python monitor.py
    python monitor.py --output monitoring/reporte_monitoreo.md
    python monitor.py --model-dir model --rolling-window 30

Pensado para ejecutarse periódicamente (p. ej. un cron semanal) sobre los
logs que la API (app.py) va acumulando en producción.
"""

import argparse
import datetime
from pathlib import Path

import monitoring_utils as mon
from retraining_policy import DIAS_ENTRE_REENTRENAMIENTOS_CALENDARIO, evaluate_retraining_policy


def evaluate_calendar_rule(versiones: list[dict]) -> dict:
    if not versiones:
        return {
            "recomendacion": "reentrenar",
            "razon": "No hay ninguna versión de modelo entrenada todavía.",
        }

    ultima = versiones[-1]
    fecha_entrenamiento = datetime.datetime.strptime(ultima["trained_at"], "%Y-%m-%d %H:%M:%S")
    dias_transcurridos = (datetime.datetime.now() - fecha_entrenamiento).days

    if dias_transcurridos >= DIAS_ENTRE_REENTRENAMIENTOS_CALENDARIO:
        return {
            "recomendacion": "reentrenar",
            "razon": (
                f"Han pasado {dias_transcurridos} días desde el último entrenamiento "
                f"(versión {ultima['version']}, {ultima['trained_at']}), superando el "
                f"límite de {DIAS_ENTRE_REENTRENAMIENTOS_CALENDARIO} días de la política de calendario."
            ),
        }
    return {
        "recomendacion": "sin_accion",
        "razon": (
            f"Han pasado {dias_transcurridos} días desde el último entrenamiento "
            f"(versión {ultima['version']}), dentro del límite de "
            f"{DIAS_ENTRE_REENTRENAMIENTOS_CALENDARIO} días."
        ),
    }


def generar_reporte(model_dir: str = "model", rolling_window: int = 30) -> str:
    metricas = mon.compute_metrics(rolling_window=rolling_window)
    versiones = mon.list_model_versions(model_dir)

    regla_error = evaluate_retraining_policy(metricas["rmse_rodante"], metricas["n_predicciones_con_feedback"])
    regla_calendario = evaluate_calendar_rule(versiones)

    recomendacion_final = "reentrenar" if "reentrenar" in (
        regla_error["recomendacion"], regla_calendario["recomendacion"]
    ) else "sin_accion"

    lineas = []
    lineas.append("# Reporte de monitoreo — Modelo de predicción del IPC")
    lineas.append("")
    lineas.append(f"_Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lineas.append("")

    lineas.append("## 1. Indicadores de operación")
    lineas.append("")
    lineas.append(f"- Solicitudes totales: **{metricas['total_solicitudes']}**")
    lineas.append(f"- Solicitudes fallidas (4xx/5xx): **{metricas['solicitudes_fallidas']}**")
    lineas.append(f"- Porcentaje de fallos: **{metricas['mensaje_porcentaje']}**")
    lineas.append("")

    lineas.append("## 2. Volumen de predicciones")
    lineas.append("")
    lineas.append(f"- Predicciones servidas (total): **{metricas['n_predicciones_totales']}**")
    lineas.append(f"- Predicciones con valor real reportado (feedback): **{metricas['n_predicciones_con_feedback']}**")
    lineas.append("")
    if metricas["predicciones_por_ciudad"]:
        lineas.append("**Predicciones por ciudad:**")
        for ciudad, n in sorted(metricas["predicciones_por_ciudad"].items(), key=lambda x: -x[1]):
            lineas.append(f"- {ciudad}: {n}")
        lineas.append("")
    if metricas["predicciones_por_grupo_gasto"]:
        lineas.append("**Predicciones por grupo de gasto:**")
        for grupo, n in sorted(metricas["predicciones_por_grupo_gasto"].items(), key=lambda x: -x[1]):
            lineas.append(f"- {grupo}: {n}")
        lineas.append("")

    lineas.append("## 3. Error rodante")
    lineas.append("")
    if metricas["mae_rodante"] is not None:
        lineas.append(f"Sobre las últimas {min(rolling_window, metricas['n_predicciones_con_feedback'])} predicciones con feedback:")
        lineas.append(f"- MAE rodante: **{metricas['mae_rodante']:.4f}**")
        lineas.append(f"- RMSE rodante: **{metricas['rmse_rodante']:.4f}**")
    else:
        lineas.append("Todavía no hay suficiente feedback (valores reales reportados) para calcular el error rodante.")
    lineas.append("")

    lineas.append("## 4. Historial de versiones del modelo")
    lineas.append("")
    if versiones:
        lineas.append("| Versión | Entrenado | Tipo | RMSE (test) | Observaciones |")
        lineas.append("|---|---|---|---|---|")
        for v in versiones:
            lineas.append(
                f"| {v['version']} | {v['trained_at']} | {v['model_type']} "
                f"| {v['metrics']['RMSE']:.4f} | {v['n_observations']:,} |"
            )
    else:
        lineas.append("No hay versiones de modelo registradas todavía.")
    lineas.append("")

    lineas.append("## 5. Recomendación de reentrenamiento")
    lineas.append("")
    lineas.append(f"- **Regla de error:** {regla_error['recomendacion']} — {regla_error['razon']}")
    lineas.append(f"- **Regla de calendario:** {regla_calendario['recomendacion']} — {regla_calendario['razon']}")
    lineas.append("")
    if recomendacion_final == "reentrenar":
        lineas.append("### ⚠️ Recomendación final: **reentrenar el modelo**")
    else:
        lineas.append("### ✅ Recomendación final: **sin acción requerida**")
    lineas.append("")
    lineas.append("_Ver `POLITICA_REENTRENAMIENTO.md` para el detalle completo de las reglas y sus umbrales._")

    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description="Genera el reporte de monitoreo del modelo de IPC.")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--rolling-window", type=int, default=30)
    parser.add_argument("--output", default="monitoring/reporte_monitoreo.md")
    args = parser.parse_args()

    reporte = generar_reporte(model_dir=args.model_dir, rolling_window=args.rolling_window)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(reporte, encoding="utf-8")

    print(reporte)
    print(f"\n[Reporte guardado en {output_path}]")


if __name__ == "__main__":
    main()
