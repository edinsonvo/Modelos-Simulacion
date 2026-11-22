# Reporte de monitoreo — Modelo de predicción del IPC

_Generado: 2026-11-21 20:15:05_

## 1. Indicadores de operación

- Solicitudes totales: **3**
- Solicitudes fallidas (4xx/5xx): **1**
- Porcentaje de fallos: **33,3 %**

## 2. Volumen de predicciones

- Predicciones servidas (total): **2**
- Predicciones con valor real reportado (feedback): **2**

**Predicciones por ciudad:**
- Medellín: 2

**Predicciones por grupo de gasto:**
- Transporte: 2

## 3. Error rodante

Sobre las últimas 2 predicciones con feedback:
- MAE rodante: **0.1831**
- RMSE rodante: **0.1831**

## 4. Historial de versiones del modelo

| Versión | Entrenado | Tipo | RMSE (test) | Observaciones |
|---|---|---|---|---|
| 1 | 2026-08-14 19:58:29 | random_forest | 0.3781 | 60,480 |
| 2 | 2026-11-21 20:14:49 | ridge | 0.4543 | 60,480 |

## 5. Recomendación de reentrenamiento

- **Regla de error:** esperar — Solo hay 2 predicciones con feedback; se requieren al menos 15 para evaluar la regla de error con suficiente evidencia.
- **Regla de calendario:** sin_accion — Han pasado 0 días desde el último entrenamiento (versión 2), dentro del límite de 30 días.

### ✅ Recomendación final: **sin acción requerida**

_Ver `POLITICA_REENTRENAMIENTO.md` para el detalle completo de las reglas y sus umbrales._