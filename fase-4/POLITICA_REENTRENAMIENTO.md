# Política de reentrenamiento — Modelo de predicción del IPC

**Proyecto:** Predicción de la variación mensual del IPC en Colombia por ciudad y grupo de gasto
**Curso:** Modelos y Simulación de Sistemas I — Universidad de Antioquia

## 1. Objetivo

Definir criterios simples y verificables para decidir **cuándo reentrenar**
el modelo en producción, en vez de dejarlo librado al criterio manual. La
política combina dos reglas independientes: una basada en el **error
observado** y otra basada en un **calendario fijo**. Basta con que una de
las dos se cumpla para recomendar reentrenamiento.

## 2. Regla 1 — Umbral de error (detección de deterioro / drift)

### Métrica de referencia

En la Fase 1, el modelo (Random Forest) obtuvo **RMSE = 0.378** en el
conjunto de prueba temporal (2023-2025), frente a **RMSE = 0.663** del
baseline ingenuo (predecir el valor del mes anterior).

### Umbral de alerta

```
RMSE_UMBRAL = (RMSE_baseline_ingenuo + RMSE_modelo_original) / 2
            = (0.663 + 0.378) / 2
            ≈ 0.520
```

Se eligió el punto medio entre el modelo original y el baseline ingenuo
como umbral: representa un deterioro claro del desempeño original, pero
todavía mejor que no usar modelo alguno. Cruzar este umbral es una señal
razonable de que el modelo dejó de capturar bien la relación entre las
variables macro y el IPC (por ejemplo, ante un choque estructural como una
crisis cambiaria no vista en el entrenamiento).

### Cómo se mide

- Cada predicción servida por `/predict` se registra (`monitoring/predictions_log.jsonl`).
- Cuando se conoce el valor real de esa variación de IPC (por ejemplo, al
  publicarse el dato oficial del DANE del mes correspondiente), se reporta
  vía `POST /feedback` y se calcula su error absoluto.
- Se calcula el **RMSE rodante** sobre las últimas **30** predicciones con
  feedback disponible (ventana configurable en `monitor.py` y
  `retraining_policy.py`).

### Condición mínima de evidencia

La regla de error solo se evalúa si hay al menos **15** predicciones con
feedback acumulado. Con menos observaciones, el RMSE rodante es demasiado
ruidoso para tomar una decisión confiable, y la recomendación es "esperar"
en vez de "reentrenar" o "sin acción".

## 3. Regla 2 — Calendario fijo

Independientemente del error, se recomienda reentrenar si han pasado
**30 días** desde el último entrenamiento registrado (`trained_at` en
`model/model_v{N}_metadata.json`). Esto asegura que el modelo incorpore
periódicamente los meses más recientes del panel IPC, sin depender
exclusivamente de que se detecte deterioro.

*Nota:* 30 días es un valor de referencia razonable dado que el IPC se
publica mensualmente; en producción real se ajustaría al ciclo de
publicación del DANE (aproximadamente a inicios de cada mes).

## 4. Decisión final

```
SI (regla_error == "reentrenar") O (regla_calendario == "reentrenar"):
    recomendación final = "reentrenar"
SINO:
    recomendación final = "sin acción"
```

## 5. Implementación

- `retraining_policy.py` — implementa la regla de error (umbral, mínimo de evidencia).
- `monitor.py` — implementa la regla de calendario y combina ambas reglas en el reporte.
- El endpoint `GET /metrics` de la API expone la evaluación de la regla de error en tiempo real.
- Ejecutar `python monitor.py` genera el reporte completo (`monitoring/reporte_monitoreo.md`) con ambas reglas y la recomendación final.

## 6. Política operativa en dos frases

Cuando se reúnan al menos 15 predicciones con valor real, o cuando pasen 30 días desde el último entrenamiento, el equipo revisa el reporte y la persona responsable del modelo ejecuta `POST /train` para generar una versión candidata. La candidata se activa únicamente si su RMSE sobre el mismo conjunto reservado es menor que el de la versión vigente; de lo contrario se mantiene la versión actual y se corrige `model/latest.json` para apuntar a ella. Para volver a la versión 1 basta editar `model/latest.json` con `"version": 1`, `"model_file": "model_v1.joblib"` y `"metadata_file": "model_v1_metadata.json"`, y reiniciar la API.

## 7. Qué NO cubre esta política (alcance del proyecto)

- No implementa reentrenamiento **automático** (disparar `/train` solo);
  la recomendación queda en el reporte para que un humano decida, lo cual
  es apropiado para el alcance de este proyecto académico.
- No evalúa *data drift* en la distribución de las variables de entrada
  (por ejemplo, si empiezan a llegar solicitudes de ciudades con patrones
  muy distintos a los vistos en entrenamiento), solo el error de salida.
  Sería una extensión natural con más tiempo de desarrollo.
