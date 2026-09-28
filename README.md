# Fase 1 — Modelo predictivo

**Proyecto:** Predicción de la variación mensual del IPC en Colombia por ciudad y grupo de gasto
**Curso:** Modelos y Simulación de Sistemas I — Universidad de Antioquia

## Contenido de la carpeta

```
fase-1/
├── data/
│   ├── generar_datos.py          # genera el dataset de trabajo
│   └── ipc_panel_sintetico.csv   # dataset (panel ciudad x grupo de gasto x mes)
├── model/
│   ├── model_ipc_v1.joblib       # pipeline entrenado (preprocesamiento + modelo)
│   └── model_metadata.json       # métricas, features y periodo de entrenamiento
├── proyecto_fase1.ipynb          # notebook ejecutable de la Fase 1 (EDA -> modelo)
└── README.md
```

## ⚠️ Nota importante sobre los datos

El dataset `data/ipc_panel_sintetico.csv` es **sintético**. Se generó con
`data/generar_datos.py` porque el entorno donde se construyó este proyecto
no tenía acceso de red a las APIs del DANE (datos.gov.co) ni del Banco de la
República. Está calibrado con órdenes de magnitud realistas (volatilidad por
grupo de gasto, estacionalidad de enero/matrículas, choques cambiarios de
2014-2016 y 2020, escalones de la tasa de Banrep, etc.) para poder construir
y validar todo el pipeline de extremo a extremo.

**Antes de la entrega final del proyecto** se debe reemplazar este CSV por
datos reales. Al final de `data/generar_datos.py` están las instrucciones y
fuentes exactas:

- **IPC por ciudad y grupo de gasto:** DANE, vía API Socrata de datos.gov.co
- **TRM histórica:** Banco de la República — https://www.banrep.gov.co/es/estadisticas/trm
- **Tasa de intervención:** Banco de la República — https://www.banrep.gov.co/es/estadisticas/tasas-interes-politica-monetaria
- **Petróleo Brent/WTI:** por ejemplo con `yfinance` (`yf.download("BZ=F")`)

El notebook y el pipeline de modelado **no requieren ningún cambio**: basta
con producir un CSV con las mismas columnas
(`fecha, ciudad, grupo_gasto, ipc_variacion_mensual, trm_promedio_mensual, tasa_banrep, petroleo_brent_usd`)
y reemplazar el archivo de entrada.

## Cómo reproducir

```bash
pip install pandas numpy scikit-learn matplotlib joblib jupyter --break-system-packages

# (opcional) regenerar el dataset sintético
python3 data/generar_datos.py

# ejecutar el notebook de punta a punta
jupyter nbconvert --to notebook --execute --inplace proyecto_fase1.ipynb
```

## Resumen de la Fase 1

1. **Datos:** panel de 60.480 observaciones (ciudad x grupo de gasto x mes,
   2008-2025, tras generar rezagos), muy por encima del mínimo de 1.000
   exigido, con 8 variables predictoras numéricas + categóricas (ciudad,
   grupo de gasto).
2. **EDA:** distribución de la variación mensual del IPC, evolución en el
   tiempo, diferencias por grupo de gasto y correlación con TRM/tasa
   Banrep/petróleo.
3. **Feature engineering:** rezagos propios de la serie (t-1, t-3, t-6),
   estacionalidad cíclica del mes, variación de TRM y petróleo, codificación
   One-Hot de ciudad y grupo de gasto.
4. **Modelos comparados:** Ridge Regression (línea base) vs. Random Forest.
5. **Validación temporal (walk-forward):** entrenamiento 2008-2022, prueba
   2023-2025 — sin mezclar pasado y futuro, evitando fuga de información.
6. **Resultado:** el mejor modelo (Random Forest) obtuvo **RMSE ≈ 0,38**
   frente a **RMSE ≈ 0,66** del baseline ingenuo de persistencia (predecir
   el valor del mes anterior), validando el uso de aprendizaje supervisado.
7. **Modelo guardado:** pipeline completo (preprocesamiento + modelo) en
   `model/model_ipc_v1.joblib`, listo para ser cargado por `predict.py` en
   la Fase 2.

## Próximos pasos (Fase 2)

- Envolver el entrenamiento y la predicción en `train.py` / `predict.py`
  reutilizables desde línea de comandos.
- Crear el `Dockerfile` para contenerizar el pipeline.
- Reemplazar el dataset sintético por los datos reales de DANE/Banrep.
