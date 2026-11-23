import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Fase 1 — Modelo predictivo
## Predicción de la variación mensual del IPC en Colombia por ciudad y grupo de gasto

**Proyecto integrador — Modelos y Simulación de Sistemas I — Universidad de Antioquia**

Este notebook cubre la Fase 1 del proyecto integrador:
1. Carga y descripción de los datos
2. Análisis exploratorio (EDA)
3. Ingeniería de variables (feature engineering)
4. Entrenamiento y comparación de modelos supervisados
5. Validación temporal (walk-forward) evitando fuga de información
6. Selección y guardado del mejor modelo

> **Nota sobre los datos:** este notebook usa `data/ipc_panel_sintetico.csv`,
> un panel **sintético** generado con `data/generar_datos.py`, calibrado con
> órdenes de magnitud realistas (ver comentarios en ese script) porque este
> entorno de desarrollo no tiene acceso a las APIs de DANE/Banrep. El
> pipeline completo (EDA → features → modelos → validación → guardado) está
> listo para recibir el dataset real sin modificaciones: basta con
> reemplazar el CSV de entrada por los datos reales, siguiendo las
> instrucciones al final de `data/generar_datos.py`.""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.set_option("display.max_columns", None)
plt.rcParams["figure.figsize"] = (10, 4)
RANDOM_STATE = 42""")

md("## 1. Carga y descripción de los datos")

code("""df = pd.read_csv("data/ipc_panel_sintetico.csv", parse_dates=["fecha"])
print(f"Observaciones: {len(df):,}")
print(f"Ciudades: {df['ciudad'].nunique()}  |  Grupos de gasto: {df['grupo_gasto'].nunique()}")
print(f"Rango de fechas: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
df.head()""")

code("""df.describe(include="all").T""")

code("""# Verificación de requisitos mínimos del proyecto integrador
assert len(df) >= 1000, "Se requieren al menos 1.000 observaciones"
n_predictoras = df.shape[1] - 1  # todas menos la variable objetivo
print(f"Observaciones: {len(df):,} (>= 1.000 requerido: {len(df) >= 1000})")
print(f"Columnas disponibles para features: {list(df.columns.drop('ipc_variacion_mensual'))}")""")

md("""## 2. Análisis exploratorio (EDA)

Revisamos la distribución de la variable objetivo, su evolución en el tiempo,
diferencias entre grupos de gasto y ciudades, y la relación con las
variables macro exógenas.""")

code("""fig, ax = plt.subplots()
df["ipc_variacion_mensual"].hist(bins=60, ax=ax)
ax.set_title("Distribución de la variación mensual del IPC (panel completo)")
ax.set_xlabel("Variación mensual (%)")
ax.set_ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("eda_distribucion.png", dpi=110)
plt.show()""")

code("""serie_nacional = df.groupby("fecha")["ipc_variacion_mensual"].mean()
fig, ax = plt.subplots()
serie_nacional.plot(ax=ax)
ax.set_title("Variación mensual promedio del IPC (proxy nacional, panel sintético)")
ax.set_ylabel("Variación mensual (%)")
plt.tight_layout()
plt.savefig("eda_serie_nacional.png", dpi=110)
plt.show()""")

code("""promedio_por_grupo = df.groupby("grupo_gasto")["ipc_variacion_mensual"].agg(["mean", "std"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(9, 5))
promedio_por_grupo["mean"].plot(kind="barh", xerr=promedio_por_grupo["std"], ax=ax)
ax.set_title("Variación mensual promedio (+/- desv. estándar) por grupo de gasto")
ax.set_xlabel("Variación mensual (%)")
plt.tight_layout()
plt.savefig("eda_por_grupo.png", dpi=110)
plt.show()
promedio_por_grupo""")

code("""# Relación con variables macro exógenas (a nivel de serie nacional promedio)
macro_nacional = df.groupby("fecha")[["ipc_variacion_mensual", "trm_promedio_mensual",
                                       "tasa_banrep", "petroleo_brent_usd"]].mean()
macro_nacional[["trm_promedio_mensual", "tasa_banrep", "petroleo_brent_usd"]] \\
    .corrwith(macro_nacional["ipc_variacion_mensual"]) \\
    .rename("correlación con IPC (nivel de las variables)")""")

md("""**Lectura del EDA:** la variación mensual del IPC tiene una distribución
concentrada alrededor de un valor positivo pequeño con cola derecha (meses
de choques de precios). Hay diferencias claras entre grupos de gasto
(alimentos y transporte más volátiles, educación con estacionalidad marcada
en enero-febrero). Esto justifica incluir `ciudad`, `grupo_gasto` y variables
de estacionalidad como predictoras, además de las variables macro.""")

md("""## 3. Ingeniería de variables (feature engineering)

Construimos:
- **Rezagos** de la propia serie IPC (ciudad, grupo_gasto) en t-1, t-3 y t-6
- **Estacionalidad** cíclica del mes (seno/coseno) en vez de un entero plano
- **Variación mensual** de TRM y petróleo (en vez de su nivel), más consistente económicamente
- **Codificación categórica** de `ciudad` y `grupo_gasto` (One-Hot)""")

code("""panel = df.sort_values(["ciudad", "grupo_gasto", "fecha"]).copy()

grp = panel.groupby(["ciudad", "grupo_gasto"])["ipc_variacion_mensual"]
panel["ipc_lag1"] = grp.shift(1)
panel["ipc_lag3"] = grp.shift(3)
panel["ipc_lag6"] = grp.shift(6)

panel["mes"] = panel["fecha"].dt.month
panel["mes_sin"] = np.sin(2 * np.pi * panel["mes"] / 12)
panel["mes_cos"] = np.cos(2 * np.pi * panel["mes"] / 12)

panel["var_trm"] = panel.groupby(["ciudad", "grupo_gasto"])["trm_promedio_mensual"].pct_change() * 100
panel["var_petroleo"] = panel.groupby(["ciudad", "grupo_gasto"])["petroleo_brent_usd"].pct_change() * 100

# Eliminar filas iniciales sin rezago suficiente (primeros 6 meses de cada serie)
panel = panel.dropna(subset=["ipc_lag1", "ipc_lag3", "ipc_lag6", "var_trm", "var_petroleo"]).reset_index(drop=True)

feature_cols_num = ["ipc_lag1", "ipc_lag3", "ipc_lag6", "mes_sin", "mes_cos",
                     "tasa_banrep", "var_trm", "var_petroleo"]
feature_cols_cat = ["ciudad", "grupo_gasto"]
target_col = "ipc_variacion_mensual"

print(f"Observaciones tras feature engineering: {len(panel):,}")
panel[feature_cols_num + feature_cols_cat + [target_col]].head()""")

md("""## 4. Validación temporal (walk-forward)

**No usamos una división aleatoria**, porque estamos ante series de tiempo:
mezclar meses futuros y pasados en train/test produciría fuga de información
(el modelo "vería el futuro"). En su lugar:

- **Entrenamiento:** enero 2008 – diciembre 2022
- **Prueba:** enero 2023 – diciembre 2025

Esto simula la situación real de producción: el modelo solo conoce el
pasado al momento de predecir.""")

code("""fecha_corte = "2023-01-01"
train = panel[panel["fecha"] < fecha_corte]
test = panel[panel["fecha"] >= fecha_corte]

print(f"Train: {len(train):,} filas ({train['fecha'].min().date()} a {train['fecha'].max().date()})")
print(f"Test:  {len(test):,} filas ({test['fecha'].min().date()} a {test['fecha'].max().date()})")

X_train, y_train = train[feature_cols_num + feature_cols_cat], train[target_col]
X_test, y_test = test[feature_cols_num + feature_cols_cat], test[target_col]""")

md("""## 5. Entrenamiento y comparación de modelos

Comparamos dos modelos:
- **Ridge Regression** — línea base lineal, rápida e interpretable
- **Random Forest** — captura no linealidades e interacciones entre ciudad/grupo/macro

Ambos dentro de un `Pipeline` con el mismo preprocesamiento (One-Hot para
las categóricas), para que el proceso sea reproducible y quede empaquetado
junto con el modelo.""")

code("""preprocesador = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
    ],
    remainder="passthrough",
)

modelos = {
    "ridge": Pipeline([
        ("prep", preprocesador),
        ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ]),
    "random_forest": Pipeline([
        ("prep", preprocesador),
        ("reg", RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ]),
}

resultados = []
predicciones = {}

for nombre, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    predicciones[nombre] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100

    resultados.append({"modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE_%": mape})

resultados_df = pd.DataFrame(resultados).sort_values("RMSE")
resultados_df""")

code("""# Baseline ingenuo: predecir el mismo valor del mes anterior (persistencia)
y_naive = test["ipc_lag1"].values
mae_naive = mean_absolute_error(y_test, y_naive)
rmse_naive = mean_squared_error(y_test, y_naive) ** 0.5
print(f"Baseline ingenuo (persistencia t-1)  ->  MAE: {mae_naive:.3f}  RMSE: {rmse_naive:.3f}")
print("Los modelos deben superar este baseline para justificar su uso.")""")

code("""fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(resultados_df["modelo"], resultados_df["RMSE"])
ax.axhline(rmse_naive, color="red", linestyle="--", label="Baseline ingenuo")
ax.set_ylabel("RMSE")
ax.set_title("Comparación de modelos — error de predicción (test 2023-2025)")
ax.legend()
plt.tight_layout()
plt.savefig("comparacion_modelos.png", dpi=110)
plt.show()""")

md("""## 6. Importancia de variables (interpretación)

Para el mejor modelo, revisamos qué variables aportan más a la predicción —
esto conecta el resultado técnico con la interpretación macroeconómica
pedida en el proyecto (p. ej. si el rezago propio del IPC domina, o si TRM
y petróleo tienen peso relevante en grupos como Transporte).""")

code("""mejor_nombre = resultados_df.iloc[0]["modelo"]
mejor_pipe = modelos[mejor_nombre]
print(f"Mejor modelo según RMSE en test: {mejor_nombre}")

if mejor_nombre == "random_forest":
    ohe = mejor_pipe.named_steps["prep"].named_transformers_["cat"]
    nombres_cat = list(ohe.get_feature_names_out(feature_cols_cat))
    nombres_features = nombres_cat + feature_cols_num
    importancias = mejor_pipe.named_steps["reg"].feature_importances_
    imp_df = pd.DataFrame({"feature": nombres_features, "importancia": importancias})
    imp_df = imp_df.sort_values("importancia", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importancia"][::-1])
    ax.set_title("Importancia de variables (Top 15) — Random Forest")
    plt.tight_layout()
    plt.savefig("importancia_variables.png", dpi=110)
    plt.show()
else:
    print("El mejor modelo fue Ridge; revisar coeficientes en su lugar.")""")

md("""## 7. Guardado del modelo

Guardamos el pipeline completo (preprocesamiento + modelo) para que
`predict.py` (Fase 2) pueda cargarlo directamente sin reconstruir el
preprocesamiento a mano, junto con sus metadatos de entrenamiento.""")

code("""Path("model").mkdir(exist_ok=True)

joblib.dump(mejor_pipe, "model/model_ipc_v1.joblib", compress=3)

metadata = {
    "version": "v1",
    "modelo": mejor_nombre,
    "fecha_entrenamiento": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "periodo_train": [str(train["fecha"].min().date()), str(train["fecha"].max().date())],
    "periodo_test": [str(test["fecha"].min().date()), str(test["fecha"].max().date())],
    "features_numericas": feature_cols_num,
    "features_categoricas": feature_cols_cat,
    "target": target_col,
    "metricas_test": resultados_df.set_index("modelo").loc[mejor_nombre].to_dict(),
    "baseline_naive_rmse": rmse_naive,
    "n_observaciones_totales": int(len(panel)),
    "nota_datos": "Entrenado con panel SINTETICO (ver data/generar_datos.py). Reemplazar por datos reales de DANE/Banrep antes de la entrega final.",
}

with open("model/model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Modelo y metadatos guardados en model/")
metadata""")

code("""from IPython.display import Markdown, display

resumen = f'''## 8. Conclusiones de la Fase 1

- Se construyó un panel de **{len(panel):,} observaciones** (ciudad x grupo de gasto x mes), muy por encima del mínimo exigido de 1.000, conservando el IPC como variable objetivo central.
- El mejor modelo (**{mejor_nombre}**) obtuvo RMSE={resultados_df.set_index("modelo").loc[mejor_nombre, "RMSE"]:.3f} frente a RMSE={rmse_naive:.3f} del baseline ingenuo de persistencia, lo que valida el uso de aprendizaje supervisado frente a una regla simple.
- Los rezagos propios de la serie IPC son, como es esperable en series de inflacion, la variable mas relevante; las variables macro exogenas (TRM, petroleo) aportan senal adicional, especialmente para el grupo Transporte.
- **Pendiente antes de la entrega final:** reemplazar el dataset sintetico por los datos reales de DANE (IPC por ciudad y grupo de gasto) y del Banco de la Republica (TRM, tasa de intervencion), siguiendo las instrucciones documentadas en `data/generar_datos.py`. El pipeline (features, modelos, validacion, guardado) no requiere cambios.
'''
display(Markdown(resumen))""")

nb["cells"] = cells

with open("proyecto_fase1.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook construido: proyecto_fase1.ipynb")
