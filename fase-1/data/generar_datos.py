"""
Generador de datos SINTÉTICOS para el proyecto:
"Predicción de la variación mensual del IPC en Colombia por ciudad y grupo de gasto"

IMPORTANTE
----------
Este entorno de ejecución (sandbox) no tiene acceso de red a datos.gov.co
ni a la API del Banco de la República, por lo que este script genera un
panel SINTÉTICO calibrado con órdenes de magnitud reales (ver comentarios)
para poder construir, probar y validar todo el pipeline de la Fase 1.

Antes de la entrega final del proyecto, este dataset debe reemplazarse por
datos reales. Al final de este archivo se documenta exactamente cómo
descargarlos (el estudiante sí tiene acceso a internet en su propio equipo).

Estructura del panel: ciudad x grupo_de_gasto x mes
- 24 ciudades (dominios geográficos históricos del IPC-DANE)
- 12 grupos de gasto (divisiones COICOP)
- 216 meses (enero 2008 - diciembre 2025)
Total: 24 x 12 x 216 = 62.208 observaciones
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

CIUDADES = [
    "Bogotá D.C.", "Medellín", "Cali", "Barranquilla", "Bucaramanga",
    "Manizales", "Pereira", "Pasto", "Cúcuta", "Cartagena", "Neiva",
    "Montería", "Villavicencio", "Tunja", "Florencia", "Popayán",
    "Valledupar", "Riohacha", "Santa Marta", "Armenia", "Ibagué",
    "Sincelejo", "Quibdó", "San Andrés",
]

GRUPOS_GASTO = [
    "Alimentos y bebidas no alcohólicas",
    "Bebidas alcohólicas y tabaco",
    "Prendas de vestir y calzado",
    "Alojamiento, agua, electricidad, gas y otros combustibles",
    "Muebles, artículos para el hogar y para la conservación ordinaria del hogar",
    "Salud",
    "Transporte",
    "Información y comunicación",
    "Recreación y cultura",
    "Educación",
    "Restaurantes y hoteles",
    "Bienes y servicios diversos",
]

# Ponderación aproximada (peso relativo dentro de la canasta) — orden de
# magnitud real, no exacta.
PESO_GRUPO = {
    "Alimentos y bebidas no alcohólicas": 0.15,
    "Bebidas alcohólicas y tabaco": 0.02,
    "Prendas de vestir y calzado": 0.03,
    "Alojamiento, agua, electricidad, gas y otros combustibles": 0.30,
    "Muebles, artículos para el hogar y para la conservación ordinaria del hogar": 0.06,
    "Salud": 0.03,
    "Transporte": 0.15,
    "Información y comunicación": 0.04,
    "Recreación y cultura": 0.03,
    "Educación": 0.05,
    "Restaurantes y hoteles": 0.09,
    "Bienes y servicios diversos": 0.05,
}

# Volatilidad relativa por grupo (alimentos y transporte son más volátiles)
VOL_GRUPO = {
    "Alimentos y bebidas no alcohólicas": 1.8,
    "Bebidas alcohólicas y tabaco": 0.8,
    "Prendas de vestir y calzado": 0.9,
    "Alojamiento, agua, electricidad, gas y otros combustibles": 1.1,
    "Muebles, artículos para el hogar y para la conservación ordinaria del hogar": 0.9,
    "Salud": 0.7,
    "Transporte": 1.6,
    "Información y comunicación": 0.6,
    "Recreación y cultura": 0.8,
    "Educación": 0.5,
    "Restaurantes y hoteles": 1.0,
    "Bienes y servicios diversos": 0.9,
}

fechas = pd.date_range("2008-01-01", "2025-12-01", freq="MS")

# ---------------------------------------------------------------------
# 1) Variables macro nacionales (exógenas), una serie por mes
# ---------------------------------------------------------------------
n_meses = len(fechas)

# Tasa de intervención de Banrep: caminata con "escalones" (como ocurre en
# la realidad, el Banco mueve la tasa en reuniones, no continuamente).
tasa_banrep = np.zeros(n_meses)
tasa_actual = 9.5
for i in range(n_meses):
    if rng.random() < 0.12:  # probabilidad de cambio de tasa ese mes
        tasa_actual += rng.choice([-0.75, -0.5, -0.25, 0.25, 0.5, 0.75])
        tasa_actual = float(np.clip(tasa_actual, 1.75, 13.25))
    tasa_banrep[i] = tasa_actual

# TRM: caminata aleatoria con deriva leve y choques (crisis 2014-2016, 2020, 2025)
trm = np.zeros(n_meses)
trm_actual = 2000
for i, f in enumerate(fechas):
    choque = 0
    if f.year in (2014, 2015, 2016):
        choque = rng.normal(15, 10)
    if f.year == 2020 and f.month in (3, 4, 5):
        choque = rng.normal(60, 20)
    trm_actual = trm_actual * (1 + rng.normal(0.002, 0.02)) + choque
    trm_actual = float(np.clip(trm_actual, 1600, 5200))
    trm[i] = trm_actual

# Petróleo Brent (USD/barril): caminata con reversión a la media
petroleo = np.zeros(n_meses)
p_actual = 90
for i, f in enumerate(fechas):
    if f.year == 2020 and f.month in (3, 4):
        p_actual *= 0.5
    p_actual = p_actual + 0.05 * (75 - p_actual) + rng.normal(0, 6)
    p_actual = float(np.clip(p_actual, 20, 140))
    petroleo[i] = p_actual

macro = pd.DataFrame({
    "fecha": fechas,
    "trm_promedio_mensual": trm.round(2),
    "tasa_banrep": tasa_banrep.round(2),
    "petroleo_brent_usd": petroleo.round(2),
})

# ---------------------------------------------------------------------
# 2) Panel ciudad x grupo_gasto x mes de variación mensual del IPC
# ---------------------------------------------------------------------
filas = []
efecto_ciudad = {c: rng.normal(0, 0.05) for c in CIUDADES}

for grupo in GRUPOS_GASTO:
    efecto_grupo_base = rng.normal(0.35, 0.15)  # nivel medio propio del grupo
    vol = VOL_GRUPO[grupo]
    for ciudad in CIUDADES:
        efecto_c = efecto_ciudad[ciudad]
        valor_anterior = 0.0
        for i, f in enumerate(fechas):
            estacional = 0.0
            if f.month == 1:
                estacional += 0.25  # ajustes de tarifas y matrículas en enero
            if grupo == "Educación" and f.month in (1, 2):
                estacional += 0.9  # matrículas
            if grupo == "Restaurantes y hoteles" and f.month == 12:
                estacional += 0.4  # temporada navideña

            # sensibilidad a variables macro nacionales (con rezago simple)
            infl_trm = 0.0
            infl_petroleo = 0.0
            if i > 0:
                var_trm = (macro["trm_promedio_mensual"][i] / macro["trm_promedio_mensual"][i - 1] - 1) * 100
                infl_trm = 0.06 * var_trm if grupo in (
                    "Transporte", "Alimentos y bebidas no alcohólicas",
                    "Muebles, artículos para el hogar y para la conservación ordinaria del hogar",
                ) else 0.02 * var_trm
                var_pet = (macro["petroleo_brent_usd"][i] / macro["petroleo_brent_usd"][i - 1] - 1) * 100
                infl_petroleo = 0.08 * var_pet if grupo == "Transporte" else 0.01 * var_pet

            ruido = rng.normal(0, 0.35 * vol)
            persistencia = 0.15 * valor_anterior  # autocorrelación leve

            valor = (
                efecto_grupo_base + efecto_c + estacional
                + infl_trm + infl_petroleo + persistencia + ruido
            )
            valor = float(np.clip(valor, -3.5, 6.0))
            valor_anterior = valor

            filas.append((f, ciudad, grupo, round(valor, 3)))

panel = pd.DataFrame(filas, columns=["fecha", "ciudad", "grupo_gasto", "ipc_variacion_mensual"])
panel = panel.merge(macro, on="fecha", how="left")
panel = panel.sort_values(["ciudad", "grupo_gasto", "fecha"]).reset_index(drop=True)

out_path = "data/ipc_panel_sintetico.csv"
panel.to_csv(out_path, index=False)
print(f"Dataset sintético generado: {out_path}")
print(f"Filas: {len(panel):,} | Columnas: {list(panel.columns)}")
print(panel.head())

# ---------------------------------------------------------------------
# CÓMO REEMPLAZAR ESTE DATASET POR DATOS REALES (ejecutar en un equipo
# con acceso a internet, no en este sandbox):
#
# 1) IPC por ciudad y grupo de gasto — DANE / datos.gov.co (Socrata API):
#    Buscar en https://www.datos.gov.co el dataset "Índice de Precios al
#    Consumidor (IPC)" del DANE y usar su endpoint de Socrata, por ejemplo:
#      import pandas as pd
#      url = "https://www.datos.gov.co/resource/<ID_DEL_DATASET>.json?$limit=200000"
#      df = pd.read_json(url)
#
# 2) TRM histórica — Banco de la República (series estadísticas, serie
#    "Tasa de cambio representativa del mercado (TRM)"):
#      https://www.banrep.gov.co/es/estadisticas/trm
#
# 3) Tasa de intervención de política monetaria — Banco de la República:
#      https://www.banrep.gov.co/es/estadisticas/tasas-interes-politica-monetaria
#
# 4) Petróleo Brent/WTI — por ejemplo con la librería yfinance:
#      import yfinance as yf
#      brent = yf.download("BZ=F", start="2008-01-01")
#
# Una vez descargados, unir todo en un único panel con las mismas columnas
# que produce este script (fecha, ciudad, grupo_gasto, ipc_variacion_mensual,
# trm_promedio_mensual, tasa_banrep, petroleo_brent_usd) y usarlo como
# reemplazo directo de data/ipc_panel_sintetico.csv — el resto del notebook
# no necesita cambios.
# ---------------------------------------------------------------------
