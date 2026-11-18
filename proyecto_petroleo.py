# ============================================================
# PROYECTO INTEGRADOR: SECTOR PETROLERO COLOMBIANO
# Modelos: VAR, Cadenas de Markov y Monte Carlo
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. CARGAR Y LIMPIAR LAS SERIES
# ============================================================
# Usamos rutas relativas para que cualquiera pueda ejecutarlo
RUTA_BASE = "../base de datos" 

# --- 1.1 EXPORTACIONES ---
ruta_exportaciones = RUTA_BASE + "/Exportaciones_petroleo_crudo_ene2015_sep2026.xlsx"
df_base = pd.read_excel(ruta_exportaciones, sheet_name="Base completa", header=0)
df_base = df_base[["Periodo", "Valor FOB (millones USD)"]].copy()
df_base["Valor FOB (millones USD)"] = pd.to_numeric(df_base["Valor FOB (millones USD)"], errors="coerce")
df_export = df_base.groupby("Periodo", as_index=False)["Valor FOB (millones USD)"].sum()
df_export.columns = ["Periodo", "Exportaciones_MillonesUSD"]
df_export["Periodo"] = df_export["Periodo"].astype(str).str.strip()

# --- 1.2 PRECIO BRENT ---
ruta_precios = RUTA_BASE + "/Precios al contado referencia bren usd por barril 2015_2026.xlsx"
df_precios = pd.read_excel(ruta_precios, sheet_name="Hoja1", header=0)
df_precios = df_precios.melt(id_vars=["Year"], var_name="Mes", value_name="Precio_Brent_USD")
df_precios = df_precios.dropna(subset=["Precio_Brent_USD"])
df_precios.columns = ["Año", "Mes", "Precio_Brent_USD"]
meses_map = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06",
             "Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
df_precios["Mes_Num"] = df_precios["Mes"].map(meses_map)
df_precios["Periodo"] = df_precios["Año"].astype(str) + "-" + df_precios["Mes_Num"]

# --- 1.3 PRODUCCIÓN ---
ruta_produccion = RUTA_BASE + "/Producción_Fiscalizada_Crudo_2015-2026.xlsx"
df_prod = pd.read_excel(ruta_produccion, sheet_name="Hoja1", header=0)
meses_es = {"enero":"01","febrero":"02","marzo":"03","abril":"04","mayo":"05","junio":"06",
            "julio":"07","agosto":"08","septiembre":"09","octubre":"10","noviembre":"11","diciembre":"12",
            "Enero":"01","Febrero":"02","Marzo":"03","Abril":"04","Mayo":"05","Junio":"06",
            "Julio":"07","Agosto":"08","Septiembre":"09","Octubre":"10","Noviembre":"11","Diciembre":"12"}
df_prod["Mes_Num"] = df_prod["Mes"].map(meses_es)
df_prod["Periodo"] = df_prod["Año"].astype(str) + "-" + df_prod["Mes_Num"]
df_prod = df_prod[["Periodo", "BARRILES POR DIA CALENDARIO - BPDC"]].copy()
df_prod.columns = ["Periodo", "Produccion_BPDC"]

# --- 1.4 UNIÓN ---
df_final = df_export.merge(df_precios[["Periodo","Precio_Brent_USD"]], on="Periodo", how="inner")
df_final = df_final.merge(df_prod, on="Periodo", how="inner")
df_final = df_final.sort_values(by="Periodo").reset_index(drop=True)
df_final["Fecha"] = pd.to_datetime(df_final["Periodo"] + "-01")
df_final = df_final.set_index("Fecha")

# ============================================================
# 2. PRUEBAS ADF Y TEST DE CHOW
# ============================================================
df_niveles = df_final[["Exportaciones_MillonesUSD", "Precio_Brent_USD", "Produccion_BPDC"]].copy()
df_var = df_niveles.diff().dropna()

# Guardar resumen ADF
resumen_adf = {
    "Variable": list(df_niveles.columns),
    "ADF_Niveles_pvalor": [adfuller(df_niveles[c].dropna(), autolag="AIC")[1] for c in df_niveles.columns],
    "ADF_Diferencias_pvalor": [adfuller(df_var[c].dropna(), autolag="AIC")[1] for c in df_var.columns],
}
pd.DataFrame(resumen_adf).to_csv("resumen_adf.csv", index=False)

# Test de Chow (COVID-19) para la producción
df_chow = df_niveles.copy()
df_chow["dummy_covid"] = (df_chow.index >= "2020-04-01").astype(int)
X_chow = sm.add_constant(df_chow[["dummy_covid"]])
modelo_chow = sm.OLS(df_chow["Produccion_BPDC"], X_chow).fit()
with open("resultados_test_chow.txt", "w") as f:
    f.write(modelo_chow.summary().as_text())

# ============================================================
# 3. MODELO VAR(1)
# ============================================================
modelo_var = VAR(df_var)
resultado_var = modelo_var.fit(1)

# Guardar pronóstico
pronostico_dif = resultado_var.forecast(df_var.values[-resultado_var.k_ar:], steps=12)
df_pronostico_dif = pd.DataFrame(pronostico_dif, columns=df_var.columns)
ultimo_export = df_niveles["Exportaciones_MillonesUSD"].iloc[-1]
pronostico_niveles = ultimo_export + df_pronostico_dif["Exportaciones_MillonesUSD"].cumsum()
pronostico_niveles.to_csv("resultados_pronostico.csv", index=False, header=["Pronostico"])

# ============================================================
# 4. CADENAS DE MARKOV
# ============================================================
def clasificar_precio(precio):
    if precio < 50: return "Bajo"
    elif precio <= 80: return "Medio"
    else: return "Alto"

df_final["Estado_Precio"] = df_final["Precio_Brent_USD"].apply(clasificar_precio)
estados = ["Bajo", "Medio", "Alto"]
matriz_transicion = pd.DataFrame(0, index=estados, columns=estados)

for i in range(1, len(df_final)):
    ea = df_final["Estado_Precio"].iloc[i-1]
    es = df_final["Estado_Precio"].iloc[i]
    matriz_transicion.loc[ea, es] += 1

matriz_prob = matriz_transicion.div(matriz_transicion.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
matriz_prob.to_csv("resultados_markov.csv")

# ============================================================
# 5. SIMULACIÓN MONTE CARLO
# ============================================================
n_sim = 5000
n_meses = 12
precio_actual = df_niveles["Precio_Brent_USD"].iloc[-1]
media_ret = df_niveles["Precio_Brent_USD"].pct_change().mean()
volat = df_niveles["Precio_Brent_USD"].pct_change().std()

np.random.seed(42)
simulaciones = np.zeros((n_sim, n_meses))
for i in range(n_sim):
    precios = [precio_actual]
    for t in range(n_meses):
        r = np.random.normal(media_ret, volat)
        precios.append(precios[-1] * (1 + r))
    simulaciones[i, :] = precios[1:]

indicador_final = simulaciones[:, -1] < 50
prob_final = indicador_final.mean()
indicador_cruce = (simulaciones < 50).any(axis=1)
prob_cruce = indicador_cruce.mean()

# Guardar resultados de Monte Carlo en un CSV
df_mc = pd.DataFrame({
    "Indicador": ["Terminar bajo 50", "Cruzar bajo 50"],
    "Probabilidad": [prob_final, prob_cruce],
    "Conteo": [indicador_final.sum(), indicador_cruce.sum()]
})
df_mc.to_csv("resultados_montecarlo.csv", index=False)

# Guardar base final
df_final.to_csv("resultados_proyecto_petroleo.csv")