import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sector Petrolero Colombiano", layout="wide")
st.title("🛢️ Análisis y Simulación del Sector Petrolero Colombiano")
st.markdown("**Modelos:** VAR(1), Cadenas de Markov y Monte Carlo | **Periodo:** 2015 - 2026")
st.markdown("**Integrantes:** Luis Carlos Camargo, Danny Pulido, Cristian Camilo González")

# Cargar datos base
@st.cache_data
def cargar_datos():
    return pd.read_csv("resultados_proyecto_petroleo.csv", parse_dates=["Fecha"], index_col="Fecha")

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error("⚠️ Ejecuta primero 'proyecto_petroleo.py' para generar los resultados.")
    st.stop()

st.sidebar.header("⚙️ Navegación")
opcion = st.sidebar.radio("Selecciona una sección:", 
    ["Inicio", "Series Históricas", "Modelo VAR", "Markov", "Monte Carlo", "Limitaciones"])

if opcion == "Inicio":
    st.header("📌 Resumen del Proyecto")
    st.markdown("""
    **Pregunta central:** ¿Cómo se relacionan dinámicamente la producción nacional de crudo, el precio internacional Brent y las exportaciones petroleras de Colombia entre 2015 y 2026?
    **Usuario:** Equipo de análisis de política energética.
    **Decisión:** Activar alertas de riesgo exportador.
    """)
    st.subheader("Últimos datos registrados")
    col1, col2, col3 = st.columns(3)
    ultimo = df.iloc[-1]
    col1.metric("Exportaciones (Millones USD)", f"${ultimo['Exportaciones_MillonesUSD']:,.2f}")
    col2.metric("Precio Brent (USD/barril)", f"${ultimo['Precio_Brent_USD']:.2f}")
    col3.metric("Producción (Barriles/día)", f"{ultimo['Produccion_BPDC']:,.0f}")

elif opcion == "Series Históricas":
    st.header("📈 Series Históricas")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(df.index, df["Exportaciones_MillonesUSD"], color="blue"); axes[0].set_title("Exportaciones")
    axes[1].plot(df.index, df["Precio_Brent_USD"], color="orange"); axes[1].set_title("Precio Brent")
    axes[2].plot(df.index, df["Produccion_BPDC"], color="green"); axes[2].set_title("Producción")
    plt.tight_layout(); st.pyplot(fig)
    st.dataframe(df.describe().round(2))

elif opcion == "Modelo VAR":
    st.header("📊 Modelo VAR(1)")
    st.markdown("**Coeficiente clave:** Brent L1 = 11.19 (p < 0.001).")
    pronostico = pd.read_csv("resultados_pronostico.csv")["Pronostico"].values
    meses = pd.date_range(df.index[-1], periods=13, freq="ME")[1:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-12:], df["Exportaciones_MillonesUSD"].iloc[-12:], label="Histórico", marker="o")
    ax.plot(meses, pronostico, label="Pronóstico", color="red", marker="x")
    ax.legend(); st.pyplot(fig)

elif opcion == "Markov":
    st.header("🔄 Cadenas de Markov")
    matriz = pd.read_csv("resultados_markov.csv", index_col=0)
    st.subheader("Matriz de transición")
    st.dataframe(matriz.style.format("{:.1%}"))

elif opcion == "Monte Carlo":
    st.header("🎲 Simulación Monte Carlo")
    mc = pd.read_csv("resultados_montecarlo.csv")
    col1, col2 = st.columns(2)
    col1.metric("Terminar mes 12 bajo USD 50", f"{mc.iloc[0]['Probabilidad']:.2%}", f"{mc.iloc[0]['Conteo']} de 5000")
    col2.metric("Cruzar bajo USD 50", f"{mc.iloc[1]['Probabilidad']:.2%}", f"{mc.iloc[1]['Conteo']} de 5000")

elif opcion == "Limitaciones":
    st.header("⚠️ Limitaciones")
    st.markdown("""
    **1. Quiebres estructurales:** COVID-19 (2020) y guerra de Ucrania (2022).
    **2. No normalidad de residuos:** Test multivariado rechaza normalidad.
    **3. Asociación vs. causalidad:** El coeficiente 11.19 es asociación de corto plazo.
    **4. ADF de Exportaciones:** p=0.0134 sugiere posible estacionariedad en niveles (I(0)).
    **5. Supuestos de Monte Carlo:** Probabilidades simuladas, no garantías.
    """)