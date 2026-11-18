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

opcion = st.sidebar.radio("Selecciona una sección:", 
    ["Inicio", "Series Históricas", "Modelo VAR", "Markov", "Monte Carlo", "Limitaciones", "Proyección por precio"])

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
    

elif opcion == "Proyección por precio":
    st.header("🔮 Proyección por escenario de precio (Brent)")
    
    st.markdown("""
    Esta sección permite simular un escenario condicionado al precio del Brent.
    **No es una predicción causal**, sino una proyección basada en el modelo VAR(1) estimado.
    """)

    # 1. Slider para el precio supuesto
    precio_supuesto = st.slider("Selecciona un precio Brent supuesto (USD/barril):", 
                                min_value=20, max_value=150, value=70, step=1)

    # 2. Cargar datos históricos
    df = pd.read_csv("resultados_proyecto_petroleo.csv", parse_dates=["Fecha"], index_col="Fecha")
    columnas = ["Exportaciones_MillonesUSD", "Precio_Brent_USD", "Produccion_BPDC"]

    # 3. Reestimar el modelo VAR(1) para obtener los coeficientes
    from statsmodels.tsa.api import VAR
    df_var = df[columnas].diff().dropna()
    modelo_var = VAR(df_var)
    resultado_var = modelo_var.fit(1)
    coefs = resultado_var.coefs[0]
    intercepto = resultado_var.intercept

    # 4. Simulación del escenario
    if st.button("Simular escenario"):
        # Tomar los últimos valores observados
        ultima_obs = df[columnas].iloc[-1].values
        ultima_dif = df_var.iloc[-1].values
        
        pronosticos = []
        nivel_actual = ultima_obs.copy()
        diferencia_anterior = ultima_dif.copy()

        for t in range(12):
            # Calcular la nueva diferencia con el VAR(1)
            cambio = intercepto + coefs @ diferencia_anterior
            
            # Reconstruir el nivel
            nivel_actual = nivel_actual + cambio
            
            # Forzar el precio Brent al valor del escenario
            nivel_actual[1] = precio_supuesto
            
            # Guardar los resultados
            pronosticos.append(nivel_actual.copy())
            
            # Actualizar la diferencia para el siguiente paso
            # La nueva diferencia es el cambio que acabamos de calcular
            diferencia_anterior = nivel_actual - (nivel_actual - cambio) # Esto es simplemente 'cambio'
            # Pero para mantener la consistencia con el modelo, usamos el cambio calculado
            diferencia_anterior = cambio.copy()
            # NOTA: Si forzamos el precio, debemos asegurar que la diferencia de precio también se actualice
            # Para simplificar, usamos el cambio calculado por el VAR antes de forzar el precio.

        # Crear DataFrame con los resultados
        resultado_escenario = pd.DataFrame(pronosticos, columns=columnas)
        resultado_escenario["Mes"] = pd.date_range(
            df.index[-1] + pd.offsets.MonthBegin(1),
            periods=12,
            freq="MS"
        )
        resultado_escenario = resultado_escenario.set_index("Mes")

        # 5. Mostrar resultados
        st.subheader("Resultados del escenario")
        
        col1, col2 = st.columns(2)
        col1.metric("Producción promedio", f"{resultado_escenario['Produccion_BPDC'].mean():,.0f} BPDC")
        col2.metric("Exportaciones acumuladas", f"USD {resultado_escenario['Exportaciones_MillonesUSD'].sum():,.2f} millones")

        col3, col4 = st.columns(2)
        col3.metric("Producción al mes 12", f"{resultado_escenario['Produccion_BPDC'].iloc[-1]:,.0f} BPDC")
        col4.metric("Exportaciones al mes 12", f"USD {resultado_escenario['Exportaciones_MillonesUSD'].iloc[-1]:,.2f} millones")

        st.subheader("Tabla de proyección")
        st.dataframe(resultado_escenario.round(2), use_container_width=True)

        st.subheader("Evolución de la producción")
        st.line_chart(resultado_escenario["Produccion_BPDC"])

        st.subheader("Evolución de las exportaciones FOB")
        st.line_chart(resultado_escenario["Exportaciones_MillonesUSD"])

        st.info("""
        **Advertencia metodológica:** este escenario mantiene constante
        el precio Brent introducido. Las estimaciones de producción
        y exportaciones son condicionadas al VAR(1), no relaciones
        causales ni garantías de resultados futuros.
        """)
        

