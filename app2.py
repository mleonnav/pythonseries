import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

st.set_page_config(page_title="Predicción SARIMA", layout="wide")

st.title("📈 Predicción de Series Temporales con SARIMA")
st.markdown("Carga tu archivo de Excel, configura el periodo de la serie y define el rango de fechas para predecir.")

# Sidebar - Configuración de datos de entrada
st.sidebar.header("1. Carga y Configuración del Archivo")
uploaded_file = st.sidebar.file_uploader("Cargar archivo Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Leer hojas del Excel
    xl = pd.ExcelFile(uploaded_file)
    sheet_name = st.sidebar.selectbox("Selecciona la hoja de Excel", xl.sheet_names)
    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    
    st.sidebar.subheader("Columnas del conjunto de datos")
    date_col = st.sidebar.selectbox("Columna de Fechas", df_raw.columns)
    val_col = st.sidebar.selectbox("Columna de Valores (Serie Temporal)", [c for c in df_raw.columns if c != date_col])

    # Convertir fechas
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    df = df_raw.sort_values(by=date_col).copy()
    
    # Frecuencia de la serie
    st.sidebar.subheader("2. Frecuencia y Fechas de la Serie")
    freq_option = st.sidebar.selectbox(
        "Frecuencia de la serie temporal",
        options=["D (Diaria)", "W (Semanal)", "MS (Mensual - inicio de mes)", "M (Mensual - fin de mes)", "Q (Trimestral)", "Y (Anual)"],
        index=2
    )
    freq_code = freq_option.split(" ")[0]

    # Filtrar rango de fechas de la serie histórica
    min_date_hist = df[date_col].min().date()
    max_date_hist = df[date_col].max().date()
    
    start_hist, end_hist = st.sidebar.date_input(
        "Rango de fechas históricas a utilizar",
        value=(min_date_hist, max_date_hist),
        min_value=min_date_hist,
        max_value=max_date_hist
    )

    # Filtrar DataFrame por el rango seleccionado
    mask = (df[date_col].dt.date >= start_hist) & (df[date_col].dt.date <= end_hist)
    df_filtered = df.loc[mask].set_index(date_col)[[val_col]]
    
    # Asegurar frecuencia del índice
    df_filtered = df_filtered.asfreq(freq_code)
    df_filtered[val_col] = df_filtered[val_col].ffill() # Rellenar faltantes si los hay

    # Configuración de Predicción
    st.sidebar.subheader("3. Fechas de Predicción")
    pred_type = st.sidebar.radio("Método para especificar la predicción:", ["Por rango de fechas", "Por número de periodos"])
    
    if pred_type == "Por rango de fechas":
        default_start_pred = end_hist + pd.Timedelta(days=1)
        pred_end_date = st.sidebar.date_input("Fecha fin de la predicción", value=default_start_pred + pd.DateOffset(months=12))
        
        # Generar rango futuro
        future_dates = pd.date_range(start=df_filtered.index[-1], end=pred_end_date, freq=freq_code)
        # Excluir la última fecha histórica si coincide
        future_dates = future_dates[future_dates > df_filtered.index[-1]]
        n_periods = len(future_dates)
    else:
        n_periods = st.sidebar.number_input("Número de periodos a predecir", min_value=1, max_value=240, value=12)
        future_dates = pd.date_range(start=df_filtered.index[-1], periods=n_periods+1, freq=freq_code)[1:]

    # Parámetros del modelo SARIMA
    st.sidebar.subheader("4. Configuración SARIMA")
    auto_arima = st.sidebar.checkbox("Ajuste automático de parámetros (Auto-ARIMA)", value=True)
    
    seasonal_period = st.sidebar.number_input("Periodo estacional (m) [Ej: 12 para mensual, 4 para trimestral]", min_value=1, value=12)

    if not auto_arima:
        p = st.sidebar.number_input("p (AR)", 0, 5, 1)
        d = st.sidebar.number_input("d (Diferenciación)", 0, 2, 1)
        q = st.sidebar.number_input("q (MA)", 0, 5, 1)
        P = st.sidebar.number_input("P (AR Estacional)", 0, 5, 1)
        D = st.sidebar.number_input("D (Dif. Estacional)", 0, 2, 1)
        Q = st.sidebar.number_input("Q (MA Estacional)", 0, 5, 1)

    # Botón de ejecución
    if st.sidebar.button("🚀 Ejecutar Modelo y Predicción"):
        with st.spinner("Entrenando el modelo SARIMA..."):
            try:
                if auto_arima:
                    # Búsqueda automática con pmdarima
                    model_auto = pm.auto_arima(
                        df_filtered[val_col],
                        m=seasonal_period,
                        seasonal=True,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore"
                    )
                    order = model_auto.order
                    seasonal_order = model_auto.seasonal_order
                    st.success(f"Modelo Auto-SARIMA óptimo encontrado: SARIMA{order}x{seasonal_order}_{seasonal_period}")
                else:
                    order = (p, d, q)
                    seasonal_order = (P, D, Q, seasonal_period)

                # Ajuste del modelo SARIMAX
                model = SARIMAX(df_filtered[val_col], order=order, seasonal_order=seasonal_order)
                results = model.fit(disp=False)

                # Generar predicciones
                forecast_res = results.get_forecast(steps=n_periods)
                forecast_df = forecast_res.summary_frame(alpha=0.05) # Intervalo de confianza del 95%
                forecast_df.index = future_dates

                # Mostrar Resultados
                st.subheader("📊 Gráfico de Histórico y Predicción")
                
                fig = go.Figure()
                
                # Datos Históricos
                fig.add_trace(go.Scatter(
                    x=df_filtered.index, 
                    y=df_filtered[val_col], 
                    mode='lines', 
                    name='Histórico'
                ))
                
                # Predicción
                fig.add_trace(go.Scatter(
                    x=forecast_df.index, 
                    y=forecast_df['mean'], 
                    mode='lines', 
                    name='Predicción',
                    line=dict(color='red')
                ))
                
                # Intervalo de confianza (Límite superior e inferior)
                fig.add_trace(go.Scatter(
                    x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                    y=list(forecast_df['mean_ci_upper']) + list(forecast_df['mean_ci_lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='Intervalo de Confianza (95%)'
                ))

                fig.update_layout(
                    title="Predicción SARIMA",
                    xaxis_title="Fecha",
                    yaxis_title="Valor",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Tabla de datos
                st.subheader("📋 Tabla de Valores Predichos")
                output_table = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].rename(
                    columns={
                        'mean': 'Predicción',
                        'mean_ci_lower': 'Límite Inferior (95%)',
                        'mean_ci_upper': 'Límite Superior (95%)'
                    }
                )
                st.dataframe(output_table)

                # Opción de descarga
                csv_data = output_table.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Descargar Predicciones en CSV",
                    data=csv_data,
                    file_name="prediccion_sarima.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Ocurrió un error durante la ejecución del modelo: {e}")
else:
    st.info("Por favor, sube un archivo de Excel para comenzar.")
