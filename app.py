import streamlit as st
import yfinance as yf
from lmfit import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Análisis LPPLS de Burbujas Financieras", layout="wide")

def lppls(t, A, B, C, tc, m, omega, phi):
    with np.errstate(invalid='ignore'):
        return A + B * (tc - t)*m + C * (tc - t)*m * np.cos(omega * np.log(tc - t) - phi)

def realizar_analisis_lppls(symbol, start_date, end_date):
    try:
        # Cargar datos
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No se pudieron obtener datos para el símbolo especificado")
            return None
            
        data = data[['Close']]
        data.index = pd.to_datetime(data.index)
        data.index.name = 'Date'
        data['LogPrice'] = np.log(data['Close'])
        
        # Preparar datos para el modelo
        t = (data.index - data.index[0]).days.values
        log_price = data['LogPrice'].values
        
        # Crear y ajustar el modelo
        model = Model(lppls)
        params = model.make_params(A=10, B=-10, C=0.1, tc=t[-1] + 50, m=0.5, omega=6, phi=0)
        params['tc'].set(min=t[-1] + 1, max=t[-1] + 100)
        
        result = model.fit(log_price, params, t=t)
        best_params = result.best_values
        tc = best_params['tc']
        
        # Predicciones
        t_pred = np.linspace(t[0], t[-1] + 100, 500)
        log_price_pred = lppls(t_pred, **best_params)
        
        # Cálculos intermedios para evitar problemas con Series
        precio_inicial = data['Close'].iloc[0]
        precio_final = data['Close'].iloc[-1]
        
        # Métricas con manejo seguro de tipos
        ci = result.chisqr / result.ndata
        bs = (precio_final - precio_inicial) / precio_inicial
        bd = t[-1] - t[0]
        
        # Cálculo seguro de bg
        try:
            bg = (np.exp(np.log(1 + bs) / (bd / 365)) - 1) * 100
        except:
            bg = 0.0
            
        # Cálculo seguro de bp
        try:
            bp = (t[-1] - t[0]) / (tc - t[0]) * 100
        except:
            bp = 0.0
            
        # Cálculos finales con verificación de valores negativos
        try:
            geometric_avg = (abs(ci) * abs(bg) * abs(bp)) ** (1/3)
        except:
            geometric_avg = 0.0
            
        scenario_prob = (ci + bg + bp) / 3
        tc_date = data.index[0] + pd.to_timedelta(tc, unit='D')
        
        return {
            'data': data,
            't_pred': t_pred,
            'log_price_pred': log_price_pred,
            'tc': tc,
            'tc_date': tc_date,
            'metrics': {
                'ci': float(ci),
                'bs': float(bs * 100),  # Convertir a porcentaje
                'bd': float(bd),
                'bg': float(bg),
                'bp': float(bp),
                'geometric_avg': float(geometric_avg),
                'scenario_prob': float(scenario_prob)
            }
        }
    except Exception as e:
        st.error(f"Error en el análisis: {str(e)}")
        st.error("Detalles del error para depuración:", exc_info=True)
        return None

# Interfaz de usuario
st.title("📈 Análisis de Burbujas Financieras con LPPLS")

col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("Símbolo del activo", "VNQ")

with col2:
    start_date = st.date_input(
        "Fecha de inicio",
        datetime.now() - timedelta(days=365)
    )

with col3:
    end_date = st.date_input(
        "Fecha final",
        datetime.now()
    )

if st.button("Realizar Análisis"):
    with st.spinner("Analizando datos..."):
        resultados = realizar_analisis_lppls(symbol, start_date, end_date)
        
        if resultados:
            # Mostrar gráfico
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(resultados['data'].index, resultados['data']['LogPrice'], label='Datos originales')
            ax.plot(
                resultados['data'].index[0] + pd.to_timedelta(resultados['t_pred'], unit='D'),
                resultados['log_price_pred'],
                label='Predicción LPPLS',
                linestyle='--'
            )
            ax.axvline(resultados['tc_date'], color='r', linestyle='--', label='Tiempo Crítico (tc)')
            ax.set_xlabel('Fecha')
            ax.set_ylabel('Precio logarítmico')
            ax.legend()
            st.pyplot(fig)
            
            # Mostrar métricas en columnas
            st.subheader("📊 Métricas del Análisis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Confianza (CI)", f"{resultados['metrics']['ci']:.2f}%")
                st.metric("Tamaño de la Burbuja (BS)", f"{resultados['metrics']['bs']:.2f}%")
                
            with col2:
                st.metric("Duración de la Burbuja", f"{resultados['metrics']['bd']} días")
                st.metric("CAGR de la Burbuja", f"{resultados['metrics']['bg']:.2f}%")
                
            with col3:
                st.metric("Progreso de la Burbuja", f"{resultados['metrics']['bp']:.2f}%")
                st.metric("Probabilidad del Escenario", f"{resultados['metrics']['scenario_prob']:.2f}%")
            
            st.info(f"🎯 Tiempo Crítico Estimado: {resultados['tc_date'].strftime('%Y-%m-%d')}")

st.sidebar.markdown("""
### 📝 Instrucciones
1. Ingresa el símbolo del activo (ej. VNQ, AAPL, BTC-USD)
2. Selecciona el rango de fechas
3. Haz clic en "Realizar Análisis"
""")

st.sidebar.markdown("""
### ℹ️ Acerca del Análisis LPPLS
El modelo Log-Periodic Power Law Singularity (LPPLS) se utiliza para identificar burbujas financieras y predecir posibles puntos críticos de reversión.
""") 