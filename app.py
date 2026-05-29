import os
import pandas as pd
import numpy as np
import streamlit as streamlit_app
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Configuración de la página web
st = streamlit_app
st.set_page_config(page_title="Scouting Analytics | Clones de Jugadores", layout="wide")


# =====================================================================
# 1. MOTOR DE INTELIGENCIA DE DATOS (Caché activo para alta velocidad)
# =====================================================================
@st.cache_data
def inicializar_modelo_y_datos():
    archivos_en_carpeta = [f for f in os.listdir('.') if f.endswith('.csv')]
    archivos_filtrados = [f for f in archivos_en_carpeta if 'Porteros' not in f]

    df_maestro = None
    for archivo in sorted(archivos_filtrados):
        df_temp = pd.read_csv(archivo)
        df_temp.columns = df_temp.columns.str.strip()
        if 'Nombre' in df_temp.columns:
            df_temp = df_temp.rename(columns={'Nombre': 'jugador(a)'})

        if df_maestro is None:
            df_maestro = df_temp
        else:
            columnas_comunes = ['Posiciones', 'Edad', 'Minutos disputados',
                                'Ganancia de duelos aéreos de campo ganados']
            columnas_interseccion = [col for col in columnas_comunes if
                                     col in df_maestro.columns and col in df_temp.columns]
            df_temp = df_temp.drop(columns=columnas_interseccion, errors='ignore')
            df_maestro = pd.merge(df_maestro, df_temp, on=['jugador(a)', 'Equipo'], how='outer')

    # Filtro táctico por minutos competitivos
    df_filtrado = df_maestro[df_maestro['Minutos disputados'] >= 600].copy()
    df_filtrado = df_filtrado.fillna(0).reset_index(drop=True)

    # Separar identidad de rendimiento
    columnas_identidad = ['jugador(a)', 'Equipo', 'Posiciones', 'Edad', 'Minutos disputados']
    metricas_features = [col for col in df_filtrado.columns if col not in columnas_identidad]

    # Pipeline Scikit-Learn
    X = df_filtrado[metricas_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar Vecinos Cercanos con métrica Coseno
    modelo_scouting = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
    modelo_scouting.fit(X_scaled)

    return df_filtrado, X_scaled, modelo_scouting, metricas_features


# Carga inicial de datos
try:
    df_filtrado, X_scaled, modelo_scouting, metricas_features = inicializar_modelo_y_datos()
except Exception as e:
    st.error(f"Error al procesar los archivos CSV: {e}")
    st.stop()

# =====================================================================
# 2. INTERFAZ GRÁFICA DE LA APLICACIÓN
# =====================================================================
st.title("⚽ Plataforma de Scouting Inteligente")
st.markdown(
    "Identifica perfiles y clones de rendimiento utilizando algoritmos de **Machine Learning (Nearest Neighbors & Cosine Distance)**.")
st.divider()

# Barra lateral para parámetros de búsqueda
st.sidebar.header("Filtros de Selección")

# --- NUEVA LÓGICA DE FILTRADO ENCADENADO ---
# 1. Selector de Equipo
lista_equipos = sorted(df_filtrado['Equipo'].unique())
equipo_seleccionado = st.sidebar.selectbox("1. Selecciona un Equipo:", lista_equipos)

# 2. Filtrar el DataFrame temporalmente por el equipo seleccionado para obtener sus jugadores
df_equipo_actual = df_filtrado[df_filtrado['Equipo'] == equipo_seleccionado]
lista_jugadores_filtrados = sorted(df_equipo_actual['jugador(a)'].unique())

# 3. Selector de Jugador (solo muestra los del equipo elegido)
jugador_seleccionado = st.sidebar.selectbox("2. Selecciona el jugador objetivo:", lista_jugadores_filtrados)
# --------------------------------------------

st.sidebar.header("Configuración del Radar")
# Control de cantidad de clones a mostrar
top_n = st.sidebar.slider("Número de clones a buscar:", min_value=1, max_value=10, value=5)

# Filtro opcional por edad en la barra lateral
edad_maxima = st.sidebar.slider("Filtrar candidatos hasta la edad de:", min_value=17, max_value=42, value=42)

# =====================================================================
# 3. PROCESAMIENTO DE SIMILITUD Y RENDERIZADO
# =====================================================================
if jugador_seleccionado:
    # Encontrar índice del jugador target en el DataFrame maestro (df_filtrado)
    idx_jugador = df_filtrado[df_filtrado['jugador(a)'] == jugador_seleccionado].index[0]
    info_target = df_filtrado.iloc[idx_jugador]

    # Contenedor del perfil seleccionado
    st.subheader(f"👤 Perfil Analizado: {info_target['jugador(a)']}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Club Actual", value=info_target['Equipo'])
    with col2:
        st.metric(label="Posición de Campo", value=str(info_target['Posiciones']).upper())
    with col3:
        st.metric(label="Edad", value=f"{int(info_target['Edad'])} años")
    with col4:
        st.metric(label="Minutos Jugados", value=f"{int(info_target['Minutos disputados'])}'")

    st.divider()

    # Calcular vecinos cercanos para el vector del jugador seleccionado
    vector_jugador = X_scaled[idx_jugador].reshape(1, -1)

    # Solicitamos vecinos internamente para tener margen con los filtros de edad
    distancias, indices = modelo_scouting.kneighbors(vector_jugador, n_neighbors=min(30, len(df_filtrado)))

    distancias = distancias.flatten()
    indices = indices.flatten()

    resultados = []
    for i in range(1, len(indices)):
        idx_vecino = indices[i]
        candidato = df_filtrado.iloc[idx_vecino]

        # Filtro de edad dinámico aplicado sobre los resultados del modelo
        if candidato['Edad'] <= edad_maxima:
            porcentaje_similitud = round((1 - distancias[i]) * 100, 2)

            resultados.append({
                'Ranking': len(resultados) + 1,
                'Jugador': candidato['jugador(a)'],
                'Equipo': candidato['Equipo'],
                'Edad': int(candidato['Edad']),
                'Posiciones': str(candidato['Posiciones']).upper(),
                'Minutos': int(candidato['Minutos disputados']),
                'Similitud %': porcentaje_similitud
            })

        if len(resultados) == top_n:
            break

    # Mostrar la tabla de resultados interactiva
    st.subheader(f"🎯 Top {len(resultados)} Clones Potenciales Encontrados")
    if resultados:
        df_resultados = pd.DataFrame(resultados).set_index('Ranking')

        # Formatear el DataFrame visualmente
        st.dataframe(
            df_resultados,
            column_config={
                "Similitud %": st.column_config.ProgressColumn(
                    "Porcentaje de Similitud",
                    help="Calculado mediante distancia del coseno matemática",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                )
            },
            use_container_width=True
        )
    else:
        st.warning("No se encontraron jugadores que cumplan con los filtros de edad seleccionados.")