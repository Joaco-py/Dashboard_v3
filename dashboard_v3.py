import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.dates as mdates
import calendar
from scipy import stats

# Configuración inicial
st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title="Dashboard Supermarket Sales")
sns.set_style("whitegrid")

# Estilos CSS globales
st.markdown("""
<style>
    /* Estilos para el fondo de la aplicación y texto */
    .stApp {
        background-color: #0d1117;
        color: #e2e8f0;
    }
    
    /* Estilos para todos los encabezados */
    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Reducción de espacios verticales en toda la aplicación */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Estilo para los contenedores de sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid rgba(100,116,139,0.6);
    }
    
    /* Estilo para elementos de filtro */
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stDateInput,
    [data-testid="stSidebar"] .stMultiselect {
        background-color: rgba(27, 38, 59, 0.8);
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 8px;
        border: 1px solid rgba(100,116,139,0.5);
    }
    
    /* Estilo para los elementos seleccionados del multiselect */
    [data-testid="stSidebar"] div[data-baseweb="chip"] {
        margin-top: 5px;
    }
    
    /* Fix para eliminar el margen verde adicional del multiselect */
    [data-testid="stSidebar"] div.stMultiSelect > div > div:has(div[data-baseweb="select"]) {
        background-color: transparent !important;
        padding: 0 !important;
        border: none !important;
    }
    
    /* Reducir anidamiento y márgenes excesivos */
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stDateInput > div,
    [data-testid="stSidebar"] .stMultiselect > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Estilo para selectbox y multiselect */
    div[data-baseweb="select"] {
        background-color: #1b263b;
        border-radius: 5px;
    }
    
    /* Estilo para botones de navegación uniformes */
    div[data-testid*="stHorizontalBlock"] button {
        min-height: 80px !important; /* Altura uniforme basada en el botón más grande */
        width: 100% !important;
        border-radius: 8px !important;
        white-space: normal !important;
        padding: 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 500 !important;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif !important; /* Tipografía profesional */
        letter-spacing: 0.3px !important;
    }
    
    /* Estilo para botones */
    button[kind="primaryFormSubmit"] {
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    /* Estilo para cajas de información */
    div[data-testid="stInfo"] {
        background-color: rgba(59, 130, 246, 0.2);
        color: #e2e8f0;
        border-left-color: #3b82f6;
    }
    
    /* Estilo para warnings */
    div[data-testid="stWarning"] {
        background-color: rgba(245, 158, 11, 0.2);
        color: #e2e8f0;
    }
    
    /* Estilo para expanders */
    details {
        background-color: #1b263b;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Estilo para dataframes */
    .stDataFrame {
        background-color: #1b263b;
        border-radius: 5px;
    }
    
    .stDataFrame td, .stDataFrame th {
        color: #e2e8f0 !important;
    }
    
    /* Estilo para botones de navegación con margen uniforme */
    div[data-testid*="stHorizontalBlock"] {
        gap: 10px !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    div[data-testid*="stHorizontalBlock"] > div {
        flex: 1;
    }
    
    /* Estilo específico para los títulos de los widgets */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stMultiselect label {
        color: #e2e8f0 !important;
        font-weight: 500;
        margin-bottom: 5px;
    }
    
    /* Reducir el espacio entre secciones */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Estilo para títulos de sección con barra roja */
    div[style*="border-left: 4px solid #FF3545"] {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Estilo para encabezado de información del grupo - alineado a la izquierda con nuevo estilo */
    .group-info {
        background: linear-gradient(90deg, rgba(13,17,23,0.6) 0%, rgba(27,38,59,0.6) 100%);
        border-radius: 7px;
        padding: 8px 15px;
        margin-bottom: 12px;
        margin-left: 0;
        margin-right: auto;
        width: max-content;
        max-width: 100%;
        border: 2px solid #FF3545; /* Barra roja alrededor de todo el contenedor */
    }
    
    .group-info p {
        margin: 3px 0;
        font-size: 14px;
        color: #e2e8f0;
    }
    
    .group-info ul {
        margin: 2px 0 5px 0;
        padding-left: 20px;
    }
    
    .group-info li {
        font-size: 13px;
        margin: 1px 0;
        color: #e2e8f0;
    }
    
    /* Reducir espacio entre título de navegación y botones */
    div.navigation-container {
        margin-bottom: 0.2rem !important;
    }
    
    div.navigation-container + div[data-testid*="stHorizontalBlock"] {
        margin-top: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

def parse_date(x):
    if pd.isna(x) or not isinstance(x, str):
        return pd.NaT
    s = x.strip()
    if '-' in s:
        d, m, y = s.split('-')
        return pd.to_datetime(f"{d}-{m}-{y}", format='%d-%m-%Y', errors='coerce')
    if '/' in s:
        a, b, y = s.split('/')
        ia, ib = int(a), int(b)
        if ia > 12:
            return pd.to_datetime(f"{ia}-{ib}-{y}", format='%d-%m-%Y', errors='coerce')
        if ib > 12:
            return pd.to_datetime(f"{ib}-{ia}-{y}", format='%d-%m-%Y', errors='coerce')
        return pd.to_datetime(f"{ia}-{ib}-{y}", format='%d-%m-%Y', errors='coerce')
    return pd.to_datetime(s, infer_datetime_format=True, errors='coerce')

def clean_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    mask = s.str.count(r'\.') > 1
    cleaned = np.where(mask, s.str.replace('.', '', regex=False), s)
    return pd.to_numeric(cleaned, errors='coerce')

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, dtype=str)
    df['Date'] = df['Date'].apply(parse_date)
    df['Time'] = df['Time'].astype(str)
    for col in ['Branch','City','Customer type','Gender','Product line','Payment']:
        df[col] = df[col].astype('category')
    df['Invoice ID'] = df['Invoice ID'].astype(str)
    df['Quantity'] = pd.to_numeric(df['Quantity'], downcast='integer', errors='coerce')
    for col in ['Unit price','Tax 5%','cogs','Rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['Total','gross margin percentage','gross income']:
        df[col] = clean_numeric(df[col])
    return df

@st.cache_data
def load_raw(path):
    return pd.read_csv(path)

df = load_data("data.csv")
df_raw = load_raw("data.csv")

if 'seccion' not in st.session_state:
    st.session_state.seccion = "EDA"
if 'sub_eda' not in st.session_state:
    st.session_state.sub_eda = "EDA INICIAL"
if 'grafico_compuesto' not in st.session_state:
    st.session_state.grafico_compuesto = "1️⃣ Ingreso Bruto por Sucursal"

# Información del grupo
st.markdown("""
<div class="group-info">
    <p><strong>Curso:</strong> Visualización de Datos en Python</p>
    <p><strong>Grupo:</strong> 14</p>
    <p><strong>Equipo:</strong></p>
    <ul>
        <li>Joaquín Alonso Martínez Palma</li>
        <li>Rodrigo Antonio Olivares González</li>
        <li>Sebastian Ignacio Rozas Cifuentes</li>
        <li>Ignacio Andres Sandoval Zapata</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Título principal
st.title("📊 Dashboard Supermarket Sales")

# Navegación
st.markdown("""
<div style="background: linear-gradient(90deg, rgba(13,17,23,0.9) 0%, rgba(27,38,59,0.9) 100%); 
            border-radius: 10px; 
            padding: 12px 20px 12px 20px; 
            margin-bottom: 0.2rem;
            border: 1px solid rgba(100,116,139,0.5);
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);"
            class="navigation-container">
    <h2 style="color: #e2e8f0; margin: 0; font-size: 22px;">
        <span style="font-size: 22px;">🔍</span> Navegación
    </h2>
    <div id="navigation-buttons" style="display: flex; gap: 10px; justify-content: space-between;">
    </div>
</div>
""", unsafe_allow_html=True)

# Secciones para navegación
secciones = ["Introducción", "EDA", "Justificación de variables claves", "Visualizaciones Básicas", "Gráfico compuesto y multivariado", "Gráfico 3D", "Conclusiones"]
cols = st.columns(len(secciones))
button_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        const buttonContainer = document.querySelector('[data-testid="stHorizontalBlock"]');
        if (buttonContainer) {
            const navContainer = document.getElementById('navigation-buttons');
            if (navContainer) {
                navContainer.appendChild(buttonContainer);
            }
        }
    }, 500);
});
</script>
"""

st.markdown(button_js, unsafe_allow_html=True)

for i, nombre in enumerate(secciones):
    if st.session_state.seccion == nombre:
        button_style = """
        <style>
        div[data-testid*="stHorizontalBlock"] > div:nth-child({}) button {{
            background-color: #FF3545 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
        }}
        </style>
        """.format(i+1)
        st.markdown(button_style, unsafe_allow_html=True)
    else:
        button_style = """
        <style>
        div[data-testid*="stHorizontalBlock"] > div:nth-child({}) button {{
            background-color: rgba(27, 38, 59, 0.8) !important;
            border: 1px solid rgba(100,116,139,0.4) !important;
        }}
        div[data-testid*="stHorizontalBlock"] > div:nth-child({}) button:hover {{
            background-color: rgba(255, 53, 69, 0.7) !important;
            color: white !important;
        }}
        </style>
        """.format(i+1, i+1)
        st.markdown(button_style, unsafe_allow_html=True)
    
    if cols[i].button(nombre):
        st.session_state.seccion = nombre
        if nombre == "EDA":
            st.session_state.sub_eda = "EDA INICIAL"
        st.rerun()

# Estilos para multiselect
st.markdown("""
<style>
    /* Aseguramos que el multiselect no tenga estilos adicionales */
    [data-testid="stMultiSelect"] {
        background-color: transparent !important;
        padding: 0 !important;
        margin-bottom: 15px !important;
        border: none !important;
    }
    
    /* El contenedor del multiselect debe tener el mismo estilo que los otros filtros */
    [data-testid="stMultiSelect"] > div {
        background-color: rgba(27, 38, 59, 0.8) !important;
        padding: 10px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(100,116,139,0.5) !important;
    }
    
    /* Estilo para los tags seleccionados */
    [data-testid="stMultiSelect"] div[data-baseweb="tag"] {
        background-color: #FF3545 !important;
        border-radius: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar con filtros
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(180deg, rgba(27,38,59,0.95) 0%, rgba(27,38,59,0.95) 100%);
               border-radius: 8px;
               padding: 15px;
               margin-bottom: 15px;
               border: 1px solid rgba(100,116,139,0.6);
               box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h3 style="color: #e2e8f0; margin: 0; font-size: 20px;">
            <span style="margin-right: 8px;">📋</span>Filtros
        </h3>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        branch_opt = ['Todas'] + list(df['Branch'].unique())
        branch = st.selectbox("Sucursal", branch_opt)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        valid_dates = df['Date'].dropna()
        if not valid_dates.empty:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.date_input("Rango de Fechas", (min_d, max_d), min_value=min_d, max_value=max_d)
            if len(date_range) == 2:
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            else:
                start_date, end_date = pd.to_datetime(min_d), pd.to_datetime(max_d)
        else:
            start_date = pd.to_datetime("2019-01-01")
            end_date = pd.to_datetime("2019-12-31")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        products = st.multiselect("Líneas de Producto", options=df['Product line'].unique(), default=list(df['Product line'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)

# Filtrado de datos
df_filtered = df.copy()
if branch != 'Todas':
    df_filtered = df_filtered[df_filtered['Branch'] == branch]
df_filtered = df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]
if products:
    df_filtered = df_filtered[df_filtered['Product line'].isin(products)]

# Sección: Introducción
if st.session_state.seccion == "Introducción":
    st.markdown("""
    <h2 style="margin: 0.5rem 0; color: #e2e8f0; background: linear-gradient(90deg, rgba(13,17,23,0.6) 0%, rgba(27,38,59,0.6) 100%); 
                padding: 8px 15px; border-radius: 7px; border-left: 4px solid #FF3545;">
        Introducción
    </h2>
    """, unsafe_allow_html=True)
    
    st.info("Esta sección está pendiente de ser completada.")

# Sección: EDA
elif st.session_state.seccion == "EDA":
    st.markdown("""
    <h2 style="margin: 0.5rem 0; color: #e2e8f0; background: linear-gradient(90deg, rgba(13,17,23,0.6) 0%, rgba(27,38,59,0.6) 100%); 
                padding: 8px 15px; border-radius: 7px; border-left: 4px solid #FF3545;">
        Exploración de Datos
    </h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    div[data-testid="stRadio"] > div {
        background-color: rgba(13,17,23,0.3);
        border-radius: 7px;
        padding: 10px;
    }
    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    sub = st.radio("Subsección EDA", ["EDA INICIAL", "EDA POST-TRATAMIENTO"], horizontal=True)
    st.session_state.sub_eda = sub

    if sub == "EDA INICIAL":
        st.write(f"Total de registros: {len(df_raw)}")
        st.write("**Tipo de dato por columna:**")
        st.dataframe(pd.DataFrame(df_raw.dtypes, columns=["Tipo"]))
        st.write("**Valores no nulos:**")
        st.dataframe(pd.DataFrame(df_raw.count(), columns=["No Nulos"]))
        st.write("**Primeros 10 registros:**")
        st.dataframe(df_raw.head(10))
        st.markdown("""
        ### Observaciones
        Al revisar los primeros 10 valores del campo "Date" vemos cosas como:

        "1/5/2019", "3/8/2019", "3/3/2019", "1/27/2019",...

        Esto confirma que:

        - Algunos registros siguen el formato day/month/year y otros month/day/year.
        - Se aplicó parseo inteligente y se normalizó a "DD-MM-YYYY".
        """)
    else:
        st.write("**Tipos de dato finales:**")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Tipo"]))
        st.write("**Primeros 50 registros post-limpieza:**")
        st.dataframe(df.head(50))
        st.markdown("### Observaciones sobre los datos procesados")
        st.info("[Ingresar texto de análisis aquí]")

# Sección: Justificación de variables claves
elif st.session_state.seccion == "Justificación de variables claves":
    # Código para generar la matriz de correlación
    num_cols = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']
    cat_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
    
    # Crear copia para no afectar el dataframe original
    df_codes = df[num_cols + cat_cols].copy()
    
    # Convertir categorías a códigos numéricos para correlación
    for c in cat_cols:
        df_codes[c] = df_codes[c].cat.codes
    
    # Generar matriz de correlación
    corr_ext = df_codes.corr()
    
    # Crear la figura con matplotlib/seaborn
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_ext, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.title("Correlación (numéricas + categorías codificadas)")
    
    # Mostrar la matriz de correlación
    st.pyplot(fig)
    
    # Texto de análisis
    st.markdown("""
    ### Decisión y justificación de variables

    Al analizar la tabla estadística conjunta y la matriz de correlación extendida, observamos que:

    - Entre las variables numéricas, "Total", "Quantity" y "Unit price" muestran fuertes relaciones mutuas (r ≥ 0.63) y capturan tanto volumen como importe de la venta.
    - "Rating" aporta información independiente (corr ≈ 0) sobre satisfacción del cliente.
    - La variable "gross margin percentage" es constante y no aporta variabilidad, por lo que la descartamos.
    - De las variables categóricas, las correlaciones con las numéricas son muy bajas (<|0.1|), lo que refleja que el método de pago o la ciudad no explica directamente variaciones en el monto o cantidad de venta. Sin embargo, por su importancia de negocio, incluiremos "Product line" y "Payment" en los análisis posteriores para comparar desempeño entre segmentos.
    - "Gender" a pesar de mostrar baja correlación con las métricas de venta, es clave para entender diferencias de comportamiento de compra entre hombres y mujeres y enriquecer el análisis de segmentos.

    Por tanto, para el siguiente paso nos quedamos con las siguientes selecciones:
    - Variables Numéricas: Total, Quantity, Unit price, Rating
    - Variables Categóricas: Product line, Payment, Gender
    """)

# Sección: Visualizaciones Básicas
elif st.session_state.seccion == "Visualizaciones Básicas":
    st.markdown("""
    <div style="margin: 0.5rem 0; color: #e2e8f0; background: linear-gradient(90deg, rgba(13,17,23,0.6) 0%, rgba(27,38,59,0.6) 100%); 
               padding: 8px 15px; border-radius: 7px; border-left: 4px solid #FF3545; display:inline-block;">
        <p style="color:#e2e8f0; font-size:16px; margin:0;">
            <span style="color:#FF3545; font-size:18px;">📌</span>
            <span style="margin-left:5px;">Selecciona el gráfico para visualizar</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    grafico = st.radio(
        "Selecciona el gráfico",
        [
            "1️⃣ Evolución Ventas", 
            "2️⃣ Ingresos por Línea Producto", 
            "3️⃣ Distribución Rating", 
            "4️⃣ Gasto por Tipo Cliente", 
            "5️⃣ Precio vs Rating"
        ], 
        horizontal=True,
        label_visibility="collapsed"
    )

    if grafico == "1️⃣ Evolución Ventas":
        sales = df_filtered.groupby(pd.Grouper(key='Date', freq='MS'))['Total'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales['Date'], y=sales['Total'], mode='lines+markers', marker=dict(color='orange')))
        fig.update_layout(
            title="Evolución Mensual de Ventas Totales", 
            xaxis_title="Fecha", 
            yaxis_title="Ventas",
            xaxis=dict(
                tickmode='array',
                tickvals=sales['Date'],
                tickformat='%b %Y',
                tickangle=-45,
                showgrid=True
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("[Ingresar texto de análisis aquí]")

    elif grafico == "2️⃣ Ingresos por Línea Producto":
        revenue_by_product = df_filtered.groupby('Product line')['Total'].sum().sort_values(ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=revenue_by_product.index,
            y=revenue_by_product.values,
            marker_color='orange',
            marker_line_color='black',
            marker_line_width=1
        ))
        fig.update_layout(
            title="Ingresos Totales por Línea de Producto",
            xaxis_title="",
            yaxis_title="Ingresos Totales",
            xaxis={'categoryorder':'total descending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("[Ingresar texto de análisis aquí]")

    elif grafico == "3️⃣ Distribución Rating":
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df_filtered['Rating'], 
            nbinsx=10, 
            marker_color='orange',
            marker_line_color='black',
            marker_line_width=1,
            opacity=1.0,
            name="Frecuencia"
        ))
        
        if len(df_filtered) > 1:
            kde = stats.gaussian_kde(df_filtered['Rating'].dropna())
            x_range = np.linspace(df_filtered['Rating'].min()-0.5, df_filtered['Rating'].max()+0.5, 1000)
            y_kde = kde(x_range)
            
            hist, bin_edges = np.histogram(df_filtered['Rating'].dropna(), bins=10)
            scale_factor = max(hist) / max(y_kde)
            y_kde_scaled = y_kde * scale_factor
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                line=dict(color='#0099FF', width=3),
                name="Densidad"
            ))
        
        fig.update_layout(
            title="Distribución de Calificaciones",
            xaxis_title="Rating",
            yaxis_title="Frecuencia",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("[Ingresar texto de análisis aquí]")

    elif grafico == "4️⃣ Gasto por Tipo Cliente":
        fig = go.Figure()
        
        colors = {'Member': '#FF8C00', 'Normal': '#1f77b4'}
        
        for tipo in df_filtered['Customer type'].unique():
            fig.add_trace(go.Box(
                y=df_filtered[df_filtered['Customer type'] == tipo]['Total'], 
                name=tipo,
                marker_color=colors.get(tipo, 'gray')
            ))
            
        fig.update_layout(
            title="Gasto Total por Tipo de Cliente",
            yaxis_title="Gasto"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("[Ingresar texto de análisis aquí]")

    elif grafico == "5️⃣ Precio vs Rating":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Unit price'], 
            y=df_filtered['Rating'], 
            mode='markers', 
            marker=dict(color='orange', opacity=1.0)
        ))
        fig.update_layout(
            title="Precio Unitario vs Rating", 
            xaxis_title="Precio Unitario", 
            yaxis_title="Rating"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Análisis: Relación entre Precio y Evaluación del Cliente
        
        El diagrama muestra una nube de puntos muy dispersa sin una pendiente clara, lo que indica que el precio unitario y la valoración del cliente son prácticamente independientes. Tanto en productos baratos como en los más caros encontramos valoraciones altas y bajas, sin importar cuánto pague el cliente. Esto sugiere que otros factores como la calidad percibida, el servicio o la familiaridad con la marca tienen más peso en la satisfacción que el propio precio.
        
        Este hallazgo contradice la creencia común de que "lo caro es mejor", ampliamente estudiada en la literatura sobre comportamiento del consumidor. Según investigaciones en el campo de la psicología del consumidor, existe una tendencia a asociar precios más altos con mayor calidad ("Los consumidores a menudo asocian un precio más alto con una mayor calidad"), pero nuestros datos no respaldan esta asociación para los productos analizados.
        
        Estudios sobre la relación precio-calidad percibida, como los recopilados por investigadores académicos, han encontrado que esta relación "no es universal y, además, no es siempre positiva" (Peterson y Wilson, 1985). Nuestro análisis coincide con estas observaciones, demostrando que al menos en este supermercado, factores distintos al precio determinan la satisfacción del cliente.
        
        Esta información es valiosa para estrategias comerciales, ya que indica que competir únicamente en precio podría no ser eficaz para mejorar la percepción del cliente. En su lugar, enfocarse en otros atributos que contribuyen a la experiencia del usuario podría generar mayor impacto en las evaluaciones y la fidelidad de los clientes.
        """)


# Sección: Gráfico compuesto y multivariado
elif st.session_state.seccion == "Gráfico compuesto y multivariado":
    st.markdown("""
    <div style="margin: 0.5rem 0; color: #e2e8f0; background: linear-gradient(90deg, rgba(13,17,23,0.6) 0%, rgba(27,38,59,0.6) 100%); 
               padding: 8px 15px; border-radius: 7px; border-left: 4px solid #FF3545; display:inline-block;">
        <p style="color:#e2e8f0; font-size:16px; margin:0;">
            <span style="color:#FF3545; font-size:18px;">📌</span>
            <span style="margin-left:5px;">Selecciona el gráfico para visualizar</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    grafico_compuesto = st.radio(
        "Tipo de gráfico compuesto",
        [
            "1️⃣ Ingreso Bruto por Sucursal",
            "2️⃣ Ventas por Género y Tipo Cliente"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.session_state.grafico_compuesto = grafico_compuesto
    
    if grafico_compuesto == "1️⃣ Ingreso Bruto por Sucursal":
        pivot_income = df_filtered.groupby(['Branch', 'Product line'])['gross income'].sum().reset_index()
        fig = px.bar(
            pivot_income, 
            x='Branch', 
            y='gross income', 
            color='Product line', 
            barmode='stack', 
            text_auto=True
        )
        fig.update_layout(
            title="Composición del Ingreso Bruto por Sucursal y Línea de Producto",
            legend_title="Línea Producto",
            yaxis_title="Ingreso Bruto"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("[Ingresar texto de análisis aquí]")

    elif grafico_compuesto == "2️⃣ Ventas por Género y Tipo Cliente":
        grouped = df_filtered.groupby(['Product line', 'Gender', 'Customer type'])['Total'].mean().reset_index()
        if not grouped.empty:
            # Variables para controlar si es el primer gráfico
            first_graph = True
            
            for ctype in grouped['Customer type'].unique():
                data_ctype = grouped[grouped['Customer type'] == ctype]
                
                color_map = {'Male': '#1f77b4', 'Female': '#FF69B4'}
                
                fig = px.bar(
                    data_ctype, 
                    x='Product line', 
                    y='Total', 
                    color='Gender',
                    barmode='group', 
                    title=f"Tipo de Cliente: {ctype}",
                    color_discrete_map=color_map
                )
                fig.update_layout(
                    xaxis_title="Línea de Producto",
                    yaxis_title="Total Promedio de Ventas"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Solo mostrar el texto de análisis después del último gráfico
                if not first_graph:
                    st.info("[Ingresar texto de análisis aquí]")
                first_graph = False

# Sección: Gráfico 3D
elif st.session_state.seccion == "Gráfico 3D":
    st.markdown("""
    <h4 style="color: #e2e8f0; margin: 0.5rem 0; background: linear-gradient(90deg, rgba(13,17,23,0.6) 0%, rgba(27,38,59,0.6) 100%); 
               padding: 8px 15px; border-radius: 7px; border-left: 4px solid #FF3545; display:inline-block;">
        <span style="margin-left:5px;">Transacciones por Mes y Hora</span>
    </h4>
    """, unsafe_allow_html=True)
    
    df_temp = df_filtered.copy()
    if not df_temp.empty:
        df_temp['MonthNum'] = df_temp['Date'].dt.month
        df_temp['HourInt'] = df_temp['Time'].astype(str).str.split(':').str[0].astype(int)
        pivot = df_temp.groupby(['MonthNum','HourInt']).size().unstack(fill_value=0)
        if not pivot.empty:
            months = pivot.index.values
            hours = pivot.columns.values
            Z = pivot.values
            fig3d = go.Figure(data=go.Surface(x=months, y=hours, z=Z, colorscale='Viridis'))
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(
                        title='Mes',
                        tickmode='array',
                        tickvals=list(months),
                        ticktext=[calendar.month_abbr[m] for m in months]
                    ),
                    yaxis=dict(title='Hora del día'),
                    zaxis=dict(title='Número de transacciones'),
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, t=50, b=0),
                height=700
            )
            st.plotly_chart(fig3d, use_container_width=True)
            st.info("[Ingresar texto de análisis aquí]")
        else:
            st.warning("No hay suficientes datos para crear la visualización 3D con el rango seleccionado.")
    else:
        st.warning("No hay datos para el rango seleccionado.")

# Sección: Conclusiones
elif st.session_state.seccion == "Conclusiones":
    st.markdown("""
    <h2 style="margin: 0.5rem 0; color: #e2e8f0; background: linear-gradient(90deg, rgba(13,17,23,0.6) 0%, rgba(27,38,59,0.6) 100%); 
                padding: 8px 15px; border-radius: 7px; border-left: 4px solid #FF3545;">
        Conclusiones
    </h2>
    """, unsafe_allow_html=True)
    
    st.info("Esta sección está pendiente de ser completada.")

# Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 10px; padding: 10px; background-color: rgba(13,17,23,0.5); border-radius: 5px;">
    <p style="color: #94a3b8; font-size: 12px; margin: 0;">
        Dashboard Supermarket Sales | Datos: data.csv | Desarrollado con Streamlit
    </p>
</div>
""", unsafe_allow_html=True)