import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import shap

# Inject project root in system path to resolve relative imports when loading pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config

# Set page configuration
st.set_page_config(
    page_title="Predicción de Homicidios - Guayaquil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    .kpi-card {
        background-color: #fdfefe;
        border: 1px solid #e5e8e8;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a5276;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        font-weight: 500;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load pickle data
@st.cache_resource
def load_pipeline_data():
    pkl_path = os.path.join("models", "pipeline_outputs.pkl")
    if not os.path.exists(pkl_path):
        st.error("No se encontraron los datos precalculados del pipeline. Por favor, ejecute 'python src/pipeline.py' primero.")
        st.stop()
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

data = load_pipeline_data()
df = data['df']
prep_data = data['prep_data']
models = data['models']
df_metrics = data['metrics']
cv_results = data['cv_results']
bootstrap_results = data['bootstrap']
comparisons = data['comparisons']
df_importance = data['importance']
best_model_name = data['best_model']
conclusions_text = data['conclusions']
shap_explainer = data['shap_explainer']
shap_bg = data['shap_bg']
shap_values = data['shap_values']

# Navigation
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", [
    "Inicio",
    "Exploración de Datos",
    "Análisis Espacial",
    "Balance de Clases (SMOTE)",
    "Modelos Individuales",
    "Comparación de Modelos",
    "Interpretación SHAP Global",
    "Importancia de Variables",
    "Bootstrap & Incertidumbre",
    "Pruebas Estadísticas",
    "Predicción en Tiempo Real",
    "Exportación de Reportes"
])

classes = list(prep_data['label_encoder'].classes_)

# ----------------- PAGE: INICIO -----------------
if page == "Inicio":
    st.markdown('<div class="main-header">Estudio de Predicción del Mecanismo del Homicidio en Guayaquil</div>', unsafe_allow_html=True)
    
    st.markdown(
        "Este proyecto presenta una investigación científica que compara tres modelos de clasificación multiclase "
        "para predecir si un homicidio en la Zona 8 (Guayaquil, Durán, Samborondón) se llevará a cabo mediante "
        "**Arma de Fuego**, **Arma Blanca** u **Otros** mecanismos."
    )
    
    # KPIs Row
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-value">4,681</div><div class="kpi-label">Homicidios Registrados</div></div>', 
            unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-value">88.5%</div><div class="kpi-label">Uso de Arma de Fuego</div></div>', 
            unsafe_allow_html=True
        )
    with cols[2]:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-value">Nueva Prosperina</div><div class="kpi-label">Distrito Crítico (1,030 casos)</div></div>', 
            unsafe_allow_html=True
        )
    with cols[3]:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color: #27ae60;">{best_model_name}</div><div class="kpi-label">Modelo Ganador (Val. Cruzada)</div></div>', 
            unsafe_allow_html=True
        )
        
    st.markdown('<div class="sub-header">Objetivo del Estudio</div>', unsafe_allow_html=True)
    st.write(
        "El núcleo de esta investigación es determinar si la integración de variables de georreferenciación "
        "(coordenadas exactas Latitud/Longitud y agrupaciones administrativas de distrito, zona y parroquia) "
        "aporta un valor predictivo estadísticamente significativo comparado con variables puramente demográficas y temporales. "
        "Comparamos:"
    )
    st.markdown(
        "1. **Regresión Logística Multinomial**: Baseline matemático de clasificación estadística.\n"
        "2. **XGBoost Multiclase**: Modelo tradicional basado en boosting de árboles, sin variables geográficas.\n"
        "3. **G-XGBoost Espacial**: Modelo XGBoost que integra coordenadas de latitud/longitud y divisiones territoriales."
    )
    
    st.markdown('<div class="sub-header">Resumen de Datos</div>', unsafe_allow_html=True)
    st.dataframe(df.head(5))

# ----------------- PAGE: EXPLORACION DE DATOS -----------------
elif page == "Exploración de Datos":
    st.markdown('<div class="main-header">Exploración Interactiva de Datos (EDA)</div>', unsafe_allow_html=True)
    
    # Filters
    st.sidebar.subheader("Filtros de Exploración")
    year_filter = st.sidebar.multiselect("Filtrar por Año:", options=sorted(df['Anio_i'].unique()), default=sorted(df['Anio_i'].unique()))
    sex_filter = st.sidebar.multiselect("Filtrar por Sexo:", options=df['Sexo'].unique(), default=df['Sexo'].unique())
    district_filter = st.sidebar.multiselect("Filtrar por Distrito:", options=sorted(df['Distrito'].unique()), default=sorted(df['Distrito'].unique())[:5])
    
    filtered_df = df[
        df['Anio_i'].isin(year_filter) &
        df['Sexo'].isin(sex_filter) &
        df['Distrito'].isin(district_filter)
    ]
    
    st.subheader(f"Muestra de Registros Filtrados ({len(filtered_df)} filas)")
    st.dataframe(filtered_df.head(100))
    
    st.markdown('<div class="sub-header">Distribuciones y Relaciones</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            filtered_df, x="Edad", color="Mecanismo", barmode="stack", 
            title="Distribución de Edad por Mecanismo del Homicidio",
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Gender counts
        fig_sex = px.histogram(
            filtered_df, x="Sexo", color="Mecanismo", barmode="group",
            title="Distribución de Mecanismos por Sexo",
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig_sex, use_container_width=True)
        
    with col2:
        # Hour distribution
        fig_hour = px.histogram(
            filtered_df, x="Hora", color="Mecanismo", barmode="stack",
            title="Distribución de Horarios por Mecanismo del Homicidio",
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        
        # Boxplot age
        fig_box = px.box(
            filtered_df, x="Mecanismo", y="Edad", color="Mecanismo",
            title="Boxplot de Edad por Mecanismo",
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ----------------- PAGE: ANALISIS ESPACIAL -----------------
elif page == "Análisis Espacial":
    st.markdown('<div class="main-header">Análisis Espacial y Geográfico</div>', unsafe_allow_html=True)
    
    st.sidebar.subheader("Filtros Geográficos")
    year_f = st.sidebar.multiselect("Año:", options=sorted(df['Anio_i'].unique()), default=sorted(df['Anio_i'].unique()))
    sex_f = st.sidebar.multiselect("Sexo:", options=df['Sexo'].unique(), default=df['Sexo'].unique())
    mech_f = st.sidebar.multiselect("Mecanismo:", options=df['Mecanismo'].unique(), default=df['Mecanismo'].unique())
    
    geo_filtered = df[
        df['Anio_i'].isin(year_f) &
        df['Sexo'].isin(sex_f) &
        df['Mecanismo'].isin(mech_f)
    ]
    
    tab1, tab2, tab3 = st.tabs(["Mapa de Calor (Folium)", "Distribución Mapbox", "Densidad por Sectores"])
    
    with tab1:
        st.subheader("Mapa de Densidad y Concentración (Folium)")
        st.write("Visualice los clusters calientes de delincuencia armada en la urbe:")
        folium_file = os.path.join("dashboard", "folium_map.html")
        if os.path.exists(folium_file):
            with open(folium_file, "r", encoding="utf-8") as f:
                folium_html = f.read()
            st.components.v1.html(folium_html, height=600)
        else:
            st.warning("El mapa de Folium precalculado no existe. Se utilizará un mapa dinámico alternativo.")
            
    with tab2:
        st.subheader("Visualización en Plotly Mapbox")
        # Generate Mapbox scatter dynamically with filtered data
        fig_map = px.scatter_mapbox(
            geo_filtered, lat="Coord_Y", lon="Coord_X", color="Mecanismo",
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
            hover_data=["Edad", "Sexo", "Distrito", "Parroquia"],
            zoom=11, center={"lat": -2.1883, "lon": -79.9474},
            title="Distribución Geográfica Exacta de Homicidios"
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
    with tab3:
        st.subheader("Homicidios Agrupados por Parroquias y Distritos")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            dist_counts = geo_filtered['Distrito'].value_counts().reset_index()
            dist_counts.columns = ['Distrito', 'Homicidios']
            fig_bar_d = px.bar(dist_counts.head(10), x='Homicidios', y='Distrito', orientation='h', 
                               title='Top 10 Distritos por Frecuencia', color='Homicidios', color_continuous_scale='Reds')
            st.plotly_chart(fig_bar_d, use_container_width=True)
        with col_s2:
            parr_counts = geo_filtered['Parroquia'].value_counts().reset_index()
            parr_counts.columns = ['Parroquia', 'Homicidios']
            fig_bar_p = px.bar(parr_counts.head(10), x='Homicidios', y='Parroquia', orientation='h',
                               title='Top 10 Parroquias por Frecuencia', color='Homicidios', color_continuous_scale='Oranges')
            st.plotly_chart(fig_bar_p, use_container_width=True)

# ----------------- PAGE: BALANCE DE CLASES -----------------
elif page == "Balance de Clases (SMOTE)":
    st.markdown('<div class="main-header">Balanceo de Clases Mediante SMOTE</div>', unsafe_allow_html=True)
    
    st.write(
        "Debido a la naturaleza asimétrica del conjunto de datos original, donde los homicidios por Arma de Fuego "
        "representan más del 88.5% del total, entrenar modelos de aprendizaje automático directamente generaría "
        "clasificadores altamente sesgados. Para remediar esto, implementamos SMOTE (Synthetic Minority Over-sampling Technique) "
        "en los conjuntos de entrenamiento."
    )
    
    # Load comparison image or recreate dynamically
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Distribución Comparativa")
        before_counts = prep_data['before_smote_counts']
        after_counts = prep_data['after_smote_counts']
        
        b_df = pd.DataFrame(list(before_counts.items()), columns=['Mecanismo', 'Cantidad'])
        b_df['Fase'] = 'Antes de SMOTE'
        
        a_df = pd.DataFrame(list(after_counts.items()), columns=['Mecanismo', 'Cantidad'])
        a_df['Fase'] = 'Después de SMOTE (Balanceado)'
        
        comparison_df = pd.concat([b_df, a_df])
        
        fig_sm = px.bar(comparison_df, x='Mecanismo', y='Cantidad', color='Fase', barmode='group',
                        color_discrete_sequence=['#ff7f0e', '#2ca02c'], title="Clases de Entrenamiento")
        st.plotly_chart(fig_sm, use_container_width=True)
        
    with col2:
        st.subheader("Registros Totales en Entrenamiento")
        st.write("**Antes de SMOTE:**")
        for k, v in before_counts.items():
            st.write(f"- {k}: {v}")
        st.write(f"**Total:** {sum(before_counts.values())}")
        
        st.write("**Después de SMOTE:**")
        for k, v in after_counts.items():
            st.write(f"- {k}: {v}")
        st.write(f"**Total:** {sum(after_counts.values())}")

# ----------------- PAGE: MODELOS INDIVIDUALES -----------------
elif page == "Modelos Individuales":
    st.markdown('<div class="main-header">Métricas y Rendimiento Individual de Modelos</div>', unsafe_allow_html=True)
    
    selected_model = st.selectbox("Seleccione el Modelo para ver sus detalles:", [
        "Regresión Logística",
        "XGBoost",
        "G-XGBoost Espacial"
    ])
    
    # Extract row
    model_metrics = df_metrics[df_metrics['Model'] == selected_model].iloc[0]
    
    col_m1, col_m2 = st.columns([1, 2])
    
    with col_m1:
        st.subheader(f"Métricas Clave: {selected_model}")
        st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
        st.metric("Balanced Accuracy", f"{model_metrics['Balanced Accuracy']:.4f}")
        st.metric("F1-Score (Macro)", f"{model_metrics['F1 (Macro)']:.4f}")
        st.metric("ROC AUC", f"{model_metrics['ROC AUC']:.4f}")
        st.metric("Log Loss", f"{model_metrics['Log Loss']:.4f}")
        
    with col_m2:
        st.subheader("Métricas de Tiempo y Generalización")
        st.write(f"- **Tiempo de Entrenamiento:** {model_metrics['Train Time (s)']:.4f} segundos")
        st.write(f"- **Tiempo de Predicción (Test):** {model_metrics['Prediction Time (s)']:.4f} segundos")
        
        # Validation cross validation score
        cv_res = cv_results[selected_model]
        st.write(f"- **Precisión de Validación Cruzada (10-Fold):** {cv_res['mean']:.4f} (+/- {cv_res['std']:.4f})")
        st.write(f"- **Intervalo de Confianza 95% CV:** [{cv_res['ci'][0]:.4f}, {cv_res['ci'][1]:.4f}]")
        
        # Explanation of model features
        st.subheader("Variables Utilizadas")
        if selected_model == "G-XGBoost Espacial":
            st.write("Características temporales, demográficas Y geográficas:")
            st.write("`Edad`, `Hora`, `Anio_i`, `Mes_i`, `Sexo`, `Tipo_Lugar`, `Presunta_Motivacion` + `Coord_Y`, `Coord_X`, `Parroquia`, `Zona`, `Distrito`")
        else:
            st.write("Únicamente características temporales y demográficas:")
            st.write("`Edad`, `Hora`, `Anio_i`, `Mes_i`, `Sexo`, `Tipo_Lugar`, `Presunta_Motivacion`")

# ----------------- PAGE: COMPARACION DE MODELOS -----------------
elif page == "Comparación de Modelos":
    st.markdown('<div class="main-header">Comparación de Modelos Científica</div>', unsafe_allow_html=True)
    
    st.write(
        "A continuación se presenta la tabla comparativa científica que evalúa exhaustivamente el rendimiento de los tres modelos "
        "en el conjunto de datos de prueba."
    )
    
    # Show comparison table sorted from best to worst
    df_sorted = df_metrics.sort_values(by='Balanced Accuracy', ascending=False)
    st.dataframe(df_sorted.style.highlight_max(axis=0, color='#d4efdf', subset=['Accuracy', 'Balanced Accuracy', 'F1 (Macro)', 'ROC AUC'])
                          .highlight_min(axis=0, color='#f9e79f', subset=['Log Loss', 'Train Time (s)']))
                          
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Matrices de Confusión de los Modelos")
        fig_file = os.path.join("outputs", "figures", "confusion_matrices.png")
        if os.path.exists(fig_file):
            st.image(fig_file, caption="Matriz de Confusión Normal y Normalizada", use_container_width=True)
            
        st.subheader("Curvas de Calibración y Brier Scores")
        fig_cal = os.path.join("outputs", "figures", "calibration_curves.png")
        if os.path.exists(fig_cal):
            st.image(fig_cal, caption="Calibración por clase", use_container_width=True)
            
    with col_c2:
        st.subheader("Curvas ROC Multiclase (One-vs-Rest)")
        fig_roc = os.path.join("outputs", "figures", "roc_curves.png")
        if os.path.exists(fig_roc):
            st.image(fig_roc, caption="Comparación Curvas ROC", use_container_width=True)
            
        st.subheader("Curvas Precision-Recall por Clase")
        fig_pr = os.path.join("outputs", "figures", "pr_curves.png")
        if os.path.exists(fig_pr):
            st.image(fig_pr, caption="Comparación Curvas PR", use_container_width=True)

# ----------------- PAGE: INTERPRETACION SHAP GLOBAL -----------------
elif page == "Interpretación SHAP Global":
    st.markdown('<div class="main-header">Explicaciones Globales de Modelos (SHAP)</div>', unsafe_allow_html=True)
    
    st.write(
        "Utilizando valores SHAP (SHapley Additive exPlanations), descomponemos las predicciones del modelo "
        "G-XGBoost Espacial para comprender la contribución marginal de cada variable geográfica, demográfica "
        "y temporal."
    )
    
    selected_class = st.selectbox("Seleccione la clase para ver explicaciones de SHAP:", classes)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("SHAP Beeswarm Plot (Distribución de Impacto)")
        st.write(
            "Muestra cómo valores altos o bajos de cada característica aumentan o disminuyen la probabilidad "
            "de pertenecer a la clase seleccionada:"
        )
        beeswarm_img = os.path.join("outputs", "figures", f"shap_beeswarm_{selected_class.replace(' ', '_')}.png")
        if os.path.exists(beeswarm_img):
            st.image(beeswarm_img, use_container_width=True)
        else:
            st.info("Gráfico beeswarm de SHAP no encontrado.")
            
    with col_s2:
        st.subheader("SHAP Bar Plot (Importancia Promedio)")
        st.write("Cuantifica la magnitud promedio del impacto de cada variable sobre la clase:")
        bar_img = os.path.join("outputs", "figures", f"shap_bar_{selected_class.replace(' ', '_')}.png")
        if os.path.exists(bar_img):
            st.image(bar_img, use_container_width=True)
        else:
            st.info("Gráfico de barras de SHAP no encontrado.")
            
    st.markdown('<div class="sub-header">Gráficos de Dependencia de SHAP</div>', unsafe_allow_html=True)
    dep_img = os.path.join("outputs", "figures", "shap_dependence_latitude.png")
    if os.path.exists(dep_img):
        st.image(dep_img, caption="SHAP Dependence Plot para Latitud (Coord_Y)", width=600)

# ----------------- PAGE: IMPORTANCIA DE VARIABLES -----------------
elif page == "Importancia de Variables":
    st.markdown('<div class="main-header">Importancia de Variables (XGBoost)</div>', unsafe_allow_html=True)
    
    st.write(
        "Evaluamos el peso, ganancia y cobertura (Gain, Weight, Cover) del modelo G-XGBoost Espacial. "
        "La ganancia (Gain) es la métrica científica preferida, ya que indica la mejora relativa en la precisión "
        "de predicción al dividir los nodos de los árboles."
    )
    
    col_i1, col_i2 = st.columns([3, 2])
    
    with col_i1:
        st.subheader("Top 20 Variables por Ganancia (Gain)")
        fig_imp_file = os.path.join("outputs", "figures", "feature_importance.png")
        if os.path.exists(fig_imp_file):
            st.image(fig_imp_file, use_container_width=True)
            
    with col_i2:
        st.subheader("Tabla de Clasificación")
        st.dataframe(df_importance)

# ----------------- PAGE: BOOTSTRAP -----------------
elif page == "Bootstrap & Incertidumbre":
    st.markdown('<div class="main-header">Análisis de Estabilidad mediante Bootstrap</div>', unsafe_allow_html=True)
    
    st.write(
        "Ejecutamos bootstrap con 1000 iteraciones (remuestreo con reemplazo) sobre el conjunto de test para "
        "estimar empíricamente la estabilidad y los intervalos de confianza del 95% para cada métrica predictiva."
    )
    
    selected_metric = st.selectbox("Seleccione la Métrica para evaluar la distribución de bootstrap:", [
        "Accuracy", "Balanced Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"
    ])
    
    # Plotly distribution comparison
    fig_b = go.Figure()
    colors_dict = {'Regresión Logística': '#1f77b4', 'XGBoost': '#ff7f0e', 'G-XGBoost Espacial': '#2ca02c'}
    
    for name, b_res in bootstrap_results.items():
        scores = b_res[selected_metric]['scores']
        ci_l = b_res[selected_metric]['ci_lower']
        ci_u = b_res[selected_metric]['ci_upper']
        mean_v = b_res[selected_metric]['mean']
        
        fig_b.add_trace(go.Histogram(
            x=scores, name=f"{name} (IC95% = [{ci_l:.3f}, {ci_u:.3f}])",
            xbins=dict(start=0.3, end=1.0, size=0.005),
            marker_color=colors_dict[name], opacity=0.6
        ))
        
    fig_b.update_layout(
        barmode='overlay',
        title=f"Distribución de Bootstrap para {selected_metric} (1000 iteraciones)",
        xaxis_title="Score de Desempeño",
        yaxis_title="Frecuencia",
        legend_title="Modelos e Intervalos de Confianza",
        height=500
    )
    st.plotly_chart(fig_b, use_container_width=True)
    
    # CIs Table display
    st.subheader("Resumen de Intervalos de Confianza (IC95%)")
    ci_table_data = []
    for model_name, metrics_dict in bootstrap_results.items():
        for m_name, vals in metrics_dict.items():
            ci_table_data.append({
                'Modelo': model_name,
                'Métrica': m_name,
                'Media Bootstrap': f"{vals['mean']:.4f}",
                'Límite Inferior (2.5%)': f"{vals['ci_lower']:.4f}",
                'Límite Superior (97.5%)': f"{vals['ci_upper']:.4f}"
            })
    st.dataframe(pd.DataFrame(ci_table_data))

# ----------------- PAGE: PRUEBAS ESTADISTICAS -----------------
elif page == "Pruebas Estadísticas":
    st.markdown('<div class="main-header">Pruebas Estadísticas de Hipótesis</div>', unsafe_allow_html=True)
    
    st.write(
        "Para validar rigurosamente la superioridad de la incorporación de variables geográficas (G-XGBoost Espacial), "
        "comparamos las predicciones del test utilizando el **Test de McNemar** (diseñado para evaluar proporciones marginales "
        "en muestras pareadas de clasificación) y el **Test de Wilcoxon** (sobre las distribuciones de validación cruzada 10-fold)."
    )
    
    for name, comp in comparisons.items():
        st.markdown(f'<div class="sub-header">{name}</div>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            st.write(f"- **P-Valor McNemar:** {comp['mcnemar_p']:.6f}")
            st.write(f"- **Estadístico McNemar:** {comp['mcnemar_stat']:.4f}")
            st.write(f"- **P-Valor Wilcoxon (CV):** {comp['wilcoxon_p']:.6f}")
        with col_t2:
            st.info(f"**Interpretación:** {comp['interpretation']}")

# ----------------- PAGE: PREDICCION EN TIEMPO REAL -----------------
elif page == "Predicción en Tiempo Real":
    st.markdown('<div class="main-header">Predicción en Tiempo Real y SHAP Local</div>', unsafe_allow_html=True)
    
    st.write(
        "Ingrese las características del evento a continuación para predecir probabilísticamente el mecanismo "
        "del homicidio y visualizar el gráfico de explicabilidad SHAP para esta predicción particular."
    )
    
    # Load preprocessors and features
    preprocessor_spatial = prep_data['preprocessor_spatial']
    spatial_features = config.NON_SPATIAL_FEATURES + config.SPATIAL_FEATURES
    
    # Read raw unique options for fields to build nice dropdown selectors
    raw_df_spatial = prep_data['X_train_df_spatial']
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("Datos de Entrada")
        # Age
        age_in = st.slider("Edad de la víctima:", min_value=1, max_value=110, value=30)
        # Hour
        hour_in = st.slider("Hora de la infracción:", min_value=0, max_value=23, value=12)
        # Sex
        sex_in = st.selectbox("Sexo:", raw_df_spatial['Sexo'].unique())
        # Location type
        place_type_in = st.selectbox("Tipo de Lugar:", raw_df_spatial['Tipo_Lugar'].unique())
        # Presumed motive
        motive_in = st.selectbox("Presunta Motivación:", raw_df_spatial['Presunta_Motivacion'].unique())
        
        # Spatial Variables
        st.markdown("**Variables Espaciales**")
        lat_in = st.number_input("Latitud (Coord_Y):", min_value=-4.0, max_value=0.0, value=-2.1883, format="%.5f")
        lon_in = st.number_input("Longitud (Coord_X):", min_value=-81.5, max_value=-78.0, value=-79.9474, format="%.5f")
        
        parroquia_in = st.selectbox("Parroquia (Circuito):", sorted(raw_df_spatial['Parroquia'].unique()))
        distrito_in = st.selectbox("Distrito:", sorted(raw_df_spatial['Distrito'].unique()))
        zona_in = st.selectbox("Zona:", sorted(raw_df_spatial['Zona'].unique()))
        
        # Temporal variables
        st.markdown("**Variables Temporales Adicionales**")
        anio_in = st.selectbox("Año:", sorted(raw_df_spatial['Anio_i'].unique()))
        mes_in = st.selectbox("Mes:", sorted(raw_df_spatial['Mes_i'].unique()))
        
        predict_clicked = st.button("Predecir Mecanismo", type="primary")
        
    with col_p2:
        st.subheader("Resultados de Predicción")
        if predict_clicked:
            # Build input dictionary
            input_dict = {
                'Edad': age_in,
                'Hora': hour_in,
                'Sexo': sex_in,
                'Tipo_Lugar': place_type_in,
                'Presunta_Motivacion': motive_in,
                'Anio_i': anio_in,
                'Mes_i': mes_in,
                'Coord_Y': lat_in,
                'Coord_X': lon_in,
                'Parroquia': parroquia_in,
                'Distrito': distrito_in,
                'Zona': zona_in
            }
            
            # Map into DataFrame
            df_input = pd.DataFrame([input_dict])
            
            # Preprocess using spatial ColumnTransformer
            # Need to transform
            try:
                x_trans = preprocessor_spatial.transform(df_input)
                
                # Predict probabilities
                spatial_model = models['G-XGBoost Espacial']
                probs = spatial_model.predict_proba(x_trans)[0]
                
                predicted_class_idx = np.argmax(probs)
                predicted_class = classes[predicted_class_idx]
                
                st.success(f"**Mecanismo Predicho:** {predicted_class}")
                
                # Show probabilities bar chart
                prob_df = pd.DataFrame({
                    'Mecanismo': classes,
                    'Probabilidad': probs
                })
                fig_prob = px.bar(prob_df, x='Probabilidad', y='Mecanismo', orientation='h', 
                                  color='Mecanismo', color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
                                  xlim=[0, 1])
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # SHAP local waterfall plot
                st.subheader(f"Explicación SHAP Local ({predicted_class})")
                
                # Compute SHAP value for this single point
                expl = shap_explainer(x_trans)
                # expl.shape will be (1, n_features, n_classes) or list of length n_classes with (1, n_features)
                
                # Get spatial feature names
                feat_names = prep_data['spatial_feature_names']
                
                is_list = isinstance(expl, list) or isinstance(data['shap_values'], list)
                
                # Extract SHAP values for the predicted class
                if is_list:
                    # In list format, explainer(x_trans) returns a list of Explanations or similar. 
                    # If it's a list, we can compute it using shap_explainer.shap_values(x_trans)
                    s_vals = shap_explainer.shap_values(x_trans)
                    sh_class = s_vals[predicted_class_idx][0]
                    base_v = shap_explainer.expected_value[predicted_class_idx]
                else:
                    # Array representation
                    sh_class = expl.values[0, :, predicted_class_idx]
                    base_v = expl.base_values[0, predicted_class_idx]
                
                # Create Explanation
                single_expl = shap.Explanation(
                    values=sh_class,
                    base_values=base_v,
                    data=x_trans[0],
                    feature_names=feat_names
                )
                
                fig_sh, ax_sh = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(single_expl, max_display=10, show=False)
                plt.title(f"Impacto de Variables en Predicción de {predicted_class}")
                plt.tight_layout()
                st.pyplot(fig_sh)
                plt.close()
                
                st.write(
                    "**Interpretación:** Las barras rojas empujan la predicción hacia arriba (mayor probabilidad "
                    "del mecanismo), mientras que las barras azules la jalan hacia abajo."
                )
            except Exception as e:
                st.error(f"Error durante el preprocesamiento/predicción: {e}")
        else:
            st.info("Presione 'Predecir Mecanismo' para generar predicción y SHAP.")

# ----------------- PAGE: EXPORTACION DE REPORTES -----------------
elif page == "Exportación de Reportes":
    st.markdown('<div class="main-header">Exportar Resultados y Reportes Científicos</div>', unsafe_allow_html=True)
    
    st.write(
        "Puede descargar las tablas de rendimiento, el artículo científico pre-formateado en PDF, "
        "o el manuscrito académico en Word (.docx) para su revisión."
    )
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("Documentos y Artículos Académicos")
        
        # Word report download
        word_path = os.path.join("reports", "Articulo_Homicidios_Guayaquil.docx")
        if os.path.exists(word_path):
            with open(word_path, "rb") as f:
                word_bytes = f.read()
            st.download_button(
                label="Descargar Manuscrito Científico (Word .docx)",
                data=word_bytes,
                file_name="Manuscrito_Homicidios_Guayaquil.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="word_dl"
            )
        else:
            st.warning("Word report not found.")
            
        # PDF report download
        pdf_path = os.path.join("reports", "Articulo_Homicidios_Guayaquil.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Descargar Artículo Formateado (PDF)",
                data=pdf_bytes,
                file_name="Articulo_Homicidios_Guayaquil.pdf",
                mime="application/pdf",
                key="pdf_dl"
            )
        else:
            st.warning("PDF report not found.")
            
    with col_d2:
        st.subheader("Métricas y Tablas CSV/Excel")
        
        # CSV Metrics Download
        csv_metrics = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Métricas Comparativas (CSV)",
            data=csv_metrics,
            file_name="metricas_modelos.csv",
            mime="text/csv",
            key="csv_dl"
        )
        
        # Excel Metrics Download
        excel_path = os.path.join("outputs", "metrics", "model_comparison_metrics.xlsx")
        if os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                excel_bytes = f.read()
            st.download_button(
                label="Descargar Métricas de Modelos (Excel .xlsx)",
                data=excel_bytes,
                file_name="metricas_modelos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="xlsx_dl"
            )
            
        # Cleaned dataset download
        processed_csv_path = os.path.join("data", "processed", "homicidios_clean.csv")
        if os.path.exists(processed_csv_path):
            with open(processed_csv_path, "rb") as f:
                processed_bytes = f.read()
            st.download_button(
                label="Descargar Base de Datos Preprocesada (CSV)",
                data=processed_bytes,
                file_name="homicidios_limpios.csv",
                mime="text/csv",
                key="clean_dl"
            )
