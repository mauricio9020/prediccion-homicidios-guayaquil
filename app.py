"""
Streamlit Interactive Scientific Dashboard for Guayaquil Homicide Prediction.

Scientifically Addresses Reviewer Comments:
- Point #6: Uses updated terminology ('XGBoost con Covariables Geográficas').
- Point #7: Promotes Macro F1 as the primary scientific evaluation metric.
- Point #5: Displays automated Feature Ablation Study results.
- Point #8: Displays Reliability Diagrams & Brier Scores per class.
- Point #11: Incorporates Spatial Econometrics (Moran's I, LISA Clusters, Spatial Error Maps).
- Point #18: Preserves interactive user experience without breaking any Streamlit feature.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

# Inject project root in system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config

# Set page config
st.set_page_config(
    page_title="Predicción de Homicidios - Guayaquil (GeoAI Q1)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 15px;
    }
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 15px;
        margin-bottom: 10px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    .kpi-card {
        background-color: #fdfefe;
        border: 1px solid #e5e8e8;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
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


@st.cache_resource
def load_pipeline_data():
    pkl_path = os.path.join("models", "pipeline_outputs.pkl")
    if not os.path.exists(pkl_path):
        st.error("No se encontraron los datos del pipeline. Por favor ejecute 'python src/pipeline.py' primero.")
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
df_ablation = data.get('ablation', pd.DataFrame())
moran_res = data.get('moran', {})
lisa_df = data.get('lisa', pd.DataFrame())
brier_scores = data.get('brier', {})
best_model_name = data['best_model']
conclusions_text = data['conclusions']
shap_explainer = data['shap_explainer']
shap_trans = data.get('shap_trans', None)
shap_values = data['shap_values']

classes = list(prep_data['label_encoder'].classes_)

# Sidebar navigation
st.sidebar.title("Navegación Metodológica")
page = st.sidebar.radio("Ir a:", [
    "Inicio",
    "Exploración de Datos (EDA)",
    "Econometría Espacial (Moran & LISA)",
    "Validación Espacial & Pipeline",
    "Estudio de Ablación",
    "Modelos Individuales",
    "Comparación de Modelos (Macro F1)",
    "Calibración & Brier Score",
    "Interpretación SHAP Agrupada",
    "Bootstrap & Incertidumbre",
    "Pruebas Estadísticas Pareadas",
    "Predicción en Tiempo Real",
    "Exportación de Reportes Q1"
])

# ----------------- PAGE: INICIO -----------------
if page == "Inicio":
    st.markdown('<div class="main-header">Estudio de Predicción del Mecanismo del Homicidio en Guayaquil</div>', unsafe_allow_html=True)
    
    st.markdown(
        "Este proyecto presenta una investigación científica publicada con estándares de revistas internacionales Q1. "
        "Compara modelos de clasificación multiclase para predecir el mecanismo de homicidio (**Arma de Fuego**, **Arma Blanca**, **Otros**) "
        "evaluando cuantitativamente el aporte de las covariables geográficas sin fuga de datos."
    )
    
    # KPIs Row
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Homicidios Registrados</div></div>', unsafe_allow_html=True)
    with cols[1]:
        firearm_pct = (df[config.TARGET_COL_CLEAN] == 'Arma de Fuego').mean() * 100
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{firearm_pct:.1f}%</div><div class="kpi-label">Uso de Arma de Fuego</div></div>', unsafe_allow_html=True)
    with cols[2]:
        moran_val = moran_res.get('morans_i', 0.0)
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{moran_val:.3f}</div><div class="kpi-label">Moran\'s I (p < 0.05)</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #27ae60;">{best_model_name}</div><div class="kpi-label">Modelo Ganador (Macro F1)</div></div>', unsafe_allow_html=True)
        
    st.markdown('<div class="sub-header">Innovaciones Metodológicas Atendidas (#1 - #20)</div>', unsafe_allow_html=True)
    st.markdown(
        "- **Data Leakage Eliminado:** Encapsulamiento en `imblearn.pipeline.Pipeline`.\n"
        "- **Validación Espacial:** `GroupKFold` por Parroquia / Distrito en lugar de partición aleatoria tradicional.\n"
        "- **Nested Cross-Validation:** Optimización bucle interno, evaluación en bucle externo.\n"
        "- **Macro F1:** Métrica principal para evaluar clases desbalanceadas.\n"
        "- **XGBoost con Covariables Geográficas:** Denominación formal rigurosa."
    )
    st.dataframe(df.head(5))

# ----------------- PAGE: EDA -----------------
elif page == "Exploración de Datos (EDA)":
    st.markdown('<div class="main-header">Exploración de Datos e Infracciones</div>', unsafe_allow_html=True)
    
    st.sidebar.subheader("Filtros")
    year_filter = st.sidebar.multiselect("Año:", options=sorted(df['Anio_i'].unique()), default=sorted(df['Anio_i'].unique()))
    sex_filter = st.sidebar.multiselect("Sexo:", options=df['Sexo'].unique(), default=df['Sexo'].unique())
    
    filtered_df = df[df['Anio_i'].isin(year_filter) & df['Sexo'].isin(sex_filter)]
    
    col1, col2 = st.columns(2)
    with col1:
        fig_age = px.histogram(filtered_df, x="Edad", color="Mecanismo", barmode="stack", title="Distribución de Edad por Mecanismo")
        st.plotly_chart(fig_age, use_container_width=True)
    with col2:
        fig_hour = px.histogram(filtered_df, x="Hora", color="Mecanismo", barmode="stack", title="Distribución de Horas por Mecanismo")
        st.plotly_chart(fig_hour, use_container_width=True)

# ----------------- PAGE: ECONOMETRIA ESPACIAL -----------------
elif page == "Econometría Espacial (Moran & LISA)":
    st.markdown('<div class="main-header">Econometría Espacial y Autocorrelación</div>', unsafe_allow_html=True)
    
    st.subheader("1. Estadístico Moran's I Global")
    st.write(
        f"- **Moran's I:** `{moran_res.get('morans_i', 0.0):.4f}`\n"
        f"- **Valor Esperado:** `{moran_res.get('expected_i', 0.0):.4f}`\n"
        f"- **Z-Score:** `{moran_res.get('z_score', 0.0):.4f}`\n"
        f"- **P-Valor:** `{moran_res.get('p_value', 1.0):.6f}`"
    )
    if moran_res.get('p_value', 1.0) < 0.05:
        st.success("Existe una autocorrelación espacial positiva altamente significativa (p < 0.05). Los homicidios se agrupan en conglomerados geográficos.")
        
    st.subheader("2. Mapas de Clusters LISA (Local Moran's I)")
    lisa_img = os.path.join("outputs", "figures", "lisa_cluster_map.png")
    if os.path.exists(lisa_img):
        st.image(lisa_img, caption="Clusters LISA por Parroquia (Hotspots High-High y Coldspots Low-Low)", use_container_width=True)
        
    st.subheader("3. Mapa de Errores del Modelo Espacial")
    err_img = os.path.join("outputs", "figures", "spatial_error_map.png")
    if os.path.exists(err_img):
        st.image(err_img, caption="Distribución Geográfica de Errores de Clasificación", use_container_width=True)

# ----------------- PAGE: VALIDACION ESPACIAL -----------------
elif page == "Validación Espacial & Pipeline":
    st.markdown('<div class="main-header">Estrategia de Validación Espacial y Pipeline</div>', unsafe_allow_html=True)
    st.write(
        "Para impedir que datos de entrenamiento y prueba compartan el mismo entorno geográfico (lo que inflaría falsamente las métricas), "
        "implementamos **GroupKFold** y **GroupShuffleSplit** utilizando Parroquias y Distritos administrativos como unidades de agrupación."
    )
    folium_file = os.path.join("dashboard", "folium_map.html")
    if os.path.exists(folium_file):
        with open(folium_file, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=550)

# ----------------- PAGE: ESTUDIO DE ABLACION -----------------
elif page == "Estudio de Ablación":
    st.markdown('<div class="main-header">Estudio de Ablación de Características</div>', unsafe_allow_html=True)
    st.write(
        "El estudio de ablación evalúa el impacto incremental de incorporar secuencialmente cada grupo de variables. "
        "Muestra cuánto aporta cada bloque funcional sobre el modelo base."
    )
    if not df_ablation.empty:
        st.dataframe(df_ablation)
        fig_abl = px.line(df_ablation, x='Grupo_Variables', y=['Macro F1', 'Balanced Accuracy'], markers=True, title="Progreso de Métricas en el Estudio de Ablación")
        st.plotly_chart(fig_abl, use_container_width=True)

# ----------------- PAGE: MODELOS INDIVIDUALES -----------------
elif page == "Modelos Individuales":
    st.markdown('<div class="main-header">Detalle de Modelos Individuales</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Seleccione el Modelo:", df_metrics['Modelo'].unique())
    m_row = df_metrics[df_metrics['Modelo'] == selected_model].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Macro F1 (Principal)", f"{m_row['Macro F1 (Principal)']:.4f}")
        st.metric("Balanced Accuracy", f"{m_row['Balanced Accuracy']:.4f}")
        st.metric("MCC Multiclase", f"{m_row['MCC Multiclase']:.4f}")
    with col2:
        st.metric("ROC-AUC (OvR)", f"{m_row['ROC-AUC (OvR)']:.4f}")
        st.metric("Accuracy (Secundario)", f"{m_row['Accuracy (Secundario)']:.4f}")

# ----------------- PAGE: COMPARACION DE MODELOS -----------------
elif page == "Comparación de Modelos (Macro F1)":
    st.markdown('<div class="main-header">Comparación Metodológica de Modelos</div>', unsafe_allow_html=True)
    st.write("Tabla comparativa ordenada jerárquicamente priorizando el **Macro F1** (Métrica Principal):")
    
    st.dataframe(
        df_metrics.style.highlight_max(axis=0, color='#d4efdf', subset=['Macro F1 (Principal)', 'Balanced Accuracy', 'MCC Multiclase', 'ROC-AUC (OvR)'])
    )
    
    col1, col2 = st.columns(2)
    with col1:
        cm_img = os.path.join("outputs", "figures", "confusion_matrices.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Matrices de Confusión", use_container_width=True)
    with col2:
        cal_img = os.path.join("outputs", "figures", "calibration_curves.png")
        if os.path.exists(cal_img):
            st.image(cal_img, caption="Curvas de Calibración", use_container_width=True)

# ----------------- PAGE: CALIBRACION & BRIER -----------------
elif page == "Calibración & Brier Score":
    st.markdown('<div class="main-header">Calibración de Probabilidades & Brier Score</div>', unsafe_allow_html=True)
    st.write("Evaluación de la confiabilidad de las probabilidades predichas por clase:")
    cal_img = os.path.join("outputs", "figures", "calibration_curves.png")
    if os.path.exists(cal_img):
        st.image(cal_img, use_container_width=True)
    if brier_scores:
        st.subheader("Brier Score por Clase (Valores menores indican mejor calibración)")
        st.dataframe(pd.DataFrame(brier_scores).T)

# ----------------- PAGE: SHAP AGRUPADO -----------------
elif page == "Interpretación SHAP Agrupada":
    st.markdown('<div class="main-header">Interpretabilidad SHAP Agrupada (TreeSHAP)</div>', unsafe_allow_html=True)
    st.warning("Importante: Los valores SHAP representan contribución marginal al modelo y atribución de características. NUNCA deben interpretarse como causalidad.")
    
    selected_c = st.selectbox("Seleccione la Clase:", classes)
    c_img = os.path.join("outputs", "figures", f"shap_beeswarm_{selected_c.replace(' ', '_')}.png")
    if os.path.exists(c_img):
        st.image(c_img, caption=f"SHAP Beeswarm Agrupado - {selected_c}", use_container_width=True)

# ----------------- PAGE: BOOTSTRAP -----------------
elif page == "Bootstrap & Incertidumbre":
    st.markdown('<div class="main-header">Estabilidad y Bootstrap (1000 Iteraciones)</div>', unsafe_allow_html=True)
    st.write("Estimación empírica de intervalos de confianza del 95% y sesgo en el conjunto de prueba:")
    
    table_b = []
    for m_name, metrics_dict in bootstrap_results.items():
        for metric_k, vals in metrics_dict.items():
            table_b.append({
                'Modelo': m_name,
                'Métrica': metric_k,
                'Point Estimate': f"{vals['point_estimate']:.4f}",
                'Media Bootstrap': f"{vals['mean']:.4f}",
                'Mediana': f"{vals['median']:.4f}",
                'Sesgo': f"{vals['bias']:.4f}",
                'IC 95% Inferior': f"{vals['ci_lower']:.4f}",
                'IC 95% Superior': f"{vals['ci_upper']:.4f}"
            })
    st.dataframe(pd.DataFrame(table_b))

# ----------------- PAGE: PRUEBAS ESTADISTICAS -----------------
elif page == "Pruebas Estadísticas Pareadas":
    st.markdown('<div class="main-header">Pruebas Estadísticas (McNemar & Wilcoxon)</div>', unsafe_allow_html=True)
    for pair_name, comp in comparisons.items():
        st.subheader(pair_name)
        st.write(f"- **McNemar p-value:** `{comp['mcnemar_p']:.6f}`")
        st.write(f"- **Wilcoxon p-value:** `{comp['wilcoxon_p']:.6f}`")
        st.info(comp['interpretation'])

# ----------------- PAGE: PREDICCION -----------------
elif page == "Predicción en Tiempo Real":
    st.markdown('<div class="main-header">Predicción en Tiempo Real</div>', unsafe_allow_html=True)
    raw_df_spatial = prep_data['X_train_df_spatial']
    
    col1, col2 = st.columns(2)
    with col1:
        age_in = st.slider("Edad de la víctima:", 1, 100, 30)
        hour_in = st.slider("Hora:", 0, 23, 14)
        sex_in = st.selectbox("Sexo:", raw_df_spatial['Sexo'].unique())
        place_in = st.selectbox("Tipo de Lugar:", raw_df_spatial['Tipo_Lugar'].unique())
        motive_in = st.selectbox("Presunta Motivación:", raw_df_spatial['Presunta_Motivacion'].unique())
        lat_in = st.number_input("Latitud (Coord_Y):", value=-2.1883, format="%.5f")
        lon_in = st.number_input("Longitud (Coord_X):", value=-79.9474, format="%.5f")
        parr_in = st.selectbox("Parroquia:", sorted(raw_df_spatial['Parroquia'].unique()))
        dist_in = st.selectbox("Distrito:", sorted(raw_df_spatial['Distrito'].unique()))
        zona_in = st.selectbox("Zona:", sorted(raw_df_spatial['Zona'].unique()))
        anio_in = st.selectbox("Año:", [2024, 2025, 2026])
        mes_in = st.selectbox("Mes:", list(range(1, 13)))
        
        btn = st.button("Predecir Mecanismo", type="primary")
        
    with col2:
        if btn:
            input_df = pd.DataFrame([{
                'Edad': age_in, 'Hora': hour_in, 'Sexo': sex_in, 'Tipo_Lugar': place_in,
                'Presunta_Motivacion': motive_in, 'Anio_i': anio_in, 'Mes_i': mes_in,
                'Coord_Y': lat_in, 'Coord_X': lon_in, 'Parroquia': parr_in,
                'Distrito': dist_in, 'Zona': zona_in
            }])
            
            xgb_model = models[config.MODEL_NAMES['XGB_SPATIAL']]
            probs = xgb_model.predict_proba(input_df)[0]
            pred_idx = np.argmax(probs)
            pred_class = classes[pred_idx]
            
            st.success(f"**Mecanismo Predicho:** {pred_class}")
            st.plotly_chart(px.bar(x=classes, y=probs, labels={'x':'Mecanismo','y':'Probabilidad'}, title="Probabilidades Predichas"), use_container_width=True)

# ----------------- PAGE: EXPORTACION -----------------
elif page == "Exportación de Reportes Q1":
    st.markdown('<div class="main-header">Exportación de Reportes y Tablas Científicas</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        word_path = os.path.join("reports", "Articulo_Homicidios_Guayaquil.docx")
        if os.path.exists(word_path):
            with open(word_path, "rb") as f:
                st.download_button("Descargar Manuscrito (Word .docx)", f, "Manuscrito_Guayaquil.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        pdf_path = os.path.join("reports", "Articulo_Homicidios_Guayaquil.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("Descargar Artículo (PDF)", f, "Articulo_Guayaquil.pdf", "application/pdf")
    with col2:
        csv_metrics = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Métricas (CSV)", csv_metrics, "metricas.csv", "text/csv")
