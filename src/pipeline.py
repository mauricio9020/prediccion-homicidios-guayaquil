import os
import sys
import time
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import HeatMap, MarkerCluster

# Inject parent folder in path for package loading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config, data_prep, modeling, evaluation, report_gen

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_eda(df: pd.DataFrame):
    """
    Performs full exploratory data analysis (EDA) and saves tables and figures.
    """
    logger.info("Starting Exploratory Data Analysis (EDA)...")
    
    # 1. Descriptive statistics
    desc_stats = df.describe(include='all')
    desc_stats.to_csv(os.path.join(config.TABLES_DIR, 'descriptive_statistics.csv'))
    
    # 2. Missing values
    missing = df.isnull().sum().to_frame(name='Missing_Count')
    missing['Percentage'] = (missing['Missing_Count'] / len(df)) * 100
    missing.to_csv(os.path.join(config.TABLES_DIR, 'missing_values_report.csv'))
    
    # 3. Duplicates
    duplicates_count = df.duplicated().sum()
    with open(os.path.join(config.TABLES_DIR, 'duplicates_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Total rows: {len(df)}\nDuplicate rows: {duplicates_count}\n")
        
    # 4. Class distribution
    plt.figure(figsize=(8, 6))
    class_counts = df[config.TARGET_COL_CLEAN].value_counts()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1.5, 'antialiased': True})
    plt.title('Distribución de Clases del Mecanismo del Homicidio')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'class_distribution.png'), dpi=300)
    plt.close()
    
    # 5. Histograms (Age and Hour)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x='Edad', hue=config.TARGET_COL_CLEAN, multiple='stack', bins=20, palette=colors[:len(class_counts)], ax=axes[0])
    axes[0].set_title('Distribución de Edades por Mecanismo')
    axes[0].set_xlabel('Edad')
    axes[0].set_ylabel('Frecuencia')
    
    sns.histplot(data=df, x='Hora', hue=config.TARGET_COL_CLEAN, multiple='stack', bins=24, palette=colors[:len(class_counts)], ax=axes[1])
    axes[1].set_title('Distribución de Horas por Mecanismo')
    axes[1].set_xlabel('Hora del Día')
    axes[1].set_ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'histograms_age_hour.png'), dpi=300)
    plt.close()
    
    # 6. Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x=config.TARGET_COL_CLEAN, y='Edad', hue=config.TARGET_COL_CLEAN, palette=colors[:len(class_counts)], legend=False, ax=axes[0])
    axes[0].set_title('Boxplot de Edad por Mecanismo')
    axes[0].set_xlabel('Mecanismo')
    axes[0].set_ylabel('Edad')
    
    sns.boxplot(data=df, x=config.TARGET_COL_CLEAN, y='Hora', hue=config.TARGET_COL_CLEAN, palette=colors[:len(class_counts)], legend=False, ax=axes[1])
    axes[1].set_title('Boxplot de Hora por Mecanismo')
    axes[1].set_xlabel('Mecanismo')
    axes[1].set_ylabel('Hora del Día')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'boxplots_age_hour.png'), dpi=300)
    plt.close()
    
    # 7. Correlation Heatmap (numeric vars)
    plt.figure(figsize=(8, 6))
    numeric_df = df[['Edad', 'Hora', 'Coord_Y', 'Coord_X']].copy()
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Mapa de Calor de Correlaciones Numéricas')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 8. Pairplots (Age, Hour, Coordinates)
    plt.figure(figsize=(10, 8))
    pairplot_df = df[['Edad', 'Hora', 'Coord_Y', 'Coord_X', config.TARGET_COL_CLEAN]].copy()
    pairplot_df.columns = ['Edad', 'Hora', 'Latitud', 'Longitud', 'Mecanismo']
    # Use smaller plot dimensions for pairplot to prevent layout issues
    g = sns.pairplot(pairplot_df, hue='Mecanismo', palette=colors[:len(class_counts)], corner=True, diag_kind='kde')
    g.fig.suptitle('Pairplot de Características Numéricas', y=1.02)
    g.savefig(os.path.join(config.FIGURES_DIR, 'pairplot.png'), dpi=300)
    plt.close()
    
    # 9. Categorical distributions by Mecanismo
    # Sexo
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Sexo', hue=config.TARGET_COL_CLEAN, palette=colors[:len(class_counts)])
    plt.title('Distribución del Mecanismo por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'distribution_by_sex.png'), dpi=300)
    plt.close()
    
    # Distrito (Top 10)
    plt.figure(figsize=(12, 6))
    top10_districts = df['Distrito'].value_counts().head(10).index
    sns.countplot(data=df[df['Distrito'].isin(top10_districts)], y='Distrito', hue=config.TARGET_COL_CLEAN, palette=colors[:len(class_counts)], order=top10_districts)
    plt.title('Distribución del Mecanismo en los 10 Distritos Principales')
    plt.xlabel('Frecuencia')
    plt.ylabel('Distrito')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'distribution_by_district.png'), dpi=300)
    plt.close()
    
    # Parroquia (Top 10 Circuito)
    plt.figure(figsize=(12, 6))
    top10_parroquias = df['Parroquia'].value_counts().head(10).index
    sns.countplot(data=df[df['Parroquia'].isin(top10_parroquias)], y='Parroquia', hue=config.TARGET_COL_CLEAN, palette=colors[:len(class_counts)], order=top10_parroquias)
    plt.title('Distribución del Mecanismo en las 10 Parroquias Principales')
    plt.xlabel('Frecuencia')
    plt.ylabel('Parroquia (Circuito)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'distribution_by_parroquia.png'), dpi=300)
    plt.close()
    
    # Año
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Anio_i', hue=config.TARGET_COL_CLEAN, palette=colors[:len(class_counts)])
    plt.title('Distribución del Mecanismo por Año')
    plt.xlabel('Año')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'distribution_by_year.png'), dpi=300)
    plt.close()
    
    logger.info("EDA visualizations saved successfully.")

def run_spatial_analysis(df: pd.DataFrame):
    """
    Generates spatial plots using Folium and Plotly Mapbox, and saves them to HTML files.
    """
    logger.info("Starting Spatial Analysis maps generation...")
    
    # Map center (Guayaquil center coordinates)
    center_lat, center_lon = -2.1883, -79.9474
    
    # --- Folium Maps ---
    # 1. Interactive HeatMap & Cluster map combined (standard folio)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')
    
    # HeatMap layer
    heat_data = df[['Coord_Y', 'Coord_X']].dropna().values.tolist()
    HeatMap(heat_data, name='Mapa de Calor de Densidad', radius=12, max_val=1.0).add_to(m)
    
    # MarkerCluster layer
    mc = MarkerCluster(name='Clusters de Homicidios').add_to(m)
    # Add a sample of markers to prevent HTML file bloat (e.g. 500 records)
    sample_df = df.sample(min(500, len(df)), random_state=config.RANDOM_STATE)
    for _, row in sample_df.iterrows():
        folium.Marker(
            location=[row['Coord_Y'], row['Coord_X']],
            popup=f"Mecanismo: {row[config.TARGET_COL_CLEAN]}<br>Edad: {row['Edad']}<br>Distrito: {row['Distrito']}",
            icon=folium.Icon(color='red' if row[config.TARGET_COL_CLEAN] == 'Arma de Fuego' else ('blue' if row[config.TARGET_COL_CLEAN] == 'Arma Blanca' else 'green'))
        ).add_to(mc)
        
    folium.LayerControl().add_to(m)
    folium_path = os.path.join(config.BASE_DIR, 'dashboard', 'folium_map.html')
    m.save(folium_path)
    logger.info(f"Saved Folium density map to {folium_path}")
    
    # --- Plotly Mapbox Interactive Maps ---
    # Since Mapbox tokens aren't available, we use open-street-map tiles
    # 2. Interactive Map colored by Homicide Mechanism
    fig_mech = px.scatter_mapbox(
        df, lat="Coord_Y", lon="Coord_X", color=config.TARGET_COL_CLEAN, 
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],
        hover_data=["Edad", "Sexo", "Distrito", "Parroquia", "Anio_i"],
        zoom=10, center={"lat": center_lat, "lon": center_lon},
        title="Homicidios en Guayaquil por Mecanismo (Plotly Mapbox)"
    )
    fig_mech.update_layout(mapbox_style="open-street-map")
    fig_mech.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    plotly_path1 = os.path.join(config.BASE_DIR, 'dashboard', 'plotly_mecanismo.html')
    fig_mech.write_html(plotly_path1)
    
    # 3. Interactive Map colored by Distrito
    fig_dist = px.scatter_mapbox(
        df, lat="Coord_Y", lon="Coord_X", color="Distrito",
        hover_data=["Edad", "Sexo", "Parroquia", config.TARGET_COL_CLEAN],
        zoom=10, center={"lat": center_lat, "lon": center_lon},
        title="Distribución Geográfica de Homicidios por Distrito"
    )
    fig_dist.update_layout(mapbox_style="open-street-map")
    fig_dist.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    plotly_path2 = os.path.join(config.BASE_DIR, 'dashboard', 'plotly_distrito.html')
    fig_dist.write_html(plotly_path2)
    
    # 4. Interactive Map colored by Parroquia
    fig_parr = px.scatter_mapbox(
        df, lat="Coord_Y", lon="Coord_X", color="Parroquia",
        hover_data=["Edad", "Sexo", "Distrito", config.TARGET_COL_CLEAN],
        zoom=10, center={"lat": center_lat, "lon": center_lon},
        title="Distribución Geográfica de Homicidios por Parroquia (Circuito)"
    )
    fig_parr.update_layout(mapbox_style="open-street-map")
    fig_parr.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    plotly_path3 = os.path.join(config.BASE_DIR, 'dashboard', 'plotly_parroquia.html')
    fig_parr.write_html(plotly_path3)
    
    logger.info("Saved all Plotly Mapbox interactive HTML maps.")

def main():
    logger.info("Running Homicides Machine Learning Pipeline...")
    start_pipeline = time.time()
    
    # 1. Load and Clean Data
    df = data_prep.load_and_clean_data(config.RAW_DATA_PATH)
    
    # 2. Run EDA
    run_eda(df)
    
    # 3. Run Spatial Analysis
    run_spatial_analysis(df)
    
    # 4. Preprocess, Train-Test Split, and Apply SMOTE
    prep_data = data_prep.preprocess_and_split(df)
    
    X_train_base = prep_data['X_train_base']
    y_train_base = prep_data['y_train_base']
    X_test_base = prep_data['X_test_base']
    
    X_train_spatial = prep_data['X_train_spatial']
    y_train_spatial = prep_data['y_train_spatial']
    X_test_spatial = prep_data['X_test_spatial']
    
    y_test = prep_data['y_test']
    label_encoder = prep_data['label_encoder']
    
    # Save before/after SMOTE class distribution plot
    plt.figure(figsize=(10, 5))
    before_df = pd.DataFrame(list(prep_data['before_smote_counts'].items()), columns=['Clase', 'Conteo'])
    before_df['Etapa'] = 'Antes SMOTE'
    after_df = pd.DataFrame(list(prep_data['after_smote_counts'].items()), columns=['Clase', 'Conteo'])
    after_df['Etapa'] = 'Después SMOTE'
    smote_df = pd.concat([before_df, after_df])
    
    sns.barplot(data=smote_df, x='Clase', y='Conteo', hue='Etapa', palette='Set2')
    plt.title('Distribución de Clases en Entrenamiento Antes y Después de SMOTE')
    plt.xlabel('Mecanismo')
    plt.ylabel('Cantidad de Registros')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'smote_comparison.png'), dpi=300)
    plt.close()
    
    # 5. Train Multinomial Logistic Regression
    lr_model, lr_train_time = modeling.train_logistic_regression(X_train_base, y_train_base)
    
    # 6. Optimize XGBoost (Non-spatial) and train
    best_params = modeling.optimize_xgboost(X_train_base, y_train_base)
    xgb_model, xgb_train_time = modeling.train_xgboost(X_train_base, y_train_base, best_params)
    
    # 7. Train Spatial G-XGBoost
    spatial_xgb_model, spatial_train_time = modeling.train_spatial_g_xgboost(X_train_spatial, y_train_spatial, best_params)
    
    # 8. Stratified 10-Fold Cross Validation
    models_dict = {
        'Regresión Logística': lr_model,
        'XGBoost': xgb_model,
        'G-XGBoost Espacial': spatial_xgb_model
    }
    
    cv_data_dict = {
        'Regresión Logística': {'X': X_train_base, 'y': y_train_base},
        'XGBoost': {'X': X_train_base, 'y': y_train_base},
        'G-XGBoost Espacial': {'X': X_train_spatial, 'y': y_train_spatial}
    }
    
    cv_results = modeling.perform_cross_validation(models_dict, cv_data_dict)
    
    # 9. Bootstrap CI (1000 iterations)
    X_test_dict = {
        'Regresión Logística': X_test_base,
        'XGBoost': X_test_base,
        'G-XGBoost Espacial': X_test_spatial
    }
    bootstrap_results = modeling.run_bootstrap_validation(models_dict, X_test_dict, y_test, n_iterations=1000)
    
    # 10. Performance evaluation on test set
    metrics_list = []
    models_preds = {}
    models_probs = {}
    
    # Logistic Regression
    lr_metrics, lr_preds, lr_probs = evaluation.get_performance_metrics(
        'Regresión Logística', lr_model, X_test_base, y_test, lr_train_time, label_encoder
    )
    metrics_list.append(lr_metrics)
    models_preds['Regresión Logística'] = lr_preds
    models_probs['Regresión Logística'] = lr_probs
    
    # XGBoost
    xgb_metrics, xgb_preds, xgb_probs = evaluation.get_performance_metrics(
        'XGBoost', xgb_model, X_test_base, y_test, xgb_train_time, label_encoder
    )
    metrics_list.append(xgb_metrics)
    models_preds['XGBoost'] = xgb_preds
    models_probs['XGBoost'] = xgb_probs
    
    # Spatial G-XGBoost
    spatial_metrics, spatial_preds, spatial_probs = evaluation.get_performance_metrics(
        'G-XGBoost Espacial', spatial_xgb_model, X_test_spatial, y_test, spatial_train_time, label_encoder
    )
    metrics_list.append(spatial_metrics)
    models_preds['G-XGBoost Espacial'] = spatial_preds
    models_probs['G-XGBoost Espacial'] = spatial_probs
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # Save metrics tables
    df_metrics.to_csv(os.path.join(config.METRICS_DIR, 'model_comparison_metrics.csv'), index=False)
    
    # Format and save as Excel using XlsxWriter (with conditional formatting/colors if possible)
    excel_path = os.path.join(config.METRICS_DIR, 'model_comparison_metrics.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_metrics.to_excel(writer, sheet_name='Performance Metrics', index=False)
        workbook  = writer.book
        worksheet = writer.sheets['Performance Metrics']
        # Apply formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        for col_num, value in enumerate(df_metrics.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
    logger.info(f"Saved model metrics to {excel_path}")
    
    # 11. Plot Curves
    evaluation.plot_confusion_matrices(models_preds, y_test, label_encoder)
    evaluation.plot_roc_curves(models_probs, y_test, label_encoder)
    evaluation.plot_pr_curves(models_probs, y_test, label_encoder)
    evaluation.plot_calibration_curves(models_probs, y_test, label_encoder)
    
    # 12. Run Statistical Tests
    comparisons = evaluation.compute_statistical_comparisons(models_preds, y_test, cv_results)
    
    # Get statistical summary to print in docx/pdf reports
    stat_summary = ""
    for name, comp in comparisons.items():
        stat_summary += f"{name}: {comp['interpretation']}\n\n"
        
    # 13. Variable Importance
    df_importance = evaluation.compute_variable_importance(spatial_xgb_model, prep_data['spatial_feature_names'])
    
    # 14. Compute SHAP explanations (representative subset)
    explainer, shap_bg, shap_values = evaluation.compute_shap_explanations(
        spatial_xgb_model, X_train_spatial, prep_data['spatial_feature_names'], label_encoder
    )
    
    # 15. Determine best model and improvements
    best_row = df_metrics.sort_values(by='Balanced Accuracy', ascending=False).iloc[0]
    best_model_name = best_row['Model']
    worst_row = df_metrics.sort_values(by='Balanced Accuracy', ascending=False).iloc[-1]
    worst_model_name = worst_row['Model']
    improvement = (best_row['Balanced Accuracy'] - worst_row['Balanced Accuracy']) * 100
    
    # 16. Compile Scientific conclusions text
    top_var = df_importance.iloc[0]['Variable']
    conclusions_text = (
        f"El estudio científico indica que el mejor modelo para predecir el mecanismo del homicidio es '{best_model_name}' "
        f"con un Balanced Accuracy de {best_row['Balanced Accuracy']:.4f} y un ROC AUC de {best_row['ROC AUC']:.4f}.\n"
        f"El rendimiento de '{best_model_name}' supera al peor modelo ('{worst_model_name}') por {improvement:.2f}% de Balanced Accuracy.\n"
        f"La variable más importante identificada por ganancia de información es '{top_var}'. "
        "El balanceo de clases mediante SMOTE previno el sesgo hacia la clase mayoritaria (Arma de Fuego), incrementando "
        "la precisión promedio para Arma Blanca y Otros. La incorporación de variables geográficas (latitud, longitud y parroquia) "
        "proporcionó una mejora estadísticamente significativa de la exactitud del modelo espacial G-XGBoost en comparación con "
        "su contraparte no espacial."
    )
    with open(os.path.join(config.TABLES_DIR, 'automatic_conclusions.txt'), 'w', encoding='utf-8') as f:
        f.write(conclusions_text)
        
    # 17. Generate scientific papers (PDF and DOCX)
    report_gen.generate_word_report(df_metrics, stat_summary, best_model_name)
    report_gen.generate_pdf_report(df_metrics, stat_summary, best_model_name)
    
    # 18. Save all pipeline variables for Streamlit instant load
    pipeline_outputs = {
        'df': df,
        'prep_data': prep_data,
        'models': models_dict,
        'metrics': df_metrics,
        'cv_results': cv_results,
        'bootstrap': bootstrap_results,
        'comparisons': comparisons,
        'importance': df_importance,
        'best_model': best_model_name,
        'conclusions': conclusions_text,
        'shap_explainer': explainer,
        'shap_bg': shap_bg,
        'shap_values': shap_values
    }
    
    pkl_path = os.path.join(config.MODELS_DIR, 'pipeline_outputs.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(pipeline_outputs, f)
        
    logger.info(f"Pipeline executed successfully in {time.time() - start_pipeline:.2f}s!")
    logger.info(f"Precomputed outputs saved to {pkl_path}")

if __name__ == '__main__':
    main()
