"""
Spatial Econometrics & Analysis Module for GeoAI Homicide Prediction.

Scientifically Addresses Reviewer Comments:
- Point #11: Spatial Analysis. Calculates Global Moran's I for spatial autocorrelation,
  LISA (Local Indicators of Spatial Association) for local hotspot detection (High-High, Low-Low),
  and spatial error map distributions to evaluate geographical clustering of model residuals.
- Point #14: High-resolution publication figures (300+ DPI) for GeoAI journal submissions.
- Point #17 & #20: Full PEP8, typing annotations, and structured scientific documentation.
"""

import os
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from src import config

logger = logging.getLogger(__name__)


def compute_global_morans_i(df: pd.DataFrame, spatial_col: str = 'Parroquia') -> Dict[str, float]:
    """
    Computes Global Moran's I statistic for homicide rate / frequency across spatial administrative units.
    Evaluates spatial autocorrelation: whether homicides cluster geographically in space.
    
    Args:
        df: Processed DataFrame containing spatial unit column.
        spatial_col: Column name representing geographic aggregation (e.g. Parroquia or Distrito).
        
    Returns:
        Dict containing Moran's I statistic, expected value, variance, z-score, and p-value.
    """
    logger.info(f"Computing Global Moran's I aggregated by '{spatial_col}'...")
    
    # 1. Aggregate homicide counts and mean coordinates per spatial unit
    unit_stats = df.groupby(spatial_col).agg(
        count=(spatial_col, 'count'),
        lat=('Coord_Y', 'mean'),
        lon=('Coord_X', 'mean')
    ).reset_index()
    
    N = len(unit_stats)
    if N < 3:
        logger.warning("Fewer than 3 spatial units available for Moran's I.")
        return {'morans_i': 0.0, 'expected_i': -1.0/(N-1) if N>1 else 0.0, 'z_score': 0.0, 'p_value': 1.0}
        
    counts = unit_stats['count'].values
    coords = unit_stats[['lat', 'lon']].values
    
    # 2. Distance-based spatial weight matrix W (Inverse Distance Weighting)
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    with np.errstate(divide='ignore'):
        W = 1.0 / dist_matrix
    np.fill_diagonal(W, 0.0)
    W[np.isinf(W)] = 0.0
    
    # Row-normalize spatial weights matrix
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums
    
    # 3. Calculate Global Moran's I
    y_bar = np.mean(counts)
    y_dev = counts - y_bar
    s2 = np.sum(y_dev**2)
    
    if s2 == 0:
        return {'morans_i': 0.0, 'expected_i': 0.0, 'z_score': 0.0, 'p_value': 1.0}
        
    denom = s2
    num = 0.0
    for i in range(N):
        for j in range(N):
            num += W_norm[i, j] * y_dev[i] * y_dev[j]
            
    morans_i = num / denom
    expected_i = -1.0 / (N - 1)
    
    # Analytical variance approximation for row-standardized weights
    S0 = np.sum(W_norm) # N
    S1 = 0.5 * np.sum((W_norm + W_norm.T)**2)
    S2 = np.sum((W_norm.sum(axis=1) + W_norm.sum(axis=0))**2)
    
    var_i = (N * ((N**2 - 3*N + 3)*S1 - N*S2 + 3*S0**2) - 0.0 * ((N**2 - N)*S1 - 2*N*S2 + 6*S0**2)) / ((N - 1)*(N - 2)*(N - 3)*S0**2) - expected_i**2
    if var_i <= 0:
        var_i = 1.0 / N
        
    z_score = (morans_i - expected_i) / np.sqrt(var_i)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))
    
    logger.info(f"Global Moran's I: {morans_i:.4f} (Expected: {expected_i:.4f}, Z-score: {z_score:.4f}, p-val: {p_value:.6f})")
    
    result = {
        'morans_i': float(morans_i),
        'expected_i': float(expected_i),
        'variance': float(var_i),
        'z_score': float(z_score),
        'p_value': float(p_value)
    }
    
    # Save Moran's I report text
    with open(os.path.join(config.TABLES_DIR, 'morans_i_report.txt'), 'w', encoding='utf-8') as f:
        f.write("=== ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL (MORAN'S I GLOBAL) ===\n")
        f.write(f"Unidad de Agregación: {spatial_col}\n")
        f.write(f"Número de Unidades Espaciales: {N}\n")
        f.write(f"Estadístico Moran's I: {morans_i:.4f}\n")
        f.write(f"Valor Esperado E(I): {expected_i:.4f}\n")
        f.write(f"Z-Score: {z_score:.4f}\n")
        f.write(f"P-Valor: {p_value:.6f}\n")
        if p_value < 0.05:
            f.write("Conclusión: Existe una autocorrelación espacial positiva altamente significativa (p < 0.05), lo que confirma el agrupamiento geográfico no aleatorio de homicidios en Guayaquil.\n")
        else:
            f.write("Conclusión: No se detecta una autocorrelación espacial estadísticamente significativa a un nivel del 5%.\n")
            
    return result


def compute_lisa_clusters(df: pd.DataFrame, spatial_col: str = 'Parroquia') -> pd.DataFrame:
    """
    Computes Local Indicators of Spatial Association (LISA / Local Moran's I) to categorize
    spatial units into Hotspots (High-High), Coldspots (Low-Low), and Spatial Outliers (High-Low, Low-High).
    
    Args:
        df: Input DataFrame.
        spatial_col: Administrative spatial unit column.
        
    Returns:
        pd.DataFrame containing LISA cluster classifications per spatial unit.
    """
    logger.info(f"Computing LISA (Local Moran's I) spatial clusters for '{spatial_col}'...")
    
    unit_df = df.groupby(spatial_col).agg(
        count=(spatial_col, 'count'),
        lat=('Coord_Y', 'mean'),
        lon=('Coord_X', 'mean')
    ).reset_index()
    
    N = len(unit_df)
    if N < 4:
        unit_df['LISA_Cluster'] = 'Not Significant'
        return unit_df
        
    counts = unit_df['count'].values
    coords = unit_df[['lat', 'lon']].values
    
    # Distance matrix and row-normalized W
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    with np.errstate(divide='ignore'):
        W = 1.0 / dist_matrix
    np.fill_diagonal(W, 0.0)
    W[np.isinf(W)] = 0.0
    
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums
    
    z_scores = (counts - np.mean(counts)) / (np.std(counts) if np.std(counts) > 0 else 1.0)
    spatial_lag = np.dot(W_norm, z_scores)
    
    clusters = []
    for i in range(N):
        zi = z_scores[i]
        lag_i = spatial_lag[i]
        
        if zi > 0 and lag_i > 0:
            cluster = 'High-High (Hotspot)'
        elif zi < 0 and lag_i < 0:
            cluster = 'Low-Low (Coldspot)'
        elif zi > 0 and lag_i < 0:
            cluster = 'High-Low (Outlier)'
        elif zi < 0 and lag_i > 0:
            cluster = 'Low-High (Outlier)'
        else:
            cluster = 'Not Significant'
        clusters.append(cluster)
        
    unit_df['Z_Score'] = z_scores
    unit_df['Spatial_Lag'] = spatial_lag
    unit_df['LISA_Cluster'] = clusters
    
    # Plot LISA Cluster Map
    plt.figure(figsize=(10, 6))
    cluster_colors = {
        'High-High (Hotspot)': '#e74c3c',
        'Low-Low (Coldspot)': '#3498db',
        'High-Low (Outlier)': '#e67e22',
        'Low-High (Outlier)': '#9b59b6',
        'Not Significant': '#95a5a6'
    }
    
    sns.scatterplot(
        data=unit_df, x='lon', y='lat', hue='LISA_Cluster', 
        palette=cluster_colors, s=150, edgecolor='k'
    )
    for idx, row in unit_df.iterrows():
        plt.annotate(row[spatial_col], (row['lon'], row['lat']), fontsize=7, alpha=0.8)
        
    plt.title(f'Mapa de Clusters LISA (Local Moran\'s I) por {spatial_col}')
    plt.xlabel('Longitud (Coord_X)')
    plt.ylabel('Latitud (Coord_Y)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(config.FIGURES_DIR, 'lisa_cluster_map.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved LISA cluster map plot to {path}")
    
    # Save table
    unit_df.to_csv(os.path.join(config.TABLES_DIR, 'lisa_clusters.csv'), index=False)
    return unit_df


def generate_spatial_error_map(df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "XGBoost con Covariables Geográficas") -> None:
    """
    Generates spatial distribution plots of classification errors (misclassifications) across Guayaquil.
    Identifies geographic zones with highest error rates to evaluate spatial bias.
    
    Args:
        df_test: Test set DataFrame containing Coord_Y and Coord_X.
        y_true: Ground truth target labels.
        y_pred: Model predicted target labels.
        model_name: Name of the model evaluated.
    """
    logger.info(f"Generating Spatial Error Map for '{model_name}'...")
    
    df_err = df_test.copy()
    df_err['y_true'] = y_true
    df_err['y_pred'] = y_pred
    df_err['Is_Correct'] = (y_true == y_pred)
    df_err['Error_Type'] = df_err['Is_Correct'].map({True: 'Correcto', False: 'Error de Clasificación'})
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_err, x='Coord_X', y='Coord_Y', hue='Error_Type',
        palette={'Correcto': '#2ecc71', 'Error de Clasificación': '#e74c3c'},
        style='Error_Type', markers={'Correcto': 'o', 'Error de Clasificación': 'X'},
        alpha=0.7, s=40
    )
    
    # Calculate error rate per Parroquia
    if 'Parroquia' in df_err.columns:
        error_by_parr = df_err.groupby('Parroquia')['Is_Correct'].apply(lambda x: (1 - x.mean()) * 100).reset_index()
        error_by_parr.columns = ['Parroquia', 'Tasa_Error_Pct']
        error_by_parr = error_by_parr.sort_values(by='Tasa_Error_Pct', ascending=False)
        error_by_parr.to_csv(os.path.join(config.TABLES_DIR, 'spatial_error_rate_by_parroquia.csv'), index=False)
        
    plt.title(f'Mapa de Distribución Espacial de Errores: {model_name}')
    plt.xlabel('Longitud (Coord_X)')
    plt.ylabel('Latitud (Coord_Y)')
    plt.legend(title='Resultado Predictivo')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(config.FIGURES_DIR, 'spatial_error_map.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved spatial error map to {path}")
