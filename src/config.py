import os

"""
Configuration module for the Homicide Mechanism Prediction study in Guayaquil.
Refactored to meet international Q1 journal publication standards (IEEE, Elsevier, Springer, ISPRS, PLOS ONE).

Scientifically Addresses Reviewer Comments:
- Point #6: Renamed 'G-XGBoost' to 'XGBoost con Covariables Geográficas' to avoid false claims of creating a novel algorithm.
- Point #16: Added dedicated output structures (results, article_tables, metrics, figures) for automated paper generation.
- Point #19: Fixed RANDOM_STATE = 123 for full experimental reproducibility.
- Point #20: Full scientific documentation of parameters and feature groups.
"""

# Project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data folders
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Muertes_Violentas_Guayaquil_Nuevos_Solo_2024_2026.xlsx')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'homicidios_clean.csv')

# Output folders for high-resolution artifacts and paper tables
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
TABLES_DIR = os.path.join(BASE_DIR, 'outputs', 'tables')
METRICS_DIR = os.path.join(BASE_DIR, 'outputs', 'metrics')
ARTICLE_TABLES_DIR = os.path.join(BASE_DIR, 'outputs', 'article_tables')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure all output directories exist
for directory in [FIGURES_DIR, TABLES_DIR, METRICS_DIR, ARTICLE_TABLES_DIR, RESULTS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Fixed Random Seed for strict reproducibility
RANDOM_STATE = 123

# Target column definition
TARGET_COL_RAW = 'Arma'
TARGET_COL_CLEAN = 'Mecanismo'
TARGET_CLASSES = {
    'ARMA DE FUEGO': 'Arma de Fuego',
    'ARMA BLANCA': 'Arma Blanca',
    'OTROS': 'Otros',
    'ARMA CONTUNDENTE': 'Otros',
    'CONSTRICTORA': 'Otros'
}

# Standard nomenclature for models (Point #6: No false claims of a new algorithm)
MODEL_NAMES = {
    'MAJORITY': 'Baseline Clase Mayoritaria',
    'TERRITORIAL': 'Baseline Territorial (Parroquia)',
    'LOG_REG': 'Regresión Logística Multinomial',
    'RANDOM_FOREST': 'Random Forest',
    'EXTRA_TREES': 'Extra Trees',
    'XGB_BASE': 'XGBoost Base (No Espacial)',
    'XGB_SPATIAL': 'XGBoost con Covariables Geográficas' # Renamed from G-XGBoost Espacial
}

# Variable categories
NUM_COLS = ['Edad', 'Hora', 'Coord_Y', 'Coord_X']
CAT_COLS = ['Sexo', 'Genero', 'Estado_Civil', 'Instruccion', 'Antecedentes', 'Tipo_Lugar', 'Area_Hecho', 'Presunta_Motivacion', 'Parroquia', 'Distrito', 'Zona']

# Spatial grouping columns for Spatial Block Validation (Point #2: Prevent spatial data leakage)
SPATIAL_GROUP_COL = 'Parroquia'
SECONDARY_SPATIAL_GROUP_COL = 'Distrito'

# Standard feature sets
NON_SPATIAL_FEATURES = ['Edad', 'Hora', 'Anio_i', 'Mes_i', 'Sexo', 'Tipo_Lugar', 'Presunta_Motivacion']
SPATIAL_FEATURES = ['Edad', 'Hora', 'Anio_i', 'Mes_i', 'Sexo', 'Tipo_Lugar', 'Presunta_Motivacion', 'Coord_Y', 'Coord_X', 'Parroquia', 'Zona', 'Distrito']

# Feature groups for Ablation Study (Point #5)
ABLATION_GROUPS = {
    '1_Base_Demografica': ['Edad', 'Sexo'],
    '2_+Temporal': ['Edad', 'Sexo', 'Hora', 'Anio_i', 'Mes_i', 'Dia_Semana'],
    '3_+Motivacion': ['Edad', 'Sexo', 'Hora', 'Anio_i', 'Mes_i', 'Dia_Semana', 'Presunta_Motivacion'],
    '4_+Tipo_Lugar': ['Edad', 'Sexo', 'Hora', 'Anio_i', 'Mes_i', 'Dia_Semana', 'Presunta_Motivacion', 'Tipo_Lugar', 'Area_Hecho'],
    '5_+Coordenadas': ['Edad', 'Sexo', 'Hora', 'Anio_i', 'Mes_i', 'Dia_Semana', 'Presunta_Motivacion', 'Tipo_Lugar', 'Area_Hecho', 'Coord_Y', 'Coord_X'],
    '6_+Parroquia': ['Edad', 'Sexo', 'Hora', 'Anio_i', 'Mes_i', 'Dia_Semana', 'Presunta_Motivacion', 'Tipo_Lugar', 'Area_Hecho', 'Coord_Y', 'Coord_X', 'Parroquia'],
    '7_+Distrito_Zona': ['Edad', 'Sexo', 'Hora', 'Anio_i', 'Mes_i', 'Dia_Semana', 'Presunta_Motivacion', 'Tipo_Lugar', 'Area_Hecho', 'Coord_Y', 'Coord_X', 'Parroquia', 'Zona', 'Distrito']
}

