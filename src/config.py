import os

# Project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data folders
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Muertes_Violentas_Guayaquil_Nuevos_Solo_2024_2026.xlsx')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'homicidios_clean.csv')

# Output folders
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
TABLES_DIR = os.path.join(BASE_DIR, 'outputs', 'tables')
METRICS_DIR = os.path.join(BASE_DIR, 'outputs', 'metrics')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure directories exist
for directory in [FIGURES_DIR, TABLES_DIR, METRICS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Random seed
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

# Variable categories
NUM_COLS = ['Edad', 'Hora']
CAT_COLS = ['Sexo', 'Genero', 'Estado_Civil', 'Instruccion', 'Antecedentes', 'Tipo_Lugar', 'Area_Hecho', 'Presunta_Motivacion']
# Standard features used for base model (non-spatial)
NON_SPATIAL_FEATURES = ['Edad', 'Hora', 'Anio_i', 'Mes_i', 'Sexo', 'Tipo_Lugar', 'Presunta_Motivacion']
# Spatial features
SPATIAL_FEATURES = ['Coord_Y', 'Coord_X', 'Parroquia', 'Zona', 'Distrito']
