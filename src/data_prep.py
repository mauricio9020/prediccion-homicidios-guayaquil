import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from src import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names to avoid encoding issues and maps them to standard names.
    """
    logger.info("Cleaning column names...")
    new_cols = []
    for c in df.columns:
        c_lower = c.lower()
        if 'tipo muert' in c_lower:
            name = 'Tipo_Muerte'
        elif 'subzona' in c_lower:
            name = 'Subzona'
        elif 'zona' in c_lower and 'sub' not in c_lower:
            name = 'Zona'
        elif 'distrito' in c_lower:
            name = 'Distrito'
        elif 'subcircuito' in c_lower:
            name = 'Subcircuito'
        elif 'circuito' in c_lower:
            name = 'Circuito'
        elif 'provincia' in c_lower and 'c' in c_lower and 'd' in c_lower:
            name = 'Codigo_Provincia'
        elif 'provincia' in c_lower:
            name = 'Provincia'
        elif 'cant' in c_lower and 'c' in c_lower and 'd' in c_lower:
            name = 'Codigo_Canton'
        elif 'cant' in c_lower:
            name = 'Canton'
        elif 'coord. y' in c_lower and 'rev' not in c_lower:
            name = 'Coord_Y'
        elif 'coord. x' in c_lower and 'rev' not in c_lower:
            name = 'Coord_X'
        elif 'coord. y' in c_lower and 'rev' in c_lower:
            name = 'Coord_Y_Rev'
        elif 'coord. x' in c_lower and 'rev' in c_lower:
            name = 'Coord_X_Rev'
        elif 'area' in c_lower:
            name = 'Area_Hecho'
        elif 'lugar' in c_lower and 'tipo' not in c_lower:
            name = 'Lugar'
        elif 'lugar' in c_lower and 'tipo' in c_lower:
            name = 'Tipo_Lugar'
        elif 'fecha' in c_lower:
            name = 'Fecha_Infraccion'
        elif 'hora' in c_lower and 'inf' in c_lower:
            name = 'Hora_Infraccion'
        elif 'tipo arma' in c_lower:
            name = 'Tipo_Arma'
        elif 'arma' in c_lower:
            name = 'Arma'
        elif 'presun. motiva.' in c_lower and 'obser' not in c_lower:
            name = 'Presunta_Motivacion'
        elif 'presun. motiva.' in c_lower and 'obser' in c_lower:
            name = 'Presunta_Motivacion_Obs'
        elif 'causa' in c_lower:
            name = 'Probable_Causa'
        elif 'edad' in c_lower and 'rango' in c_lower:
            name = 'Rango_Edad'
        elif 'edad' in c_lower and 'med' in c_lower:
            name = 'Medida_Edad'
        elif 'edad' in c_lower:
            name = 'Edad'
        elif 'g' in c_lower and 'ner' in c_lower: # genero
            name = 'Genero'
        elif 'sexo' in c_lower:
            name = 'Sexo'
        elif 'etnia' in c_lower:
            name = 'Etnia'
        elif 'estado civil' in c_lower:
            name = 'Estado_Civil'
        elif 'nacionalidad' in c_lower:
            name = 'Nacionalidad'
        elif 'discapacidad' in c_lower:
            name = 'Discapacidad'
        elif 'prof reg civ' in c_lower:
            name = 'Profesion'
        elif 'instru' in c_lower:
            name = 'Instruccion'
        elif 'antecedentes' in c_lower:
            name = 'Antecedentes'
        elif 'anio' in c_lower:
            name = 'Anio_i'
        elif 'mes' in c_lower:
            name = 'Mes_i'
        elif 'hora' in c_lower:
            name = 'Hora'
        else:
            name = c.strip().replace(' ', '_')
        new_cols.append(name)
    df.columns = new_cols
    return df

def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Coord_Y and Coord_X from string/comma format to numeric floats.
    """
    logger.info("Cleaning coordinates...")
    for col in ['Coord_Y', 'Coord_X']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing or invalid coords
    # Impute missing lat/lon with Guayaquil defaults if any are missing
    guayaquil_lat_default = -2.1883
    guayaquil_lon_default = -79.9474
    df['Coord_Y'] = df['Coord_Y'].fillna(guayaquil_lat_default)
    df['Coord_X'] = df['Coord_X'].fillna(guayaquil_lon_default)
    
    return df

def clean_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Edad column. Converts text like 'SD' or invalid values to NaN and imputes them.
    """
    logger.info("Cleaning age...")
    if 'Edad' in df.columns:
        df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
        # Impute with median
        median_age = df['Edad'].median()
        if pd.isna(median_age):
            median_age = 30.0
        df['Edad'] = df['Edad'].fillna(median_age).astype(int)
    return df

def clean_antecedents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Antecedentes column.
    """
    logger.info("Cleaning antecedents...")
    if 'Antecedentes' in df.columns:
        df['Antecedentes'] = df['Antecedentes'].fillna('NO').astype(str).str.upper().str.strip()
        # standardizing to SI / NO
        df['Antecedentes'] = df['Antecedentes'].apply(lambda x: 'SI' if 'SI' in x or 'SÍ' in x else 'NO')
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies custom business logic mapping and creates geographic Parroquia.
    """
    logger.info("Applying feature engineering...")
    # 1. Target mapping: map raw 'Arma' values to clean mechanisms
    if 'Arma' in df.columns:
        df[config.TARGET_COL_CLEAN] = df['Arma'].map(config.TARGET_CLASSES).fillna('Otros')
    else:
        raise ValueError("Target column 'Arma' not found in raw data")
        
    # 2. Geographic mapping: Set Parroquia = Circuito
    if 'Circuito' in df.columns:
        df['Parroquia'] = df['Circuito']
    else:
        df['Parroquia'] = 'Guayaquil_Parroquia_Desconocida'
        
    # 3. Handle datetime parsing to ensure month and hour distributions are proper
    if 'Fecha_Infraccion' in df.columns:
        df['Fecha_Infraccion'] = pd.to_datetime(df['Fecha_Infraccion'], errors='coerce')
        # Fill missing dates with modern placeholder
        df['Fecha_Infraccion'] = df['Fecha_Infraccion'].fillna(pd.Timestamp('2024-01-01'))
        df['Dia_Semana'] = df['Fecha_Infraccion'].dt.day_name()
    else:
        df['Dia_Semana'] = 'Unknown'
        
    return df

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data, cleans columns, ages, coordinates, and saves processed data.
    """
    logger.info(f"Loading raw dataset from {file_path}")
    df = pd.read_excel(file_path)
    
    # 1. Clean columns
    df = clean_column_names(df)
    
    # 2. Clean numeric variables
    df = clean_coordinates(df)
    df = clean_age(df)
    df = clean_antecedents(df)
    
    # 3. Feature engineering
    df = feature_engineering(df)
    
    # 4. Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_len - len(df)} duplicates. Remaining: {len(df)}")
    
    # Save processed data
    df.to_csv(config.PROCESSED_DATA_PATH, index=False, encoding='utf-8')
    logger.info(f"Saved processed data to {config.PROCESSED_DATA_PATH}")
    
    return df

def get_preprocessor(numeric_cols, categorical_cols):
    """
    Creates an sklearn ColumnTransformer preprocessor with imputer, scaler and encoder.
    """
    numeric_transformer = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_cols)
        ],
        remainder='drop'
    )
    
    # Impute categorical variables with mode and then OneHotEncode
    categorical_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_pipeline', MinMaxScaler(), numeric_cols),
            ('cat_pipeline', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor

def preprocess_and_split(df: pd.DataFrame):
    """
    Extracts features, pre-processes, splits (70/30 stratified) and balances classes using SMOTE.
    """
    logger.info("Splitting and preprocessing features...")
    
    # Non-spatial features
    non_spatial_features = config.NON_SPATIAL_FEATURES
    # Spatial features
    spatial_features = non_spatial_features + config.SPATIAL_FEATURES
    
    y_raw = df[config.TARGET_COL_CLEAN]
    
    # Target Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # Class mapping review
    classes = list(label_encoder.classes_)
    logger.info(f"Mapped target classes: {classes} -> {list(range(len(classes)))}")
    
    # Numeric and categorical feature categorization
    num_features_base = [col for col in non_spatial_features if col in ['Edad', 'Hora']]
    cat_features_base = [col for col in non_spatial_features if col not in ['Edad', 'Hora']]
    
    num_features_spatial = [col for col in spatial_features if col in ['Edad', 'Hora', 'Coord_Y', 'Coord_X']]
    cat_features_spatial = [col for col in spatial_features if col not in ['Edad', 'Hora', 'Coord_Y', 'Coord_X']]
    
    # Column transformers
    preprocessor_base = get_preprocessor(num_features_base, cat_features_base)
    preprocessor_spatial = get_preprocessor(num_features_spatial, cat_features_spatial)
    
    # Prepare DataFrames for fit
    df_base = df[non_spatial_features].copy()
    df_spatial = df[spatial_features].copy()
    
    # Handle missing values prior to fit/transform
    for col in num_features_spatial:
        if col in df.columns:
            median_val = df[col].median()
            df_base[col] = df_base[col].fillna(median_val) if col in df_base.columns else None
            df_spatial[col] = df_spatial[col].fillna(median_val)
            
    for col in cat_features_spatial:
        if col in df.columns:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Desconocido'
            if col in df_base.columns:
                df_base[col] = df_base[col].fillna(mode_val)
            df_spatial[col] = df_spatial[col].fillna(mode_val)
            
    # Train/Test Split (70/30, stratified)
    X_train_df_base, X_test_df_base, y_train, y_test = train_test_split(
        df_base, y, test_size=0.3, stratify=y, random_state=config.RANDOM_STATE
    )
    
    # For spatial model
    X_train_df_spatial, X_test_df_spatial, _, _ = train_test_split(
        df_spatial, y, test_size=0.3, stratify=y, random_state=config.RANDOM_STATE
    )
    
    # Fit and transform
    X_train_base = preprocessor_base.fit_transform(X_train_df_base)
    X_test_base = preprocessor_base.transform(X_test_df_base)
    
    X_train_spatial = preprocessor_spatial.fit_transform(X_train_df_spatial)
    X_test_spatial = preprocessor_spatial.transform(X_test_df_spatial)
    
    # Get feature names after OneHotEncoding
    try:
        base_cat_names = preprocessor_base.named_transformers_['cat_pipeline'].get_feature_names_out(cat_features_base)
        base_feature_names = num_features_base + list(base_cat_names)
    except Exception as e:
        logger.warning("Could not extract feature names for base preprocessor: " + str(e))
        base_feature_names = [f"f_{i}" for i in range(X_train_base.shape[1])]
        
    try:
        spatial_cat_names = preprocessor_spatial.named_transformers_['cat_pipeline'].get_feature_names_out(cat_features_spatial)
        spatial_feature_names = num_features_spatial + list(spatial_cat_names)
    except Exception as e:
        logger.warning("Could not extract feature names for spatial preprocessor: " + str(e))
        spatial_feature_names = [f"f_{i}" for i in range(X_train_spatial.shape[1])]
    
    # Record class counts before SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    before_smote_counts = dict(zip(label_encoder.inverse_transform(unique), counts))
    logger.info(f"Class distribution before SMOTE: {before_smote_counts}")
    
    # Apply SMOTE to training sets
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_train_base_res, y_train_base_res = smote.fit_resample(X_train_base, y_train)
    X_train_spatial_res, y_train_spatial_res = smote.fit_resample(X_train_spatial, y_train)
    
    # Record class counts after SMOTE
    unique_res, counts_res = np.unique(y_train_base_res, return_counts=True)
    after_smote_counts = dict(zip(label_encoder.inverse_transform(unique_res), counts_res))
    logger.info(f"Class distribution after SMOTE: {after_smote_counts}")
    
    return {
        'X_train_base': X_train_base_res,
        'y_train_base': y_train_base_res,
        'X_test_base': X_test_base,
        'X_train_spatial': X_train_spatial_res,
        'y_train_spatial': y_train_spatial_res,
        'X_test_spatial': X_test_spatial,
        'y_test': y_test,
        'preprocessor_base': preprocessor_base,
        'preprocessor_spatial': preprocessor_spatial,
        'label_encoder': label_encoder,
        'base_feature_names': base_feature_names,
        'spatial_feature_names': spatial_feature_names,
        'before_smote_counts': before_smote_counts,
        'after_smote_counts': after_smote_counts,
        'X_test_df_spatial': X_test_df_spatial, # Keep raw spatial test frame for visualisations
        'X_train_df_spatial': X_train_df_spatial # Keep raw spatial train frame
    }
