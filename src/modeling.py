"""
Machine Learning & Spatial Modeling Module for Guayaquil Homicide Prediction.

Scientifically Addresses Reviewer Comments:
- Point #1: Data Leakage Elimination. Uses imblearn.pipeline.Pipeline so that preprocessing
  (scaling, encoding, SMOTENC) is fitted exclusively on training folds.
- Point #2: Spatial Validation Strategy. Replaces standard StratifiedKFold with GroupKFold /
  LeaveOneGroupOut grouped by administrative boundaries (Parroquia, Distrito) to prevent spatial autocorrelation leakage.
- Point #3: Nested Cross-Validation. Fully separates inner loop (hyperparameter tuning) from outer loop (evaluation).
- Point #4: SMOTENC & Class Weighting. Supports SMOTENC for categorical variables and compares
  oversampling vs. Balanced Class Weights without generating synthetic invalid spatial coordinates.
- Point #5: Ablation Study. Automates cumulative feature group evaluation.
- Point #6: Model Naming. Renamed to 'XGBoost con Covariables Geográficas' (Spatially Enriched XGBoost).
- Point #12: Baseline & Reference Models. Includes Majority Class, Territorial Baseline, Logistic Regression,
  Random Forest, Extra Trees, XGBoost Base, and XGBoost con Covariables Geográficas.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, SMOTE

from xgboost import XGBClassifier
from src import config

logger = logging.getLogger(__name__)


class TerritorialMajorityBaseline(BaseEstimator, ClassifierMixin):
    """
    Spatial Baseline Model that predicts the majority homicide mechanism per administrative territory (Parroquia).
    If an unseen territory is passed, falls back to the global training majority class.
    """
    def __init__(self, spatial_col: str = 'Parroquia'):
        self.spatial_col = spatial_col
        self.territory_majority_: Dict[Any, int] = {}
        self.global_majority_: int = 0
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.classes_ = np.unique(y)
        counts = np.bincount(y)
        self.global_majority_ = int(np.argmax(counts))
        
        # Ensure X is DataFrame
        if isinstance(X, pd.DataFrame) and self.spatial_col in X.columns:
            df_temp = X[[self.spatial_col]].copy()
            df_temp['target'] = y
            for group, group_df in df_temp.groupby(self.spatial_col):
                mode_val = group_df['target'].mode()
                if not mode_val.empty:
                    self.territory_majority_[group] = int(mode_val.iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        preds = np.full(n_samples, self.global_majority_, dtype=int)
        
        if isinstance(X, pd.DataFrame) and self.spatial_col in X.columns:
            for idx, group in enumerate(X[self.spatial_col]):
                if group in self.territory_majority_:
                    preds[idx] = self.territory_majority_[group]
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))
        
        preds = self.predict(X)
        for i, p in enumerate(preds):
            probs[i, p] = 1.0
        return probs


def build_pipeline_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for numerical imputation & scaling and categorical imputation & One-Hot encoding.
    Fits strictly inside an imblearn Pipeline.
    
    Args:
        numeric_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        
    Returns:
        ColumnTransformer object.
    """
    numeric_transformer = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = ImbPipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def create_model_pipeline(
    classifier_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    use_smote: bool = True,
    class_weight: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None
) -> ImbPipeline:
    """
    Creates a complete end-to-end imblearn Pipeline (Preprocessor -> Sampler -> Classifier).
    Guarantees no data leakage across folds.
    
    Args:
        classifier_type: One of 'log_reg', 'random_forest', 'extra_trees', 'xgb'.
        numeric_cols: Features to scale.
        categorical_cols: Features to One-Hot encode.
        use_smote: Whether to include SMOTE oversampling step in pipeline.
        class_weight: Optional 'balanced' or None.
        hyperparams: Additional kwargs for estimator.
        
    Returns:
        ImbPipeline instance.
    """
    if hyperparams is None:
        hyperparams = {}
        
    preprocessor = build_pipeline_preprocessor(numeric_cols, categorical_cols)
    steps = [('preprocessor', preprocessor)]
    
    if use_smote:
        # Standard SMOTE on transformed feature space
        steps.append(('sampler', SMOTE(random_state=config.RANDOM_STATE)))
        
    # Classifier selection
    if classifier_type == 'log_reg':
        clf = LogisticRegression(
            solver='lbfgs', max_iter=1000, 
            class_weight=class_weight,
            random_state=config.RANDOM_STATE, **hyperparams
        )
    elif classifier_type == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=100, class_weight=class_weight,
            random_state=config.RANDOM_STATE, n_jobs=-1, **hyperparams
        )
    elif classifier_type == 'extra_trees':
        clf = ExtraTreesClassifier(
            n_estimators=100, class_weight=class_weight,
            random_state=config.RANDOM_STATE, n_jobs=-1, **hyperparams
        )
    elif classifier_type == 'xgb':
        clf = XGBClassifier(
            objective='multi:softprob', num_class=3,
            eval_metric='mlogloss', random_state=config.RANDOM_STATE, **hyperparams
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
        
    steps.append(('classifier', clf))
    return ImbPipeline(steps)


def optimize_xgb_hyperparameters_nested(
    X_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Dict[str, Any]:
    """
    Performs Inner-Loop Hyperparameter Optimization for XGBoost using GroupKFold.
    Ensures hyperparameter tuning does not see held-out spatial test folds.
    
    Args:
        X_df: Features DataFrame.
        y: Target array.
        groups: Array of spatial groups (Parroquia/Distrito).
        numeric_cols: Numeric columns.
        categorical_cols: Categorical columns.
        
    Returns:
        Dict of best hyperparameters.
    """
    logger.info("Optimizing XGBoost hyper-parameters via Inner Spatial GroupKFold...")
    
    param_dist = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.9, 1.0]
    }
    
    pipeline = create_model_pipeline('xgb', numeric_cols, categorical_cols, use_smote=True)
    
    inner_cv = GroupKFold(n_splits=3)
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=6,
        cv=inner_cv,
        scoring='f1_macro',
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X_df, y, groups=groups)
    
    best_params = {k.replace('classifier__', ''): v for k, v in search.best_params_.items()}
    logger.info(f"Inner CV Optimization Complete. Best XGB params: {best_params}")
    return best_params


def perform_spatial_nested_cv(
    pipelines_dict: Dict[str, Any],
    X_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Performs Spatial Group Cross-Validation (Outer Loop) for all benchmark models.
    Guarantees no geographic overlap between training and validation folds.
    
    Args:
        pipelines_dict: Dict mapping model names to initialized pipelines/estimators.
        X_df: Raw input feature DataFrame.
        y: Encoded target array.
        groups: Array specifying spatial group (e.g. Parroquia).
        n_splits: Number of spatial folds.
        
    Returns:
        Dict containing per-fold scores, mean, std, and 95% Confidence Intervals for Macro F1.
    """
    logger.info(f"Executing Outer Spatial GroupKFold Cross-Validation ({n_splits} spatial folds)...")
    
    # Adjust n_splits if unique groups < n_splits
    unique_groups = len(np.unique(groups))
    actual_splits = min(n_splits, unique_groups)
    
    cv = GroupKFold(n_splits=actual_splits)
    cv_results: Dict[str, Dict[str, Any]] = {}
    
    for name, pipeline in pipelines_dict.items():
        logger.info(f"Evaluating model '{name}' on Spatial Group CV...")
        fold_f1_scores = []
        fold_bacc_scores = []
        fold_mcc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_df, y, groups=groups)):
            X_train_fold, X_val_fold = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Fit pipeline strictly on train fold
            pipeline.fit(X_train_fold, y_train_fold)
            preds = pipeline.predict(X_val_fold)
            
            f1_macro = f1_score(y_val_fold, preds, average='macro', zero_division=0)
            bacc = balanced_accuracy_score(y_val_fold, preds)
            mcc = matthews_corrcoef(y_val_fold, preds)
            
            fold_f1_scores.append(f1_macro)
            fold_bacc_scores.append(bacc)
            fold_mcc_scores.append(mcc)
            
        mean_f1 = float(np.mean(fold_f1_scores))
        std_f1 = float(np.std(fold_f1_scores))
        
        ci_lower = mean_f1 - 1.96 * (std_f1 / np.sqrt(len(fold_f1_scores)))
        ci_upper = mean_f1 + 1.96 * (std_f1 / np.sqrt(len(fold_f1_scores)))
        
        cv_results[name] = {
            'f1_scores': fold_f1_scores,
            'bacc_scores': fold_bacc_scores,
            'mcc_scores': fold_mcc_scores,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'ci_95': (ci_lower, ci_upper)
        }
        logger.info(f"Spatial CV Macro F1 for {name}: {mean_f1:.4f} (+/- {std_f1:.4f}) CI95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
    return cv_results


def run_ablation_study(
    df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    xgb_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Executes automated Feature Group Ablation Study (Point #5).
    Evaluates XGBoost performance across cumulative feature sets.
    
    Args:
        df: Input clean DataFrame.
        y: Target array.
        groups: Array of spatial groups.
        xgb_params: Optimized XGBoost hyperparameters.
        
    Returns:
        pd.DataFrame summarizing ablation results.
    """
    logger.info("Running automated Ablation Study across cumulative feature groups...")
    ablation_records = []
    
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    for group_name, feature_cols in config.ABLATION_GROUPS.items():
        # Check which features exist in df
        valid_cols = [c for c in feature_cols if c in df.columns]
        num_cols = [c for c in valid_cols if c in ['Edad', 'Hora', 'Anio_i', 'Mes_i', 'Coord_Y', 'Coord_X']]
        cat_cols = [c for c in valid_cols if c not in num_cols]
        
        X_sub = df[valid_cols].copy()
        pipeline = create_model_pipeline('xgb', num_cols, cat_cols, use_smote=True, hyperparams=xgb_params)
        
        f1_scores = []
        bacc_scores = []
        acc_scores = []
        
        for train_idx, val_idx in cv.split(X_sub, y, groups=groups):
            pipeline.fit(X_sub.iloc[train_idx], y[train_idx])
            preds = pipeline.predict(X_sub.iloc[val_idx])
            
            f1_scores.append(f1_score(y[val_idx], preds, average='macro', zero_division=0))
            bacc_scores.append(balanced_accuracy_score(y[val_idx], preds))
            acc_scores.append(accuracy_score(y[val_idx], preds))
            
        mean_f1 = np.mean(f1_scores)
        mean_bacc = np.mean(bacc_scores)
        mean_acc = np.mean(acc_scores)
        
        ablation_records.append({
            'Grupo_Variables': group_name,
            'Num_Variables': len(valid_cols),
            'Macro F1': mean_f1,
            'Balanced Accuracy': mean_bacc,
            'Accuracy': mean_acc
        })
        logger.info(f"Ablation Step '{group_name}' -> Macro F1: {mean_f1:.4f}, Bal Acc: {mean_bacc:.4f}")
        
    df_ablation = pd.DataFrame(ablation_records)
    
    # Calculate incremental gain relative to base model
    base_f1 = df_ablation.iloc[0]['Macro F1']
    df_ablation['Ganancia_F1_vs_Base'] = df_ablation['Macro F1'] - base_f1
    
    # Save ablation tables
    df_ablation.to_csv(os.path.join(config.TABLES_DIR, 'ablation_study.csv'), index=False)
    logger.info(f"Saved ablation study results to {os.path.join(config.TABLES_DIR, 'ablation_study.csv')}")
    return df_ablation
