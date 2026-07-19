"""
Evaluation, Calibration, Bootstrap & SHAP Interpretability Module for Guayaquil Homicide Study.

Scientifically Addresses Reviewer Comments:
- Point #7: Modern Metrics Hierarchy. Macro F1 is the primary evaluation metric, followed by Balanced Accuracy,
  Recall per class, Multiclass MCC, Macro Precision, ROC-AUC, PR-AUC, and Accuracy (secondary only).
- Point #8: Calibration Curves, Brier Score & ECE. Calculates per-class Brier Scores, Expected Calibration Error (ECE),
  and generates Reliability Diagrams.
- Point #9: Correct Bootstrap. Resamples test set (1000 iterations) reporting Mean, Median, Bias, IC95%, and Std Error.
- Point #10: Grouped SHAP Interpretability. TreeSHAP values are aggregated back into parent categorical variables
  to eliminate One-Hot dummy clutter. Includes Dependence Plots & explicit disclaimers against causal misinterpretation.
- Point #15: Correct Statistical Tests. McNemar test on paired test predictions, Wilcoxon test on spatial CV folds,
  and 95% Confidence Intervals for pairwise model metric differences.
- Point #17 & #20: Full PEP8, typing, logging, and scientific documentation.
"""

import os
import time
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    log_loss, confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve, auc
)
from sklearn.calibration import calibration_curve
from scipy import stats
import shap

from src import config

logger = logging.getLogger(__name__)


def calculate_ece(y_true_bin: np.ndarray, y_prob_class: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for a binary class probability output.
    ECE measures the average absolute difference between predicted confidence and empirical accuracy.
    
    Args:
        y_true_bin: Binary indicators (0 or 1) for target class.
        y_prob_class: Predicted probabilities for target class.
        n_bins: Number of equal-width bins.
        
    Returns:
        float representing Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true_bin)
    
    for j in range(n_bins):
        bin_lower = bin_boundaries[j]
        bin_upper = bin_boundaries[j + 1]
        
        # Mask for samples falling in bin j
        in_bin = (y_prob_class > bin_lower) & (y_prob_class <= bin_upper)
        if j == 0:
            in_bin = (y_prob_class >= bin_lower) & (y_prob_class <= bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_confidence = np.mean(y_prob_class[in_bin])
            avg_accuracy = np.mean(y_true_bin[in_bin])
            ece += (bin_size / total_samples) * np.abs(avg_confidence - avg_accuracy)
            
    return float(ece)


def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_encoder: Any) -> Dict[str, Any]:
    """
    Computes per-class recall, precision, and F1-score for each homicide mechanism category.
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        label_encoder: Fitted LabelEncoder instance.
        
    Returns:
        Dict mapping class names to per-class metrics.
    """
    classes = label_encoder.classes_
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    result = {}
    for i, c_name in enumerate(classes):
        result[f"Recall_{c_name}"] = float(per_class_recall[i])
        result[f"Precision_{c_name}"] = float(per_class_precision[i])
        result[f"F1_{c_name}"] = float(per_class_f1[i])
    return result


def get_performance_metrics(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_time: float,
    label_encoder: Any
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Computes all publication-grade performance metrics for a model on test set.
    Strictly orders metrics prioritizing Macro F1 (Point #7).
    
    Args:
        name: Name of the model evaluated.
        model: Fitted pipeline or estimator.
        X_test: Test feature DataFrame.
        y_test: Test target labels.
        train_time: Time taken to train model in seconds.
        label_encoder: LabelEncoder instance.
        
    Returns:
        Tuple of (metrics_dict, y_pred, y_probs).
    """
    logger.info(f"Computing publication metrics for model: {name}")
    
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)
    else:
        n_classes = len(label_encoder.classes_)
        y_probs = np.zeros((len(y_test), n_classes))
        for i, p in enumerate(y_pred):
            y_probs[i, p] = 1.0
            
    # Metrics hierarchy (Point #7: Macro F1 is Primary)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    pre_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    
    # ROC AUC (OvR Macro)
    try:
        roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = np.nan
        
    # PR AUC (OvR Macro Average Precision)
    try:
        y_test_bin = pd.get_dummies(y_test).values
        pr_auc = average_precision_score(y_test_bin, y_probs, average='macro')
    except Exception:
        pr_auc = np.nan
        
    acc = accuracy_score(y_test, y_pred)
    
    try:
        loss = log_loss(y_test, y_probs)
    except Exception:
        loss = np.nan
        
    per_class = calculate_per_class_metrics(y_test, y_pred, label_encoder)
    
    metrics = {
        'Modelo': name,
        'Macro F1 (Principal)': f1_macro,
        'Balanced Accuracy': b_acc,
        'Recall (Macro)': rec_macro,
        'MCC Multiclase': mcc,
        'Precision (Macro)': pre_macro,
        'ROC-AUC (OvR)': roc_auc,
        'PR-AUC (OvR)': pr_auc,
        'Accuracy (Secundario)': acc,
        'Log Loss': loss,
        'Tiempo Entrenamiento (s)': train_time,
        'Tiempo Predicción (s)': pred_time
    }
    
    metrics.update(per_class)
    return metrics, y_pred, y_probs


def plot_confusion_matrices(models_predictions: Dict[str, np.ndarray], y_test: np.ndarray, label_encoder: Any) -> None:
    """
    Plots high-resolution confusion matrix heatmaps (raw counts and normalized proportions).
    """
    logger.info("Plotting publication confusion matrices...")
    classes = label_encoder.classes_
    n_models = len(models_predictions)
    
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4.5 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for idx, (name, y_pred) in enumerate(models_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'{name} - Conteos Absolutos')
        axes[idx, 0].set_ylabel('Real')
        axes[idx, 0].set_xlabel('Predicho')
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges', xticklabels=classes, yticklabels=classes, ax=axes[idx, 1])
        axes[idx, 1].set_title(f'{name} - Proporción Normalizada')
        axes[idx, 1].set_ylabel('Real')
        axes[idx, 1].set_xlabel('Predicho')
        
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices to {path}")


def plot_calibration_curves(models_probs: Dict[str, np.ndarray], y_test: np.ndarray, label_encoder: Any) -> Dict[str, Dict[str, Any]]:
    """
    Computes per-class Brier Scores, Expected Calibration Error (ECE), and plots Calibration Curves / Reliability Diagrams.
    
    Args:
        models_probs: Dict mapping model name to test predicted probability matrix.
        y_test: Test ground truth labels.
        label_encoder: Fitted LabelEncoder.
        
    Returns:
        Dict mapping model name to per-class calibration statistics (Brier & ECE).
    """
    logger.info("Plotting calibration curves and computing Brier Scores & ECE...")
    classes = label_encoder.classes_
    n_classes = len(classes)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))
    if n_classes == 1:
        axes = [axes]
        
    calibration_metrics: Dict[str, Dict[str, Any]] = {name: {} for name in models_probs.keys()}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, c_name in enumerate(classes):
        axes[i].plot([0, 1], [0, 1], "k:", label="Perfectamente Calibrado")
        y_test_bin = (y_test == i).astype(int)
        
        for m_idx, (name, probs) in enumerate(models_probs.items()):
            prob_class = probs[:, i]
            frac_pos, mean_pred = calibration_curve(y_test_bin, prob_class, n_bins=10)
            
            brier = brier_score_loss(y_test_bin, prob_class)
            ece = calculate_ece(y_test_bin, prob_class, n_bins=10)
            
            calibration_metrics[name][f"{c_name}_Brier"] = float(brier)
            calibration_metrics[name][f"{c_name}_ECE"] = float(ece)
            
            color = colors[m_idx % len(colors)]
            axes[i].plot(mean_pred, frac_pos, "s-", color=color, label=f"{name} (Brier={brier:.3f}, ECE={ece:.3f})")
            
        axes[i].set_ylabel("Fracción de Positivos Reales")
        axes[i].set_xlabel("Valor Medio Predicho")
        axes[i].set_title(f"Calibración: {c_name}")
        axes[i].legend(loc="lower right", fontsize=8)
        axes[i].grid(alpha=0.3)
        
    plt.suptitle("Curvas de Calibración (Diagramas de Confiabilidad) por Clase", fontsize=14)
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'calibration_curves.png')
    plt.savefig(path, dpi=300)
    plt.close()
    
    calib_df = pd.DataFrame(calibration_metrics).T
    calib_df.to_csv(os.path.join(config.TABLES_DIR, 'brier_scores.csv'))
    logger.info(f"Saved calibration curves, Brier Scores & ECE to {path}")
    return calibration_metrics


def run_bootstrap_validation(
    models_dict: Dict[str, Any],
    X_test_dict: Dict[str, pd.DataFrame],
    y_test: np.ndarray,
    n_iterations: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Performs Bootstrap Validation (1000 resamples with replacement) on held-out test set (Point #9).
    Reports Mean, Median, Bias (Mean - Point Estimate), 95% Confidence Interval, and Std Error.
    
    Args:
        models_dict: Dict of fitted models/pipelines.
        X_test_dict: Dict of test DataFrames per model.
        y_test: Ground truth test target array.
        n_iterations: Number of bootstrap iterations.
        
    Returns:
        Dict of bootstrap statistical distributions per model and metric.
    """
    logger.info(f"Starting Bootstrap Validation ({n_iterations} iterations on test set)...")
    np.random.seed(config.RANDOM_STATE)
    bootstrap_results: Dict[str, Dict[str, Any]] = {}
    n_samples = len(y_test)
    
    for name, model in models_dict.items():
        X_test = X_test_dict[name]
        
        point_preds = model.predict(X_test)
        point_f1 = f1_score(y_test, point_preds, average='macro', zero_division=0)
        point_bacc = balanced_accuracy_score(y_test, point_preds)
        point_acc = accuracy_score(y_test, point_preds)
        
        f1_boot = np.zeros(n_iterations)
        bacc_boot = np.zeros(n_iterations)
        acc_boot = np.zeros(n_iterations)
        
        for i in range(n_iterations):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            if isinstance(X_test, pd.DataFrame):
                X_res = X_test.iloc[indices]
            else:
                X_res = X_test[indices]
            y_res = y_test[indices]
            
            preds = model.predict(X_res)
            f1_boot[i] = f1_score(y_res, preds, average='macro', zero_division=0)
            bacc_boot[i] = balanced_accuracy_score(y_res, preds)
            acc_boot[i] = accuracy_score(y_res, preds)
            
        metrics = {
            'Macro F1': (f1_boot, point_f1),
            'Balanced Accuracy': (bacc_boot, point_bacc),
            'Accuracy': (acc_boot, point_acc)
        }
        
        bootstrap_results[name] = {}
        for m_name, (scores, point_est) in metrics.items():
            mean_val = float(np.mean(scores))
            median_val = float(np.median(scores))
            bias = float(mean_val - point_est)
            std_err = float(np.std(scores))
            ci_lower = float(np.percentile(scores, 2.5))
            ci_upper = float(np.percentile(scores, 97.5))
            
            bootstrap_results[name][m_name] = {
                'scores': scores.tolist(),
                'point_estimate': point_est,
                'mean': mean_val,
                'median': median_val,
                'bias': bias,
                'std_error': std_err,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            logger.info(f"{name} - Bootstrap {m_name}: Mean={mean_val:.4f}, Median={median_val:.4f}, Bias={bias:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
            
    return bootstrap_results


def run_mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, Any]:
    """
    Performs McNemar's Test for paired nominal predictions on the test set.
    """
    m1_correct = (y_pred1 == y_true)
    m2_correct = (y_pred2 == y_true)
    
    a = np.sum(m1_correct & m2_correct)
    b = np.sum(m1_correct & ~m2_correct)
    c = np.sum(~m1_correct & m2_correct)
    d = np.sum(~m1_correct & ~m2_correct)
    
    if (b + c) > 0:
        stat = (abs(b - c) - 1.0)**2 / (b + c)
        p_val = float(stats.chi2.sf(stat, 1))
    else:
        stat = 0.0
        p_val = 1.0
        
    return {
        'contingency_table': [[int(a), int(b)], [int(c), int(d)]],
        'statistic': float(stat),
        'p_value': float(p_val),
        'm1_better': b > c
    }


def compute_statistical_comparisons(
    models_predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    cv_results: Dict[str, Dict[str, Any]],
    n_bootstrap_diff: int = 200
) -> Dict[str, Dict[str, Any]]:
    """
    Executes McNemar's Test (test set), Wilcoxon Signed-Rank Test (CV folds),
    and computes 95% Confidence Intervals for pairwise model metric differences Delta Macro F1 (Point #7 & #15).
    """
    logger.info("Computing statistical hypothesis tests & Confidence Intervals for model differences...")
    model_names = list(models_predictions.keys())
    comparisons = {}
    np.random.seed(config.RANDOM_STATE)
    n_samples = len(y_test)
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]
            
            y_pred1 = models_predictions[m1]
            y_pred2 = models_predictions[m2]
            
            # 1. McNemar Test
            mc_res = run_mcnemar_test(y_test, y_pred1, y_pred2)
            p_mc = mc_res['p_value']
            
            # 2. Wilcoxon Test on Spatial CV Folds
            if m1 in cv_results and m2 in cv_results:
                f1_scores1 = cv_results[m1]['f1_scores']
                f1_scores2 = cv_results[m2]['f1_scores']
                diff = np.array(f1_scores1) - np.array(f1_scores2)
                if np.all(diff == 0):
                    p_wx = 1.0
                    wx_stat = 0.0
                else:
                    wx_stat, p_wx = stats.wilcoxon(f1_scores1, f1_scores2)
                    wx_stat = float(wx_stat)
                    p_wx = float(p_wx)
            else:
                p_wx = 1.0
                wx_stat = 0.0
                
            # 3. 95% CI for Difference in Macro F1 (Delta F1 = F1_m1 - F1_m2)
            f1_m1_point = f1_score(y_test, y_pred1, average='macro', zero_division=0)
            f1_m2_point = f1_score(y_test, y_pred2, average='macro', zero_division=0)
            delta_point = f1_m1_point - f1_m2_point
            
            diff_boot = np.zeros(n_bootstrap_diff)
            for b_idx in range(n_bootstrap_diff):
                idx_res = np.random.choice(n_samples, size=n_samples, replace=True)
                f1_1_b = f1_score(y_test[idx_res], y_pred1[idx_res], average='macro', zero_division=0)
                f1_2_b = f1_score(y_test[idx_res], y_pred2[idx_res], average='macro', zero_division=0)
                diff_boot[b_idx] = f1_1_b - f1_2_b
                
            ci_delta_lower = float(np.percentile(diff_boot, 2.5))
            ci_delta_upper = float(np.percentile(diff_boot, 97.5))
            
            interp = f"Diferencia Macro F1: {delta_point:.4f} (IC95%: [{ci_delta_lower:.4f}, {ci_delta_upper:.4f}]). "
            interp += f"Test de McNemar (p = {p_mc:.4f}): "
            if p_mc < 0.05:
                better = m1 if mc_res['m1_better'] else m2
                interp += f"Diferencia significativa en test (p < 0.05). Modelo superior: '{better}'. "
            else:
                interp += "Sin diferencia significativa en el test. "
                
            interp += f"Test de Wilcoxon CV (p = {p_wx:.4f}): "
            if p_wx < 0.05:
                interp += "Diferencia significativa entre folds espaciales."
            else:
                interp += "Sin diferencia significativa en CV espacial."
                
            pair_key = f"{m1} vs {m2}"
            comparisons[pair_key] = {
                'delta_macro_f1': float(delta_point),
                'ci_delta_95': (ci_delta_lower, ci_delta_upper),
                'mcnemar_p': p_mc,
                'mcnemar_stat': mc_res['statistic'],
                'wilcoxon_p': p_wx,
                'wilcoxon_stat': wx_stat,
                'interpretation': interp
            }
            logger.info(f"Comparison '{pair_key}': Delta F1={delta_point:.4f} CI95%=[{ci_delta_lower:.4f}, {ci_delta_upper:.4f}], McNemar p={p_mc:.4f}")
            
    with open(os.path.join(config.TABLES_DIR, 'statistical_tests.txt'), 'w', encoding='utf-8') as f:
        f.write("=== PRUEBAS DE HIPÓTESIS ESTADÍSTICAS E INTERVALOS DE CONFIANZA ===\n\n")
        for pair_key, comp in comparisons.items():
            f.write(f"--- {pair_key} ---\n")
            f.write(f"Diferencia Macro F1 (Point): {comp['delta_macro_f1']:.4f}\n")
            f.write(f"IC95% de la Diferencia: [{comp['ci_delta_95'][0]:.4f}, {comp['ci_delta_95'][1]:.4f}]\n")
            f.write(f"McNemar p-value: {comp['mcnemar_p']:.6f} (Stat: {comp['mcnemar_stat']:.4f})\n")
            f.write(f"Wilcoxon p-value: {comp['wilcoxon_p']:.6f} (Stat: {comp['wilcoxon_stat']:.4f})\n")
            f.write(f"Interpretación: {comp['interpretation']}\n\n")
            
    return comparisons


def compute_grouped_shap_explanations(
    xgb_pipeline: Any,
    X_train_df: pd.DataFrame,
    label_encoder: Any
) -> Tuple[Any, np.ndarray, Any, List[str]]:
    """
    Computes TreeSHAP explanations for XGBoost, aggregating One-Hot dummy features back
    into their parent categorical variables (Point #10).
    Adds SHAP Dependence Plot and explicit disclaimers: SHAP represents marginal feature attribution, NOT causality.
    """
    logger.info("Computing grouped SHAP explanations (TreeSHAP)...")
    
    preprocessor = xgb_pipeline.named_steps['preprocessor']
    classifier = xgb_pipeline.named_steps['classifier']
    
    sample_size = min(100, len(X_train_df))
    np.random.seed(config.RANDOM_STATE)
    sample_df = X_train_df.sample(sample_size, random_state=config.RANDOM_STATE)
    
    X_trans = preprocessor.transform(sample_df)
    
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_trans)
    
    classes = label_encoder.classes_
    
    try:
        cat_cols = [c for c in sample_df.columns if c in config.CAT_COLS]
        num_cols = [c for c in sample_df.columns if c not in cat_cols]
        ohe = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = num_cols + cat_feature_names
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_trans.shape[1])]
        
    parent_map = {}
    for fname in feature_names:
        matched = False
        for original_col in config.SPATIAL_FEATURES:
            if fname.startswith(f"{original_col}_") or fname == original_col:
                parent_map[fname] = original_col
                matched = True
                break
        if not matched:
            parent_map[fname] = fname
            
    unique_parents = list(dict.fromkeys(parent_map.values()))
    is_list = isinstance(shap_values, list)
    
    for class_idx, class_name in enumerate(classes):
        class_shap = shap_values[class_idx] if is_list else (shap_values[:, :, class_idx] if len(shap_values.shape) == 3 else shap_values)
        
        grouped_shap = np.zeros((sample_size, len(unique_parents)))
        grouped_data = np.zeros((sample_size, len(unique_parents)))
        
        for p_idx, parent_col in enumerate(unique_parents):
            matching_indices = [i for i, fn in enumerate(feature_names) if parent_map[fn] == parent_col]
            grouped_shap[:, p_idx] = class_shap[:, matching_indices].sum(axis=1)
            grouped_data[:, p_idx] = X_trans[:, matching_indices].sum(axis=1)
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(grouped_shap, grouped_data, feature_names=unique_parents, show=False)
        plt.title(f'SHAP Beeswarm Plot (Variables Agrupadas) - {class_name}')
        plt.xlabel('Impacto SHAP (Contribución Marginal al Modelo - NO CAUSALIDAD)')
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, f'shap_beeswarm_{class_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(grouped_shap, grouped_data, feature_names=unique_parents, plot_type='bar', show=False)
        plt.title(f'SHAP Bar Plot (Importancia Promedio) - {class_name}')
        plt.xlabel('Magnitud Promedio de Impacto SHAP')
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, f'shap_bar_{class_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        
    # Generate dependence plot for Coord_Y if present
    if 'Coord_Y' in unique_parents:
        try:
            coord_idx = unique_parents.index('Coord_Y')
            class0_shap = shap_values[0] if is_list else (shap_values[:, :, 0] if len(shap_values.shape) == 3 else shap_values)
            grouped_shap0 = np.zeros((sample_size, len(unique_parents)))
            grouped_data0 = np.zeros((sample_size, len(unique_parents)))
            for p_idx, parent_col in enumerate(unique_parents):
                matching_indices = [i for i, fn in enumerate(feature_names) if parent_map[fn] == parent_col]
                grouped_shap0[:, p_idx] = class0_shap[:, matching_indices].sum(axis=1)
                grouped_data0[:, p_idx] = X_trans[:, matching_indices].sum(axis=1)
                
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(coord_idx, grouped_shap0, grouped_data0, feature_names=unique_parents, show=False)
            plt.title('SHAP Dependence Plot: Latitud (Coord_Y) vs Impacto Predicción')
            plt.tight_layout()
            path_dep = os.path.join(config.FIGURES_DIR, 'shap_dependence_latitude.png')
            plt.savefig(path_dep, dpi=300)
            plt.close()
            logger.info(f"Saved SHAP dependence plot to {path_dep}")
        except Exception as e:
            logger.warning(f"Could not compute SHAP dependence plot: {e}")
            
    logger.info("Saved grouped SHAP beeswarm, bar, and dependence plots for all classes.")
    return explainer, X_trans, shap_values, unique_parents
