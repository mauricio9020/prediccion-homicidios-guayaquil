import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score, log_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve
from scipy import stats
import shap
from src import config

logger = logging.getLogger(__name__)

def calculate_multiclass_specificity_sensitivity(y_true, y_pred, n_classes=3):
    """
    Computes sensitivity (recall) and specificity for each class,
    and returns their macro averages.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    sensitivities = []
    specificities = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = sum(sum(cm)) - tp - fp - fn
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sens)
        specificities.append(spec)
        
    return {
        'sensitivity_per_class': sensitivities,
        'specificity_per_class': specificities,
        'sensitivity_macro': np.mean(sensitivities),
        'specificity_macro': np.mean(specificities)
    }

def get_performance_metrics(name, model, X_test, y_test, train_time, label_encoder):
    """
    Computes all standard performance metrics for a model on test set.
    """
    logger.info(f"Computing metrics for model: {name}")
    
    # Measure prediction time
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    y_probs = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Precision, Recall, F1
    pre_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    pre_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Specificity and Sensitivity
    sens_spec = calculate_multiclass_specificity_sensitivity(y_test, y_pred, n_classes=len(label_encoder.classes_))
    spec_macro = sens_spec['specificity_macro']
    sens_macro = sens_spec['sensitivity_macro']
    
    # MCC and Cohen's Kappa
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # ROC AUC (One vs Rest)
    try:
        roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC for {name}: {e}")
        roc_auc = np.nan
        
    # Log Loss
    try:
        loss = log_loss(y_test, y_probs)
    except Exception as e:
        logger.warning(f"Could not compute Log Loss for {name}: {e}")
        loss = np.nan
        
    metrics = {
        'Model': name,
        'Accuracy': acc,
        'Balanced Accuracy': b_acc,
        'Precision (Macro)': pre_macro,
        'Precision (Weighted)': pre_weighted,
        'Recall (Macro)': rec_macro,
        'Recall (Weighted)': rec_weighted,
        'F1 (Macro)': f1_macro,
        'F1 (Weighted)': f1_weighted,
        'Sensitivity (Macro)': sens_macro,
        'Specificity (Macro)': spec_macro,
        'MCC': mcc,
        'Cohen Kappa': kappa,
        'ROC AUC': roc_auc,
        'Log Loss': loss,
        'Train Time (s)': train_time,
        'Prediction Time (s)': pred_time
    }
    
    return metrics, y_pred, y_probs

def plot_confusion_matrices(models_predictions, y_test, label_encoder):
    """
    Plots normal and normalized confusion matrix heatmaps for each model.
    """
    logger.info("Plotting confusion matrices...")
    classes = label_encoder.classes_
    n_models = len(models_predictions)
    
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for idx, (name, y_pred) in enumerate(models_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
        
        # Absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[idx, 0])
        axes[idx, 0].set_title(f'{name} - Matriz de Confusión (Conteos)')
        axes[idx, 0].set_ylabel('Real')
        axes[idx, 0].set_xlabel('Predicho')
        
        # Proportions
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges', xticklabels=classes, yticklabels=classes, ax=axes[idx, 1])
        axes[idx, 1].set_title(f'{name} - Matriz de Confusión (Normalizada)')
        axes[idx, 1].set_ylabel('Real')
        axes[idx, 1].set_xlabel('Predicho')
        
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices plot to {path}")

def plot_roc_curves(models_probs, y_test, label_encoder):
    """
    Plots multi-class One-vs-Rest ROC curves for all models.
    """
    logger.info("Plotting ROC curves...")
    classes = label_encoder.classes_
    n_classes = len(classes)
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    styles = ['-', '--', '-.']
    
    for model_idx, (name, probs) in enumerate(models_probs.items()):
        # Calculate ROC curve for each class vs Rest
        for i in range(n_classes):
            # Binarize labels
            y_test_bin = (y_test == i).astype(int)
            y_prob_class = probs[:, i]
            
            # Compute ROC
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test_bin, y_prob_class)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], linestyle=styles[model_idx],
                     label=f'{name} - {classes[i]} (AUC = {roc_auc:.2f})')
                     
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC Multiclase (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    path = os.path.join(config.FIGURES_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves to {path}")

def plot_pr_curves(models_probs, y_test, label_encoder):
    """
    Plots multi-class One-vs-Rest Precision-Recall curves for all models.
    """
    logger.info("Plotting Precision-Recall curves...")
    classes = label_encoder.classes_
    n_classes = len(classes)
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    styles = ['-', '--', '-.']
    
    for model_idx, (name, probs) in enumerate(models_probs.items()):
        for i in range(n_classes):
            y_test_bin = (y_test == i).astype(int)
            y_prob_class = probs[:, i]
            
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_test_bin, y_prob_class)
            ap = average_precision_score(y_test_bin, y_prob_class)
            
            plt.plot(recall, precision, color=colors[i], linestyle=styles[model_idx],
                     label=f'{name} - {classes[i]} (AP = {ap:.2f})')
                     
    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precision (Precisión)')
    plt.title('Curvas Precision-Recall Multiclase (One-vs-Rest)')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    path = os.path.join(config.FIGURES_DIR, 'pr_curves.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall curves to {path}")

def plot_calibration_curves(models_probs, y_test, label_encoder):
    """
    Plots Calibration Curves and calculates Brier scores for each class.
    """
    logger.info("Plotting calibration curves...")
    classes = label_encoder.classes_
    n_classes = len(classes)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))
    if n_classes == 1:
        axes = [axes]
        
    from sklearn.metrics import brier_score_loss
    
    colors = {'Regresión Logística': '#1f77b4', 'XGBoost': '#ff7f0e', 'G-XGBoost Espacial': '#2ca02c'}
    
    for i in range(n_classes):
        axes[i].plot([0, 1], [0, 1], "k:", label="Perfectamente Calibrado")
        y_test_bin = (y_test == i).astype(int)
        
        for name, probs in models_probs.items():
            prob_class = probs[:, i]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test_bin, prob_class, n_bins=10)
            
            brier = brier_score_loss(y_test_bin, prob_class)
            
            axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", color=colors.get(name, None),
                         label=f"{name} (Brier = {brier:.4f})")
                         
        axes[i].set_ylabel("Fracción de Positivos")
        axes[i].set_xlabel("Valor Medio Predicho")
        axes[i].set_title(f"Mecanismo: {classes[i]}")
        axes[i].legend(loc="lower right")
        axes[i].grid(alpha=0.3)
        
    plt.suptitle("Curvas de Calibración de Modelos por Clase")
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'calibration_curves.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved calibration curves to {path}")

def run_mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Performs McNemar's test for two sets of predictions.
    """
    # Contingency table
    # y_pred1 == y_true | y_pred2 == y_true
    # Yes / Yes -> Both correct
    # Yes / No -> Model 1 correct, Model 2 wrong
    # No / Yes -> Model 1 wrong, Model 2 correct
    # No / No -> Both wrong
    
    m1_correct = (y_pred1 == y_true)
    m2_correct = (y_pred2 == y_true)
    
    a = np.sum(m1_correct & m2_correct) # Both correct
    b = np.sum(m1_correct & ~m2_correct) # M1 correct, M2 wrong
    c = np.sum(~m1_correct & m2_correct) # M1 wrong, M2 correct
    d = np.sum(~m1_correct & ~m2_correct) # Both wrong
    
    # Calculate chi2 statistic with continuity correction
    if (b + c) > 0:
        stat = (abs(b - c) - 1)**2 / (b + c)
        p_val = stats.chi2.sf(stat, 1)
    else:
        stat = 0.0
        p_val = 1.0
        
    return {
        'contingency_table': [[a, b], [c, d]],
        'statistic': stat,
        'p_value': p_val,
        'm1_better_m2': b > c
    }

def run_wilcoxon_test(cv_scores1, cv_scores2):
    """
    Performs Wilcoxon signed-rank test on 10-fold cross validation scores.
    """
    diff = np.array(cv_scores1) - np.array(cv_scores2)
    if np.all(diff == 0):
        return {'statistic': 0.0, 'p_value': 1.0}
    stat, p_val = stats.wilcoxon(cv_scores1, cv_scores2)
    return {
        'statistic': stat,
        'p_value': p_val
    }

def compute_statistical_comparisons(models_predictions, y_test, cv_results):
    """
    Compares models statistically using McNemar and Wilcoxon tests.
    """
    logger.info("Computing statistical comparisons...")
    pairs = [
        ('Regresión Logística', 'XGBoost'),
        ('Regresión Logística', 'G-XGBoost Espacial'),
        ('XGBoost', 'G-XGBoost Espacial')
    ]
    
    comparisons = {}
    
    for m1, m2 in pairs:
        # McNemar (Test set predictions)
        y_pred1 = models_predictions[m1]
        y_pred2 = models_predictions[m2]
        mc_res = run_mcnemar_test(y_test, y_pred1, y_pred2)
        
        # Wilcoxon (CV scores)
        cv1 = cv_results[m1]['scores']
        cv2 = cv_results[m2]['scores']
        wx_res = run_wilcoxon_test(cv1, cv2)
        
        p_mc = mc_res['p_value']
        p_wx = wx_res['p_value']
        
        interpretation = ""
        if p_mc < 0.05:
            better_model = m1 if mc_res['m1_better_m2'] else m2
            interpretation += f"El test de McNemar indica una diferencia estadísticamente significativa (p = {p_mc:.4f}) en el rendimiento predictivo del conjunto de prueba. El modelo '{better_model}' superó al otro. "
        else:
            interpretation += f"El test de McNemar no muestra una diferencia estadísticamente significativa (p = {p_mc:.4f}) en el conjunto de prueba. "
            
        if p_wx < 0.05:
            interpretation += f"El test de Wilcoxon en validación cruzada (CV 10-Fold) confirma que la diferencia de desempeño es estadísticamente significativa (p = {p_wx:.4f})."
        else:
            interpretation += f"El test de Wilcoxon en validación cruzada indica que las diferencias en CV no son significativas a nivel del 5% (p = {p_wx:.4f})."
            
        comparisons[f"{m1} vs {m2}"] = {
            'mcnemar_p': p_mc,
            'mcnemar_stat': mc_res['statistic'],
            'wilcoxon_p': p_wx,
            'wilcoxon_stat': wx_res['statistic'],
            'interpretation': interpretation
        }
        logger.info(f"Comparison {m1} vs {m2}: McNemar p = {p_mc:.4f}, Wilcoxon p = {p_wx:.4f}")
        
    # Save statistical comparisons to a text file
    with open(os.path.join(config.TABLES_DIR, 'statistical_tests.txt'), 'w', encoding='utf-8') as f:
        for name, comp in comparisons.items():
            f.write(f"=== {name} ===\n")
            f.write(f"McNemar p-value: {comp['mcnemar_p']:.6f} (Stat: {comp['mcnemar_stat']:.4f})\n")
            f.write(f"Wilcoxon p-value: {comp['wilcoxon_p']:.6f} (Stat: {comp['wilcoxon_stat']:.4f})\n")
            f.write(f"Interpretación: {comp['interpretation']}\n\n")
            
    return comparisons

def compute_variable_importance(xgb_model, spatial_feature_names):
    """
    Computes and saves feature importances for G-XGBoost (Gain, Weight, Cover).
    """
    logger.info("Computing G-XGBoost variable importances...")
    booster = xgb_model.get_booster()
    
    # Get importance metrics
    importance_types = ['gain', 'weight', 'cover']
    importance_dfs = []
    
    for imp_t in importance_types:
        imp_scores = booster.get_score(importance_type=imp_t)
        # Map back to feature names (XGBoost uses f0, f1, etc if fit with matrix)
        # Check if imp_scores keys are integers or f0, f1 style
        mapped_scores = {}
        for k, v in imp_scores.items():
            try:
                idx = int(k.replace('f', ''))
                if idx < len(spatial_feature_names):
                    feat_name = spatial_feature_names[idx]
                    mapped_scores[feat_name] = v
            except ValueError:
                mapped_scores[k] = v
                
        df_imp = pd.DataFrame(list(mapped_scores.items()), columns=['Variable', imp_t.capitalize()])
        importance_dfs.append(df_imp)
        
    # Merge importances
    df_importance = importance_dfs[0]
    for df in importance_dfs[1:]:
        df_importance = pd.merge(df_importance, df, on='Variable', how='outer')
        
    df_importance = df_importance.fillna(0.0)
    
    # Add Ranking based on Gain (traditional main variable importance metric)
    df_importance = df_importance.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    df_importance['Ranking'] = df_importance.index + 1
    
    # Save top 20
    df_importance.head(20).to_csv(os.path.join(config.TABLES_DIR, 'feature_importance_top20.csv'), index=False)
    
    # Plot feature importance (Top 20 Gain)
    plt.figure(figsize=(10, 6))
    top20 = df_importance.head(20)
    sns.barplot(x='Gain', y='Variable', data=top20, hue='Variable', palette='viridis', legend=False)
    plt.title('Importancia de Variables (G-XGBoost - Top 20 por Gain)')
    plt.xlabel('Ganancia de Información (Gain)')
    plt.ylabel('Variable')
    plt.tight_layout()
    path = os.path.join(config.FIGURES_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to {path}")
    
    return df_importance

def compute_shap_explanations(xgb_model, X_train_spatial, spatial_feature_names, label_encoder):
    """
    Computes SHAP values for the G-XGBoost Espacial model and generates interpretability plots.
    To ensure speed, we run SHAP on a background sample of 100 observations.
    """
    logger.info("Computing SHAP explanations (representative subset)...")
    
    # Background dataset for SHAP (100 representative samples)
    np.random.seed(config.RANDOM_STATE)
    if X_train_spatial.shape[0] > 100:
        idx = np.random.choice(X_train_spatial.shape[0], 100, replace=False)
        shap_bg = X_train_spatial[idx]
    else:
        shap_bg = X_train_spatial
        
    # Fit SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(shap_bg)
    
    classes = label_encoder.classes_
    
    # In SHAP 0.45+, shap_values is a list for multi-class, or a single array with shape [n_samples, n_features, n_classes]
    # Let's handle both cases gracefully
    is_list = isinstance(shap_values, list)
    
    # Plot global summary for each class
    for class_idx, class_name in enumerate(classes):
        plt.figure(figsize=(10, 6))
        class_shap = shap_values[class_idx] if is_list else (shap_values[:, :, class_idx] if len(shap_values.shape) == 3 else shap_values)
        
        # Summary Beeswarm Plot
        shap.summary_plot(class_shap, shap_bg, feature_names=spatial_feature_names, show=False)
        plt.title(f'SHAP Beeswarm Plot - {class_name}')
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, f'shap_beeswarm_{class_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        
        # Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(class_shap, shap_bg, feature_names=spatial_feature_names, plot_type='bar', show=False)
        plt.title(f'SHAP Bar Plot - {class_name}')
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, f'shap_bar_{class_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        
    logger.info("Saved SHAP beeswarm and bar plots for all target classes")
    
    # Generate dependence plot for age and latitude
    # Find indices for Coord_Y (Latitude) and Edad
    try:
        lat_idx = spatial_feature_names.index('Coord_Y')
        class_shap_firearm = shap_values[0] if is_list else (shap_values[:, :, 0] if len(shap_values.shape) == 3 else shap_values)
        
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(lat_idx, class_shap_firearm, shap_bg, feature_names=spatial_feature_names, show=False)
        plt.title('SHAP Dependence Plot: Latitud (Coord_Y) vs Firearm Probability')
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, 'shap_dependence_latitude.png')
        plt.savefig(path, dpi=300)
        plt.close()
        logger.info(f"Saved SHAP dependence plot for Latitude to {path}")
    except Exception as e:
        logger.warning(f"Could not compute SHAP dependence plot: {e}")
        
    # Store background data, explainer and sample SHAP values for streamlit dashboard local prediction
    return explainer, shap_bg, shap_values
