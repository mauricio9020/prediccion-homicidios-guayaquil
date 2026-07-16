import time
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier
from src import config

logger = logging.getLogger(__name__)

def train_logistic_regression(X_train, y_train):
    """
    Trains a Multinomial Logistic Regression model.
    """
    logger.info("Training Multinomial Logistic Regression model...")
    model = LogisticRegression(
        solver='lbfgs', 
        max_iter=1000, 
        random_state=config.RANDOM_STATE
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"Logistic Regression trained in {train_time:.4f}s")
    return model, train_time

def optimize_xgboost(X_train, y_train):
    """
    Applies RandomizedSearchCV to optimize hyper-parameters for multiclass XGBoost.
    """
    logger.info("Optimizing XGBoost hyperparameters via RandomizedSearchCV...")
    
    # Simple search space to run in reasonable time
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(
        objective='multi:softprob', 
        num_class=3, 
        random_state=config.RANDOM_STATE,
        eval_metric='mlogloss'
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    
    search = RandomizedSearchCV(
        estimator=xgb, 
        param_distributions=param_dist, 
        n_iter=8, 
        cv=cv, 
        scoring='accuracy', 
        random_state=config.RANDOM_STATE, 
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    logger.info(f"XGBoost optimization complete. Best params: {search.best_params_}")
    return search.best_params_

def train_xgboost(X_train, y_train, best_params):
    """
    Trains a Multiclass XGBoost model using the best optimized parameters.
    """
    logger.info("Training Multiclass XGBoost model...")
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=config.RANDOM_STATE,
        eval_metric='mlogloss',
        **best_params
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"XGBoost trained in {train_time:.4f}s")
    return model, train_time

def train_spatial_g_xgboost(X_train, y_train, best_params):
    """
    Trains a G-XGBoost Espacial model (incorporating spatial/geographical features).
    """
    logger.info("Training Spatial G-XGBoost model...")
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=config.RANDOM_STATE,
        eval_metric='mlogloss',
        **best_params
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"Spatial G-XGBoost trained in {train_time:.4f}s")
    return model, train_time

def perform_cross_validation(models_dict, cv_data_dict):
    """
    Performs Stratified 10-Fold Cross-Validation for each model on their respective training sets.
    """
    logger.info("Performing Stratified 10-Fold Cross-Validation...")
    cv_results = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=config.RANDOM_STATE)
    
    for name, model in models_dict.items():
        X = cv_data_dict[name]['X']
        y = cv_data_dict[name]['y']
        
        # Calculate CV accuracy
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 95% Confidence Interval for mean score
        ci_lower = mean_score - 1.96 * (std_score / np.sqrt(len(scores)))
        ci_upper = mean_score + 1.96 * (std_score / np.sqrt(len(scores)))
        
        cv_results[name] = {
            'scores': scores.tolist(),
            'mean': mean_score,
            'std': std_score,
            'ci': (ci_lower, ci_upper)
        }
        logger.info(f"CV 10-Fold Accuracy for {name}: {mean_score:.4f} (+/- {std_score:.4f})")
        
    return cv_results

def run_bootstrap_validation(models_dict, X_test_dict, y_test, n_iterations=1000):
    """
    Performs bootstrap resampling (n_iterations) on the test set to compute 
    95% Confidence Intervals for Accuracy, Recall, Precision, F1, and Balanced Accuracy.
    """
    logger.info(f"Starting Bootstrap Validation ({n_iterations} iterations)...")
    np.random.seed(config.RANDOM_STATE)
    bootstrap_results = {}
    
    n_samples = len(y_test)
    
    for name, model in models_dict.items():
        X_test = X_test_dict[name]
        
        # Arrays to store metrics
        acc_s = np.zeros(n_iterations)
        rec_s = np.zeros(n_iterations)
        pre_s = np.zeros(n_iterations)
        f1_s = np.zeros(n_iterations)
        b_acc_s = np.zeros(n_iterations)
        
        for i in range(n_iterations):
            # Sample indices with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_res = X_test[indices]
            y_res = y_test[indices]
            
            # Predict
            preds = model.predict(X_res)
            
            # Compute macro-averaged metrics
            acc_s[i] = accuracy_score(y_res, preds)
            b_acc_s[i] = balanced_accuracy_score(y_res, preds)
            pre_s[i] = precision_score(y_res, preds, average='macro', zero_division=0)
            rec_s[i] = recall_score(y_res, preds, average='macro', zero_division=0)
            f1_s[i] = f1_score(y_res, preds, average='macro', zero_division=0)
            
        # Calculate 95% Confidence Intervals (2.5th and 97.5th percentiles)
        metrics = {
            'Accuracy': (acc_s, np.percentile(acc_s, 2.5), np.percentile(acc_s, 97.5)),
            'Balanced Accuracy': (b_acc_s, np.percentile(b_acc_s, 2.5), np.percentile(b_acc_s, 97.5)),
            'Precision (Macro)': (pre_s, np.percentile(pre_s, 2.5), np.percentile(pre_s, 97.5)),
            'Recall (Macro)': (rec_s, np.percentile(rec_s, 2.5), np.percentile(rec_s, 97.5)),
            'F1-Score (Macro)': (f1_s, np.percentile(f1_s, 2.5), np.percentile(f1_s, 97.5))
        }
        
        bootstrap_results[name] = {}
        for m_name, (scores, p_low, p_high) in metrics.items():
            bootstrap_results[name][m_name] = {
                'scores': scores.tolist(),
                'ci_lower': p_low,
                'ci_upper': p_high,
                'mean': np.mean(scores)
            }
            logger.info(f"{name} - Bootstrap {m_name}: Mean = {np.mean(scores):.4f}, 95% CI = [{p_low:.4f}, {p_high:.4f}]")
            
    return bootstrap_results
